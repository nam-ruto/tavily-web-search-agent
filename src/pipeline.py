from __future__ import annotations

import logging
from typing import List, Dict

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.status import Status
from rich.markdown import Markdown

from agents.answerer import answer
from agents.evaluator import evaluate_context
from models import Document, Passage
from processing.text import chunk_documents, rank_passages, select_top_passages
from services.web_search import search_web, fetch_documents

logger = logging.getLogger(__name__)
console = Console()


def run_pipeline(question: str, max_iters: int = 3, model: str = "gemma3:latest") -> str:
    """
    Run the iterative web-retrieval RAG-style pipeline and return the final answer string.
    """
    logger.info("Starting pipeline for question: %s using model: %s", question, model)
    url_cache: Dict[str, Document] = {}

    original_question = question
    current_queries: List[str] = [question]

    selected_passages: List[Passage] = []

    for iteration in range(1, max_iters + 1):
        if not current_queries:
            logger.info("No more queries to run; stopping iterations.")
            break

        console.print(f"\n[bold blue]--- Iteration {iteration}/{max_iters} ---[/bold blue]")
        
        query_list = "\n".join([f"• {q}" for q in current_queries])
        console.print(Panel(query_list, title="[bold cyan]Queries this iteration[/]", border_style="cyan", expand=False))

        logger.info("=== Iteration %d/%d ===", iteration, max_iters)
        logger.info("Queries this iteration: %s", current_queries)

        # 1) Search the web and fetch documents for each query
        docs_before = len(url_cache)
        
        with Status(f"[bold green]Processing searches...", console=console) as status:
            for i, q in enumerate(current_queries, 1):
                status.update(f"[bold green]Searching ({i}/{len(current_queries)}): [cyan]{q}[/cyan]...")
                results = search_web(q, max_results=6)
                
                status.update(f"[bold green]Fetching results for: [cyan]{q}[/cyan]...")
                fetch_documents(results, url_cache=url_cache, progress_callback=status.update)
        
        docs_after = len(url_cache)
        new_docs = docs_after - docs_before
        console.print(f"[dim]Fetched {new_docs} new documents this iteration (total cached: {docs_after})[/dim]")

        # 2) Build passages from all known documents so far
        documents: List[Document] = list(url_cache.values())
        passages: List[Passage] = chunk_documents(documents)

        # 3) Rank passages relative to the original user question
        ranked_passages = rank_passages(original_question, passages)

        # 4) Apply dedupe + context budgeting
        selected_passages = select_top_passages(ranked_passages, max_passages=12, max_total_chars=8000)

        # 5) Evaluate whether context is sufficient
        with Status(f"[bold yellow]Evaluating context sufficiency with {model}...", console=console):
            evaluation = evaluate_context(original_question, selected_passages, model=model)
        
        logger.info("Evaluation: %s", evaluation.rationale)

        if evaluation.sufficient:
            console.print(Panel(
                f"[bold green]PASSED[/bold green]\n[dim]Rationale: {evaluation.rationale}[/dim]",
                title="Evaluation Result",
                border_style="green",
                expand=False
            ))
        else:
            msg = f"[bold red]NOT SUFFICIENT[/bold red]\n[dim]Rationale: {evaluation.rationale}[/dim]"
            if evaluation.new_queries:
                msg += "\n\n[bold]Next iteration queries:[/bold]"
                for rq in evaluation.new_queries:
                    msg += f"\n- {rq}"
            
            console.print(Panel(msg, title="Evaluation Result", border_style="red", expand=False))

        if evaluation.sufficient:
            logger.info(
                "Evaluator judged context sufficient on iteration %d; stopping search.",
                iteration,
            )
            break

        if iteration >= max_iters:
            logger.info("Reached maximum iterations (%d); stopping search.", max_iters)
            break

        current_queries = evaluation.new_queries
        if not current_queries:
            logger.info("Evaluator did not propose any new queries; stopping search.")
            break

    # --- Display Top Relevant Passages ---
    if selected_passages:
        table = Table(
            title="[bold yellow]Top Relevant Context Found[/bold yellow]",
            show_header=True,
            header_style="bold magenta",
            box=None,
            expand=True  # Expand to terminal width
        )
        table.add_column("Rank", justify="center", style="dim", width=4)
        table.add_column("Score", justify="right", style="green", width=8)
        table.add_column("Source", style="blue", overflow="ellipsis", max_width=40)
        table.add_column("Snippet", style="italic", ratio=1) # Give snippet the remaining space

        for i, p in enumerate(selected_passages[:3], 1):
            snippet = (p.text[:120] + "...") if len(p.text) > 120 else p.text
            table.add_row(
                str(i),
                f"{p.rank_score:.3f}" if p.rank_score else "N/A",
                p.url,
                snippet
            )
        console.print("\n")
        console.print(table)
        console.print("\n")

    # 6) Generate an answer using the final selected passages
    with Status(f"[bold magenta]Generating final answer with {model}...", console=console):
        final_answer = answer(original_question, selected_passages, model=model)
    
    logger.info("Pipeline finished.")
    return final_answer
