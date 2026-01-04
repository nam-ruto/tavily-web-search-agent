from __future__ import annotations

import logging
from typing import List, Dict

from agents.answerer import answer
from agents.evaluator import evaluate_context
from models import Document, Passage
from processing.text import chunk_documents, rank_passages, select_top_passages
from services.web_search import search_web, fetch_documents

logger = logging.getLogger(__name__)


def run_pipeline(question: str, max_iters: int = 3) -> str:
    """
    Run the iterative web-retrieval RAG-style pipeline and return the final answer string.
    """
    logger.info("Starting pipeline for question: %s", question)
    url_cache: Dict[str, Document] = {}

    original_question = question
    current_queries: List[str] = [question]

    selected_passages: List[Passage] = []

    for iteration in range(1, max_iters + 1):
        if not current_queries:
            logger.info("No more queries to run; stopping iterations.")
            break

        print()
        print(f"--- Iteration {iteration}/{max_iters} ---")
        print(f"Queries this iteration:")
        for q in current_queries:
            print(f"  - {q}")

        logger.info("=== Iteration %d/%d ===", iteration, max_iters)
        logger.info("Queries this iteration: %s", current_queries)

        # 1) Search the web and fetch documents for each query
        docs_before = len(url_cache)
        for q in current_queries:
            results = search_web(q, max_results=6)
            fetch_documents(results, url_cache=url_cache)
        docs_after = len(url_cache)
        new_docs = docs_after - docs_before
        print(f"Fetched {new_docs} new documents this iteration (total cached: {docs_after})")

        # 2) Build passages from all known documents so far
        documents: List[Document] = list(url_cache.values())
        passages: List[Passage] = chunk_documents(documents)

        # 3) Rank passages relative to the original user question
        ranked_passages = rank_passages(original_question, passages)

        # 4) Apply dedupe + context budgeting
        selected_passages = select_top_passages(ranked_passages, max_passages=12, max_total_chars=8000)

        # 5) Evaluate whether context is sufficient
        evaluation = evaluate_context(original_question, selected_passages)
        logger.info("Evaluation: %s", evaluation.rationale)

        if evaluation.sufficient:
            print("Evaluation result: PASSED")
            print(f"  Rationale: {evaluation.rationale}")
        else:
            print("Evaluation result: NOT SUFFICIENT")
            print(f"  Rationale: {evaluation.rationale}")
            if evaluation.new_queries:
                print("  Refined queries suggested for next iteration:")
                for rq in evaluation.new_queries:
                    print(f"    - {rq}")

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

    # 6) Generate an answer using the final selected passages
    final_answer = answer(original_question, selected_passages)
    logger.info("Pipeline finished.")
    return final_answer
