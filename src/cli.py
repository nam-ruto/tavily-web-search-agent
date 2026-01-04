from __future__ import annotations

import argparse
import logging

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

from pipeline import run_pipeline

console = Console()


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    # Redirect logs to a file to keep the console clean for Rich
    logging.basicConfig(
        level=numeric_level,
        filename="tavily_agent.log",
        filemode="a",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Optional: still log critical errors to stderr if needed, 
    # but normally we want the Rich UI to handle the display.


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iterative web-retrieval RAG demo powered by Tavily."
    )
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Question to ask. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=3,
        help="Maximum number of search / refinement iterations (default: 3).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma3:latest",
        help="Ollama model to use for evaluation and answering (default: gemma3:latest).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )

    args = parser.parse_args()
    _configure_logging(args.log_level)

    if args.question:
        question = args.question
    else:
        question = Prompt.ask("[bold green]Enter your question[/bold green]").strip()

    if not question:
        console.print("[red]No question provided; aborting.[/red]")
        return

    answer_text = run_pipeline(question=question, max_iters=args.max_iters, model=args.model)
    
    console.print("\n" + "=" * 40)
    console.print(Panel(Markdown(answer_text), title="[bold green]FINAL ANSWER[/bold green]", border_style="green"))
    console.print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
