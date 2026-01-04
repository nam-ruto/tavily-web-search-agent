from __future__ import annotations

import argparse
import logging

from pipeline import run_pipeline


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


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
        question = input("Enter your question: ").strip()

    if not question:
        print("No question provided; aborting.")
        return

    answer_text = run_pipeline(question=question, max_iters=args.max_iters)
    print()
    print("=" * 80)
    print("FINAL ANSWER:")
    print(answer_text)
    print("=" * 80)


if __name__ == "__main__":
    main()
