from __future__ import annotations

import logging
from typing import List

from models import Passage

logger = logging.getLogger(__name__)


def _shorten(text: str, max_chars: int = 400) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _first_sentences(text: str, max_sentences: int = 2) -> str:
    # Very naive sentence splitting.
    parts = text.split(".")
    sentences = [p.strip() for p in parts if p.strip()]
    joined = ". ".join(sentences[:max_sentences])
    if joined and not joined.endswith("."):
        joined += "."
    return joined


def build_context_pack(passages: List[Passage]) -> str:
    """
    Build a human-readable 'context pack' string with numbered passages and URLs.
    """
    lines = []
    for idx, p in enumerate(passages, start=1):
        score_str = f"{p.rank_score:.3f}" if p.rank_score is not None else "n/a"
        lines.append(
            f"[{idx}] {p.title} ({p.url})  score={score_str}\n"
            f"{_shorten(p.text)}\n"
        )
    return "\n".join(lines)


def answer(question: str, passages: List[Passage]) -> str:
    """
    Placeholder answerer that summarizes top passages and returns citations.
    """
    if not passages:
        logger.info("No passages provided to answerer; returning fallback answer.")
        return (
            "I couldn't retrieve enough relevant information from the web to answer this "
            "question in a grounded way."
        )

    logger.info("Building answer from %d passages", len(passages))

    # Build a lightweight summary from the top passages.
    summary_points: List[str] = []
    for p in passages[:5]:
        snippet = _first_sentences(p.text, max_sentences=2)
        if snippet:
            summary_points.append(f"- {snippet}")

    sources_lines = [f"[{i+1}] {p.url}" for i, p in enumerate(passages)]

    answer_text = [
        f"Question: {question}",
        "",
        "Based on the retrieved web context, here is a synthesized answer:",
        "",
        *summary_points,
        "",
        "Sources:",
        *sources_lines,
    ]

    return "\n".join(answer_text)
