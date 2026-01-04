from __future__ import annotations

import logging
from typing import List

from models import Passage
from services.ollama_service import get_answer_from_llm, chat_completion

logger = logging.getLogger(__name__)


def _shorten(text: str, max_chars: int = 400) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


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


def answer(question: str, passages: List[Passage], model: str = "gemma3:latest") -> str:
    """
    Generate an answer using Ollama based on the retrieved passages.
    """
    if not passages:
        logger.info("No passages provided to answerer; returning fallback answer.")
        return (
            "I couldn't retrieve enough relevant information from the web to answer this "
            "question in a grounded way."
        )

    logger.info("Building answer from %d passages using Ollama (%s)", len(passages), model)

    # Build the context string
    context = build_context_pack(passages)
    
    try:
        # Generate answer from LLM
        answer_text = get_answer_from_llm(question, context, model=model)
        
        # Append sources for transparency
        sources_lines = [f"- [{i+1}] {p.url}" for i, p in enumerate(passages)]
        sources_block = "\n".join(sources_lines)
        
        return f"{answer_text}\n\n### Sources\n{sources_block}"
        
    except Exception as e:
        logger.error(f"Failed to get answer from Ollama: {e}")
        return "Sorry, I encountered an error while processing the answer with the LLM."
