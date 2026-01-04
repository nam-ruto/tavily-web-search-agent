from __future__ import annotations

import logging
from typing import List
from urllib.parse import urlparse

from models import Passage, EvaluationResult

logger = logging.getLogger(__name__)


SIMILARITY_THRESHOLD = 0.22
MIN_PASSAGES = 3
MIN_UNIQUE_DOMAINS = 2


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or url


def evaluate_context(question: str, passages: List[Passage]) -> EvaluationResult:
    """
    Heuristic evaluator, no LLM involved.

    Sufficient if:
      - at least 3 passages
      - at least 2 unique domains
      - top passage similarity score >= SIMILARITY_THRESHOLD
    """
    if not passages:
        rationale = "No passages available yet; context clearly insufficient."
        logger.info("Context insufficient: %s", rationale)
        return EvaluationResult(
            sufficient=False,
            rationale=rationale,
            new_queries=_generate_refined_queries(question, passages),
            missing_aspects=["No relevant documents retrieved"],
        )

    top_score = passages[0].rank_score or 0.0
    unique_domains = {_extract_domain(p.url) for p in passages}

    sufficient = (
        len(passages) >= MIN_PASSAGES
        and len(unique_domains) >= MIN_UNIQUE_DOMAINS
        and top_score >= SIMILARITY_THRESHOLD
    )

    if sufficient:
        rationale = (
            f"Sufficient context: {len(passages)} passages from "
            f"{len(unique_domains)} domains; top similarity={top_score:.3f}."
        )
        logger.info(rationale)
        return EvaluationResult(
            sufficient=True,
            rationale=rationale,
            new_queries=[],
            missing_aspects=[],
        )

    missing: List[str] = []
    if len(passages) < MIN_PASSAGES:
        missing.append(f"Need at least {MIN_PASSAGES} passages (have {len(passages)})")
    if len(unique_domains) < MIN_UNIQUE_DOMAINS:
        missing.append(
            f"Need at least {MIN_UNIQUE_DOMAINS} unique domains "
            f"(have {len(unique_domains)})"
        )
    if top_score < SIMILARITY_THRESHOLD:
        missing.append(
            f"Top passage similarity below threshold "
            f"({top_score:.3f} < {SIMILARITY_THRESHOLD:.2f})"
        )

    rationale = "Context likely insufficient: " + "; ".join(missing)
    logger.info(rationale)

    return EvaluationResult(
        sufficient=False,
        rationale=rationale,
        new_queries=_generate_refined_queries(question, passages),
        missing_aspects=missing,
    )


def _generate_refined_queries(question: str, passages: List[Passage]) -> List[str]:
    """
    Generate 2–4 refined queries using simple heuristics.
    """
    base = question.strip()
    refined: List[str] = [
        f"{base} official documentation",
        f"{base} specification",
        f"{base} 2025",
    ]

    # Optionally enrich with a few keywords from the top passages.
    if passages:
        top_text = " ".join(p.text for p in passages[:3])
        keywords = _extract_keywords_from_text(top_text, limit=4)
        if keywords:
            refined.append(f"{base} {' '.join(keywords)}")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_refined: List[str] = []
    for q in refined:
        if q not in seen:
            seen.add(q)
            unique_refined.append(q)

    logger.info("Generated %d refined queries", len(unique_refined))
    return unique_refined


def _extract_keywords_from_text(text: str, limit: int = 4) -> List[str]:
    """
    Very light-weight keyword extractor: pick distinct longer words.
    """
    words = text.split()
    candidates: List[str] = []
    seen: set[str] = set()

    for w in words:
        token = "".join(ch for ch in w.lower() if ch.isalnum())
        if len(token) <= 4:
            continue
        if token in seen:
            continue
        seen.add(token)
        candidates.append(token)
        if len(candidates) >= limit:
            break

    return candidates
