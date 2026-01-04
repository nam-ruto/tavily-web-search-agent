from __future__ import annotations

import logging
import json
from typing import List
from urllib.parse import urlparse

from models import Passage, EvaluationResult
from services.ollama_service import chat_completion

logger = logging.getLogger(__name__)


def evaluate_context(question: str, passages: List[Passage], model: str = "gemma3:latest") -> EvaluationResult:
    """
    Evaluate context sufficiency using Ollama.
    """
    if not passages:
        rationale = "No passages available yet; context clearly insufficient."
        logger.info("Context insufficient: %s", rationale)
        return EvaluationResult(
            sufficient=False,
            rationale=rationale,
            new_queries=_generate_refined_queries_heuristic(question, passages),
            missing_aspects=["No relevant documents retrieved"],
        )

    logger.info("Evaluating context sufficiency using Ollama (%s)", model)

    # Build context for evaluation
    context_lines = []
    for idx, p in enumerate(passages, start=1):
        context_lines.append(f"[{idx}] {p.title}: {p.text[:300]}...")
    context_str = "\n".join(context_lines)

    system_prompt = """
        You are an expert researcher evaluating search results. 
        Determine if the provided context is sufficient to answer the user's question accurately and comprehensively. 
        You MUST respond with a valid JSON object with the following structure:
        {
        "sufficient": boolean,
        "rationale": "a brief explanation of why the context is or isn't sufficient",
        "new_queries": ["a list of 2-3 specific search queries to fill gaps, if not sufficient"],
        "missing_aspects": ["list of what information is still missing"]
        }
        Keep new_queries focused and diverse.
        """

    user_prompt = f"Question: {question}\n\nContext:\n{context_str}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response_text = chat_completion(
            model=model,
            messages=messages,
            format="json"
        )
        
        logger.debug("Raw Ollama evaluation response: %r", response_text)
        
        # Parse JSON response
        data = json.loads(response_text)
        
        return EvaluationResult(
            sufficient=data.get("sufficient", False),
            rationale=data.get("rationale", "No rationale provided."),
            new_queries=data.get("new_queries", []),
            missing_aspects=data.get("missing_aspects", [])
        )

    except Exception as e:
        logger.error(f"Ollama evaluation failed: {e}. Raw response: {response_text if 'response_text' in locals() else 'N/A'}. Falling back to heuristic.")
        # Fallback to heuristic
        return _evaluate_heuristic(question, passages)


def _evaluate_heuristic(question: str, passages: List[Passage]) -> EvaluationResult:
    """
    Heuristic evaluator fallback.
    """
    top_score = passages[0].rank_score or 0.0
    unique_domains = {urlparse(p.url).netloc for p in passages}

    sufficient = (
        len(passages) >= 3
        and len(unique_domains) >= 2
        and top_score >= 0.22
    )

    if sufficient:
        return EvaluationResult(
            sufficient=True,
            rationale="Heuristic: sufficient passages and domains.",
            new_queries=[],
            missing_aspects=[]
        )
    
    return EvaluationResult(
        sufficient=False,
        rationale="Heuristic: insufficient passages, domains, or score.",
        new_queries=_generate_refined_queries_heuristic(question, passages),
        missing_aspects=["Heuristic check failed"]
    )


def _generate_refined_queries_heuristic(question: str, passages: List[Passage]) -> List[str]:
    """
    Generate refined queries using simple heuristics (renamed from original).
    """
    base = question.strip()
    refined = [f"{base} official documentation", f"{base} detailed explanation"]
    return refined
