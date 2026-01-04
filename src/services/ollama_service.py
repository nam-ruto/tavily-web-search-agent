from __future__ import annotations

import logging
from typing import List, Optional
import ollama

logger = logging.getLogger(__name__)

def chat_completion(
    model: str,
    messages: List[dict],
    options: Optional[dict] = None,
    format: Optional[str] = None
) -> str:
    """
    Simple wrapper around ollama.chat.
    """
    try:
        logger.info(f"Calling Ollama chat with model: {model}")
        response = ollama.chat(
            model=model,
            messages=messages,
            options=options or {},
            format=format
        )
        return response['message']['content']
    except Exception as e:
        logger.error(f"Ollama chat failed: {e}")
        raise

def get_answer_from_llm(
    question: str,
    context: str,
    model: str = "gemma3:latest"
) -> str:
    """
    Generates an answer based on the provided context.
    """
    system_prompt = (
        "You are a helpful assistant that answers questions based on the provided context. "
        "Use the provided context passages (labeled [1], [2], etc.) to answer the user question. "
        "Cite your sources inline using the [n] format. "
        "If the context does not contain enough information, say so. "
        "Format your response using Markdown for better readability."
    )
    
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return chat_completion(model=model, messages=messages)
