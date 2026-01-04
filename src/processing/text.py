from __future__ import annotations

import hashlib
import logging
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from models import Document, Passage

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size_words: int = 500,
    overlap_words: int = 100,
) -> List[Passage]:
    """
    Chunk each document into overlapping passages of roughly `chunk_size_words`.
    """
    passages: List[Passage] = []
    step = max(chunk_size_words - overlap_words, 1)

    for doc in documents:
        words = doc.text.split()
        if not words:
            continue

        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size_words]
            # Skip very short chunks which are unlikely to be useful
            if len(chunk_words) < 50:
                continue
            text = " ".join(chunk_words)
            passage_id = f"{doc.url}#chunk-{start}"
            passages.append(
                Passage(
                    id=passage_id,
                    url=doc.url,
                    title=doc.title,
                    text=text,
                )
            )

    logger.info("Chunked %d documents into %d passages", len(documents), len(passages))
    return passages


def rank_passages(question: str, passages: List[Passage]) -> List[Passage]:
    """
    Rank passages by TF-IDF cosine similarity with the user's question.
    """
    if not passages:
        return []

    texts = [p.text for p in passages]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    doc_matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([question])

    # cosine_similarity returns shape (1, n_passages)
    scores = cosine_similarity(query_vec, doc_matrix)[0]

    for passage, score in zip(passages, scores):
        passage.rank_score = float(score)

    ranked = sorted(passages, key=lambda p: p.rank_score or 0.0, reverse=True)
    logger.info("Ranked %d passages by relevance to the question", len(ranked))
    return ranked


def _fingerprint_text(text: str) -> str:
    normalized = " ".join(text.split()).lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def select_top_passages(
    ranked_passages: List[Passage],
    max_passages: int = 12,
    max_total_chars: int = 8000,
) -> List[Passage]:
    """
    Apply deduplication and simple token/length budgeting to select final context passages.
    """
    selected: List[Passage] = []
    seen_hashes: set[str] = set()
    total_chars = 0

    for passage in ranked_passages:
        if len(selected) >= max_passages:
            break

        fp = _fingerprint_text(passage.text)
        if fp in seen_hashes:
            continue

        passage.fingerprint = fp
        projected_chars = total_chars + len(passage.text)
        if projected_chars > max_total_chars and selected:
            # budget exceeded; only allow the very first passage to exceed if nothing selected yet
            logger.debug(
                "Stopping selection due to max_total_chars limit (%d)", max_total_chars
            )
            break

        seen_hashes.add(fp)
        selected.append(passage)
        total_chars = projected_chars

    logger.info(
        "Selected %d passages after deduplication and budgeting (chars=%d)",
        len(selected),
        total_chars,
    )
    return selected
