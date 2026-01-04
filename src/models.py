from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List


@dataclass
class SearchResult:
    """Single Tavily search hit."""

    title: str
    url: str
    snippet: Optional[str] = None
    score: Optional[float] = None
    published_date: Optional[datetime] = None


@dataclass
class Document:
    """Fetched and cleaned web page."""

    url: str
    title: str
    text: str
    extracted_at: datetime
    word_count: int


@dataclass
class Passage:
    """Chunk of text taken from a document."""

    id: str
    url: str
    title: str
    text: str
    rank_score: Optional[float] = None
    fingerprint: Optional[str] = None  # used for deduplication


@dataclass
class EvaluationResult:
    """Heuristic evaluation of whether context is sufficient."""

    sufficient: bool
    rationale: str
    new_queries: List[str]
    missing_aspects: List[str]
