from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Metadata = dict[str, Any]


@dataclass(frozen=True)
class Chunk:
    """A contiguous raw-text span plus budget and provenance metadata."""

    chunk_id: str
    text: str
    token_count: int
    start_char: int | None = None
    end_char: int | None = None
    breadcrumbs: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    """A chunk with retrieval, risk, and optional reranker scores."""

    chunk: Chunk
    retrieval_score: float = 0.0
    risk_score: float = 0.0
    rerank_score: float | None = None
    source: str = "unknown"
    scores: Metadata = field(default_factory=dict)

    @property
    def utility(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.retrieval_score


@dataclass(frozen=True)
class CompressionResult:
    """Final selected context and audit metadata."""

    query: str
    compressed_context: str
    selected_chunks: tuple[Chunk, ...]
    dropped_chunks: tuple[Chunk, ...] = ()
    metrics: Metadata = field(default_factory=dict)
