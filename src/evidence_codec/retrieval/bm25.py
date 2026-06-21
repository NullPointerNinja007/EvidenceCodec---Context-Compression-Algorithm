from __future__ import annotations

import re

from evidence_codec.core.types import Chunk

TOKEN_RE = re.compile(r"[A-Za-z0-9_.$%/-]+")


def tokenize_for_retrieval(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def bm25_scores(query: str, chunks: list[Chunk]) -> list[float]:
    if not chunks:
        return []
    from rank_bm25 import BM25Okapi

    corpus_tokens = [tokenize_for_retrieval(chunk.text) for chunk in chunks]
    query_tokens = tokenize_for_retrieval(query)
    if not query_tokens:
        return [0.0 for _ in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    return [float(score) for score in bm25.get_scores(query_tokens)]


def top_k_by_score(chunk_ids: list[str], scores: list[float], k: int) -> list[str]:
    ranked = sorted(zip(chunk_ids, scores, strict=False), key=lambda item: item[1], reverse=True)
    return [chunk_id for chunk_id, _ in ranked[: max(k, 0)]]
