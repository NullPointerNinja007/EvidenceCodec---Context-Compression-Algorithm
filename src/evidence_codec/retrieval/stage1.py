from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evidence_codec.core.config import RetrievalConfig
from evidence_codec.core.types import Candidate, Chunk
from evidence_codec.retrieval.bm25 import bm25_scores, top_k_by_score
from evidence_codec.retrieval.dense import DenseScorer, build_dense_scorer
from evidence_codec.retrieval.hybrid import reciprocal_rank_fusion
from evidence_codec.risk.features import normalize_scores, risk_prior


@dataclass(frozen=True)
class StageOneResult:
    candidates: list[Candidate]
    ranked_lists: dict[str, list[str]]
    metrics: dict[str, Any]


def generate_stage1_candidates(
    query: str,
    chunks: list[Chunk],
    config: RetrievalConfig | None = None,
    dense_scorer: DenseScorer | None = None,
) -> StageOneResult:
    """Generate high-recall candidates with BM25 + cosine + risk rescue."""

    config = config or RetrievalConfig()
    dense_scorer = dense_scorer or build_dense_scorer(
        backend=config.dense_backend,
        model_name=config.dense_model_name,
    )
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    bm25_raw = bm25_scores(query, chunks)
    dense_raw = dense_scorer.score(query, chunks)
    risk_raw = [risk_prior(chunk, query=query) for chunk in chunks]

    bm25_norm = normalize_scores(bm25_raw)
    dense_norm = normalize_scores(dense_raw)
    risk_norm = normalize_scores(risk_raw)
    hybrid_norm = [
        0.45 * sparse + 0.45 * dense + 0.10 * risk
        for sparse, dense, risk in zip(bm25_norm, dense_norm, risk_norm, strict=False)
    ]

    ranked_lists = {
        "bm25": top_k_by_score(chunk_ids, bm25_raw, config.bm25_top_k),
        "dense": top_k_by_score(chunk_ids, dense_raw, config.dense_top_k),
        "risk": top_k_by_score(chunk_ids, risk_raw, config.risk_top_k),
        "hybrid": top_k_by_score(chunk_ids, hybrid_norm, config.candidate_cap),
    }
    fused = reciprocal_rank_fusion(
        {
            "bm25": ranked_lists["bm25"],
            "dense": ranked_lists["dense"],
            "risk": ranked_lists["risk"],
        },
        weights={
            "bm25": config.bm25_weight,
            "dense": config.dense_weight,
            "risk": config.risk_weight,
        },
        c=config.rrf_c,
        limit=config.candidate_cap,
    )
    score_by_id = {
        chunk_id: {
            "bm25": bm25_raw[index],
            "bm25_norm": bm25_norm[index],
            "dense": dense_raw[index],
            "dense_norm": dense_norm[index],
            "risk": risk_raw[index],
            "risk_norm": risk_norm[index],
            "hybrid_norm": hybrid_norm[index],
        }
        for index, chunk_id in enumerate(chunk_ids)
    }
    candidates: list[Candidate] = []
    for rank, (chunk_id, rrf_score) in enumerate(fused, start=1):
        source_membership = [
            source for source in ("bm25", "dense", "risk") if chunk_id in ranked_lists[source]
        ]
        scores = {
            **score_by_id[chunk_id],
            "rrf": rrf_score,
            "stage1_rank": rank,
            "stage1_sources": source_membership,
        }
        candidates.append(
            Candidate(
                chunk=chunk_by_id[chunk_id],
                retrieval_score=rrf_score,
                risk_score=score_by_id[chunk_id]["risk_norm"],
                source="+".join(source_membership) or "stage1",
                scores=scores,
            )
        )

    metrics = {
        "input_chunks": len(chunks),
        "candidate_count": len(candidates),
        "bm25_count": len(ranked_lists["bm25"]),
        "dense_count": len(ranked_lists["dense"]),
        "risk_count": len(ranked_lists["risk"]),
        "candidate_cap": config.candidate_cap,
    }
    return StageOneResult(candidates=candidates, ranked_lists=ranked_lists, metrics=metrics)
