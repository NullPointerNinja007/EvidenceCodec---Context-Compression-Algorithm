from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from evidence_codec.chunking import chunk_text
from evidence_codec.core.config import ChunkingConfig, RerankerConfig, RetrievalConfig, SelectionConfig
from evidence_codec.core.types import Candidate, CompressionResult
from evidence_codec.rerank import CrossEncoderReranker
from evidence_codec.retrieval import generate_stage1_candidates
from evidence_codec.retrieval.dense import DenseScorer
from evidence_codec.risk import risk_prior
from evidence_codec.selection import assemble_selected_context, select_budgeted_context


class CandidateScorer(Protocol):
    def score(self, query: str, candidates: list[Candidate], batch_size: int = 32) -> list[float]:
        """Return one raw reranker logit per candidate."""


@dataclass(frozen=True)
class CompressionPipelineConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    rerank_batch_size: int = 32
    use_reranker: bool = True


def compress_context(
    context: str,
    query: str,
    *,
    document_id: str = "doc",
    config: CompressionPipelineConfig | None = None,
    scorer: CandidateScorer | None = None,
    dense_scorer: DenseScorer | None = None,
) -> CompressionResult:
    """Run quality-mode compression from raw context to selected raw-text context."""

    config = config or CompressionPipelineConfig()
    chunks = chunk_text(context, config.chunking, document_id=document_id)
    stage1 = generate_stage1_candidates(
        query=query,
        chunks=chunks,
        config=config.retrieval,
        dense_scorer=dense_scorer,
    )
    candidates = stage1.candidates
    reranker_name = "disabled"
    if config.use_reranker:
        scorer = scorer or CrossEncoderReranker(
            model_name_or_path=config.reranker.model_name,
            config=config.reranker,
        )
        candidates = rerank_candidates(
            query=query,
            candidates=candidates,
            scorer=scorer,
            batch_size=config.rerank_batch_size,
        )
        reranker_name = getattr(scorer, "model_name_or_path", scorer.__class__.__name__)

    selection = select_budgeted_context(candidates, config.selection)
    selected_chunks = tuple(candidate.chunk for candidate in selection.selected)
    dropped_chunks = tuple(candidate.chunk for candidate in selection.dropped)
    compressed_context = assemble_selected_context(selection.selected)
    return CompressionResult(
        query=query,
        compressed_context=compressed_context,
        selected_chunks=selected_chunks,
        dropped_chunks=dropped_chunks,
        metrics={
            "mode": "quality" if config.use_reranker else "retrieval_only",
            "document_id": document_id,
            "input_characters": len(context),
            "input_chunks": len(chunks),
            "candidate_count": len(candidates),
            "reranker": reranker_name,
            "rerank_batch_size": config.rerank_batch_size if config.use_reranker else 0,
            "selected_chunks": len(selected_chunks),
            "dropped_chunks": len(dropped_chunks),
            "risk_added_chunks": len(selection.risk_added),
            "used_tokens": selection.used_tokens,
            "budget_tokens": selection.budget_tokens,
            "budget_utilization": selection.metrics["budget_utilization"],
            "stage1": stage1.metrics,
            "selection": selection.metrics,
        },
    )


def rerank_candidates(
    query: str,
    candidates: list[Candidate],
    scorer: CandidateScorer,
    batch_size: int = 32,
) -> list[Candidate]:
    """Attach cross-encoder logits while exposing sigmoid probabilities as selection utility."""

    if not candidates:
        return []
    logits = scorer.score(query, candidates, batch_size=batch_size)
    if len(logits) != len(candidates):
        raise ValueError(f"Reranker returned {len(logits)} scores for {len(candidates)} candidates.")

    reranked: list[Candidate] = []
    for candidate, logit in zip(candidates, logits, strict=True):
        probability = _sigmoid(float(logit))
        scores = {
            **candidate.scores,
            "rerank_score": float(logit),
            "rerank_prob": probability,
        }
        reranked.append(
            Candidate(
                chunk=candidate.chunk,
                retrieval_score=probability,
                risk_score=candidate.risk_score,
                source=f"{candidate.source}+rerank" if candidate.source else "rerank",
                scores=scores,
            )
        )
    return sorted(reranked, key=lambda item: item.retrieval_score, reverse=True)


def assemble_prompt(query: str, candidates: list[Candidate]) -> str:
    context = assemble_selected_context(candidates)
    return f"Context:\n{context}\n\nQuestion:\n{query}"


def compress_with_scores(
    context: str,
    query: str,
    budget_tokens: int,
    candidate_scores: dict[str, float] | None = None,
) -> CompressionResult:
    """Fast deterministic compression helper for fallback and smoke tests."""

    chunks = chunk_text(context, ChunkingConfig())
    scores = candidate_scores or {}
    candidates = [
        Candidate(
            chunk=chunk,
            retrieval_score=scores.get(chunk.chunk_id, 0.0),
            risk_score=risk_prior(chunk, query=query),
            source="provided_scores",
        )
        for chunk in chunks
    ]
    selection = select_budgeted_context(candidates, SelectionConfig(budget_tokens=budget_tokens))
    selected_chunks = tuple(item.chunk for item in selection.selected)
    dropped_chunks = tuple(item.chunk for item in selection.dropped)
    return CompressionResult(
        query=query,
        compressed_context=assemble_selected_context(selection.selected),
        selected_chunks=selected_chunks,
        dropped_chunks=dropped_chunks,
        metrics={
            "input_chunks": len(chunks),
            "selected_chunks": len(selected_chunks),
            "budget_tokens": budget_tokens,
            "used_tokens": selection.used_tokens,
            "mode": "score_injected_skeleton",
        },
    )


def default_selection_config() -> SelectionConfig:
    return SelectionConfig()


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)
