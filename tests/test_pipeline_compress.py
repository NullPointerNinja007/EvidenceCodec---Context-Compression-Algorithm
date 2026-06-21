from evidence_codec.core.config import ChunkingConfig, RetrievalConfig, SelectionConfig
from evidence_codec.core.types import Candidate, Chunk
from evidence_codec.pipeline import CompressionPipelineConfig, compress_context, rerank_candidates


class KeywordScorer:
    def score(self, query: str, candidates: list[Candidate], batch_size: int = 32) -> list[float]:
        return [4.0 if "final essay" in candidate.chunk.text else -4.0 for candidate in candidates]


def test_quality_pipeline_uses_injected_reranker_and_selects_context() -> None:
    context = "\n\n".join(
        [
            "The cafeteria serves lunch from eleven to one.",
            "Students may submit late work for homework.",
            "The final essay has no extension period.",
            "The library closes early on Friday.",
        ]
    )
    config = CompressionPipelineConfig(
        chunking=ChunkingConfig(soft_max_tokens=12, hard_max_tokens=30, overlap_tokens=0),
        retrieval=RetrievalConfig(bm25_top_k=4, dense_top_k=4, risk_top_k=4, candidate_cap=4),
        selection=SelectionConfig(budget_tokens=9, risk_addback_fraction=0.0),
        use_reranker=True,
    )
    result = compress_context(
        context=context,
        query="When can students submit late work?",
        document_id="policy",
        config=config,
        scorer=KeywordScorer(),
    )
    assert result.metrics["mode"] == "quality"
    assert result.metrics["reranker"] == "KeywordScorer"
    assert "final essay" in result.compressed_context
    assert result.selected_chunks
    assert result.metrics["used_tokens"] <= 9


def test_pipeline_can_run_retrieval_only_for_cpu_smoke() -> None:
    result = compress_context(
        context="The policy allows late work unless the assignment is final.",
        query="When is late work allowed?",
        config=CompressionPipelineConfig(
            retrieval=RetrievalConfig(candidate_cap=4),
            selection=SelectionConfig(budget_tokens=64),
            use_reranker=False,
        ),
    )
    assert result.metrics["mode"] == "retrieval_only"
    assert result.selected_chunks
    assert "policy allows late work" in result.compressed_context


def test_rerank_candidates_uses_probability_as_selection_utility() -> None:
    class FixedScorer:
        def score(self, query: str, candidates: list[Candidate], batch_size: int = 32) -> list[float]:
            return [-2.0, 2.0]

    candidates = [
        Candidate(chunk=Chunk(chunk_id="a", text="A", token_count=1), retrieval_score=0.9),
        Candidate(chunk=Chunk(chunk_id="b", text="B", token_count=1), retrieval_score=0.1),
    ]
    reranked = rerank_candidates("query", candidates, FixedScorer())
    assert [candidate.chunk.chunk_id for candidate in reranked] == ["b", "a"]
    assert 0.87 < reranked[0].retrieval_score < 0.89
    assert reranked[0].rerank_score is None
    assert reranked[0].scores["rerank_score"] == 2.0
