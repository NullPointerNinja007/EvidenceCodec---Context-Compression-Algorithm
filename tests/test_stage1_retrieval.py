from evidence_codec.core.config import RetrievalConfig
from evidence_codec.core.types import Chunk
from evidence_codec.retrieval import generate_stage1_candidates
from evidence_codec.retrieval.bm25 import bm25_scores


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, text=text, token_count=len(text.split()))


def test_bm25_ranks_exact_lexical_match_first() -> None:
    chunks = [
        _chunk("a", "The deadline is March 3."),
        _chunk("b", "Bananas are yellow."),
        _chunk("c", "The policy discusses eligibility."),
    ]
    scores = bm25_scores("What is the deadline?", chunks)
    assert scores[0] == max(scores)


def test_stage1_includes_risk_rescue() -> None:
    chunks = [
        _chunk("plain", "This section gives general background about forms."),
        _chunk("risk", "Students may submit late unless the assignment is final and after 2026."),
        _chunk("lexical", "The assignment policy describes submission rules."),
    ]
    result = generate_stage1_candidates(
        "What are the submission rules?",
        chunks,
        config=RetrievalConfig(bm25_top_k=1, dense_top_k=1, risk_top_k=1, candidate_cap=3),
    )
    candidate_ids = [candidate.chunk.chunk_id for candidate in result.candidates]
    assert "risk" in candidate_ids
    risk_candidate = next(candidate for candidate in result.candidates if candidate.chunk.chunk_id == "risk")
    assert "risk" in risk_candidate.scores["stage1_sources"]


def test_stage1_respects_candidate_cap_and_scores() -> None:
    chunks = [_chunk(f"c{i}", f"chunk {i} with token {i}") for i in range(10)]
    result = generate_stage1_candidates(
        "token 7",
        chunks,
        config=RetrievalConfig(bm25_top_k=10, dense_top_k=10, risk_top_k=10, candidate_cap=4),
    )
    assert len(result.candidates) == 4
    assert all("rrf" in candidate.scores for candidate in result.candidates)
