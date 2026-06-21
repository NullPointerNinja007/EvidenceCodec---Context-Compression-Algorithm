from evidence_codec.core.config import SelectionConfig
from evidence_codec.core.types import Candidate, Chunk
from evidence_codec.selection import select_budgeted_context, selection_record_from_records


def _candidate(
    chunk_id: str,
    text: str,
    tokens: int,
    utility: float,
    risk: float = 0.0,
) -> Candidate:
    return Candidate(
        chunk=Chunk(chunk_id=chunk_id, text=text, token_count=tokens),
        retrieval_score=utility,
        risk_score=risk,
        scores={"hybrid_norm": utility, "risk_norm": risk},
    )


def test_budgeted_selector_respects_budget() -> None:
    candidates = [
        _candidate("a:chunk:00000", "high utility", 60, 0.9),
        _candidate("a:chunk:00001", "medium utility", 60, 0.8),
        _candidate("a:chunk:00002", "low utility", 60, 0.1),
    ]
    result = select_budgeted_context(
        candidates,
        SelectionConfig(budget_tokens=120, risk_addback_fraction=0.0),
    )
    assert result.used_tokens <= 120
    assert len(result.selected) == 2


def test_riskguard_adds_high_risk_dropped_chunk_with_remaining_budget() -> None:
    candidates = [
        _candidate("a:chunk:00000", "generic overview", 40, 0.9, risk=0.0),
        _candidate("a:chunk:00001", "deadline is June 21 unless final", 20, 0.36, risk=0.9),
        _candidate("a:chunk:00002", "another overview", 20, 0.8, risk=0.0),
    ]
    result = select_budgeted_context(
        candidates,
        SelectionConfig(
            budget_tokens=80,
            risk_addback_fraction=0.25,
            relevance_threshold=0.35,
            risk_threshold=0.55,
        ),
    )
    assert "a:chunk:00001" in {candidate.chunk.chunk_id for candidate in result.risk_added}
    assert result.used_tokens <= 80


def test_selection_record_reports_gold_recall() -> None:
    chunk_record = {
        "example_id": "ex",
        "dataset": "hotpotqa",
        "split": "train",
        "query": "deadline?",
        "gold_chunk_ids": ["ex:chunk:00001"],
        "chunks": [
            {"chunk_id": "ex:chunk:00000", "text": "overview", "token_count": 80},
            {"chunk_id": "ex:chunk:00001", "text": "deadline is June 21", "token_count": 20},
        ],
    }
    candidate_record = {
        "candidates": [
            {
                "chunk_id": "ex:chunk:00000",
                "hybrid_norm": 0.9,
                "risk_norm": 0.0,
                "stage1_sources": ["bm25"],
            },
            {
                "chunk_id": "ex:chunk:00001",
                "hybrid_norm": 0.8,
                "risk_norm": 0.7,
                "stage1_sources": ["risk"],
            },
        ]
    }
    record = selection_record_from_records(
        chunk_record,
        candidate_record,
        SelectionConfig(budget_tokens=100),
    )
    assert record["evidence_recall"] == 1.0
    assert record["all_gold_selected"] is True
