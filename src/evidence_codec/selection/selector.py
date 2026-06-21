from __future__ import annotations

import math
from typing import Any

from evidence_codec.core.config import SelectionConfig
from evidence_codec.core.types import Candidate, Chunk
from evidence_codec.selection.budget import SelectionResult, budgeted_select
from evidence_codec.selection.risk_addback import risk_add_back_with_added


def select_budgeted_context(
    candidates: list[Candidate],
    config: SelectionConfig | None = None,
) -> SelectionResult:
    """Select raw-text chunks under budget and run RiskGuard add-back."""

    config = config or SelectionConfig()
    addback_budget = int(config.budget_tokens * max(config.risk_addback_fraction, 0.0))
    selector_budget = max(config.budget_tokens - addback_budget, 0)
    initially_selected, dropped = budgeted_select(
        candidates,
        budget_tokens=selector_budget,
        length_exponent=config.length_exponent,
        redundancy_penalty=config.redundancy_penalty,
    )
    final_selected, risk_added = risk_add_back_with_added(
        initially_selected,
        dropped,
        total_budget_tokens=config.budget_tokens,
        addback_fraction=config.risk_addback_fraction,
        relevance_threshold=config.relevance_threshold,
        risk_threshold=config.risk_threshold,
    )
    final_ids = {candidate.chunk.chunk_id for candidate in final_selected}
    final_dropped = [candidate for candidate in candidates if candidate.chunk.chunk_id not in final_ids]
    used_tokens = sum(candidate.chunk.token_count for candidate in final_selected)
    metrics = {
        "input_candidates": len(candidates),
        "selected_before_risk": len(initially_selected),
        "selected_after_risk": len(final_selected),
        "risk_added": len(risk_added),
        "used_tokens": used_tokens,
        "selector_budget_tokens": selector_budget,
        "risk_addback_budget_tokens": addback_budget,
        "budget_tokens": config.budget_tokens,
        "budget_utilization": used_tokens / config.budget_tokens if config.budget_tokens else 0.0,
    }
    return SelectionResult(
        selected=sort_selected_in_document_order(final_selected),
        dropped=final_dropped,
        risk_added=sort_selected_in_document_order(risk_added),
        used_tokens=used_tokens,
        budget_tokens=config.budget_tokens,
        metrics=metrics,
    )


def assemble_selected_context(selected: list[Candidate]) -> str:
    return "\n\n".join(candidate.chunk.text for candidate in sort_selected_in_document_order(selected))


def sort_selected_in_document_order(selected: list[Candidate]) -> list[Candidate]:
    return sorted(selected, key=lambda candidate: _chunk_order_key(candidate.chunk.chunk_id))


def candidates_from_records(chunk_record: dict, candidate_record: dict) -> list[Candidate]:
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunk_record["chunks"]}
    candidates: list[Candidate] = []
    for item in candidate_record["candidates"]:
        chunk_data = chunk_by_id[item["chunk_id"]]
        utility = _utility_from_candidate_item(item)
        candidates.append(
            Candidate(
                chunk=_chunk_from_dict(chunk_data),
                retrieval_score=utility,
                risk_score=float(item.get("risk_norm", item.get("risk_score", 0.0))),
                source="+".join(item.get("stage1_sources") or []),
                scores=item,
            )
        )
    return candidates


def selection_record_from_records(
    chunk_record: dict,
    candidate_record: dict,
    config: SelectionConfig | None = None,
) -> dict[str, Any]:
    config = config or SelectionConfig()
    candidates = candidates_from_records(chunk_record, candidate_record)
    result = select_budgeted_context(candidates, config=config)
    selected_ids = [candidate.chunk.chunk_id for candidate in result.selected]
    selected_set = set(selected_ids)
    gold_ids = set(chunk_record.get("gold_chunk_ids") or [])
    compressed_context = assemble_selected_context(result.selected)
    return {
        "example_id": chunk_record["example_id"],
        "dataset": chunk_record["dataset"],
        "subset": chunk_record.get("subset"),
        "split": chunk_record["split"],
        "query": chunk_record["query"],
        "selected_chunk_ids": selected_ids,
        "risk_added_chunk_ids": [candidate.chunk.chunk_id for candidate in result.risk_added],
        "dropped_chunk_ids": [candidate.chunk.chunk_id for candidate in result.dropped],
        "compressed_context": compressed_context,
        "gold_chunk_ids": list(gold_ids),
        "evidence_recall": len(gold_ids & selected_set) / len(gold_ids) if gold_ids else None,
        "all_gold_selected": gold_ids.issubset(selected_set) if gold_ids else None,
        "metrics": result.metrics,
    }


def _utility_from_candidate_item(item: dict) -> float:
    if item.get("rerank_prob") is not None:
        return float(item["rerank_prob"])
    if item.get("rerank_score") is not None:
        return _sigmoid(float(item["rerank_score"]))
    if item.get("hybrid_norm") is not None:
        return float(item["hybrid_norm"])
    if item.get("rrf_score") is not None:
        return float(item["rrf_score"])
    return 0.0


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def _chunk_from_dict(item: dict) -> Chunk:
    return Chunk(
        chunk_id=item["chunk_id"],
        text=item["text"],
        token_count=item["token_count"],
        start_char=item.get("start_char"),
        end_char=item.get("end_char"),
        breadcrumbs=tuple(item.get("breadcrumbs") or []),
        metadata=item.get("metadata") or {},
    )


def _chunk_order_key(chunk_id: str) -> tuple[str, int]:
    suffix = chunk_id.rsplit(":chunk:", 1)
    if len(suffix) == 2:
        try:
            return suffix[0], int(suffix[1])
        except ValueError:
            return suffix[0], 0
    return chunk_id, 0
