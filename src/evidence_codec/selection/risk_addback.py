from __future__ import annotations

from evidence_codec.core.types import Candidate


def risk_add_back(
    selected: list[Candidate],
    dropped: list[Candidate],
    total_budget_tokens: int,
    addback_fraction: float = 0.10,
    relevance_threshold: float = 0.35,
    risk_threshold: float = 0.55,
) -> list[Candidate]:
    """Add back moderately relevant, high-risk chunks within add-back budget."""

    selected_ids = {candidate.chunk.chunk_id for candidate in selected}
    used = sum(candidate.chunk.token_count for candidate in selected)
    max_addback_tokens = int(total_budget_tokens * addback_fraction)
    addback_used = 0

    eligible = [
        candidate
        for candidate in dropped
        if candidate.chunk.chunk_id not in selected_ids
        and candidate.utility >= relevance_threshold
        and candidate.risk_score >= risk_threshold
    ]
    eligible.sort(
        key=lambda item: item.risk_score / max(item.chunk.token_count, 1),
        reverse=True,
    )

    result = list(selected)
    for candidate in eligible:
        tokens = candidate.chunk.token_count
        if used + tokens > total_budget_tokens:
            continue
        if addback_used + tokens > max_addback_tokens:
            continue
        result.append(candidate)
        used += tokens
        addback_used += tokens
    return result


def risk_add_back_with_added(
    selected: list[Candidate],
    dropped: list[Candidate],
    total_budget_tokens: int,
    addback_fraction: float = 0.10,
    relevance_threshold: float = 0.35,
    risk_threshold: float = 0.55,
) -> tuple[list[Candidate], list[Candidate]]:
    """RiskGuard add-back returning both final selection and added chunks."""

    selected_ids = {candidate.chunk.chunk_id for candidate in selected}
    used = sum(candidate.chunk.token_count for candidate in selected)
    max_addback_tokens = int(total_budget_tokens * addback_fraction)
    addback_used = 0

    eligible = [
        candidate
        for candidate in dropped
        if candidate.chunk.chunk_id not in selected_ids
        and candidate.utility >= relevance_threshold
        and candidate.risk_score >= risk_threshold
    ]
    eligible.sort(
        key=lambda item: item.risk_score / max(item.chunk.token_count, 1),
        reverse=True,
    )

    result = list(selected)
    added: list[Candidate] = []
    for candidate in eligible:
        tokens = candidate.chunk.token_count
        if used + tokens > total_budget_tokens:
            continue
        if addback_used + tokens > max_addback_tokens:
            continue
        result.append(candidate)
        added.append(candidate)
        used += tokens
        addback_used += tokens
    return result, added
