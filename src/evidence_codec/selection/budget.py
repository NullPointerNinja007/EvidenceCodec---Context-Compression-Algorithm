from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evidence_codec.core.types import Candidate


@dataclass(frozen=True)
class SelectionResult:
    selected: list[Candidate]
    dropped: list[Candidate]
    risk_added: list[Candidate]
    used_tokens: int
    budget_tokens: int
    metrics: dict[str, Any]


def greedy_budget_select(
    candidates: list[Candidate],
    budget_tokens: int,
    length_exponent: float = 0.9,
) -> list[Candidate]:
    """Select candidates by utility per token under a hard budget."""

    selected: list[Candidate] = []
    used_tokens = 0
    ranked = sorted(
        candidates,
        key=lambda item: item.utility / max(item.chunk.token_count, 1) ** length_exponent,
        reverse=True,
    )
    for candidate in ranked:
        next_tokens = used_tokens + candidate.chunk.token_count
        if next_tokens <= budget_tokens:
            selected.append(candidate)
            used_tokens = next_tokens
    return selected


def budgeted_select(
    candidates: list[Candidate],
    budget_tokens: int,
    length_exponent: float = 0.9,
    redundancy_penalty: float = 0.15,
) -> tuple[list[Candidate], list[Candidate]]:
    """Greedy value-per-token selection with lightweight redundancy control."""

    selected: list[Candidate] = []
    remaining = list(candidates)
    used_tokens = 0
    while remaining:
        scored: list[tuple[float, Candidate]] = []
        for candidate in remaining:
            if used_tokens + candidate.chunk.token_count > budget_tokens:
                continue
            adjusted_utility = max(
                0.0,
                candidate.utility - redundancy_penalty * _max_similarity(candidate, selected),
            )
            priority = adjusted_utility / max(candidate.chunk.token_count, 1) ** length_exponent
            scored.append((priority, candidate))
        if not scored:
            break
        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0][1]
        selected.append(best)
        used_tokens += best.chunk.token_count
        remaining = [candidate for candidate in remaining if candidate.chunk.chunk_id != best.chunk.chunk_id]

    selected_ids = {candidate.chunk.chunk_id for candidate in selected}
    dropped = [candidate for candidate in candidates if candidate.chunk.chunk_id not in selected_ids]
    return selected, dropped


def _max_similarity(candidate: Candidate, selected: list[Candidate]) -> float:
    if not selected:
        return 0.0
    candidate_tokens = _content_tokens(candidate.chunk.text)
    if not candidate_tokens:
        return 0.0
    return max(_jaccard(candidate_tokens, _content_tokens(item.chunk.text)) for item in selected)


def _content_tokens(text: str) -> set[str]:
    import re

    return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
