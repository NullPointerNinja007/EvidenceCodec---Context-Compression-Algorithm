from __future__ import annotations

from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[str]],
    weights: Mapping[str, float] | None = None,
    c: int = 60,
    limit: int = 256,
) -> list[tuple[str, float]]:
    """Fuse ranked candidate IDs without assuming comparable score scales."""

    weights = weights or {}
    scores: dict[str, float] = {}
    for source, chunk_ids in ranked_lists.items():
        weight = weights.get(source, 1.0)
        for rank, chunk_id in enumerate(chunk_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (c + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
