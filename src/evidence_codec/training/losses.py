from __future__ import annotations

from typing import Any


def describe_reranker_objective() -> dict:
    return {
        "bce": "BCEWithLogits over query-chunk usefulness labels.",
        "ranking": "Pairwise margin loss over positive and hard-negative chunks.",
        "risk": "Optional risk-weighted BCE for fragile evidence spans.",
        "default_weights": {"bce": 1.0, "ranking": 0.5, "margin": 0.2},
    }


def bce_plus_pairwise_margin_loss(
    logits,
    labels,
    group_ids=None,
    bce_weight: float = 1.0,
    ranking_weight: float = 0.5,
    margin: float = 0.2,
) -> tuple[Any, dict[str, float]]:
    """BCE plus same-query positive-over-negative margin loss."""

    import torch
    import torch.nn.functional as F

    labels = labels.float()
    bce = F.binary_cross_entropy_with_logits(logits.float(), labels)
    ranking = _pairwise_margin_loss(logits.float(), labels, group_ids=group_ids, margin=margin)
    total = bce_weight * bce + ranking_weight * ranking
    return total, {
        "loss": float(total.detach().cpu()),
        "bce_loss": float(bce.detach().cpu()),
        "ranking_loss": float(ranking.detach().cpu()),
    }


def _pairwise_margin_loss(logits, labels, group_ids=None, margin: float = 0.2):
    import torch
    import torch.nn.functional as F

    if group_ids is None:
        group_ids = torch.zeros_like(labels, dtype=torch.long)
    losses = []
    for group_id in torch.unique(group_ids):
        mask = group_ids == group_id
        group_logits = logits[mask]
        group_labels = labels[mask]
        pos = group_logits[group_labels >= 0.5]
        neg = group_logits[group_labels < 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        losses.append(F.relu(margin - pos[:, None] + neg[None, :]).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()
