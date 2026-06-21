from __future__ import annotations

from evidence_codec.training.reranker import train_cross_encoder


def train_reranker(
    config_path: str,
    chunk_jsonl: str | None = None,
    candidate_jsonl: str | None = None,
    max_steps: int | None = None,
) -> dict:
    return train_cross_encoder(
        config_path,
        chunk_jsonl=chunk_jsonl,
        candidate_jsonl=candidate_jsonl,
        max_steps=max_steps,
    )
