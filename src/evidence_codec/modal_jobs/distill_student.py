from __future__ import annotations

from evidence_codec.training.distill import train_student


def distill_student(config_path: str) -> dict:
    return train_student(config_path)
