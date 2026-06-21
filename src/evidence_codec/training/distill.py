from __future__ import annotations

from pathlib import Path

from evidence_codec.storage import DEFAULT_LAYOUT


def train_student(config_path: str | Path) -> dict:
    DEFAULT_LAYOUT.ensure()
    return {
        "status": "scaffold_ready",
        "config_path": str(config_path),
        "student_dir": str(DEFAULT_LAYOUT.student_models / "lightgbm"),
        "next_step": "Train LightGBM or MLP on BM25, dense, risk, overlap, and position features.",
    }
