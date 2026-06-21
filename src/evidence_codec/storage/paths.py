from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _path_from_env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


@dataclass(frozen=True)
class VolumeLayout:
    """Directory layout expected inside mounted Modal Volumes."""

    data_root: Path = _path_from_env("EVIDENCE_CODEC_DATA_DIR", "/data")
    models_root: Path = _path_from_env("EVIDENCE_CODEC_MODELS_DIR", "/models")
    runs_root: Path = _path_from_env("EVIDENCE_CODEC_RUNS_DIR", "/runs")
    cache_root: Path = _path_from_env("EVIDENCE_CODEC_CACHE_DIR", "/cache")

    @property
    def raw_data(self) -> Path:
        return self.data_root / "raw"

    @property
    def processed_data(self) -> Path:
        return self.data_root / "processed"

    @property
    def indexes(self) -> Path:
        return self.data_root / "indexes"

    @property
    def teacher_scores(self) -> Path:
        return self.data_root / "teacher_scores"

    @property
    def reranker_models(self) -> Path:
        return self.models_root / "rerankers"

    @property
    def student_models(self) -> Path:
        return self.models_root / "students"

    @property
    def evaluations(self) -> Path:
        return self.runs_root / "evaluations"

    @property
    def demos(self) -> Path:
        return self.runs_root / "demos"

    def required_dirs(self) -> tuple[Path, ...]:
        return (
            self.raw_data,
            self.processed_data,
            self.indexes,
            self.teacher_scores,
            self.reranker_models,
            self.student_models,
            self.evaluations,
            self.demos,
            self.cache_root / "huggingface",
        )

    def ensure(self) -> None:
        for path in self.required_dirs():
            path.mkdir(parents=True, exist_ok=True)


DEFAULT_LAYOUT = VolumeLayout()
