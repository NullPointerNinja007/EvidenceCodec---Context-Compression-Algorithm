from __future__ import annotations

from evidence_codec.core.config import RetrievalConfig
from evidence_codec.data.build_candidates import build_candidate_split
from evidence_codec.storage import DEFAULT_LAYOUT


def build_candidates(
    dataset: str,
    split: str,
    subset: str | None = None,
    max_examples: int | None = None,
    overwrite: bool = False,
    dense_backend: str = "hashing",
    dense_model_name: str = "BAAI/bge-small-en-v1.5",
) -> dict:
    DEFAULT_LAYOUT.ensure()
    config = RetrievalConfig(dense_backend=dense_backend, dense_model_name=dense_model_name)
    return build_candidate_split(
        dataset=dataset,
        subset=subset,
        split=split,
        max_examples=max_examples,
        overwrite=overwrite,
        config=config,
    )
