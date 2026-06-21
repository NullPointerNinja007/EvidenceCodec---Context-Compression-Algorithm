from __future__ import annotations

from evidence_codec.core.config import ChunkingConfig
from evidence_codec.data.build_chunks import build_chunked_split
from evidence_codec.storage import DEFAULT_LAYOUT


def build_chunks(
    dataset: str,
    split: str,
    subset: str | None = None,
    max_examples: int | None = None,
    overwrite: bool = False,
) -> dict:
    DEFAULT_LAYOUT.ensure()
    return build_chunked_split(
        dataset=dataset,
        subset=subset,
        split=split,
        max_examples=max_examples,
        overwrite=overwrite,
        config=ChunkingConfig(),
    )
