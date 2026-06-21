from __future__ import annotations

from evidence_codec.data.ingest import load_dataset_to_disk
from evidence_codec.storage import DEFAULT_LAYOUT


def ingest_dataset(dataset: str, split: str, subset: str | None = None) -> dict:
    DEFAULT_LAYOUT.ensure()
    safe_split = split.replace("/", "_").replace("[", "_").replace("]", "_")
    output_dir = DEFAULT_LAYOUT.raw_data / dataset
    if subset:
        output_dir = output_dir / subset
    output_dir = output_dir / safe_split
    return load_dataset_to_disk(dataset=dataset, subset=subset, split=split, output_dir=output_dir)
