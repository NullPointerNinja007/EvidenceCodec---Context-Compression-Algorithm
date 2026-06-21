"""Structure-aware deterministic chunking."""

from evidence_codec.chunking.chunker import chunk_text
from evidence_codec.chunking.dataset_chunkers import (
    chunk_hotpotqa_record,
    chunk_longbench_record,
    chunk_qasper_record,
)
from evidence_codec.chunking.units import TextUnit, pack_units

__all__ = [
    "TextUnit",
    "chunk_hotpotqa_record",
    "chunk_longbench_record",
    "chunk_qasper_record",
    "chunk_text",
    "pack_units",
]
