from __future__ import annotations

import re

from evidence_codec.core.config import ChunkingConfig
from evidence_codec.core.types import Chunk
from evidence_codec.utils.tokenization import count_tokens, truncate_by_tokens

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_RE = re.compile(r"\n\s*\n+")
NO_SPLIT_MARKERS = (
    " not ",
    " unless ",
    " except ",
    " only if ",
    " cannot ",
    " before ",
    " after ",
    " must ",
)


def chunk_text(
    text: str,
    config: ChunkingConfig | None = None,
    document_id: str = "doc",
) -> list[Chunk]:
    """Create deterministic raw-text chunks.

    This is the conservative MVP chunker. It preserves paragraphs first, then
    groups sentence-like units up to the target budget, and only falls back to
    token truncation for units over the hard maximum.
    """

    config = config or ChunkingConfig()
    units = _paragraph_units(text)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)
        if unit_tokens > config.hard_max_tokens:
            split_units = _split_long_unit(unit, config.hard_max_tokens)
        else:
            split_units = [unit]

        for split_unit in split_units:
            split_tokens = count_tokens(split_unit)
            would_exceed = buffer_tokens + split_tokens > config.soft_max_tokens
            if buffer and would_exceed and not _unsafe_boundary(buffer[-1], split_unit):
                chunks.append(_build_chunk(document_id, len(chunks), buffer))
                buffer = _overlap_tail(buffer, config.overlap_tokens)
                buffer_tokens = sum(count_tokens(item) for item in buffer)
            buffer.append(split_unit)
            buffer_tokens += split_tokens

    if buffer:
        chunks.append(_build_chunk(document_id, len(chunks), buffer))

    return chunks


def _paragraph_units(text: str) -> list[str]:
    units = [unit.strip() for unit in PARAGRAPH_RE.split(text) if unit.strip()]
    return units or [text.strip()] if text.strip() else []


def _split_long_unit(unit: str, hard_max_tokens: int) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(unit) if part.strip()]
    if len(sentences) <= 1:
        return [truncate_by_tokens(unit, hard_max_tokens)]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current and current_tokens + sentence_tokens > hard_max_tokens:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def _unsafe_boundary(left: str, right: str) -> bool:
    boundary = f" {left[-80:]} {right[:80]} ".lower()
    return any(marker in boundary for marker in NO_SPLIT_MARKERS)


def _overlap_tail(units: list[str], overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0:
        return []
    tail: list[str] = []
    total = 0
    for unit in reversed(units):
        total += count_tokens(unit)
        tail.insert(0, unit)
        if total >= overlap_tokens:
            break
    return tail


def _build_chunk(document_id: str, index: int, units: list[str]) -> Chunk:
    text = "\n\n".join(units).strip()
    return Chunk(
        chunk_id=f"{document_id}:{index:05d}",
        text=text,
        token_count=count_tokens(text),
        metadata={"unit_count": len(units)},
    )
