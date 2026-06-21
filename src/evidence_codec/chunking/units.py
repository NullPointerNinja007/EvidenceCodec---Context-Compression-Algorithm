from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from evidence_codec.core.config import ChunkingConfig
from evidence_codec.core.types import Chunk
from evidence_codec.utils.tokenization import count_tokens, token_windows

SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_RE = re.compile(r"\n\s*\n+")
NO_SPLIT_MARKERS = (
    " not ",
    " unless ",
    " except ",
    " only if ",
    " provided that ",
    " cannot ",
    " before ",
    " after ",
    " must ",
    " required ",
)


@dataclass(frozen=True)
class TextUnit:
    text: str
    unit_id: str
    breadcrumbs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    group_key: str | None = None

    @property
    def token_count(self) -> int:
        return count_tokens(self.text)


def pack_units(
    units: list[TextUnit],
    document_id: str,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """Greedily pack contiguous structure units into deterministic chunks."""

    config = config or ChunkingConfig()
    chunks: list[Chunk] = []
    buffer: list[TextUnit] = []
    buffer_tokens = 0

    for unit in _expand_oversized_units(units, config):
        unit_tokens = unit.token_count
        group_changed = bool(buffer and unit.group_key != buffer[-1].group_key)
        candidate_units = [*buffer, unit]
        candidate_prefix_tokens = _prefix_token_count(_common_breadcrumbs(candidate_units))
        candidate_separator_tokens = max(0, len(candidate_units) - 1)
        candidate_tokens = (
            buffer_tokens + unit_tokens + candidate_prefix_tokens + candidate_separator_tokens
        )
        would_exceed = candidate_tokens > config.soft_max_tokens
        would_break_hard_limit = candidate_tokens > config.hard_max_tokens
        must_flush = group_changed or would_break_hard_limit
        should_flush = bool(
            buffer and (must_flush or (would_exceed and not _unsafe_boundary(buffer[-1].text, unit.text)))
        )
        if should_flush:
            chunks.append(_build_chunk(document_id, len(chunks), buffer))
            buffer = (
                []
                if group_changed or would_break_hard_limit
                else _overlap_tail(buffer, config.overlap_tokens)
            )
            buffer_tokens = sum(item.token_count for item in buffer)
        buffer.append(unit)
        buffer_tokens += unit_tokens

    if buffer:
        chunks.append(_build_chunk(document_id, len(chunks), buffer))
    return chunks


def paragraph_units(
    text: str,
    document_id: str = "doc",
    breadcrumbs: tuple[str, ...] = (),
    group_key: str | None = None,
) -> list[TextUnit]:
    parts = [part.strip() for part in PARAGRAPH_RE.split(text) if part.strip()]
    if not parts and text.strip():
        parts = [text.strip()]
    return [
        TextUnit(
            text=part,
            unit_id=f"{document_id}:paragraph:{index}",
            breadcrumbs=breadcrumbs,
            group_key=group_key or ">".join(breadcrumbs) or document_id,
            metadata={"unit_type": "paragraph", "paragraph_index": index},
        )
        for index, part in enumerate(parts)
    ]


def sentence_units(
    text: str,
    document_id: str = "doc",
    breadcrumbs: tuple[str, ...] = (),
    group_key: str | None = None,
    base_metadata: dict[str, Any] | None = None,
) -> list[TextUnit]:
    parts = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    if not parts and text.strip():
        parts = [text.strip()]
    metadata = base_metadata or {}
    return [
        TextUnit(
            text=part,
            unit_id=f"{document_id}:sentence:{index}",
            breadcrumbs=breadcrumbs,
            group_key=group_key or ">".join(breadcrumbs) or document_id,
            metadata={**metadata, "unit_type": "sentence", "sentence_part_index": index},
        )
        for index, part in enumerate(parts)
    ]


def chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "token_count": chunk.token_count,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "breadcrumbs": list(chunk.breadcrumbs),
        "metadata": chunk.metadata,
    }


def _expand_oversized_units(units: list[TextUnit], config: ChunkingConfig) -> list[TextUnit]:
    expanded: list[TextUnit] = []
    for unit in units:
        body_limit = _available_body_tokens(unit.breadcrumbs, config.hard_max_tokens)
        if unit.token_count <= body_limit:
            expanded.append(unit)
            continue
        split_parts = _split_long_text(unit.text, body_limit)
        for index, part in enumerate(split_parts):
            expanded.append(
                TextUnit(
                    text=part,
                    unit_id=f"{unit.unit_id}:split:{index}",
                    breadcrumbs=unit.breadcrumbs,
                    group_key=unit.group_key,
                    metadata={
                        **unit.metadata,
                        "split_from": unit.unit_id,
                        "split_index": index,
                    },
                )
            )
    return expanded


def _split_long_text(text: str, hard_max_tokens: int) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_BOUNDARY_RE.split(text) if part.strip()]
    if len(sentences) <= 1:
        return token_windows(text, max_tokens=hard_max_tokens)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current and current_tokens + sentence_tokens > hard_max_tokens:
            chunks.append(" ".join(current))
            current = []
            current_tokens = 0
        if sentence_tokens > hard_max_tokens:
            chunks.extend(token_windows(sentence, max_tokens=hard_max_tokens))
            continue
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current))
    return chunks


def _unsafe_boundary(left: str, right: str) -> bool:
    boundary = f" {left[-120:]} {right[:120]} ".lower()
    return any(marker in boundary for marker in NO_SPLIT_MARKERS)


def _overlap_tail(units: list[TextUnit], overlap_tokens: int) -> list[TextUnit]:
    if overlap_tokens <= 0:
        return []
    tail: list[TextUnit] = []
    total = 0
    for unit in reversed(units):
        total += unit.token_count
        tail.insert(0, unit)
        if total >= overlap_tokens:
            break
    return tail


def _build_chunk(document_id: str, index: int, units: list[TextUnit]) -> Chunk:
    breadcrumbs = _common_breadcrumbs(units)
    prefix = _heading_prefix(breadcrumbs)
    body = "\n\n".join(unit.text.strip() for unit in units if unit.text.strip())
    text = f"{prefix}{body}" if prefix else body
    metadata = {
        "unit_ids": [unit.unit_id for unit in units],
        "unit_count": len(units),
        "unit_types": sorted({unit.metadata.get("unit_type", "unknown") for unit in units}),
        "source_units": [unit.metadata for unit in units],
    }
    return Chunk(
        chunk_id=f"{document_id}:chunk:{index:05d}",
        text=text.strip(),
        token_count=count_tokens(text),
        breadcrumbs=breadcrumbs,
        metadata=metadata,
    )


def _common_breadcrumbs(units: list[TextUnit]) -> tuple[str, ...]:
    if not units:
        return ()
    common = list(units[0].breadcrumbs)
    for unit in units[1:]:
        next_breadcrumbs = list(unit.breadcrumbs)
        keep = 0
        for left, right in zip(common, next_breadcrumbs, strict=False):
            if left != right:
                break
            keep += 1
        common = common[:keep]
    return tuple(common)


def _heading_prefix(breadcrumbs: tuple[str, ...]) -> str:
    if not breadcrumbs:
        return ""
    return " > ".join(part for part in breadcrumbs if part).strip() + "\n"


def _prefix_token_count(breadcrumbs: tuple[str, ...]) -> int:
    prefix = _heading_prefix(breadcrumbs)
    return count_tokens(prefix) if prefix else 0


def _available_body_tokens(breadcrumbs: tuple[str, ...], hard_max_tokens: int) -> int:
    return max(32, hard_max_tokens - _prefix_token_count(breadcrumbs) - 4)
