from __future__ import annotations

import re
from typing import Any

from evidence_codec.chunking.units import TextUnit, chunk_to_dict, pack_units, paragraph_units
from evidence_codec.core.config import ChunkingConfig
from evidence_codec.core.types import Chunk

WHITESPACE_RE = re.compile(r"\s+")


def chunk_hotpotqa_record(
    row: dict[str, Any],
    split: str,
    row_index: int,
    config: ChunkingConfig | None = None,
) -> dict[str, Any]:
    config = config or ChunkingConfig()
    document_id = f"hotpotqa:{split}:{row.get('id', row_index)}"
    units: list[TextUnit] = []
    context = row["context"]
    titles = context["title"]
    sentence_groups = context["sentences"]

    for title_index, title in enumerate(titles):
        group_key = f"{document_id}:{title_index}:{title}"
        for sent_id, sentence in enumerate(sentence_groups[title_index]):
            text = sentence.strip()
            if not text:
                continue
            units.append(
                TextUnit(
                    text=text,
                    unit_id=f"{document_id}:title:{title_index}:sent:{sent_id}",
                    breadcrumbs=(title,),
                    group_key=group_key,
                    metadata={
                        "unit_type": "sentence",
                        "title": title,
                        "title_index": title_index,
                        "sent_id": sent_id,
                        "hotpot_ref": f"{title}::{sent_id}",
                    },
                )
            )

    chunks = pack_units(units, document_id=document_id, config=config)
    gold_refs = {
        f"{title}::{sent_id}"
        for title, sent_id in zip(
            row["supporting_facts"]["title"],
            row["supporting_facts"]["sent_id"],
            strict=False,
        )
    }
    labels = [_hotpot_chunk_label(chunk, gold_refs) for chunk in chunks]
    return {
        "example_id": document_id,
        "dataset": "hotpotqa",
        "split": split,
        "query": row["question"],
        "answers": [row["answer"]],
        "task_type": row.get("type"),
        "level": row.get("level"),
        "chunks": [chunk_to_dict(chunk) for chunk in chunks],
        "labels": labels,
        "gold_chunk_ids": [chunk.chunk_id for chunk, label in zip(chunks, labels, strict=False) if label],
        "gold_evidence": sorted(gold_refs),
        "metadata": {
            "source_id": row.get("id"),
            "positive_chunks": sum(labels),
            "chunk_count": len(chunks),
        },
    }


def chunk_qasper_record(
    row: dict[str, Any],
    split: str,
    row_index: int,
    config: ChunkingConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or ChunkingConfig()
    paper_id = row.get("id") or f"paper-{row_index}"
    document_id = f"qasper:{split}:{paper_id}"
    title = row.get("title") or paper_id
    chunks = _qasper_document_chunks(row, document_id=document_id, config=config)
    qas = row["qas"]
    records: list[dict[str, Any]] = []
    questions = qas.get("question", [])
    question_ids = qas.get("question_id", [])
    answers_by_question = qas.get("answers", [])
    for qa_index, question in enumerate(questions):
        question_id = question_ids[qa_index] if qa_index < len(question_ids) else str(qa_index)
        raw_answer_items = answers_by_question[qa_index] if qa_index < len(answers_by_question) else []
        answer_items = _normalize_qasper_answer_items(raw_answer_items)
        evidence_spans = _qasper_evidence_spans(answer_items)
        labels = [_overlap_label(chunk.text, evidence_spans) for chunk in chunks]
        records.append(
            {
                "example_id": f"{document_id}:qa:{question_id}",
                "dataset": "qasper",
                "split": split,
                "query": question,
                "answers": _qasper_answers(answer_items),
                "paper_id": paper_id,
                "paper_title": title,
                "chunks": [chunk_to_dict(chunk) for chunk in chunks],
                "labels": labels,
                "gold_chunk_ids": [
                    chunk.chunk_id for chunk, label in zip(chunks, labels, strict=False) if label
                ],
                "gold_evidence": evidence_spans,
                "metadata": {
                    "source_id": paper_id,
                    "question_id": question_id,
                    "positive_chunks": sum(labels),
                    "chunk_count": len(chunks),
                    "answer_count": len(answer_items),
                },
            }
        )
    return records


def chunk_longbench_record(
    row: dict[str, Any],
    subset: str,
    split: str,
    row_index: int,
    config: ChunkingConfig | None = None,
) -> dict[str, Any]:
    config = config or ChunkingConfig()
    source_id = row.get("_id") or str(row_index)
    document_id = f"longbench:{subset}:{split}:{source_id}"
    units = paragraph_units(row["context"], document_id=document_id, group_key=document_id)
    chunks = pack_units(units, document_id=document_id, config=config)
    return {
        "example_id": document_id,
        "dataset": "longbench",
        "subset": subset,
        "split": split,
        "query": row["input"],
        "answers": row.get("answers") or [],
        "chunks": [chunk_to_dict(chunk) for chunk in chunks],
        "labels": None,
        "gold_chunk_ids": [],
        "gold_evidence": [],
        "metadata": {
            "source_id": source_id,
            "source_dataset": row.get("dataset"),
            "language": row.get("language"),
            "length": row.get("length"),
            "chunk_count": len(chunks),
        },
    }


def _qasper_document_chunks(
    row: dict[str, Any],
    document_id: str,
    config: ChunkingConfig,
) -> list[Chunk]:
    title = row.get("title") or document_id
    units: list[TextUnit] = []
    abstract = row.get("abstract")
    if abstract:
        units.append(
            TextUnit(
                text=abstract.strip(),
                unit_id=f"{document_id}:abstract:0",
                breadcrumbs=(title, "Abstract"),
                group_key=f"{document_id}:abstract",
                metadata={"unit_type": "abstract"},
            )
        )

    full_text = row.get("full_text") or {}
    section_names = full_text.get("section_name") or []
    section_paragraphs = full_text.get("paragraphs") or []
    for section_index, paragraphs in enumerate(section_paragraphs):
        section = section_names[section_index] if section_index < len(section_names) else "Section"
        group_key = f"{document_id}:section:{section_index}"
        for paragraph_index, paragraph in enumerate(paragraphs or []):
            text = paragraph.strip()
            if not text:
                continue
            units.append(
                TextUnit(
                    text=text,
                    unit_id=f"{document_id}:section:{section_index}:paragraph:{paragraph_index}",
                    breadcrumbs=(title, section),
                    group_key=group_key,
                    metadata={
                        "unit_type": "paragraph",
                        "section": section,
                        "section_index": section_index,
                        "paragraph_index": paragraph_index,
                    },
                )
            )

    figures = row.get("figures_and_tables") or {}
    captions = figures.get("caption") or []
    files = figures.get("file") or []
    for figure_index, caption in enumerate(captions):
        text = caption.strip()
        if not text:
            continue
        units.append(
            TextUnit(
                text=text,
                unit_id=f"{document_id}:figure_table:{figure_index}",
                breadcrumbs=(title, "Figures and Tables"),
                group_key=f"{document_id}:figures_tables",
                metadata={
                    "unit_type": "figure_or_table_caption",
                    "figure_table_index": figure_index,
                    "file": files[figure_index] if figure_index < len(files) else None,
                },
            )
        )
    return pack_units(units, document_id=document_id, config=config)


def _hotpot_chunk_label(chunk: Chunk, gold_refs: set[str]) -> int:
    refs = {
        unit.get("hotpot_ref")
        for unit in chunk.metadata.get("source_units", [])
        if unit.get("hotpot_ref")
    }
    return int(bool(refs & gold_refs))


def _qasper_evidence_spans(answer_items: list[dict[str, Any]]) -> list[str]:
    spans: list[str] = []
    for item in answer_items:
        if not isinstance(item, dict):
            continue
        answer_payload = item.get("answer") if isinstance(item.get("answer"), dict) else item
        for key in ("highlighted_evidence", "evidence"):
            value = answer_payload.get(key)
            if isinstance(value, str):
                spans.append(value)
            elif isinstance(value, list):
                spans.extend(str(span) for span in value if span)
    return _dedupe_normalized(spans)


def _normalize_qasper_answer_items(raw_answer_items) -> list[dict[str, Any]]:
    if isinstance(raw_answer_items, list):
        return [item for item in raw_answer_items if isinstance(item, dict)]
    if not isinstance(raw_answer_items, dict):
        return []

    list_lengths = [len(value) for value in raw_answer_items.values() if isinstance(value, list)]
    if not list_lengths:
        return [raw_answer_items]

    items: list[dict[str, Any]] = []
    for index in range(max(list_lengths)):
        item: dict[str, Any] = {}
        for key, value in raw_answer_items.items():
            if isinstance(value, list):
                item[key] = value[index] if index < len(value) else None
            else:
                item[key] = value
        items.append(item)
    return items


def _qasper_answers(answer_items: list[dict[str, Any]]) -> list[str]:
    answers: list[str] = []
    for item in answer_items:
        if not isinstance(item, dict):
            continue
        answer_payload = item.get("answer") if isinstance(item.get("answer"), dict) else item
        answer = answer_payload.get("free_form_answer") or answer_payload.get("answer")
        if isinstance(answer, str) and answer:
            answers.append(answer)
        elif answer_payload.get("extractive_spans"):
            answers.extend(str(span) for span in answer_payload["extractive_spans"] if span)
        elif answer_payload.get("yes_no") is not None:
            answers.append(str(answer_payload["yes_no"]))
        elif answer_payload.get("unanswerable"):
            answers.append("unanswerable")
    return _dedupe_normalized(answers)


def _overlap_label(chunk_text: str, evidence_spans: list[str]) -> int:
    if not evidence_spans:
        return 0
    normalized_chunk = _normalize_text(chunk_text)
    for span in evidence_spans:
        normalized_span = _normalize_text(span)
        if not normalized_span:
            continue
        if normalized_span in normalized_chunk:
            return 1
        if len(normalized_span) > 80 and _token_overlap(normalized_chunk, normalized_span) >= 0.72:
            return 1
    return 0


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(right_tokens)


def _dedupe_normalized(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        key = _normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip().lower()
