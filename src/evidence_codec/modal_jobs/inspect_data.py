from __future__ import annotations

from evidence_codec.storage import DEFAULT_LAYOUT


def inspect_raw_examples() -> dict:
    DEFAULT_LAYOUT.ensure()
    targets = {
        "hotpotqa": DEFAULT_LAYOUT.raw_data / "hotpotqa" / "train",
        "qasper": DEFAULT_LAYOUT.raw_data / "qasper" / "train",
        "longbench_hotpotqa": DEFAULT_LAYOUT.raw_data / "longbench" / "hotpotqa" / "test",
    }
    result: dict[str, dict] = {}
    for name, path in targets.items():
        rows = _read_arrow_rows(path, limit=1)
        row = rows[0] if rows else {}
        result[name] = {
            "rows_seen": len(rows),
            "keys": list(row.keys()),
            "sample": _shorten(row),
        }
    return result


def inspect_qasper_answers(limit_questions: int = 5) -> dict:
    from evidence_codec.chunking.dataset_chunkers import (
        _normalize_qasper_answer_items,
        _qasper_evidence_spans,
    )

    DEFAULT_LAYOUT.ensure()
    path = DEFAULT_LAYOUT.raw_data / "qasper" / "train"
    row = _read_arrow_rows(path, limit=1)[0]
    qas = row["qas"]
    answers = qas.get("answers", [])
    result = []
    for index, question in enumerate(qas.get("question", [])[:limit_questions]):
        raw_answer_items = answers[index] if index < len(answers) else []
        normalized = _normalize_qasper_answer_items(raw_answer_items)
        result.append(
            {
                "index": index,
                "question": question,
                "raw_type": type(raw_answer_items).__name__,
                "raw_preview": _shorten(raw_answer_items, max_chars=300),
                "normalized_count": len(normalized),
                "normalized_preview": _shorten(normalized, max_chars=300),
                "evidence_spans": _qasper_evidence_spans(normalized)[:3],
            }
        )
    return {"paper_id": row.get("id"), "title": row.get("title"), "questions": result}


def _read_arrow_rows(path, limit: int) -> list[dict]:
    import pyarrow as pa

    rows: list[dict] = []
    for arrow_path in sorted(path.glob("*.arrow")):
        with pa.memory_map(str(arrow_path), "r") as source:
            try:
                reader = pa.ipc.open_stream(source)
                table = reader.read_all()
            except pa.ArrowInvalid:
                source.seek(0)
                reader = pa.ipc.open_file(source)
                table = reader.read_all()
        for row in table.to_pylist():
            rows.append(row)
            if len(rows) >= limit:
                return rows
    return rows


def _shorten(value, max_chars: int = 800):
    if isinstance(value, str):
        return value[:max_chars]
    if isinstance(value, list):
        return [_shorten(item, max_chars=max_chars) for item in value[:3]]
    if isinstance(value, dict):
        return {key: _shorten(item, max_chars=max_chars) for key, item in list(value.items())[:8]}
    return value
