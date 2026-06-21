from __future__ import annotations


def evidence_recall(gold_ids: set[str], selected_ids: set[str]) -> float:
    if not gold_ids:
        return 0.0
    return len(gold_ids & selected_ids) / len(gold_ids)


def all_evidence_covered(gold_ids: set[str], selected_ids: set[str]) -> bool:
    return bool(gold_ids) and gold_ids.issubset(selected_ids)


def compression_ratio(input_tokens: int, output_tokens: int) -> float:
    if output_tokens <= 0:
        return 0.0
    return input_tokens / output_tokens
