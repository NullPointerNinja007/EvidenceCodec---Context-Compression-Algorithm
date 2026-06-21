from __future__ import annotations

from evidence_codec.evaluation.metrics import all_evidence_covered, evidence_recall


def candidate_stage_metrics(gold_ids: set[str], ranked_candidate_ids: list[str], k: int) -> dict:
    selected = set(ranked_candidate_ids[:k])
    return {
        f"EvidenceRecall@{k}": evidence_recall(gold_ids, selected),
        f"AllEvidenceCovered@{k}": all_evidence_covered(gold_ids, selected),
    }
