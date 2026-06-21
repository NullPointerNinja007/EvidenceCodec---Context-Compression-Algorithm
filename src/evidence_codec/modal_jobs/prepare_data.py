from __future__ import annotations

from evidence_codec.data.ingest import load_many_to_disk
from evidence_codec.storage import DEFAULT_LAYOUT

LONG_BENCH_ENGLISH_CONFIGS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]


def required_dataset_requests(include_longbench: bool = True) -> list[dict]:
    requests = [
        {"dataset": "hotpotqa", "split": "train"},
        {"dataset": "hotpotqa", "split": "validation"},
        {"dataset": "qasper", "split": "train"},
        {"dataset": "qasper", "split": "validation"},
        {"dataset": "qasper", "split": "test"},
    ]
    if include_longbench:
        requests.extend(
            {"dataset": "longbench", "subset": subset, "split": "test"}
            for subset in LONG_BENCH_ENGLISH_CONFIGS
        )
    return requests


def prepare_required_data(include_longbench: bool = True) -> dict:
    DEFAULT_LAYOUT.ensure()
    requests = required_dataset_requests(include_longbench=include_longbench)
    results = load_many_to_disk(requests, DEFAULT_LAYOUT.raw_data)
    return {
        "status": "ok",
        "datasets_downloaded": len(results),
        "results": results,
    }
