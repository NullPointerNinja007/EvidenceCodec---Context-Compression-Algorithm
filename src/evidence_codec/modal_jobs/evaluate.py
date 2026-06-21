from __future__ import annotations

from evidence_codec.storage import DEFAULT_LAYOUT


def run_evaluation(config_path: str, suite: str) -> dict:
    DEFAULT_LAYOUT.ensure()
    return {
        "status": "scaffold_ready",
        "suite": suite,
        "config_path": config_path,
        "output_dir": str(DEFAULT_LAYOUT.evaluations / suite),
        "next_step": "Implement candidate, reranker, compression, and end-to-end eval suites here.",
    }
