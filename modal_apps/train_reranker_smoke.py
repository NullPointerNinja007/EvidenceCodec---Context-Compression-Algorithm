from __future__ import annotations

from pprint import pprint

from modal_apps.common import app, commit_all, training_image, volumes


@app.function(image=training_image, volumes=volumes, gpu="B200", timeout=90 * 60)
def train_smoke(max_steps: int = 1) -> dict:
    from evidence_codec.modal_jobs.train_reranker import train_reranker

    result = train_reranker(
        config_path="/root/configs/experiments/modernbert_large_smoke.yaml",
        chunk_jsonl="/data/processed/chunks/hotpotqa/train.sample5.jsonl",
        candidate_jsonl="/data/processed/candidates/hotpotqa/train.sample5.k256.jsonl",
        max_steps=max_steps,
    )
    commit_all()
    return result


@app.local_entrypoint()
def main(max_steps: int = 1) -> None:
    pprint(train_smoke.remote(max_steps=max_steps))
