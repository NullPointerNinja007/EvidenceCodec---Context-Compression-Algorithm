from __future__ import annotations

from modal_apps.common import app, commit_all, training_image, volumes


@app.function(image=training_image, volumes=volumes, gpu="B200", timeout=6 * 60 * 60)
def train(
    config: str = "/root/configs/experiments/modernbert_large.yaml",
    chunk_jsonl: str | None = None,
    candidate_jsonl: str | None = None,
    max_steps: int | None = None,
) -> dict:
    from evidence_codec.modal_jobs.train_reranker import train_reranker

    result = train_reranker(
        config_path=config,
        chunk_jsonl=chunk_jsonl,
        candidate_jsonl=candidate_jsonl,
        max_steps=max_steps,
    )
    commit_all()
    return result


@app.local_entrypoint()
def main(
    config: str = "/root/configs/experiments/modernbert_large.yaml",
    chunk_jsonl: str | None = None,
    candidate_jsonl: str | None = None,
    max_steps: int | None = None,
) -> None:
    print(
        train.remote(
            config=config,
            chunk_jsonl=chunk_jsonl,
            candidate_jsonl=candidate_jsonl,
            max_steps=max_steps,
        )
    )
