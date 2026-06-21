from __future__ import annotations

from pprint import pprint
from typing import Any

import modal

from modal_apps.common import app, base_image, commit_all, reload_all, training_image, volumes


@app.function(image=base_image, volumes=volumes, timeout=60 * 60, retries=modal.Retries(max_retries=3))
def ensure_hotpotqa_data() -> dict:
    from evidence_codec.modal_jobs.prepare_data import prepare_required_data

    result = prepare_required_data(include_longbench=False)
    commit_all()
    return result


@app.function(
    image=base_image,
    volumes=volumes,
    timeout=3 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=5.0, max_delay=60.0),
    max_containers=100,
)
def build_artifact_shard(
    split: str,
    shard_id: int,
    num_shards: int,
    max_examples: int | None,
    candidate_cap: int,
    overwrite: bool,
) -> dict:
    from evidence_codec.data.parallel_hotpotqa import build_hotpotqa_artifact_shard

    result = build_hotpotqa_artifact_shard(
        split=split,
        shard_id=shard_id,
        num_shards=num_shards,
        max_examples=max_examples,
        candidate_cap=candidate_cap,
        overwrite=overwrite,
    )
    commit_all()
    return result


@app.function(image=base_image, volumes=volumes, timeout=60 * 60, retries=modal.Retries(max_retries=3))
def merge_artifacts(
    split: str,
    num_shards: int,
    max_examples: int | None,
    candidate_cap: int,
    overwrite: bool,
) -> dict:
    from evidence_codec.data.parallel_hotpotqa import merge_hotpotqa_artifacts

    reload_all()
    result = merge_hotpotqa_artifacts(
        split=split,
        num_shards=num_shards,
        max_examples=max_examples,
        candidate_cap=candidate_cap,
        overwrite=overwrite,
    )
    commit_all()
    return result


@app.function(
    image=training_image,
    volumes=volumes,
    gpu="B200",
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=30.0, max_delay=60.0),
)
def train_with_checkpoints(
    chunk_jsonl: str,
    candidate_jsonl: str,
    run_name: str,
    checkpoint_count: int,
    max_steps: int | None,
    overwrite: bool,
) -> dict:
    from evidence_codec.training.checkpoint_run import train_cross_encoder_with_checkpoints

    reload_all()
    result = train_cross_encoder_with_checkpoints(
        config_path="/root/configs/experiments/modernbert_large.yaml",
        chunk_jsonl=chunk_jsonl,
        candidate_jsonl=candidate_jsonl,
        run_name=run_name,
        checkpoint_count=checkpoint_count,
        max_steps=max_steps,
        overwrite=overwrite,
    )
    commit_all()
    return result


@app.function(
    image=training_image,
    volumes=volumes,
    gpu="B200",
    timeout=2 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10.0, max_delay=60.0),
    max_containers=8,
)
def evaluate_one_checkpoint(
    checkpoint_dir: str,
    chunk_jsonl: str,
    candidate_jsonl: str,
    max_records: int,
    budget_tokens: int,
    batch_size: int,
) -> dict:
    from evidence_codec.training.checkpoint_run import evaluate_checkpoint

    reload_all()
    result = evaluate_checkpoint(
        checkpoint_dir=checkpoint_dir,
        chunk_jsonl=chunk_jsonl,
        candidate_jsonl=candidate_jsonl,
        max_records=max_records,
        budget_tokens=budget_tokens,
        batch_size=batch_size,
    )
    commit_all()
    return result


@app.function(image=training_image, volumes=volumes, timeout=30 * 60, retries=modal.Retries(max_retries=3))
def write_eval_outputs(metrics: list[dict], run_name: str) -> dict:
    from evidence_codec.storage import DEFAULT_LAYOUT
    from evidence_codec.training.checkpoint_run import write_checkpoint_eval_outputs

    reload_all()
    result = write_checkpoint_eval_outputs(
        metrics,
        DEFAULT_LAYOUT.evaluations / run_name,
    )
    commit_all()
    return result


def _run_shards(
    split: str,
    num_shards: int,
    max_examples: int | None,
    candidate_cap: int,
    overwrite: bool,
    attempts: int = 3,
) -> list[dict]:
    remaining = list(range(num_shards))
    results_by_shard: dict[int, dict] = {}
    for attempt in range(1, attempts + 1):
        args = [
            (split, shard_id, num_shards, max_examples, candidate_cap, overwrite)
            for shard_id in remaining
        ]
        outputs = list(build_artifact_shard.starmap(args, return_exceptions=True))
        failed = []
        for shard_id, output in zip(remaining, outputs, strict=True):
            if isinstance(output, BaseException):
                failed.append(shard_id)
            else:
                results_by_shard[shard_id] = output
        if not failed:
            return [results_by_shard[index] for index in range(num_shards)]
        remaining = failed
        print(f"{split}: retrying {len(remaining)} failed shards after attempt {attempt}")
    raise RuntimeError(f"{split}: shards failed after {attempts} attempts: {remaining[:20]}")


def _run_eval(
    checkpoints: list[str],
    chunk_jsonl: str,
    candidate_jsonl: str,
    max_records: int,
    budget_tokens: int,
    batch_size: int,
    attempts: int = 3,
) -> list[dict]:
    remaining = list(checkpoints)
    metrics_by_checkpoint: dict[str, dict] = {}
    for attempt in range(1, attempts + 1):
        args = [
            (checkpoint, chunk_jsonl, candidate_jsonl, max_records, budget_tokens, batch_size)
            for checkpoint in remaining
        ]
        outputs = list(evaluate_one_checkpoint.starmap(args, return_exceptions=True))
        failed = []
        for checkpoint, output in zip(remaining, outputs, strict=True):
            if isinstance(output, BaseException):
                failed.append(checkpoint)
            else:
                metrics_by_checkpoint[checkpoint] = output
        if not failed:
            return [metrics_by_checkpoint[checkpoint] for checkpoint in checkpoints]
        remaining = failed
        print(f"eval: retrying {len(remaining)} failed checkpoints after attempt {attempt}")
    raise RuntimeError(f"eval failed after {attempts} attempts: {remaining}")


@app.local_entrypoint()
def main(
    train_shards: int = 100,
    eval_shards: int = 100,
    train_examples: int | None = None,
    validation_examples: int = 1000,
    candidate_cap: int = 256,
    checkpoint_count: int = 6,
    run_name: str = "modernbert-large-hotpotqa-full",
    max_steps: int | None = None,
    budget_tokens: int = 512,
    eval_batch_size: int = 64,
    overwrite_artifacts: bool = False,
    overwrite_training: bool = False,
) -> None:
    print("Ensuring HotpotQA raw train/validation data exists...")
    pprint(ensure_hotpotqa_data.remote())

    print(f"Building HotpotQA train artifacts with {train_shards} CPU shards...")
    train_shard_results = _run_shards(
        split="train",
        num_shards=train_shards,
        max_examples=train_examples,
        candidate_cap=candidate_cap,
        overwrite=overwrite_artifacts,
    )
    print(f"Built train shards: {len(train_shard_results)}")
    train_artifacts = merge_artifacts.remote(
        split="train",
        num_shards=train_shards,
        max_examples=train_examples,
        candidate_cap=candidate_cap,
        overwrite=overwrite_artifacts,
    )
    pprint(train_artifacts)

    print(f"Building HotpotQA validation sample artifacts with {eval_shards} CPU shards...")
    eval_shard_results = _run_shards(
        split="validation",
        num_shards=eval_shards,
        max_examples=validation_examples,
        candidate_cap=candidate_cap,
        overwrite=overwrite_artifacts,
    )
    print(f"Built validation shards: {len(eval_shard_results)}")
    eval_artifacts = merge_artifacts.remote(
        split="validation",
        num_shards=eval_shards,
        max_examples=validation_examples,
        candidate_cap=candidate_cap,
        overwrite=overwrite_artifacts,
    )
    pprint(eval_artifacts)

    print("Starting B200 training with checkpointing...")
    train_summary = train_with_checkpoints.remote(
        chunk_jsonl=train_artifacts["chunk_path"],
        candidate_jsonl=train_artifacts["candidate_path"],
        run_name=run_name,
        checkpoint_count=checkpoint_count,
        max_steps=max_steps,
        overwrite=overwrite_training,
    )
    pprint(train_summary)

    checkpoints = train_summary["checkpoints"]
    if train_summary.get("total_steps", 0) >= 5 and len(checkpoints) < 5:
        raise RuntimeError(f"Training produced fewer than 5 checkpoints: {len(checkpoints)}")
    print(f"Evaluating {len(checkpoints)} checkpoints on B200 workers...")
    metrics = _run_eval(
        checkpoints=checkpoints,
        chunk_jsonl=eval_artifacts["chunk_path"],
        candidate_jsonl=eval_artifacts["candidate_path"],
        max_records=validation_examples,
        budget_tokens=budget_tokens,
        batch_size=eval_batch_size,
    )
    pprint(metrics)
    print("Writing metrics CSV/JSONL and plots...")
    outputs = write_eval_outputs.remote(metrics=metrics, run_name=run_name)
    pprint(outputs)
