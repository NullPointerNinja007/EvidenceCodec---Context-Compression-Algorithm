from __future__ import annotations

import math
from pathlib import Path

from evidence_codec.core.config import RerankerConfig
from evidence_codec.data.reranker_pairs import load_reranker_examples
from evidence_codec.rerank import ModernBertCrossEncoder, collate_query_chunk_pairs
from evidence_codec.storage import DEFAULT_LAYOUT
from evidence_codec.training.losses import bce_plus_pairwise_margin_loss, describe_reranker_objective


def planned_reranker_outputs(run_name: str = "modernbert-large") -> dict:
    output_dir = DEFAULT_LAYOUT.reranker_models / run_name
    return {
        "checkpoint_dir": str(output_dir),
        "objective": describe_reranker_objective(),
    }


def train_cross_encoder(
    config_path: str | Path,
    chunk_jsonl: str | Path | None = None,
    candidate_jsonl: str | Path | None = None,
    max_steps: int | None = None,
) -> dict:
    """Train ModernBERT-large reranker on query-candidate chunk pairs."""

    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    DEFAULT_LAYOUT.ensure()
    raw_config = _load_yaml(config_path)
    training_cfg = raw_config.get("training", {})
    loss_cfg = training_cfg.get("losses", {})
    run_name = raw_config.get("run", {}).get("name", "modernbert-large-hotpotqa")
    reranker_config = RerankerConfig(
        model_name=training_cfg.get("model_name", "answerdotai/ModernBERT-large"),
        sequence_length=int(training_cfg.get("sequence_length", 512)),
        query_max_tokens=int(training_cfg.get("query_max_tokens", 96)),
        precision=training_cfg.get("precision", "bf16"),
        bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
        ranking_weight=float(loss_cfg.get("ranking_weight", 0.5)),
        margin=float(loss_cfg.get("margin", 0.2)),
    )
    chunk_jsonl = Path(chunk_jsonl) if chunk_jsonl else DEFAULT_LAYOUT.processed_data / "chunks" / "hotpotqa" / "train.sample5.jsonl"
    candidate_jsonl = (
        Path(candidate_jsonl)
        if candidate_jsonl
        else DEFAULT_LAYOUT.processed_data / "candidates" / "hotpotqa" / "train.sample5.k256.jsonl"
    )
    pairs = load_reranker_examples(chunk_jsonl, candidate_jsonl)
    if not pairs:
        raise RuntimeError(f"No reranker pairs found for {chunk_jsonl} and {candidate_jsonl}")

    tokenizer = AutoTokenizer.from_pretrained(reranker_config.model_name, trust_remote_code=True)
    model = ModernBertCrossEncoder(reranker_config.model_name, config=reranker_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    epochs = int(training_cfg.get("epochs", 1))
    learning_rate = float(training_cfg.get("learning_rate", 2e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.06))
    effective_batch_size = int(training_cfg.get("effective_batch_size", 128))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    batch_slices = _grouped_batch_indices(pairs)
    accumulation_windows = _accumulation_windows(batch_slices, effective_batch_size)
    available_steps = epochs * len(accumulation_windows)
    total_steps = min(max_steps, available_steps) if max_steps is not None else available_steps
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )

    step = 0
    microbatch_steps = 0
    last_metrics: dict[str, float] = {}
    autocast_enabled = device == "cuda" and reranker_config.precision == "bf16"
    for _epoch in range(epochs):
        for window in accumulation_windows:
            window_pair_count = sum(len(indices) for indices in window)
            metric_totals: dict[str, float] = {}
            optimizer.zero_grad(set_to_none=True)
            for indices in window:
                examples = [pairs[index] for index in indices]
                batch = collate_query_chunk_pairs(tokenizer, examples, config=reranker_config)
                labels = batch.pop("labels").to(device)
                group_ids = batch.pop("group_ids").to(device)
                batch = {key: value.to(device) for key, value in batch.items()}
                microbatch_weight = len(examples) / window_pair_count
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                    logits = model(**batch)
                    loss, metrics = bce_plus_pairwise_margin_loss(
                        logits,
                        labels,
                        group_ids=group_ids,
                        bce_weight=reranker_config.bce_weight,
                        ranking_weight=reranker_config.ranking_weight,
                        margin=reranker_config.margin,
                    )
                    scaled_loss = loss * microbatch_weight
                scaled_loss.backward()
                microbatch_steps += 1
                for key, value in metrics.items():
                    metric_totals[key] = metric_totals.get(key, 0.0) + value * len(examples)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()),
                    max_norm=max_grad_norm,
                )
            optimizer.step()
            scheduler.step()
            last_metrics = {
                key: value / window_pair_count
                for key, value in metric_totals.items()
            }
            last_metrics.update(
                {
                    "effective_batch_size": float(effective_batch_size),
                    "pairs_in_optimizer_step": float(window_pair_count),
                    "microbatches_in_optimizer_step": float(len(window)),
                }
            )
            step += 1
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    output_dir = DEFAULT_LAYOUT.reranker_models / run_name
    model.save_pretrained(output_dir, tokenizer=tokenizer)
    return {
        "status": "ok",
        "config_path": str(config_path),
        "chunk_jsonl": str(chunk_jsonl),
        "candidate_jsonl": str(candidate_jsonl),
        "pairs": len(pairs),
        "optimizer_steps": step,
        "steps": step,
        "microbatch_steps": microbatch_steps,
        "effective_batch_size": effective_batch_size,
        "microbatches_per_epoch": len(batch_slices),
        "optimizer_steps_per_epoch": len(accumulation_windows),
        "last_metrics": last_metrics,
        "checkpoint_dir": str(output_dir),
    }


def _load_yaml(config_path: str | Path) -> dict:
    import yaml

    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _grouped_batch_indices(examples: list[dict]) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for index, example in enumerate(examples):
        groups.setdefault(int(example["group_id"]), []).append(index)
    return list(groups.values())


def _accumulation_windows(
    microbatches: list[list[int]],
    effective_batch_size: int,
) -> list[list[list[int]]]:
    """Group query-level microbatches into optimizer steps by pair count."""

    if effective_batch_size <= 0:
        raise ValueError("effective_batch_size must be positive")
    windows: list[list[list[int]]] = []
    current: list[list[int]] = []
    current_pairs = 0
    for microbatch in microbatches:
        current.append(microbatch)
        current_pairs += len(microbatch)
        if current_pairs >= effective_batch_size:
            windows.append(current)
            current = []
            current_pairs = 0
    if current:
        windows.append(current)
    return windows
