from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from evidence_codec.core.config import RerankerConfig, SelectionConfig
from evidence_codec.core.types import Candidate, Chunk
from evidence_codec.data.reranker_pairs import load_reranker_examples
from evidence_codec.rerank import CrossEncoderReranker, ModernBertCrossEncoder, collate_query_chunk_pairs
from evidence_codec.selection import select_budgeted_context
from evidence_codec.storage import DEFAULT_LAYOUT
from evidence_codec.training.losses import bce_plus_pairwise_margin_loss
from evidence_codec.training.reranker import _accumulation_windows, _grouped_batch_indices, _load_yaml


def checkpoint_steps(total_steps: int, checkpoint_count: int = 6) -> list[int]:
    if total_steps <= 0:
        return []
    count = max(1, checkpoint_count)
    steps = {max(1, math.ceil(total_steps * index / count)) for index in range(1, count + 1)}
    steps.add(total_steps)
    return sorted(step for step in steps if step <= total_steps)


def train_cross_encoder_with_checkpoints(
    config_path: str | Path,
    chunk_jsonl: str | Path,
    candidate_jsonl: str | Path,
    run_name: str = "modernbert-large-hotpotqa-full",
    checkpoint_count: int = 6,
    max_steps: int | None = None,
    overwrite: bool = False,
) -> dict:
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    DEFAULT_LAYOUT.ensure()
    raw_config = _load_yaml(config_path)
    training_cfg = raw_config.get("training", {})
    loss_cfg = training_cfg.get("losses", {})
    reranker_config = RerankerConfig(
        model_name=training_cfg.get("model_name", "answerdotai/ModernBERT-large"),
        sequence_length=int(training_cfg.get("sequence_length", 512)),
        query_max_tokens=int(training_cfg.get("query_max_tokens", 96)),
        precision=training_cfg.get("precision", "bf16"),
        bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
        ranking_weight=float(loss_cfg.get("ranking_weight", 0.5)),
        margin=float(loss_cfg.get("margin", 0.2)),
    )
    chunk_jsonl = Path(chunk_jsonl)
    candidate_jsonl = Path(candidate_jsonl)
    output_dir = DEFAULT_LAYOUT.reranker_models / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    done_marker = output_dir / "_TRAINING_DONE"
    if done_marker.exists() and not overwrite:
        return _load_json(output_dir / "training_summary.json")

    pairs = load_reranker_examples(chunk_jsonl, candidate_jsonl)
    if not pairs:
        raise RuntimeError(f"No reranker pairs found for {chunk_jsonl} and {candidate_jsonl}")
    batch_slices = _grouped_batch_indices(pairs)
    effective_batch_size = int(training_cfg.get("effective_batch_size", 128))
    accumulation_windows = _accumulation_windows(batch_slices, effective_batch_size)
    epochs = int(training_cfg.get("epochs", 1))
    available_steps = epochs * len(accumulation_windows)
    total_steps = min(max_steps, available_steps) if max_steps is not None else available_steps
    save_steps = set(checkpoint_steps(total_steps, checkpoint_count=checkpoint_count))

    tokenizer = AutoTokenizer.from_pretrained(reranker_config.model_name, trust_remote_code=True)
    latest_checkpoint = _latest_checkpoint(output_dir)
    start_step = 0
    if latest_checkpoint is not None and not overwrite:
        model = ModernBertCrossEncoder.from_pretrained(latest_checkpoint, config=reranker_config)
        state = _load_torch_state(latest_checkpoint / "trainer_state.pt")
        start_step = int(state.get("step", 0)) if state else 0
    else:
        model = ModernBertCrossEncoder(reranker_config.model_name, config=reranker_config)
        state = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    learning_rate = float(training_cfg.get("learning_rate", 2e-5))
    weight_decay = float(training_cfg.get("weight_decay", 0.01))
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.06))
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])

    log_path = output_dir / "train_metrics.jsonl"
    microbatch_steps = 0
    step = start_step
    autocast_enabled = device == "cuda" and reranker_config.precision == "bf16"
    global_window = 0
    for epoch in range(epochs):
        for window in accumulation_windows:
            global_window += 1
            if global_window <= start_step:
                continue
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
                torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            row = {
                "step": step,
                "epoch": epoch,
                "pairs_in_optimizer_step": window_pair_count,
                "microbatches_in_optimizer_step": len(window),
                "learning_rate": scheduler.get_last_lr()[0],
                **{key: value / window_pair_count for key, value in metric_totals.items()},
            }
            _append_jsonl(log_path, row)
            if step in save_steps:
                _save_training_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=output_dir / f"checkpoint-step-{step:06d}",
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    total_steps=total_steps,
                    reranker_config=reranker_config,
                )
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    if step not in save_steps:
        _save_training_checkpoint(
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir / f"checkpoint-step-{step:06d}",
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            total_steps=total_steps,
            reranker_config=reranker_config,
        )
    checkpoints = [str(path) for path in sorted(output_dir.glob("checkpoint-step-*")) if path.is_dir()]
    summary = {
        "status": "ok",
        "run_name": run_name,
        "checkpoint_dir": str(output_dir),
        "checkpoints": checkpoints,
        "checkpoint_count": len(checkpoints),
        "chunk_jsonl": str(chunk_jsonl),
        "candidate_jsonl": str(candidate_jsonl),
        "pairs": len(pairs),
        "optimizer_steps": step,
        "total_steps": total_steps,
        "microbatch_steps": microbatch_steps,
        "effective_batch_size": effective_batch_size,
        "microbatches_per_epoch": len(batch_slices),
        "optimizer_steps_per_epoch": len(accumulation_windows),
        "train_metrics_path": str(log_path),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    done_marker.write_text("ok\n", encoding="utf-8")
    return summary


def evaluate_checkpoint(
    checkpoint_dir: str | Path,
    chunk_jsonl: str | Path,
    candidate_jsonl: str | Path,
    max_records: int = 1000,
    budget_tokens: int = 512,
    batch_size: int = 64,
) -> dict:
    checkpoint_dir = Path(checkpoint_dir)
    chunk_jsonl = Path(chunk_jsonl)
    candidate_jsonl = Path(candidate_jsonl)
    step = _step_from_checkpoint(checkpoint_dir)
    reranker = CrossEncoderReranker(model_name_or_path=str(checkpoint_dir), device="cuda")
    totals: dict[str, float] = {
        "records": 0.0,
        "MRR@10": 0.0,
        "nDCG@10": 0.0,
        "Recall@32": 0.0,
        "Recall@64": 0.0,
        "Recall@128": 0.0,
        "EvidenceRecall@Budget": 0.0,
        "AllGoldSelected@Budget": 0.0,
        "avg_used_tokens": 0.0,
        "avg_selected_chunks": 0.0,
        "avg_risk_added": 0.0,
    }
    with chunk_jsonl.open("r", encoding="utf-8") as chunk_file, candidate_jsonl.open(
        "r", encoding="utf-8"
    ) as candidate_file:
        for index, (chunk_line, candidate_line) in enumerate(zip(chunk_file, candidate_file, strict=False)):
            if index >= max_records:
                break
            if not chunk_line.strip() or not candidate_line.strip():
                continue
            chunk_record = json.loads(chunk_line)
            candidate_record = json.loads(candidate_line)
            candidates = _candidates_from_records(chunk_record, candidate_record)
            logits = reranker.score(chunk_record["query"], candidates, batch_size=batch_size)
            reranked = []
            for candidate, logit in zip(candidates, logits, strict=True):
                probability = _sigmoid(float(logit))
                reranked.append(
                    Candidate(
                        chunk=candidate.chunk,
                        retrieval_score=probability,
                        risk_score=candidate.risk_score,
                        source="eval_rerank",
                        scores={**candidate.scores, "rerank_score": float(logit), "rerank_prob": probability},
                    )
                )
            reranked.sort(key=lambda item: item.retrieval_score, reverse=True)
            ranked_ids = [candidate.chunk.chunk_id for candidate in reranked]
            gold_ids = set(chunk_record.get("gold_chunk_ids") or [])
            if not gold_ids:
                continue
            selection = select_budgeted_context(
                reranked,
                SelectionConfig(budget_tokens=budget_tokens),
            )
            selected_ids = {candidate.chunk.chunk_id for candidate in selection.selected}
            totals["records"] += 1
            totals["MRR@10"] += _mrr_at_k(ranked_ids, gold_ids, 10)
            totals["nDCG@10"] += _ndcg_at_k(ranked_ids, gold_ids, 10)
            totals["Recall@32"] += _recall_at_k(ranked_ids, gold_ids, 32)
            totals["Recall@64"] += _recall_at_k(ranked_ids, gold_ids, 64)
            totals["Recall@128"] += _recall_at_k(ranked_ids, gold_ids, 128)
            totals["EvidenceRecall@Budget"] += len(gold_ids & selected_ids) / len(gold_ids)
            totals["AllGoldSelected@Budget"] += float(gold_ids.issubset(selected_ids))
            totals["avg_used_tokens"] += selection.used_tokens
            totals["avg_selected_chunks"] += len(selection.selected)
            totals["avg_risk_added"] += len(selection.risk_added)

    records = int(totals["records"])
    metrics = {"checkpoint": str(checkpoint_dir), "step": step, "records": records}
    for key, value in totals.items():
        if key == "records":
            continue
        metrics[key] = value / records if records else 0.0
    return metrics


def write_checkpoint_eval_outputs(metrics: list[dict], output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = sorted(metrics, key=lambda row: int(row.get("step", 0)))
    jsonl_path = output_dir / "metrics_by_checkpoint.jsonl"
    csv_path = output_dir / "metrics_by_checkpoint.csv"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in metrics),
        encoding="utf-8",
    )
    if metrics:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)
    plot_paths = _write_metric_plots(metrics, output_dir / "plots")
    summary = {
        "status": "ok",
        "metrics_jsonl": str(jsonl_path),
        "metrics_csv": str(csv_path),
        "plot_paths": plot_paths,
        "checkpoints_evaluated": len(metrics),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _save_training_checkpoint(
    model,
    tokenizer,
    output_dir: Path,
    optimizer,
    scheduler,
    step: int,
    total_steps: int,
    reranker_config: RerankerConfig,
) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, tokenizer=tokenizer)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "total_steps": total_steps,
            "reranker_config": reranker_config.__dict__,
        },
        output_dir / "trainer_state.pt",
    )
    (output_dir / "_CHECKPOINT_DONE").write_text("ok\n", encoding="utf-8")


def _candidates_from_records(chunk_record: dict, candidate_record: dict) -> list[Candidate]:
    chunk_by_id = {chunk["chunk_id"]: chunk for chunk in chunk_record["chunks"]}
    candidates = []
    for item in candidate_record["candidates"]:
        chunk = chunk_by_id.get(item["chunk_id"])
        if chunk is None:
            continue
        candidates.append(
            Candidate(
                chunk=Chunk(
                    chunk_id=chunk["chunk_id"],
                    text=chunk["text"],
                    token_count=chunk["token_count"],
                    start_char=chunk.get("start_char"),
                    end_char=chunk.get("end_char"),
                    breadcrumbs=tuple(chunk.get("breadcrumbs") or []),
                    metadata=chunk.get("metadata") or {},
                ),
                retrieval_score=float(item.get("hybrid_norm", item.get("rrf_score", 0.0))),
                risk_score=float(item.get("risk_norm", item.get("risk_score", 0.0))),
                source="+".join(item.get("stage1_sources") or []),
                scores=item,
            )
        )
    return candidates


def _write_metric_plots(metrics: list[dict], output_dir: Path) -> list[str]:
    if not metrics:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    steps = [int(row["step"]) for row in metrics]
    plot_paths = []
    for metric_name in (
        "MRR@10",
        "nDCG@10",
        "Recall@32",
        "Recall@64",
        "Recall@128",
        "EvidenceRecall@Budget",
        "AllGoldSelected@Budget",
        "avg_used_tokens",
        "avg_risk_added",
    ):
        values = [float(row.get(metric_name, 0.0)) for row in metrics]
        plt.figure(figsize=(7, 4))
        plt.plot(steps, values, marker="o")
        plt.xlabel("Training step")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs training step")
        plt.grid(True, alpha=0.3)
        path = output_dir / f"{_safe_metric_name(metric_name)}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        plot_paths.append(str(path))
    return plot_paths


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = [
        path
        for path in output_dir.glob("checkpoint-step-*")
        if path.is_dir() and (path / "_CHECKPOINT_DONE").exists()
    ]
    return sorted(checkpoints)[-1] if checkpoints else None


def _load_torch_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import torch

    return torch.load(path, map_location="cpu")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _step_from_checkpoint(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return 0


def _mrr_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in gold_ids:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    dcg = 0.0
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in gold_ids:
            dcg += 1.0 / math.log2(rank + 1)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold_ids), k) + 1))
    return dcg / ideal if ideal else 0.0


def _recall_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & gold_ids) / len(gold_ids)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def _safe_metric_name(name: str) -> str:
    return name.replace("@", "at").replace("/", "_").replace(" ", "_")
