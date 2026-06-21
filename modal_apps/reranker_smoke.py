from __future__ import annotations

from pprint import pprint

from modal_apps.common import app, training_image, volumes


@app.function(image=training_image, volumes=volumes, gpu="B200", timeout=60 * 60)
def smoke(model_name: str = "answerdotai/ModernBERT-large") -> dict:
    import torch

    from evidence_codec.core.config import RerankerConfig
    from evidence_codec.core.types import Candidate, Chunk
    from evidence_codec.rerank import CrossEncoderReranker
    from evidence_codec.training.losses import bce_plus_pairwise_margin_loss

    config = RerankerConfig(model_name=model_name, precision="bf16")
    reranker = CrossEncoderReranker(model_name_or_path=model_name, device="cuda", config=config)
    query = "When can students submit late work?"
    candidates = [
        Candidate(
            chunk=Chunk(
                chunk_id="positive",
                text="Students may submit late work unless the assignment is the final essay.",
                token_count=13,
            )
        ),
        Candidate(
            chunk=Chunk(
                chunk_id="negative",
                text="The cafeteria serves lunch between 11 AM and 1 PM.",
                token_count=11,
            )
        ),
    ]
    scores = reranker.score(query, candidates, batch_size=2)
    logits = torch.tensor([2.0, -1.0], device="cuda")
    labels = torch.tensor([1.0, 0.0], device="cuda")
    groups = torch.tensor([0, 0], device="cuda")
    _, loss_metrics = bce_plus_pairwise_margin_loss(logits, labels, group_ids=groups)
    return {
        "status": "ok",
        "model_name": model_name,
        "scores": scores,
        "score_count": len(scores),
        "loss_metrics": loss_metrics,
        "cuda_device": torch.cuda.get_device_name(0),
    }


@app.local_entrypoint()
def main(model_name: str = "answerdotai/ModernBERT-large") -> None:
    pprint(smoke.remote(model_name=model_name))
