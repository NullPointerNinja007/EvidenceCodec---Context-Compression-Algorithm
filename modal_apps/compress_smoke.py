from __future__ import annotations

from pprint import pprint

from modal_apps.common import app, commit_all, training_image, volumes


@app.function(image=training_image, volumes=volumes, gpu="B200", timeout=60 * 60)
def compress_smoke(model_name: str = "answerdotai/ModernBERT-large") -> dict:
    import torch

    from evidence_codec.core.config import (
        ChunkingConfig,
        RerankerConfig,
        RetrievalConfig,
        SelectionConfig,
    )
    from evidence_codec.pipeline import CompressionPipelineConfig, compress_context

    context = "\n\n".join(
        [
            "The cafeteria serves lunch between 11 AM and 1 PM.",
            "Students may submit late work for homework.",
            "The final essay has no extension period.",
            "The library closes early on Friday during summer sessions.",
        ]
    )
    config = CompressionPipelineConfig(
        chunking=ChunkingConfig(soft_max_tokens=16, hard_max_tokens=32, overlap_tokens=0),
        retrieval=RetrievalConfig(bm25_top_k=3, dense_top_k=3, risk_top_k=3, candidate_cap=3),
        reranker=RerankerConfig(model_name=model_name, precision="bf16"),
        selection=SelectionConfig(budget_tokens=24, risk_addback_fraction=0.10),
        rerank_batch_size=4,
        use_reranker=True,
    )
    result = compress_context(
        context=context,
        query="What has no extension period?",
        document_id="modal-smoke",
        config=config,
    )
    commit_all()
    return {
        "status": "ok",
        "cuda_device": torch.cuda.get_device_name(0),
        "model_name": model_name,
        "compressed_context": result.compressed_context,
        "selected_chunk_ids": [chunk.chunk_id for chunk in result.selected_chunks],
        "metrics": result.metrics,
    }


@app.local_entrypoint()
def main(model_name: str = "answerdotai/ModernBERT-large") -> None:
    pprint(compress_smoke.remote(model_name=model_name))
