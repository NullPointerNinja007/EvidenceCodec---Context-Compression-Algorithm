# Project Structure Notes

The two reports in this directory agree on these implementation boundaries:

1. The compressor outputs selected raw text, not summaries or latent vectors.
2. Chunking is deterministic, hierarchical, and structure preserving.
3. Stage one is high-recall candidate generation with BM25, frozen dense
   retrieval, and risk rescue.
4. The quality scorer is a ModernBERT cross-encoder over query-chunk pairs.
5. Selection is budget-aware and followed by deterministic RiskGuard add-back.
6. A cheap student can be distilled from cross-encoder scores for default
   production serving.
7. LongBench is evaluation-only; HotpotQA and QASPER drive supervision.

Those boundaries map to `src/evidence_codec/` as follows:

```text
chunking/       structure-aware chunk creation
retrieval/      BM25, dense retrieval, RRF fusion, candidate caps
risk/           fragile-evidence feature extraction and risk scores
rerank/         ModernBERT cross-encoder scoring
selection/      token-budget selector and RiskGuard add-back
data/           HotpotQA/QASPER/LongBench ingestion and label projection
training/       reranker training, losses, hard negatives, student distill
evaluation/     recall, nDCG/MRR, compression, answer, and silent-loss metrics
pipeline/       end-to-end compression orchestration
serving/        API layer around the compressor
modal_jobs/     implementation functions called by thin Modal wrappers
storage/        Modal volume mount paths and layout helpers
```

Large data should not be added to this repository. Store generated artifacts in
Modal Volumes:

```text
/data/raw              source dataset downloads
/data/processed        chunk labels, candidate pools, indexes
/data/teacher_scores   cross-encoder outputs for distillation
/models/rerankers      ModernBERT checkpoints
/models/students       LightGBM or MLP students
/runs/evaluations      metrics and benchmark tables
/runs/demos            curated side-by-side demo outputs
/cache/huggingface     model and dataset cache
```
