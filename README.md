# EvidenceCodec: Context-Compression-Algorithm

My submission for Token compression challenge by The Token Company (YC 26)

EvidenceCodec is a raw-text context compression system for the Token
Compression Challenge. The architecture follows the reports in `docs/`:
deterministic structure-aware chunking, hybrid BM25 + dense + risk candidate
recall, ModernBERT cross-encoder reranking, greedy budget selection,
RiskGuard add-back, and optional student distillation.

## Repository Layout

```text
configs/                  Reproducible run and experiment settings
docs/                     Source architecture reports
modal_apps/               Thin Modal entrypoints only
scripts/                  Local helper commands that invoke Modal
src/evidence_codec/       Real implementation package
tests/                    Local unit and smoke tests
```

The repo intentionally does not contain local `data/`, `models/`, `runs/`,
or `checkpoints/` directories. Large files belong in Modal Volumes mounted as:

```text
/data    raw and processed datasets, indexes, teacher scores
/models  ModernBERT checkpoints, distilled students, tokenizer snapshots
/runs    metrics, eval tables, logs, curated demo outputs
/cache   Hugging Face, sentence-transformer, and temporary build caches
```

## Modal Policy

Modal wrappers live in `modal_apps/` and should stay small. Do not put the
training pipeline, data preparation, evaluation, or serving logic in those
files. Put implementation code under `src/evidence_codec/` and import it from
the wrapper.

No Modal job has been launched by this scaffold. Running a wrapper is what will
spend credits.

Example commands once dependencies and Modal auth are ready:

```bash
modal run modal_apps/ingest.py --dataset hotpotqa --split train
modal run modal_apps/train_reranker.py --config configs/experiments/modernbert_large.yaml
modal run modal_apps/evaluate.py --config configs/default.yaml
modal deploy modal_apps/serve.py
```
