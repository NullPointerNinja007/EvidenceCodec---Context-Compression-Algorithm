from __future__ import annotations

from modal_apps.common import app, base_image, commit_all, volumes


@app.function(image=base_image, volumes=volumes, timeout=4 * 60 * 60)
def ingest(dataset: str = "hotpotqa", split: str = "train", subset: str | None = None) -> dict:
    from evidence_codec.modal_jobs.ingest import ingest_dataset

    result = ingest_dataset(dataset=dataset, split=split, subset=subset)
    commit_all()
    return result


@app.local_entrypoint()
def main(dataset: str = "hotpotqa", split: str = "train", subset: str | None = None) -> None:
    ingest.remote(dataset=dataset, split=split, subset=subset)
