from __future__ import annotations

from pprint import pprint

from modal_apps.common import app, commit_all, training_image, volumes


@app.function(image=training_image, volumes=volumes, gpu="B200", timeout=2 * 60 * 60)
def build_candidates_sbert(
    dataset: str,
    split: str,
    subset: str | None = None,
    max_examples: int | None = None,
    overwrite: bool = False,
    dense_model_name: str = "BAAI/bge-small-en-v1.5",
) -> dict:
    from evidence_codec.modal_jobs.build_candidates import build_candidates

    result = build_candidates(
        dataset=dataset,
        subset=subset,
        split=split,
        max_examples=max_examples,
        overwrite=overwrite,
        dense_backend="sentence-transformers",
        dense_model_name=dense_model_name,
    )
    commit_all()
    return result


@app.local_entrypoint()
def main(
    dataset: str,
    split: str,
    subset: str | None = None,
    max_examples: int | None = None,
    overwrite: bool = False,
    dense_model_name: str = "BAAI/bge-small-en-v1.5",
) -> None:
    pprint(
        build_candidates_sbert.remote(
            dataset=dataset,
            subset=subset,
            split=split,
            max_examples=max_examples,
            overwrite=overwrite,
            dense_model_name=dense_model_name,
        )
    )
