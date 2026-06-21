from __future__ import annotations

from pprint import pprint

from modal_apps.common import app, base_image, commit_all, volumes


@app.function(image=base_image, volumes=volumes, timeout=2 * 60 * 60)
def build_candidates(
    dataset: str,
    split: str,
    subset: str | None = None,
    max_examples: int | None = None,
    overwrite: bool = False,
    dense_backend: str = "hashing",
    dense_model_name: str = "BAAI/bge-small-en-v1.5",
) -> dict:
    from evidence_codec.modal_jobs.build_candidates import build_candidates as run_build_candidates

    result = run_build_candidates(
        dataset=dataset,
        subset=subset,
        split=split,
        max_examples=max_examples,
        overwrite=overwrite,
        dense_backend=dense_backend,
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
    dense_backend: str = "hashing",
    dense_model_name: str = "BAAI/bge-small-en-v1.5",
) -> None:
    pprint(
        build_candidates.remote(
            dataset=dataset,
            subset=subset,
            split=split,
            max_examples=max_examples,
            overwrite=overwrite,
            dense_backend=dense_backend,
            dense_model_name=dense_model_name,
        )
    )
