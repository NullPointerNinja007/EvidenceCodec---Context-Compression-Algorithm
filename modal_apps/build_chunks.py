from __future__ import annotations

from pprint import pprint

from modal_apps.common import app, base_image, commit_all, volumes


@app.function(image=base_image, volumes=volumes, timeout=2 * 60 * 60)
def build_chunks(
    dataset: str,
    split: str,
    subset: str | None = None,
    max_examples: int | None = None,
    overwrite: bool = False,
) -> dict:
    from evidence_codec.modal_jobs.build_chunks import build_chunks as run_build_chunks

    result = run_build_chunks(
        dataset=dataset,
        subset=subset,
        split=split,
        max_examples=max_examples,
        overwrite=overwrite,
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
) -> None:
    pprint(
        build_chunks.remote(
            dataset=dataset,
            subset=subset,
            split=split,
            max_examples=max_examples,
            overwrite=overwrite,
        )
    )
