from __future__ import annotations

from pprint import pprint

from modal_apps.common import app, base_image, commit_all, volumes


@app.function(image=base_image, volumes=volumes, timeout=60 * 60)
def build_selections(
    dataset: str,
    split: str,
    subset: str | None = None,
    max_examples: int | None = None,
    candidate_cap: int = 256,
    budget_tokens: int = 512,
    overwrite: bool = False,
) -> dict:
    from evidence_codec.modal_jobs.build_selections import build_selections as run_build_selections

    result = run_build_selections(
        dataset=dataset,
        subset=subset,
        split=split,
        max_examples=max_examples,
        candidate_cap=candidate_cap,
        budget_tokens=budget_tokens,
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
    candidate_cap: int = 256,
    budget_tokens: int = 512,
    overwrite: bool = False,
) -> None:
    pprint(
        build_selections.remote(
            dataset=dataset,
            subset=subset,
            split=split,
            max_examples=max_examples,
            candidate_cap=candidate_cap,
            budget_tokens=budget_tokens,
            overwrite=overwrite,
        )
    )
