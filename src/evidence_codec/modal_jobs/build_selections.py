from __future__ import annotations

from evidence_codec.core.config import SelectionConfig
from evidence_codec.data.build_selections import build_selection_split
from evidence_codec.storage import DEFAULT_LAYOUT


def build_selections(
    dataset: str,
    split: str,
    subset: str | None = None,
    max_examples: int | None = None,
    candidate_cap: int = 256,
    budget_tokens: int = 512,
    overwrite: bool = False,
) -> dict:
    DEFAULT_LAYOUT.ensure()
    return build_selection_split(
        dataset=dataset,
        subset=subset,
        split=split,
        max_examples=max_examples,
        candidate_cap=candidate_cap,
        budget_tokens=budget_tokens,
        overwrite=overwrite,
        config=SelectionConfig(budget_tokens=budget_tokens),
    )
