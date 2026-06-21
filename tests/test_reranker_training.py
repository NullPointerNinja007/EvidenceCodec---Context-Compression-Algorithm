import pytest

from evidence_codec.training.reranker import _accumulation_windows, _grouped_batch_indices


def test_accumulation_windows_collect_microbatches_until_effective_size() -> None:
    microbatches = [[0, 1], [2, 3, 4], [5, 6, 7, 8], [9]]
    windows = _accumulation_windows(microbatches, effective_batch_size=5)
    assert windows == [
        [[0, 1], [2, 3, 4]],
        [[5, 6, 7, 8], [9]],
    ]


def test_accumulation_windows_allow_oversized_query_group() -> None:
    microbatches = [[0, 1, 2, 3, 4, 5], [6]]
    windows = _accumulation_windows(microbatches, effective_batch_size=4)
    assert windows == [
        [[0, 1, 2, 3, 4, 5]],
        [[6]],
    ]


def test_accumulation_windows_reject_invalid_effective_batch_size() -> None:
    with pytest.raises(ValueError, match="effective_batch_size"):
        _accumulation_windows([[0]], effective_batch_size=0)


def test_grouped_batch_indices_preserve_query_groups_before_accumulation() -> None:
    examples = [
        {"group_id": 3},
        {"group_id": 3},
        {"group_id": 9},
        {"group_id": 3},
        {"group_id": 9},
    ]
    assert _grouped_batch_indices(examples) == [[0, 1, 3], [2, 4]]
