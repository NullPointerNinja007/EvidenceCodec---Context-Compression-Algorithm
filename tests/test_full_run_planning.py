from evidence_codec.data.parallel_hotpotqa import _shard_range
from evidence_codec.training.checkpoint_run import (
    _mrr_at_k,
    _ndcg_at_k,
    _recall_at_k,
    checkpoint_steps,
)


def test_shard_ranges_cover_rows_without_overlap() -> None:
    ranges = [_shard_range(10, 3, shard_id) for shard_id in range(3)]
    assert ranges == [(0, 3), (3, 6), (6, 10)]
    covered = [row for start, end in ranges for row in range(start, end)]
    assert covered == list(range(10))


def test_checkpoint_steps_include_final_and_are_unique() -> None:
    assert checkpoint_steps(100, checkpoint_count=6) == [17, 34, 50, 67, 84, 100]
    assert checkpoint_steps(3, checkpoint_count=6) == [1, 2, 3]


def test_ranking_metrics() -> None:
    ranked = ["a", "b", "c", "d"]
    gold = {"c", "d"}
    assert _mrr_at_k(ranked, gold, 10) == 1 / 3
    assert _recall_at_k(ranked, gold, 3) == 0.5
    assert 0.0 < _ndcg_at_k(ranked, gold, 10) < 1.0
