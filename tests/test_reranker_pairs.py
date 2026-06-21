import json

from evidence_codec.data.reranker_pairs import load_reranker_examples, pairs_from_records


def test_pairs_from_records_keeps_positives_and_ranked_negatives() -> None:
    chunk_record = {
        "example_id": "ex",
        "query": "question",
        "chunks": [
            {"chunk_id": "pos", "text": "answer evidence"},
            {"chunk_id": "neg1", "text": "hard negative"},
            {"chunk_id": "neg2", "text": "other negative"},
        ],
        "labels": [1, 0, 0],
    }
    candidate_record = {
        "candidate_ids": ["neg1", "pos", "neg2"],
        "candidates": [
            {"chunk_id": "neg1", "stage1_rank": 1},
            {"chunk_id": "pos", "stage1_rank": 2},
            {"chunk_id": "neg2", "stage1_rank": 3},
        ],
    }
    pairs = pairs_from_records(chunk_record, candidate_record, negatives_per_positive=1)
    assert [pair.chunk_id for pair in pairs] == ["pos", "neg1"]
    assert [pair.label for pair in pairs] == [1, 0]


def test_load_reranker_examples_groups_by_example(tmp_path) -> None:
    chunk_path = tmp_path / "chunks.jsonl"
    candidate_path = tmp_path / "candidates.jsonl"
    chunk_records = [
        {
            "example_id": "ex1",
            "query": "q1",
            "chunks": [
                {"chunk_id": "ex1-pos", "text": "pos"},
                {"chunk_id": "ex1-neg", "text": "neg"},
            ],
            "labels": [1, 0],
        },
        {
            "example_id": "ex2",
            "query": "q2",
            "chunks": [
                {"chunk_id": "ex2-pos", "text": "pos"},
                {"chunk_id": "ex2-neg", "text": "neg"},
            ],
            "labels": [1, 0],
        },
    ]
    candidate_records = [
        {"candidate_ids": ["ex1-pos", "ex1-neg"], "candidates": []},
        {"candidate_ids": ["ex2-pos", "ex2-neg"], "candidates": []},
    ]
    chunk_path.write_text("\n".join(json.dumps(item) for item in chunk_records), encoding="utf-8")
    candidate_path.write_text(
        "\n".join(json.dumps(item) for item in candidate_records),
        encoding="utf-8",
    )

    examples = load_reranker_examples(chunk_path, candidate_path, negatives_per_positive=1)
    assert [example["group_id"] for example in examples] == [0, 0, 1, 1]
