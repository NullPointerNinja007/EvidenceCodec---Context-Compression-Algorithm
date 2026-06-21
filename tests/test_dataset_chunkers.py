from evidence_codec.chunking.dataset_chunkers import chunk_hotpotqa_record, chunk_qasper_record


def test_hotpotqa_supporting_fact_projection() -> None:
    row = {
        "id": "ex1",
        "question": "Where was Alice born?",
        "answer": "Paris",
        "type": "bridge",
        "level": "easy",
        "context": {
            "title": ["Alice", "Paris"],
            "sentences": [
                ["Alice is a researcher.", "Alice was born in Paris."],
                ["Paris is in France."],
            ],
        },
        "supporting_facts": {"title": ["Alice"], "sent_id": [1]},
    }
    record = chunk_hotpotqa_record(row, split="train", row_index=0)
    assert sum(record["labels"]) >= 1
    assert record["gold_chunk_ids"]
    positive = [
        chunk
        for chunk, label in zip(record["chunks"], record["labels"], strict=False)
        if label
    ][0]
    assert "Alice was born in Paris." in positive["text"]


def test_qasper_highlighted_evidence_projection() -> None:
    row = {
        "id": "paper1",
        "title": "A Test Paper",
        "abstract": "This paper studies compression.",
        "full_text": {
            "section_name": ["Introduction"],
            "paragraphs": [["The model uses discourse relations to propagate polarity."]],
        },
        "figures_and_tables": {"caption": [], "file": []},
        "qas": {
            "question_id": ["q1"],
            "question": ["How does the model propagate polarity?"],
            "answers": [
                [
                    {
                        "answer": "using discourse relations",
                        "highlighted_evidence": [
                            "The model uses discourse relations to propagate polarity."
                        ],
                        "evidence": [],
                        "unanswerable": False,
                        "yes_no": None,
                    }
                ]
            ],
        },
    }
    records = chunk_qasper_record(row, split="train", row_index=0)
    assert len(records) == 1
    assert sum(records[0]["labels"]) >= 1
    assert records[0]["gold_chunk_ids"]
