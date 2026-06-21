from evidence_codec.rerank import encode_query_chunk_pair


class FakeTokenizer:
    sep_token_id = 102

    def encode(self, text, add_special_tokens=False, truncation=False, max_length=None):
        tokens = list(range(1, len(text.split()) + 1))
        if truncation and max_length is not None:
            return tokens[:max_length]
        return tokens

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def build_inputs_with_special_tokens(self, query_ids, chunk_ids):
        return [101, *query_ids, 102, *chunk_ids, 102]


def test_pair_encoding_caps_query_and_total_length() -> None:
    tokenizer = FakeTokenizer()
    encoded = encode_query_chunk_pair(
        tokenizer,
        query=" ".join(["q"] * 100),
        chunk_text=" ".join(["c"] * 100),
        sequence_length=32,
        query_max_tokens=8,
    )
    assert len(encoded["input_ids"]) == 32
    assert encoded["input_ids"][0] == 101
    assert encoded["input_ids"][-1] == 102
    assert len(encoded["attention_mask"]) == 32
