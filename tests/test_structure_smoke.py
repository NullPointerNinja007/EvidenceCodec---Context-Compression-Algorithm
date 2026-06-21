from evidence_codec.chunking import chunk_text
from evidence_codec.pipeline.compress import compress_with_scores


def test_chunking_smoke() -> None:
    chunks = chunk_text("A first paragraph.\n\nA second paragraph with 2026 details.")
    assert chunks
    assert chunks[0].text


def test_compression_smoke() -> None:
    result = compress_with_scores(
        context="The policy allows late work unless the assignment is final.",
        query="When is late work allowed?",
        budget_tokens=64,
    )
    assert "policy allows late work" in result.compressed_context
    assert result.selected_chunks
