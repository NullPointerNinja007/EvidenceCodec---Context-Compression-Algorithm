from __future__ import annotations

from typing import Protocol

import numpy as np

from evidence_codec.core.types import Chunk


class DenseScorer(Protocol):
    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        """Return cosine-like query-chunk similarity scores."""


class HashingDenseScorer:
    """Deterministic cosine scorer used for fast CPU tests.

    This is not the final semantic model. It gives the same stage-one code a
    lightweight frozen cosine backend when sentence-transformer weights are not
    available.
    """

    def __init__(self, n_features: int = 4096, analyzer: str = "char_wb") -> None:
        self.n_features = n_features
        self.analyzer = analyzer

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.preprocessing import normalize

        vectorizer = HashingVectorizer(
            analyzer=self.analyzer,
            ngram_range=(3, 5) if self.analyzer == "char_wb" else (1, 2),
            n_features=self.n_features,
            alternate_sign=False,
            norm=None,
        )
        matrix = vectorizer.transform([query, *[chunk.text for chunk in chunks]])
        matrix = normalize(matrix, norm="l2", axis=1)
        query_vec = matrix[0]
        chunk_matrix = matrix[1:]
        return [float(score) for score in (chunk_matrix @ query_vec.T).toarray().ravel()]


class SentenceTransformerDenseScorer:
    """Frozen sentence-transformer cosine scorer for real dense retrieval."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        if not chunks:
            return []
        model = self._load()
        texts = [query, *[chunk.text for chunk in chunks]]
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embedding = embeddings[0]
        chunk_embeddings = embeddings[1:]
        return [float(score) for score in np.dot(chunk_embeddings, query_embedding)]

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model


def build_dense_scorer(backend: str = "hashing", model_name: str | None = None) -> DenseScorer:
    if backend == "hashing":
        return HashingDenseScorer()
    if backend in {"sentence-transformers", "sentence_transformers", "sbert"}:
        return SentenceTransformerDenseScorer(model_name=model_name or "BAAI/bge-small-en-v1.5")
    raise ValueError(f"Unsupported dense backend: {backend}")
