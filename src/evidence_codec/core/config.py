from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkingConfig:
    target_tokens: int = 160
    soft_max_tokens: int = 192
    hard_max_tokens: int = 224
    overlap_tokens: int = 32


@dataclass(frozen=True)
class RetrievalConfig:
    bm25_top_k: int = 96
    dense_top_k: int = 96
    risk_top_k: int = 32
    candidate_cap: int = 256
    rrf_c: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    risk_weight: float = 0.75
    dense_backend: str = "hashing"
    dense_model_name: str = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class RerankerConfig:
    model_name: str = "answerdotai/ModernBERT-large"
    sequence_length: int = 512
    query_max_tokens: int = 96
    precision: str = "bf16"
    dropout: float = 0.1
    bce_weight: float = 1.0
    ranking_weight: float = 0.5
    margin: float = 0.2


@dataclass(frozen=True)
class SelectionConfig:
    budget_tokens: int = 4096
    length_exponent: float = 0.9
    redundancy_penalty: float = 0.15
    risk_addback_fraction: float = 0.10
    relevance_threshold: float = 0.35
    risk_threshold: float = 0.55
