from __future__ import annotations

import re

from evidence_codec.core.types import Chunk

NEGATION_RE = re.compile(r"\b(no|not|never|cannot|can't|without)\b", re.IGNORECASE)
EXCEPTION_RE = re.compile(r"\b(unless|except|only if|provided that)\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*(?:%|[a-zA-Z]+)?\b")
DATE_RE = re.compile(r"\b(19|20)\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", re.IGNORECASE)
MONEY_RE = re.compile(r"[$€£]\s?\d+|\b\d+(?:\.\d+)?\s?(?:usd|eur|gbp)\b", re.IGNORECASE)
CODE_RE = re.compile(r"`[^`]+`|\b[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\)|\b[\w./-]+\.(py|js|ts|json|yaml|yml)\b")
TABLE_RE = re.compile(r"^\s*\|.+\|\s*$", re.MULTILINE)
EQUATION_RE = re.compile(r"[=<>]=?|\\frac|\\sum|\\int")
COMPARISON_RE = re.compile(
    r"\b(greater|less|before|after|higher|lower|minimum|maximum|first|last|earlier|later|more|fewer)\b",
    re.IGNORECASE,
)
ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*\b")


def risk_prior(chunk: Chunk, query: str = "") -> float:
    features = risk_features(chunk, query=query)
    return (
        2.0 * features["negation"]
        + 2.0 * features["exception"]
        + 1.5 * features["number"]
        + 1.5 * features["date"]
        + 1.5 * features["money"]
        + 1.2 * features["comparison"]
        + 1.5 * features["code"]
        + 1.5 * features["table"]
        + 1.2 * features["equation"]
        + 1.0 * features["entity_overlap"]
        + 1.0 * features["query_number_overlap"]
        + 0.5 * features["heading_overlap"]
    )


def risk_features(chunk: Chunk, query: str = "") -> dict[str, bool]:
    text = chunk.text
    return {
        "negation": bool(NEGATION_RE.search(text)),
        "exception": bool(EXCEPTION_RE.search(text)),
        "number": bool(NUMBER_RE.search(text)),
        "date": bool(DATE_RE.search(text)),
        "money": bool(MONEY_RE.search(text)),
        "comparison": bool(COMPARISON_RE.search(text)),
        "code": bool(CODE_RE.search(text)),
        "table": bool(TABLE_RE.search(text)) or _metadata_marks_table(chunk),
        "equation": bool(EQUATION_RE.search(text)),
        "entity_overlap": _entity_overlap(query, text),
        "query_number_overlap": _query_number_overlap(query, text),
        "heading_overlap": _heading_overlap(query, chunk),
    }


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    denom = high - low
    if denom <= 1e-12:
        return [0.0 for _ in scores]
    return [(score - low) / denom for score in scores]


def _query_number_overlap(query: str, text: str) -> bool:
    return bool(set(NUMBER_RE.findall(query)) & set(NUMBER_RE.findall(text)))


def _entity_overlap(query: str, text: str) -> bool:
    query_entities = {match.group(0).lower() for match in ENTITY_RE.finditer(query)}
    text_entities = {match.group(0).lower() for match in ENTITY_RE.finditer(text)}
    return bool(query_entities & text_entities)


def _heading_overlap(query: str, chunk: Chunk) -> bool:
    if not chunk.breadcrumbs:
        return False
    query_terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
    heading_terms = set(re.findall(r"[a-zA-Z0-9]+", " ".join(chunk.breadcrumbs).lower()))
    return bool(query_terms & heading_terms)


def _metadata_marks_table(chunk: Chunk) -> bool:
    unit_types = chunk.metadata.get("unit_types", [])
    return any("table" in str(unit_type).lower() for unit_type in unit_types)
