from __future__ import annotations

import re
from typing import Any

import pandas as pd


STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "another",
    "because",
    "before",
    "being",
    "between",
    "could",
    "doing",
    "during",
    "every",
    "having",
    "should",
    "their",
    "there",
    "these",
    "thing",
    "things",
    "those",
    "through",
    "under",
    "where",
    "which",
    "while",
    "would",
    "your",
    "with",
    "without",
    "people",
    "person",
    "someone",
    "something",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_text(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(value: Any, *, remove_stopwords: bool = True) -> set[str]:
    out = set()
    for token in normalize_text(value).split():
        if len(token) < 2 or token.isdigit():
            continue
        if remove_stopwords and token in STOPWORDS:
            continue
        out.add(token)
    return out


def token_jaccard(left: Any, right: Any) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def length_ratio(left: Any, right: Any) -> float:
    left_len = max(1, len(normalize_text(left).split()))
    right_len = max(1, len(normalize_text(right).split()))
    return min(left_len, right_len) / max(left_len, right_len)


def quality_bin(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"
