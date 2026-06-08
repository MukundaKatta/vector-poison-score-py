"""Score (query, document) pairs for vector/RAG poisoning signals.

Detects:

* **vector_text_mismatch** -- query/doc vectors are close (cosine >= 0.7) but
  the texts share little token overlap (Jaccard <= 0.1). The classic poisoned-
  embedding signature.
* **zero_vector** -- doc vector is all zeros (degenerate / placeholder).
* **nan_vector** -- doc vector contains NaN or +/-inf.
* **suspiciously_round** -- doc vector entries are heavily integer-valued (likely
  hand-crafted rather than learned).
* **instruction_like_payload** -- doc text contains prompt-injection markers.
  Mirrors the JS sibling's regex.
* **link_farm** -- doc text has >5 URLs (mirrors JS).
* **oversized_chunk** -- doc text > 20000 chars (mirrors JS).

The aggregate score is ``min(1.0, len(signals) / 4)``; severity tiers are
``score >= 0.66 -> high``, ``>= 0.33 -> medium``, else ``low``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal, Sequence

Severity = Literal["low", "medium", "high"]

# Mirrors the JS sibling's instruction-injection regex.
_BAD_INSTRUCTION_RE = re.compile(
    r"\b(ignore previous|system prompt|developer instruction|exfiltrate|jailbreak|override)\b",
    re.IGNORECASE,
)
_URL_RE = re.compile(r"https?://")
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Tunable knobs -- expose if users ask.
_VECTOR_CLOSE_THRESHOLD = 0.7  # cosine sim above this counts as "close"
_TEXT_DISJOINT_THRESHOLD = 0.1  # Jaccard below this counts as "disjoint"
_ROUND_FRACTION_THRESHOLD = 0.7  # >=70% near-integer entries -> suspicious
_LINK_FARM_THRESHOLD = 5
_OVERSIZED_CHUNK_THRESHOLD = 20_000


@dataclass(frozen=True)
class PoisonScore:
    """Aggregate poisoning signal for one (query, document) pair."""

    score: float  # 0..1
    signals: list[str] = field(default_factory=list)
    severity: Severity = "low"


# ----- helpers --------------------------------------------------------------


def _to_floats(vec: Sequence[float] | None) -> list[float]:
    if vec is None:
        return []
    out: list[float] = []
    for x in vec:
        try:
            out.append(float(x))
        except (TypeError, ValueError):
            # Non-numeric entries propagate as NaN so the nan signal trips.
            out.append(math.nan)
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity. Returns 0.0 if either side is zero or NaN."""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = aa = bb = 0.0
    for i in range(n):
        x, y = a[i], b[i]
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            return 0.0
        dot += x * y
        aa += x * x
        bb += y * y
    if aa == 0.0 or bb == 0.0:
        return 0.0
    return dot / (math.sqrt(aa) * math.sqrt(bb))


def _terms(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _is_zero_vector(vec: list[float]) -> bool:
    return bool(vec) and all(x == 0.0 for x in vec)


def _has_nan_or_inf(vec: list[float]) -> bool:
    return any(math.isnan(x) or math.isinf(x) for x in vec)


def _round_fraction(vec: list[float]) -> float:
    """Fraction of entries within 1e-3 of an integer (excluding NaN/inf)."""
    if not vec:
        return 0.0
    near_int = sum(
        1
        for x in vec
        if not (math.isnan(x) or math.isinf(x)) and abs(x - round(x)) < 1e-3
    )
    return near_int / len(vec)


def _severity_for(score: float) -> Severity:
    if score >= 0.66:
        return "high"
    if score >= 0.33:
        return "medium"
    return "low"


# ----- public API ------------------------------------------------------------


def score(
    query_vec: Sequence[float] | None,
    doc_vec: Sequence[float] | None,
    doc_text: str,
    query_text: str = "",
) -> PoisonScore:
    """Score one (query, document) pair for poisoning signals.

    Args:
        query_vec: Query embedding (or ``None`` to skip vector-aware checks).
        doc_vec:   Document embedding.
        doc_text:  Document text.
        query_text: Original query text (used for the vector-text mismatch check).

    Returns:
        PoisonScore with the aggregate score, list of tripped signals, and
        severity tier.
    """
    if not isinstance(doc_text, str):
        raise TypeError("doc_text must be a str")
    if not isinstance(query_text, str):
        raise TypeError("query_text must be a str")
    q = _to_floats(query_vec)
    d = _to_floats(doc_vec)
    signals: list[str] = []

    # ---- vector-only signals
    if d and _has_nan_or_inf(d):
        signals.append("nan_vector")
    if d and _is_zero_vector(d):
        signals.append("zero_vector")
    if d and not _is_zero_vector(d) and _round_fraction(d) >= _ROUND_FRACTION_THRESHOLD:
        signals.append("suspiciously_round")

    # ---- vector-text mismatch (needs both vectors and both texts)
    if q and d and query_text and doc_text:
        sim = _cosine(q, d)
        text_overlap = _jaccard(_terms(query_text), _terms(doc_text))
        if sim >= _VECTOR_CLOSE_THRESHOLD and text_overlap <= _TEXT_DISJOINT_THRESHOLD:
            signals.append("vector_text_mismatch")

    # ---- text-only signals (mirror the JS sibling)
    if _BAD_INSTRUCTION_RE.search(doc_text):
        signals.append("instruction_like_payload")
    if len(_URL_RE.findall(doc_text)) > _LINK_FARM_THRESHOLD:
        signals.append("link_farm")
    if len(doc_text) > _OVERSIZED_CHUNK_THRESHOLD:
        signals.append("oversized_chunk")

    # Aggregate: each signal contributes 0.25, capped at 1.0 (matches JS spirit:
    # JS uses ``min(1, len / 3)`` over 3 signals; we have 7, scale to 4).
    raw = min(1.0, len(signals) / 4)
    return PoisonScore(
        score=round(raw * 10000) / 10000, signals=signals, severity=_severity_for(raw)
    )


def filter_poisoned(
    records: Sequence[dict],
    *,
    max_score: float = 0.5,
    text_key: str = "text",
    vector_key: str = "embedding",
    query_vec: Sequence[float] | None = None,
    query_text: str = "",
) -> list[dict]:
    """Drop records whose ``score(...)`` exceeds ``max_score``.

    Each record is a dict with ``text_key`` and (optionally) ``vector_key`` fields.
    ``query_vec`` / ``query_text`` are used for the vector-text mismatch check;
    leave them empty to run text-only signals.
    """
    if not isinstance(records, (list, tuple)):
        raise TypeError("records must be a list or tuple")
    if not 0.0 <= max_score <= 1.0:
        raise TypeError("max_score must be in [0, 1]")
    out: list[dict] = []
    for r in records:
        if not isinstance(r, dict):
            raise TypeError("each record must be a dict")
        text = str(r.get(text_key, ""))
        vec = r.get(vector_key)
        s = score(query_vec, vec, text, query_text)
        if s.score <= max_score:
            out.append(r)
    return out
