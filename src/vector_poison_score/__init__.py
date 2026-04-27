"""vector_poison_score -- score (query, document) pairs for vector/RAG poisoning.

Public surface (Python port of @mukundakatta/vector-poison-score):

    from vector_poison_score import score, filter_poisoned, PoisonScore

* ``score(query_vec, doc_vec, doc_text, query_text)`` -- per-pair PoisonScore.
* ``filter_poisoned(records, max_score=0.5)`` -- drop records above the threshold.
* ``PoisonScore`` -- dataclass: score, signals, severity ('low'|'medium'|'high').

Pure Python, zero runtime deps. Vector ops are stdlib loops; install the optional
``[numpy]`` extra and pass ``numpy.ndarray`` for faster math.
"""

from .score import (
    PoisonScore,
    Severity,
    filter_poisoned,
    score,
)

__version__ = "0.1.0"
VERSION = __version__

__all__ = [
    "VERSION",
    "PoisonScore",
    "Severity",
    "filter_poisoned",
    "score",
]
