# vector-poison-score

[![PyPI](https://img.shields.io/pypi/v/vector-poison-score.svg)](https://pypi.org/project/vector-poison-score/)
[![Python](https://img.shields.io/pypi/pyversions/vector-poison-score.svg)](https://pypi.org/project/vector-poison-score/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Score (query, document) pairs for vector/RAG poisoning signals.** Detects vector-text mismatch, zero/NaN/suspiciously-round vectors, instruction-like payloads, link farms, oversized chunks. Pure Python, zero runtime dependencies.

Python port of [@mukundakatta/vector-poison-score](https://github.com/MukundaKatta/vector-poison-score). The JS sibling has the original heuristics; this port adds vector-aware signals on top.

## Install

```bash
pip install vector-poison-score
# Optional: faster cosine via numpy
pip install "vector-poison-score[numpy]"
```

## Usage

```python
from vector_poison_score import score, filter_poisoned

q_vec = [0.1, 0.2, 0.3, 0.4]
d_vec = [0.1, 0.2, 0.3, 0.41]   # very close to q_vec
q_text = "Who is the CEO of Anthropic?"
d_text = "Buy cheap watches at example.com"   # totally unrelated

s = score(q_vec, d_vec, d_text, q_text)
s.score        # 0.25 (one signal)
s.signals      # ["vector_text_mismatch"]
s.severity     # "low"

# Bulk-filter retrieved chunks
records = [
    {"id": "a", "embedding": q_vec, "text": "About Anthropic..."},
    {"id": "b", "embedding": d_vec, "text": d_text},
]
filter_poisoned(
    records,
    max_score=0.2,
    query_vec=q_vec,
    query_text=q_text,
)   # -> drops record 'b'
```

## Signals

| Signal | Trigger |
|---|---|
| `vector_text_mismatch` | Cosine(query, doc) >= 0.7 but Jaccard(query terms, doc terms) <= 0.1. The classic poisoned-embedding signature. |
| `zero_vector` | Doc vector is all zeros. |
| `nan_vector` | Doc vector contains NaN or +/-inf. |
| `suspiciously_round` | >= 70% of doc-vector entries are within 1e-3 of an integer (likely hand-crafted). |
| `instruction_like_payload` | Doc text matches the prompt-injection regex (mirrors JS sibling). |
| `link_farm` | Doc text has > 5 URLs (mirrors JS sibling). |
| `oversized_chunk` | Doc text > 20000 chars (mirrors JS sibling). |

`score = min(1.0, len(signals) / 4)`. Severity: `>= 0.66 -> high`, `>= 0.33 -> medium`, else `low`.

## API differences from the JS sibling

* JS: `scoreVectorPoison({ text })`, text-only. Python: `score(query_vec, doc_vec, doc_text, query_text)` -- adds vector-aware signals (mismatch, zero, NaN, round) on top of the text checks.
* JS: `filterPoisoned(docs, { maxScore })`. Python: `filter_poisoned(records, max_score=, text_key=, vector_key=, query_vec=, query_text=)` -- record-shape configurable, query inputs optional.
* `PoisonScore` is a frozen dataclass; severity is a typed string literal.

See the JS sibling for the original heuristics and broader design notes.
