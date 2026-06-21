"""Cheap lexical relevance: IDF-weighted content-word overlap.

Shared by the binder's candidate pre-filter (which span could support a claim) and the search
re-ranker (which result is most relevant to the question), so there is ONE implementation. Pure
Python, no dependencies - rarer shared words weigh more, so a distinctive term like 'sunscreen'
outweighs a common one like 'water'.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "the a an and or but if then else of to in on at by for with from into over under as is are was "
    "were be been being it its this that these those they them their he she his her him we us our you "
    "your i me my not no nor so than too very can could should would may might must will shall do does "
    "did has have had having which who whom whose what when where why how all any both each few more "
    "most other some such only own same s t will just don there here out up down off above below "
    "between through during before after about against among also another because been being get got "
    "make made many much new now one two three use used using like include including".split()
)


def content_words(text: str) -> set[str]:
    """Distinctive content words (lowercased, >2 chars, stopwords removed). Dropping stopwords is what
    stops a candidate ranking high on a bare subject overlap like 'They are ...'."""
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def idf_overlap_scores(query_words: set[str], docs_words: Sequence[set[str]]) -> list[float]:
    """IDF-weighted overlap of `query_words` with each doc, aligned to `docs_words`.

    The document frequency is computed across the given docs (a small local pool), so a query word
    that appears in almost every doc carries little weight while a rare, distinctive one dominates.
    Returns 0.0 for a doc that shares no query word (never negative), so this is safe to use as a
    pure re-ordering key: a doc that shares nothing simply sinks, it is never dropped.
    """
    if not query_words:
        return [0.0] * len(docs_words)
    n = len(docs_words)
    df: Counter[str] = Counter()
    for dw in docs_words:
        df.update(query_words & dw)
    weight = {w: math.log(1 + n / (1 + df[w])) for w in query_words}
    return [sum(weight[w] for w in (query_words & dw)) for dw in docs_words]
