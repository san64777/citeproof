"""Candidate-span splitting and verbatim anchoring with POSITIONAL disambiguation.

The red-team fix this encodes (the "locatability" property): a value that REPEATS on a page
is uniquely RE-LOCATABLE by its character offset, NOT a reason to abstain. A repeated table value
or boilerplate sentence is still anchorable; the binder already knows WHICH candidate it verified,
so anchoring is by offset and unambiguous. We abstain on locatability ONLY when a span genuinely
cannot be found at all, never merely because its text is non-unique.

candidate_spans segments text into sentences/passages while preserving exact char offsets.
find_anchor returns the offsets of an occurrence (disambiguated by an optional `near` hint when the
text repeats). is_uniquely_locatable is a DIAGNOSTIC only - it reports whether the text is globally
unique, but a False from it must never trigger an abstain on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Split on sentence-ending punctuation followed by whitespace, preserving offsets by using the
# match positions rather than re.split (which would discard them).
#
# Group 1 is the sentence-ending punctuation; the span ends THERE (citations are not part of the
# sentence). Between the punctuation and the whitespace we ALSO consume inline reference markers like
# "[182]", "[note 1]", "[citation needed]" - on Wikipedia these sit directly after the period
# ("areas.[5] Coral reefs...") with no space, so without this the boundary never matches and a whole
# paragraph collapses into ONE 600+ char span. That span verifies fine (the support is somewhere in
# it) but the receipt then highlights its FIRST sentence, not the supporting one - a mis-highlight.
# The bracket body is length-capped so this only eats real citation markers, not arbitrary "[...]".
_SENTENCE_BOUNDARY = re.compile(r"([.!?]+)(?:\[[^\]]{1,24}\])*\s+")


@dataclass(frozen=True)
class Span:
    """A candidate span with its exact char offsets into the source text."""

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Anchor:
    """The located char offsets of a span occurrence within a page."""

    start: int
    end: int


def candidate_spans(text: str) -> list[Span]:
    """Segment text into candidate spans (sentences/passages), keeping exact char offsets.

    Offsets are into the ORIGINAL text: for every returned Span, text[span.start:span.end] equals
    span.text. Leading/trailing whitespace is trimmed from each span and the offsets are tightened
    accordingly, so a span never carries surrounding whitespace.
    """
    spans: list[Span] = []
    cursor = 0
    # Walk boundaries; each chunk runs from the previous boundary end to this boundary end.
    for m in _SENTENCE_BOUNDARY.finditer(text):
        # End the span at the sentence punctuation (group 1), EXCLUDING any trailing citation
        # markers and whitespace - so the span text is the clean sentence the receipt highlights.
        chunk_end = m.start(1) + len(m.group(1))
        _append_trimmed(spans, text, cursor, chunk_end)
        cursor = m.end()
    # Trailing chunk after the last boundary.
    _append_trimmed(spans, text, cursor, len(text))
    return spans


def _append_trimmed(spans: list[Span], text: str, start: int, end: int) -> None:
    """Append text[start:end] as a Span with whitespace trimmed and offsets tightened."""
    raw = text[start:end]
    stripped = raw.strip()
    if not stripped:
        return
    lead = len(raw) - len(raw.lstrip())
    real_start = start + lead
    real_end = real_start + len(stripped)
    spans.append(Span(text=stripped, start=real_start, end=real_end))


def find_anchor(span_text: str, page: str, *, near: int | None = None) -> Anchor | None:
    """Locate `span_text` in `page`, returning its char offsets, or None if it does not occur.

    Positional disambiguation: when `span_text` occurs more than once, `near` selects the
    occurrence whose start offset is closest to `near`. A repeat is therefore RE-LOCATABLE by
    offset, never a reason to return None. With no `near` hint, the FIRST occurrence is returned.
    None is returned ONLY when the text genuinely does not occur at all.
    """
    if not span_text:
        return None

    starts: list[int] = []
    idx = page.find(span_text)
    while idx != -1:
        starts.append(idx)
        idx = page.find(span_text, idx + 1)

    if not starts:
        return None

    if near is None:
        chosen = starts[0]
    else:
        chosen = min(starts, key=lambda s: abs(s - near))

    return Anchor(start=chosen, end=chosen + len(span_text))


def is_uniquely_locatable(span_text: str, page: str) -> bool:
    """DIAGNOSTIC ONLY: True iff `span_text` occurs exactly once in `page`.

    This reports global uniqueness for telemetry (abstention-due-to-nonunique by page type). It is
    NOT an abstain trigger: a non-unique span is still anchorable by offset via find_anchor.
    """
    if not span_text:
        return False
    first = page.find(span_text)
    if first == -1:
        return False
    return page.find(span_text, first + 1) == -1
