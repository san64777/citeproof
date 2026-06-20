"""Re-anchor a verbatim quote in the snapshot artifact: the receipt's pointing finger.

The binder verifies a claim against EXTRACTED text, but the receipt must highlight the supporting
passage in the ARTIFACT (the snapshotted page), whose text differs in small ways - collapsed
whitespace, entity decoding, soft hyphens. So re-anchoring is a three-step ladder, cheapest first,
modeled on the W3C Web Annotation TextQuoteSelector (exact + prefix/suffix) with a position hint and
a fuzzy fallback (the Hypothesis-annotator strategy):

  1. EXACT     - the quote occurs verbatim; `near` disambiguates a repeated quote by offset.
  2. NORMALIZED- whitespace-insensitive match via an index map back to true artifact offsets.
  3. FUZZY     - best near-match window scored by difflib ratio; accepted only at >= 0.90 similarity.

Below the fuzzy bar we return None - ABSTAIN OVER GUESS: a receipt highlighting the wrong passage is
worse than a citation that says "could not re-locate; open the snapshot". The returned anchor carries
`exact` AS IT APPEARS IN THE ARTIFACT plus short prefix/suffix context, so a renderer can re-find it
independently (the TextQuote contract), and `strategy` so receipts can report their re-find fidelity.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Literal

_CONTEXT_LEN = 32  # prefix/suffix length, per the W3C TextQuoteSelector convention
_FUZZY_MIN_RATIO = 0.90  # abstain below this; a receipt must never highlight the wrong passage
_FUZZY_MAX_TARGET = 2_000_000  # sanity cap: do not fuzzy-scan a pathologically large artifact


@dataclass(frozen=True)
class TextAnchor:
    """A located quote in the artifact text: offsets + the TextQuote (exact/prefix/suffix) triple."""

    start: int
    end: int
    exact: str
    prefix: str
    suffix: str
    strategy: Literal["exact", "normalized", "fuzzy"]
    score: float  # 1.0 for exact/normalized; the difflib ratio for fuzzy


def _with_context(target: str, start: int, end: int, strategy: Literal["exact", "normalized", "fuzzy"],
                  score: float) -> TextAnchor:
    return TextAnchor(
        start=start,
        end=end,
        exact=target[start:end],
        prefix=target[max(0, start - _CONTEXT_LEN):start],
        suffix=target[end:end + _CONTEXT_LEN],
        strategy=strategy,
        score=score,
    )


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Collapse all whitespace runs to single spaces, returning the normalized string plus a map
    from each normalized index to its ORIGINAL index (so a normalized match converts back to true
    artifact offsets). Soft hyphens vanish entirely (they are invisible joiners, not characters a
    highlight should depend on).
    """
    out: list[str] = []
    idx_map: list[int] = []
    in_space = False
    for i, ch in enumerate(text):
        if ch == "­":  # soft hyphen
            continue
        if ch.isspace():
            if not in_space and out:  # collapse runs; drop leading whitespace
                out.append(" ")
                idx_map.append(i)
            in_space = True
            continue
        in_space = False
        out.append(ch)
        idx_map.append(i)
    # trim a trailing collapsed space
    if out and out[-1] == " ":
        out.pop()
        idx_map.pop()
    return "".join(out), idx_map


def _pick_nearest(positions: list[int], near: int | None) -> int:
    if near is None or len(positions) == 1:
        return positions[0]
    return min(positions, key=lambda p: abs(p - near))


# Only trust the positional hint when extraction kept most of the visible text. If trafilatura
# stripped a large fraction (heavy nav/footer/comment chrome, typically at the tail), the linear
# source->target map skews and could point a repeated span at the WRONG occurrence - strictly worse
# than no hint. Below this kept-ratio we return None and let the anchor take the first occurrence
# (the pre-hint behavior), so the hint is never a regression. Context-based disambiguation (W3C
# TextQuote prefix/suffix) is the more robust upgrade when this proves too conservative.
_MIN_KEPT_RATIO = 0.6


def proportional_near(span_start: int | None, source_len: int, target_len: int) -> int | None:
    """Map a span's offset in the SOURCE text (what the binder cited from) to an approximate offset
    in the TARGET text (the artifact's visible text we re-anchor in). The two texts differ - the
    source is trafilatura main-content, the target is the raw DOM's visible text - so the mapping is
    only proportional and is used purely as a tie-breaker to pick the RIGHT occurrence of a repeated
    span. Returns None (take the first occurrence) when there is nothing to scale from OR when
    extraction stripped enough that the linear map is untrustworthy (see _MIN_KEPT_RATIO).
    """
    if span_start is None or source_len <= 0 or target_len <= 0:
        return None
    if source_len < _MIN_KEPT_RATIO * target_len:
        return None
    return round(span_start / source_len * target_len)


def anchor_quote(quote: str, target: str, *, near: int | None = None) -> TextAnchor | None:
    """Locate `quote` in `target` (the artifact's text), returning offsets + TextQuote context, or
    None when no sufficiently-faithful occurrence exists (abstain over guess).
    """
    quote = quote.strip()
    if not quote or not target:
        return None

    # 1. EXACT, with positional disambiguation for repeats.
    positions: list[int] = []
    at = target.find(quote)
    while at != -1:
        positions.append(at)
        at = target.find(quote, at + 1)
    if positions:
        start = _pick_nearest(positions, near)
        return _with_context(target, start, start + len(quote), "exact", 1.0)

    # 2. NORMALIZED: whitespace-insensitive, mapped back to true offsets.
    norm_target, idx_map = _normalize_with_map(target)
    norm_quote, _ = _normalize_with_map(quote)
    if norm_quote:
        npositions: list[int] = []
        at = norm_target.find(norm_quote)
        while at != -1:
            npositions.append(at)
            at = norm_target.find(norm_quote, at + 1)
        if npositions:
            nstart = _pick_nearest(npositions, near)
            nend = nstart + len(norm_quote)
            start = idx_map[nstart]
            end = idx_map[nend - 1] + 1
            return _with_context(target, start, end, "normalized", 1.0)

    # 3. FUZZY: find the best near-match window; accept only above the similarity bar.
    if len(norm_target) > _FUZZY_MAX_TARGET or not norm_quote:
        return None
    matcher = difflib.SequenceMatcher(None, norm_target, norm_quote, autojunk=False)
    block = matcher.find_longest_match(0, len(norm_target), 0, len(norm_quote))
    if block.size == 0:
        return None
    # Window the target around the longest common block, sized to the quote.
    wstart = max(0, block.a - block.b)
    wend = min(len(norm_target), wstart + len(norm_quote))
    window = norm_target[wstart:wend]
    ratio = difflib.SequenceMatcher(None, window, norm_quote, autojunk=False).ratio()
    if ratio < _FUZZY_MIN_RATIO:
        return None
    start = idx_map[wstart]
    end = idx_map[wend - 1] + 1
    return _with_context(target, start, end, "fuzzy", round(ratio, 4))
