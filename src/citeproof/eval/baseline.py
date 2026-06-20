"""Two baseline binders: a sanity floor and a lexical-overlap stand-in for the real binder.

Neither uses model weights. They exist to exercise the harness end-to-end and, crucially, to
demonstrate the strict cite-gate (cite ONLY when verdict is Verdict.OK) before the real
MiniCheck binder lands. A baseline that respects the gate yields zero cite-gate violations.

  - AlwaysAbstainBinder: abstains on everything; the precision/recall floor.
  - LexicalOverlapBinder: cites only an OK page whose best-matching sentence has Jaccard
    token overlap with the claim at or above a threshold; the cited span is that sentence.
"""

from __future__ import annotations

import re

from veriscrape import Verdict

from citeproof.eval.models import BinderOutput, ClaimSourcePair

_WORD = re.compile(r"[a-z0-9]+")
# Split on sentence-ending punctuation followed by whitespace; keeps it dependency-free.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


class AlwaysAbstainBinder:
    """Abstains on every pair. The sanity floor: 0 citations, 0 cite-gate violations."""

    def bind(self, pair: ClaimSourcePair) -> BinderOutput:
        return BinderOutput(pair_id=pair.id, cited=False, abstained=True)


class LexicalOverlapBinder:
    """Cites an OK page when claim/best-sentence Jaccard overlap clears the threshold.

    The strict cite-gate is the FIRST check: a page whose verdict is not Verdict.OK is never
    cited, no matter how well its text matches. This is the load-bearing invariant the M0
    cite-gate coverage clause tests.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        self.threshold = threshold

    def bind(self, pair: ClaimSourcePair) -> BinderOutput:
        # Strict cite-gate: only verdict == OK pages are ever citable.
        if pair.verdict is not Verdict.OK:
            return BinderOutput(pair_id=pair.id, cited=False, abstained=True)

        claim_tokens = _tokens(pair.claim)
        best_sentence: str | None = None
        best_score = 0.0
        for sentence in _sentences(pair.source_text):
            score = _jaccard(claim_tokens, _tokens(sentence))
            if score > best_score:
                best_score = score
                best_sentence = sentence

        if best_sentence is not None and best_score >= self.threshold:
            return BinderOutput(
                pair_id=pair.id,
                cited=True,
                abstained=False,
                cited_span=best_sentence,
                source_url=pair.source_url,
                entailment_prob=best_score,
                symbolic_ok=None,
            )
        return BinderOutput(pair_id=pair.id, cited=False, abstained=True)
