"""PRUNE: ALCE leave-one-out removal of over-citations (the minimal sufficient set).

A claim may end up cited by several spans when only one is actually needed (the PRUNE pass of the
ALCE leave-one-out pipeline). Over-citation is dishonest (it implies more corroboration than exists)
and dilutes the receipt. This drops a span when the claim is STILL entailed without it - that span
was redundant - and keeps doing so until every survivor is load-bearing.

Contract:
  - NEVER drop the last surviving span (a claim cited by one span stays cited by one span).
  - A span is droppable iff, with it removed, the remaining set still entails the claim: the MAX
    entailment over the remaining spans is >= tau. (Max, not sum - entailment is satisfied by the
    single best supporting span; ALCE measures whether the citation set as a whole still supports
    the claim.)
  - Deterministic and order-stable: consider dropping the LOWEST-entailment span first, and
    re-evaluate after each drop, so the result is a minimal sufficient set reached the same way
    every run.
  - Pure logic, no heavy model: it takes any EntailmentModel (FakeEntailment / a scripted dict in
    tests). This is the binder's front-half PRUNE; re-anchoring the surviving best span and
    re-verifying happen in the binder core, out of scope here.
"""

from __future__ import annotations

from typing import Sequence

from citeproof.binder.entailment import EntailmentModel
from citeproof.binder.retrieve import Candidate


def _entails(claim: str, spans: Sequence[Candidate], entailment: EntailmentModel, tau: float) -> bool:
    """True iff the MAX entailment over `spans` clears tau (the set still supports the claim)."""
    if not spans:
        return False
    return max(entailment.score(claim, c.span_text) for c in spans) >= tau


def prune(
    claim: str,
    cited: Sequence[Candidate],
    entailment: EntailmentModel,
    *,
    tau: float = 0.7,
) -> list[Candidate]:
    """Drop over-cited (redundant) spans, returning a minimal sufficient citation set.

    Iteratively removes a span when the claim stays entailed (max entailment over the rest >= tau)
    without it. The lowest-entailment span is tried first each round, so the prune is deterministic
    and order-stable. The last surviving span is never dropped, so a non-empty input that has at
    least one entailing span never returns empty.

    Args:
        claim: the claim being cited.
        cited: the spans currently attached to the claim (over-cited or not).
        entailment: any EntailmentModel scoring how strongly a span entails the claim.
        tau: the entailment threshold a surviving set must still clear. Defaults to 0.7; the
            pre-registered threshold is tuned and frozen elsewhere, not load-bearing for this pure
            logic.

    Returns:
        The minimal sufficient subset, in the input order of the survivors. An empty input returns
        an empty list. A single citation is returned unchanged.
    """
    survivors = list(cited)
    if len(survivors) <= 1:
        return survivors

    changed = True
    while changed and len(survivors) > 1:
        changed = False
        # Standalone entailment per span (recomputed each round; the sets are tiny). Try dropping the
        # lowest-entailment span first, tie-broken by a CANONICAL key (span text, then url, then
        # position) so the minimal set is the same regardless of input order. Remove by POSITION so a
        # duplicated candidate object is dropped once, not twice.
        scores = [entailment.score(claim, c.span_text) for c in survivors]
        order = sorted(
            range(len(survivors)),
            key=lambda i: (scores[i], survivors[i].span_text, survivors[i].source_url, i),
        )
        for i in order:
            remaining = survivors[:i] + survivors[i + 1 :]
            if _entails(claim, remaining, entailment, tau):
                survivors = remaining
                changed = True
                break  # re-evaluate from scratch after each drop

    return survivors
