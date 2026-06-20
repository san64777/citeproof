"""Tests for PRUNE: ALCE leave-one-out removal of over-citations.

A claim cited by several spans should be reduced to the minimal sufficient set: a redundant span
(one that can be dropped while the claim stays entailed by the rest) is over-citation and is
removed. The last surviving span is never dropped, and a set where each span is independently
necessary is kept whole. Pure logic, driven by a scripted FakeEntailment for exact control.
"""

from __future__ import annotations

from veriscrape import Verdict

from citeproof.binder.entailment import FakeEntailment
from citeproof.binder.prune import prune
from citeproof.binder.retrieve import Candidate

CLAIM = "The merger closed in 2021."


def _cand(span_text: str, url: str) -> Candidate:
    return Candidate(span_text=span_text, source_url=url, verdict=Verdict.OK)


def test_redundant_over_citation_is_reduced_to_one() -> None:
    # Two spans BOTH strongly entail the claim. Only one is needed, so the set must shrink to one.
    a = _cand("The merger closed in 2021.", "https://x/a")
    b = _cand("The deal was completed during 2021.", "https://x/b")
    scores = {
        (CLAIM, a.span_text): 0.95,
        (CLAIM, b.span_text): 0.90,
    }
    out = prune(CLAIM, [a, b], FakeEntailment(scores=scores), tau=0.7)
    assert len(out) == 1
    # The stronger span survives (the weaker, redundant one is dropped first).
    assert out[0] is a


def test_single_citation_is_kept() -> None:
    a = _cand("The merger closed in 2021.", "https://x/a")
    out = prune(CLAIM, [a], FakeEntailment(scores={(CLAIM, a.span_text): 0.95}), tau=0.7)
    assert out == [a]


def test_each_span_independently_necessary_is_kept_whole() -> None:
    # No single span clears tau on its own except in combination is not how max-entailment works;
    # here EACH span only partially supports the claim and NONE alone reaches tau, so dropping any
    # one would leave the remainder NOT entailing -> nothing is droppable -> the set is kept whole.
    a = _cand("The merger was announced.", "https://x/a")
    b = _cand("Regulators reviewed the deal.", "https://x/b")
    c = _cand("It happened in 2021.", "https://x/c")
    scores = {
        (CLAIM, a.span_text): 0.60,
        (CLAIM, b.span_text): 0.55,
        (CLAIM, c.span_text): 0.65,
    }
    out = prune(CLAIM, [a, b, c], FakeEntailment(scores=scores), tau=0.7)
    # None alone reaches tau=0.7, so removing any span leaves max < tau -> keep all three.
    assert out == [a, b, c]


def test_never_returns_empty_when_one_span_entails() -> None:
    # Even though the strong span makes the others redundant, the last survivor is never dropped.
    a = _cand("The merger closed in 2021.", "https://x/a")
    b = _cand("The merger closed in 2021 per the filing.", "https://x/b")
    scores = {(CLAIM, a.span_text): 0.92, (CLAIM, b.span_text): 0.88}
    out = prune(CLAIM, [a, b], FakeEntailment(scores=scores), tau=0.7)
    assert len(out) >= 1
    assert out  # non-empty


def test_empty_input_returns_empty() -> None:
    assert prune(CLAIM, [], FakeEntailment(), tau=0.7) == []


def test_only_redundant_extras_dropped_keeping_minimal_set() -> None:
    # Three spans: one strong (clears tau alone), two weak (do not). The two weak ones are NOT
    # droppable individually only if removing them drops the set below tau - but the strong span
    # keeps the set entailed, so both weak spans are redundant and the set reduces to the strong one.
    strong = _cand("The merger closed in 2021.", "https://x/s")
    weak1 = _cand("There was talk of a merger.", "https://x/w1")
    weak2 = _cand("The companies discussed terms.", "https://x/w2")
    scores = {
        (CLAIM, strong.span_text): 0.95,
        (CLAIM, weak1.span_text): 0.30,
        (CLAIM, weak2.span_text): 0.25,
    }
    out = prune(CLAIM, [strong, weak1, weak2], FakeEntailment(scores=scores), tau=0.7)
    assert out == [strong]


def test_prune_is_deterministic_across_runs() -> None:
    a = _cand("The merger closed in 2021.", "https://x/a")
    b = _cand("The deal was completed during 2021.", "https://x/b")
    c = _cand("It finalized in 2021.", "https://x/c")
    scores = {
        (CLAIM, a.span_text): 0.95,
        (CLAIM, b.span_text): 0.90,
        (CLAIM, c.span_text): 0.85,
    }
    fake = FakeEntailment(scores=scores)
    out1 = prune(CLAIM, [a, b, c], fake, tau=0.7)
    out2 = prune(CLAIM, [a, b, c], fake, tau=0.7)
    assert out1 == out2
    assert out1 == [a]  # all redundant against the strongest; reduces to it


def test_duplicate_candidate_object_is_deduplicated_not_both_dropped() -> None:
    # The SAME candidate object appearing twice is an over-citation: it must reduce to ONE copy, not
    # be removed entirely. (The old identity-based removal dropped both at once and returned the
    # duplicate unchanged.)
    a = _cand("The merger closed in 2021.", "https://x/a")
    out = prune(CLAIM, [a, a], FakeEntailment(scores={(CLAIM, a.span_text): 0.95}), tau=0.7)
    assert len(out) == 1
    assert out[0] is a


def test_prune_is_order_independent_for_tied_redundant_spans() -> None:
    # Two distinct spans that BOTH fully entail and TIE on score: the survivor is chosen by a
    # canonical key (span text), so input order does not change the result.
    a = _cand("AAA the merger closed in 2021.", "https://x/a")
    b = _cand("ZZZ the merger closed in 2021.", "https://x/b")
    scores = {(CLAIM, a.span_text): 0.90, (CLAIM, b.span_text): 0.90}
    fake = FakeEntailment(scores=scores)
    out_ab = prune(CLAIM, [a, b], fake, tau=0.7)
    out_ba = prune(CLAIM, [b, a], fake, tau=0.7)
    assert out_ab == out_ba
    assert len(out_ab) == 1
