"""Tests for EntailmentBinder: the cite-gate, the orthogonal gate, and harness integration.

The headline assertion: a near-miss where the entailment model says HIGH probability but the
symbolic check finds a contradiction (claim '12%' vs span '21%') must ABSTAIN. That is the whole
point of the orthogonal gate - it catches what entailment alone missed. A high entailment score
can NEVER override a symbolic contradiction.
"""

from __future__ import annotations

from veriscrape import Verdict

from citeproof.binder.binder import EntailmentBinder
from citeproof.binder.entailment import FakeEntailment
from citeproof.eval.harness import run
from citeproof.eval.models import Bucket, ClaimSourcePair, Fold
from citeproof.eval.seed import load_seed_pairs


def _pair(
    *,
    pid: str,
    claim: str,
    source_text: str,
    verdict: Verdict,
    bucket: Bucket = Bucket.CLEAN_ENTAILED,
    entailed: bool = True,
    answerable: bool = True,
    source_url: str = "https://synthetic.example/x",
) -> ClaimSourcePair:
    return ClaimSourcePair(
        id=pid,
        bucket=bucket,
        fold=Fold.TEST,
        claim=claim,
        source_url=source_url,
        source_text=source_text,
        verdict=verdict,
        gold_span=None,
        entailed=entailed,
        answerable=answerable,
    )


def test_cite_gate_first_non_ok_always_abstains_even_when_signals_pass() -> None:
    # Entailment is forced high AND the symbolic check would pass, but the verdict is BLOCKED.
    # The cite-gate is the FIRST check and is non-negotiable: this MUST abstain.
    claim = "The service returned a maintenance notice."
    span = "The service returned a maintenance notice at the requested URL."
    fake = FakeEntailment(scores={(claim, span): 0.99})
    binder = EntailmentBinder(fake, tau_mc=0.7)

    pair = _pair(
        pid="blocked-1",
        claim=claim,
        source_text=span,
        verdict=Verdict.BLOCKED,
    )
    out = binder.bind(pair)
    assert out.abstained is True
    assert out.cited is False
    assert out.cited_span is None


def test_unverified_also_abstains_like_blocked() -> None:
    claim = "Login is required to continue."
    span = "The page states that login is required to continue."
    fake = FakeEntailment(scores={(claim, span): 0.99})
    binder = EntailmentBinder(fake, tau_mc=0.7)
    pair = _pair(pid="unv-1", claim=claim, source_text=span, verdict=Verdict.UNVERIFIED)
    out = binder.bind(pair)
    assert out.abstained is True


def test_headline_orthogonal_gate_abstains_on_number_contradiction() -> None:
    # The over-attribution trap: entailment says high-prob, but the symbolic check catches the
    # flipped number (claim 12% vs span 21%). The binder MUST abstain even with high entailment.
    claim = "Revenue grew 12% last year."
    span = "Revenue grew 21% last year, the company reported."
    # Force the entailment model to (wrongly) love this pair.
    fake = FakeEntailment(scores={(claim, span): 0.98})
    binder = EntailmentBinder(fake, tau_mc=0.7)

    pair = _pair(
        pid="nearmiss-1",
        claim=claim,
        source_text=span,
        verdict=Verdict.OK,
        bucket=Bucket.NEAR_MISS_PARAPHRASE,
        entailed=False,
        answerable=False,
    )
    out = binder.bind(pair)
    assert out.abstained is True, "orthogonal symbolic gate must veto a high-entailment near-miss"
    assert out.cited is False
    # Diagnostics: the best candidate's symbolic_ok is recorded as False on the abstention.
    assert out.symbolic_ok is False


def test_clean_entailed_pair_cites_the_right_span() -> None:
    claim = "Mercury is the closest planet to the Sun."
    span_text = "Mercury is the closest planet to the Sun in our solar system."
    page = f"{span_text} It completes one orbit every eighty-eight days."
    fake = FakeEntailment(scores={(claim, span_text): 0.95})
    binder = EntailmentBinder(fake, tau_mc=0.7)

    pair = _pair(pid="clean-1", claim=claim, source_text=page, verdict=Verdict.OK)
    out = binder.bind(pair)
    assert out.cited is True
    assert out.abstained is False
    assert out.cited_span == span_text
    assert out.source_url == pair.source_url
    assert out.symbolic_ok is True


def test_boilerplate_section_is_never_cited_even_when_scored_high() -> None:
    # A 'See also' nav list mentions the topic, so an NLI mis-scores it as support (live: 0.984 vs
    # 0.980 for the real sentence) and the receipt highlights navigation - a mis-attribution. The
    # binder must drop non-prose candidates BEFORE scoring, even if the model would score them high.
    claim = "Narendra Modi was born on 17 September 1950."
    nav = "See also\n- List of prime ministers of India\n- Opinion polling on the Narendra Modi premiership"
    prose = "Narendra Modi was born on 17 September 1950 in Vadnagar."
    page = f"{prose}\n\n{nav}"
    fake = FakeEntailment(scores={(claim, nav): 0.99, (claim, prose): 0.80})
    binder = EntailmentBinder(fake, tau_mc=0.7)
    out = binder.bind(_pair(pid="nav-1", claim=claim, source_text=page, verdict=Verdict.OK))
    assert out.cited is True
    assert "See also" not in (out.cited_span or "")  # the nav list was dropped
    assert out.cited_span == prose  # cited the real sentence despite its lower score


def test_second_signal_vetoes_a_citation_the_primary_would_attach() -> None:
    # The primary (MiniCheck) loves the supporting span and symbolic passes, so it would attach. The
    # optional orthogonal second NLI scores the claim against the FULL source, sees the contradiction
    # in ANOTHER sentence, and scores low -> the citation is vetoed.
    claim = "Jupiter is the most massive planet."
    span_text = "Jupiter is more than twice as massive as all the other planets combined."
    page = f"{span_text} However, Saturn is in fact the most massive planet."
    primary = FakeEntailment(scores={(claim, span_text): 0.95})  # primary attaches on the span
    second = FakeEntailment(scores={(claim, page): 0.05})  # orthogonal NLI vetoes on the whole source
    binder = EntailmentBinder(primary, tau_mc=0.7, second_signal=second, tau_db=0.5)

    pair = _pair(
        pid="2sig-veto",
        claim=claim,
        source_text=page,
        verdict=Verdict.OK,
        bucket=Bucket.NEAR_MISS_PARAPHRASE,
        entailed=False,
        answerable=False,
    )
    out = binder.bind(pair)
    assert out.abstained is True, "the orthogonal second NLI must veto a citation the primary attaches"
    assert out.cited is False
    # The veto records the second-signal score (the pre-registered independence diagnostic needs it).
    assert out.second_signal_prob == 0.05


def test_second_signal_above_threshold_still_cites() -> None:
    claim = "Mercury is the closest planet to the Sun."
    span_text = "Mercury is the closest planet to the Sun in our solar system."
    page = f"{span_text} It completes one orbit every eighty-eight days."
    primary = FakeEntailment(scores={(claim, span_text): 0.95})
    second = FakeEntailment(scores={(claim, page): 0.98})  # the second NLI agrees -> no veto
    binder = EntailmentBinder(primary, tau_mc=0.7, second_signal=second, tau_db=0.5)

    out = binder.bind(_pair(pid="2sig-ok", claim=claim, source_text=page, verdict=Verdict.OK))
    assert out.cited is True
    assert out.cited_span == span_text
    assert out.second_signal_prob == 0.98


def test_second_signal_exactly_at_threshold_cites() -> None:
    # The gate is inclusive (db >= tau_db cites), mirroring tau_mc. Locks the boundary.
    claim = "Mercury is the closest planet to the Sun."
    span_text = "Mercury is the closest planet to the Sun in our solar system."
    page = f"{span_text} It completes one orbit every eighty-eight days."
    primary = FakeEntailment(scores={(claim, span_text): 0.95})
    second = FakeEntailment(scores={(claim, page): 0.50})
    binder = EntailmentBinder(primary, tau_mc=0.7, second_signal=second, tau_db=0.5)
    out = binder.bind(_pair(pid="2sig-bound", claim=claim, source_text=page, verdict=Verdict.OK))
    assert out.cited is True
    assert out.second_signal_prob == 0.50


def test_second_signal_threshold_is_read_not_hardcoded() -> None:
    # second=0.80 passes a 0.5 gate but a 0.9 gate vetoes -> proves tau_db is actually wired.
    claim = "Mercury is the closest planet to the Sun."
    span_text = "Mercury is the closest planet to the Sun in our solar system."
    page = f"{span_text} It completes one orbit every eighty-eight days."
    primary = FakeEntailment(scores={(claim, span_text): 0.95})
    second = FakeEntailment(scores={(claim, page): 0.80})
    lo = EntailmentBinder(primary, tau_mc=0.7, second_signal=second, tau_db=0.5)
    hi = EntailmentBinder(primary, tau_mc=0.7, second_signal=second, tau_db=0.9)
    p = _pair(pid="2sig-thr", claim=claim, source_text=page, verdict=Verdict.OK)
    assert lo.bind(p).cited is True
    assert hi.bind(p).abstained is True


def test_second_signal_not_consulted_when_primary_abstains() -> None:
    # If the primary already abstains (entailment below tau_mc), the third gate is never reached and
    # no second-signal score is recorded - locks the ordering (the veto cannot resurrect a citation).
    claim = "Mercury is the closest planet to the Sun."
    span_text = "Mercury is the closest planet to the Sun in our solar system."
    page = f"{span_text} It completes one orbit every eighty-eight days."
    primary = FakeEntailment(scores={(claim, span_text): 0.30})  # below tau_mc -> abstain
    second = FakeEntailment(scores={(claim, page): 0.99})  # would pass, but must never be consulted
    binder = EntailmentBinder(primary, tau_mc=0.7, second_signal=second, tau_db=0.5)
    out = binder.bind(_pair(pid="2sig-skip", claim=claim, source_text=page, verdict=Verdict.OK))
    assert out.abstained is True
    assert out.second_signal_prob is None


def test_second_signal_default_off_is_inert() -> None:
    # No second signal (default) -> behavior identical to the two-signal binder.
    claim = "Mercury is the closest planet to the Sun."
    span_text = "Mercury is the closest planet to the Sun in our solar system."
    page = f"{span_text} It completes one orbit every eighty-eight days."
    primary = FakeEntailment(scores={(claim, span_text): 0.95})
    binder = EntailmentBinder(primary, tau_mc=0.7)
    out = binder.bind(_pair(pid="2sig-off", claim=claim, source_text=page, verdict=Verdict.OK))
    assert out.cited is True
    assert out.second_signal_prob is None
    assert out.entailment_prob is not None and out.entailment_prob >= 0.7


def test_below_threshold_entailment_abstains() -> None:
    claim = "Mercury is the closest planet to the Sun."
    page = "Mercury is the closest planet to the Sun in our solar system."
    fake = FakeEntailment(scores={(claim, page): 0.40})
    binder = EntailmentBinder(fake, tau_mc=0.7)
    pair = _pair(pid="low-1", claim=claim, source_text=page, verdict=Verdict.OK)
    out = binder.bind(pair)
    assert out.abstained is True


def test_picks_highest_entailment_eligible_span() -> None:
    claim = "The deluxe plan costs forty dollars."
    good = "The deluxe plan costs forty dollars per month."
    distractor = "The basic plan costs ten dollars per month."
    page = f"{distractor} {good}"
    fake = FakeEntailment(
        scores={
            (claim, distractor): 0.72,
            (claim, good): 0.93,
        }
    )
    binder = EntailmentBinder(fake, tau_mc=0.7)
    pair = _pair(pid="best-1", claim=claim, source_text=page, verdict=Verdict.OK)
    out = binder.bind(pair)
    assert out.cited is True
    assert out.cited_span == good


def test_high_entailment_but_symbolic_fail_never_attaches() -> None:
    # Even when an eligible-by-entailment span exists, a symbolic contradiction on it is fatal.
    claim = "The bridge carries 50 tonnes."
    contradicting = "The bridge carries 15 tonnes of load."
    fake = FakeEntailment(scores={(claim, contradicting): 0.99})
    binder = EntailmentBinder(fake, tau_mc=0.7)
    pair = _pair(pid="sym-veto-1", claim=claim, source_text=contradicting, verdict=Verdict.OK)
    out = binder.bind(pair)
    assert out.abstained is True


# --- HARNESS INTEGRATION -----------------------------------------------------


def test_binder_runs_through_harness_with_zero_cite_gate_violations() -> None:
    pairs = load_seed_pairs()
    assert pairs
    # A generous FakeEntailment: score every (claim, sentence) by Jaccard so the binder actually
    # cites on the clean-entailed seed pairs, exercising the full path through the real harness.
    binder = EntailmentBinder(FakeEntailment(), tau_mc=0.2)
    report = run(pairs, binder)
    # The load-bearing invariant: NEVER a citation anchored to a non-OK verdict.
    assert report.cite_gate_violations == 0
    assert report.false_ok_count == 0


def test_binder_cites_at_least_one_clean_pair_on_seed() -> None:
    pairs = load_seed_pairs()
    binder = EntailmentBinder(FakeEntailment(), tau_mc=0.2)
    report = run(pairs, binder)
    assert report.pooled_cited >= 1
