"""Harness tests: run both baselines on the synthetic seed set and prove the cite-gate.

Three things must hold:
  (a) AlwaysAbstain yields 0 citations and 0 cite-gate violations (the floor).
  (b) LexicalOverlap (which respects strict verdict==OK) yields 0 cite-gate violations.
  (c) A deliberately-broken inline binder that cites a non-OK page produces
      cite_gate_violations > 0, proving the coverage check actually fires.
"""

from __future__ import annotations

from veriscrape import Verdict

from citeproof.eval.baseline import AlwaysAbstainBinder, LexicalOverlapBinder
from citeproof.eval.harness import EvalReport, run
from citeproof.eval.models import BinderOutput, ClaimSourcePair
from citeproof.eval.seed import load_seed_pairs


def _seed() -> list[ClaimSourcePair]:
    pairs = load_seed_pairs()
    assert pairs, "seed set must be non-empty"
    return pairs


def test_seed_contains_an_entailing_non_ok_pair() -> None:
    # The cite-gate coverage check is only meaningful if such a pair exists.
    pairs = _seed()
    assert any(p.verdict is not Verdict.OK and p.entailed for p in pairs)


def test_always_abstain_floor() -> None:
    report = run(_seed(), AlwaysAbstainBinder())
    assert isinstance(report, EvalReport)
    # No citations emitted anywhere -> pooled cited count is zero.
    assert report.pooled_cited == 0
    assert report.cite_gate_violations == 0
    assert report.false_ok_count == 0
    # With zero citations, recall is zero against the answerable denominator.
    assert report.answerable_recall == 0.0


def test_lexical_overlap_respects_cite_gate() -> None:
    report = run(_seed(), LexicalOverlapBinder(threshold=0.3))
    # The strict verdict==OK gate means no non-OK page is ever cited.
    assert report.cite_gate_violations == 0
    # A reasonable lexical binder cites at least one clean-entailed pair on the seed.
    assert report.pooled_cited >= 1


class _BrokenBinder:
    """Deliberately broken: cites the source verbatim regardless of verdict.

    This is the adversary the cite-gate coverage clause exists to catch. It MUST be
    constructible (the models do not enforce the gate) so the harness check can fire.
    """

    def bind(self, pair: ClaimSourcePair) -> BinderOutput:
        return BinderOutput(
            pair_id=pair.id,
            cited=True,
            abstained=False,
            cited_span=pair.source_text[:60] or "x",
            source_url=pair.source_url,
            entailment_prob=1.0,
        )


def test_broken_binder_trips_cite_gate() -> None:
    report = run(_seed(), _BrokenBinder())
    # It cites every pair, including the non-OK ones, so the coverage check must fire.
    assert report.cite_gate_violations > 0
    # And the false-OK count (citations whose source verdict is not OK) is also positive.
    assert report.false_ok_count > 0
    # The number of non-OK citations is exactly the number of non-OK pairs in the seed.
    non_ok = sum(1 for p in _seed() if p.verdict is not Verdict.OK)
    assert report.cite_gate_violations == non_ok
    assert report.false_ok_count == non_ok


def test_report_per_bucket_and_bounds_present() -> None:
    report = run(_seed(), LexicalOverlapBinder(threshold=0.3))
    # Every bucket present in the seed appears in the per-bucket report.
    seed_buckets = {p.bucket.value for p in _seed()}
    assert seed_buckets.issubset(set(report.per_bucket.keys()))
    # Pooled precision lower bound is a valid probability and is no greater than the point.
    assert 0.0 <= report.pooled_precision_lb <= 1.0
    if report.pooled_cited > 0:
        assert report.pooled_precision_lb <= report.pooled_precision + 1e-9
    # false_ok_upper is a 95% upper bound in [0, 1].
    assert 0.0 <= report.false_ok_upper <= 1.0
