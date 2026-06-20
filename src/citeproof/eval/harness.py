"""The M0 eval runner and report.

run(pairs, binder) drives a binder over a labeled set and scores it under the pre-registered
definitions: precision excludes abstentions on both sides, recall is
against the answerable denominator, and every rate carries a Clopper-Pearson one-sided bound
(90% for precision/recall, 95% for false-OK). The cite-gate coverage count is the load-bearing
invariant: it counts citations anchored to a non-OK verdict and MUST be 0 for a correct binder.

This module computes numbers only; it does not decide GO/NO-GO and it does not gate any bar.
The synthetic seed it ships with carries no statistical weight; the real TEST fold runs later.
"""

from __future__ import annotations

from pydantic import BaseModel, Field
from veriscrape import Verdict

from citeproof.eval.models import Binder, BinderOutput, Bucket, ClaimSourcePair
from citeproof.eval.stats import clopper_pearson_lower, clopper_pearson_upper


def _is_correct_citation(pair: ClaimSourcePair, out: BinderOutput) -> bool:
    """A citation is correct iff the pair is human-entailed AND the span is non-empty AND the page is
    verdict == OK.

    The verdict==OK leg is defense in depth: a citation anchored to a non-OK page is a cite-gate
    VIOLATION, so it must never count as a correct citation (which would inflate precision). It still
    counts in the cited denominator (a wrong citation lowers precision) and is separately tallied as a
    cite_gate_violation (the HARD GO clause). The cite-gate already prevents non-OK citations, so this
    is unreachable in practice; the leg keeps the precision number honest if a future bug let one
    through.
    """
    return bool(
        pair.entailed and out.cited_span and out.cited_span.strip() and pair.verdict is Verdict.OK
    )


class BucketReport(BaseModel):
    """Per-bucket scores with their one-sided lower bounds."""

    n: int
    cited: int
    correct: int
    precision: float
    precision_lb: float
    recall: float
    recall_lb: float
    abstention_rate: float


class EvalReport(BaseModel):
    """The full scored report for one (set, binder) run. Numbers only, no GO/NO-GO."""

    per_bucket: dict[str, BucketReport] = Field(default_factory=dict)

    pooled_n: int
    pooled_cited: int
    pooled_correct: int
    pooled_precision: float
    pooled_precision_lb: float

    # NOTE: this is the binder's cite-gate LEAK rate - citations emitted onto a non-OK page, over
    # citations emitted. It is NOT the veriscrape false-OK (a junk page wrongly verdicted
    # OK), which is measured separately on the >= 200-page GATE set by gather_junk.py. The cite-gate
    # keeps both 0 in practice; false_ok_count is the same event as cite_gate_violations (the HARD
    # GO clause), reported here with a 95% upper bound.
    false_ok_count: int
    false_ok_upper: float

    cite_gate_violations: int

    answerable_total: int
    answerable_correct: int
    answerable_recall: float

    alpha: float
    false_ok_alpha: float


def _precision(correct: int, cited: int) -> float:
    return correct / cited if cited else 0.0


def _recall(correct: int, answerable: int) -> float:
    return correct / answerable if answerable else 0.0


def run(
    pairs: list[ClaimSourcePair],
    binder: Binder,
    *,
    alpha: float = 0.10,
    false_ok_alpha: float = 0.05,
) -> EvalReport:
    """Run a binder over labeled pairs and score it under the pre-registered definitions.

    Args:
        pairs: labeled (claim, source) items.
        binder: anything implementing the Binder protocol.
        alpha: one-sided tail for precision/recall bounds (0.10 -> 90% bounds).
        false_ok_alpha: one-sided tail for the false-OK upper bound (0.05 -> 95% bound).

    Returns:
        An EvalReport. Precision = correct citations / citations emitted (abstentions excluded
        both sides). Recall = correct citations / answerable claims. cite_gate_violations counts
        cited outputs whose pair verdict is not Verdict.OK.
    """
    outputs: dict[str, BinderOutput] = {p.id: binder.bind(p) for p in pairs}

    # Per-bucket accumulation.
    bucket_n: dict[Bucket, int] = {}
    bucket_cited: dict[Bucket, int] = {}
    bucket_correct: dict[Bucket, int] = {}
    bucket_abstained: dict[Bucket, int] = {}
    bucket_answerable: dict[Bucket, int] = {}
    bucket_answerable_correct: dict[Bucket, int] = {}

    pooled_cited = 0
    pooled_correct = 0
    false_ok_count = 0
    cite_gate_violations = 0
    answerable_total = 0
    answerable_correct = 0

    for pair in pairs:
        out = outputs[pair.id]
        b = pair.bucket
        bucket_n[b] = bucket_n.get(b, 0) + 1

        if pair.answerable:
            answerable_total += 1
            bucket_answerable[b] = bucket_answerable.get(b, 0) + 1

        if out.abstained:
            bucket_abstained[b] = bucket_abstained.get(b, 0) + 1
            continue

        # A citation was emitted.
        pooled_cited += 1
        bucket_cited[b] = bucket_cited.get(b, 0) + 1

        # Cite-gate: a citation anchored to a non-OK verdict is a violation AND a false-OK.
        if pair.verdict is not Verdict.OK:
            cite_gate_violations += 1
            false_ok_count += 1

        if _is_correct_citation(pair, out):
            pooled_correct += 1
            bucket_correct[b] = bucket_correct.get(b, 0) + 1
            if pair.answerable:
                answerable_correct += 1
                bucket_answerable_correct[b] = bucket_answerable_correct.get(b, 0) + 1

    per_bucket: dict[str, BucketReport] = {}
    for b, n in bucket_n.items():
        cited = bucket_cited.get(b, 0)
        correct = bucket_correct.get(b, 0)
        abstained = bucket_abstained.get(b, 0)
        ans = bucket_answerable.get(b, 0)
        ans_correct = bucket_answerable_correct.get(b, 0)

        precision = _precision(correct, cited)
        recall = _recall(ans_correct, ans)
        per_bucket[b.value] = BucketReport(
            n=n,
            cited=cited,
            correct=correct,
            precision=precision,
            precision_lb=clopper_pearson_lower(correct, cited, alpha) if cited else 0.0,
            recall=recall,
            recall_lb=clopper_pearson_lower(ans_correct, ans, alpha) if ans else 0.0,
            abstention_rate=abstained / n if n else 0.0,
        )

    pooled_precision = _precision(pooled_correct, pooled_cited)
    pooled_precision_lb = (
        clopper_pearson_lower(pooled_correct, pooled_cited, alpha) if pooled_cited else 0.0
    )
    answerable_recall = _recall(answerable_correct, answerable_total)
    # The false-OK rate is over the citations EMITTED (the chance an emitted citation is non-OK);
    # with zero citations the upper bound is the no-data 95% bound at x=0, n=0 is undefined, so we
    # report 0.0 (no citations, no false-OK risk realized on this set).
    false_ok_upper = (
        clopper_pearson_upper(false_ok_count, pooled_cited, false_ok_alpha) if pooled_cited else 0.0
    )

    return EvalReport(
        per_bucket=per_bucket,
        pooled_n=len(pairs),
        pooled_cited=pooled_cited,
        pooled_correct=pooled_correct,
        pooled_precision=pooled_precision,
        pooled_precision_lb=pooled_precision_lb,
        false_ok_count=false_ok_count,
        false_ok_upper=false_ok_upper,
        cite_gate_violations=cite_gate_violations,
        answerable_total=answerable_total,
        answerable_correct=answerable_correct,
        answerable_recall=answerable_recall,
        alpha=alpha,
        false_ok_alpha=false_ok_alpha,
    )
