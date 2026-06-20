"""Validation tests for the M0 eval data model.

The two load-bearing invariants on BinderOutput:
  - cited == (not abstained): a binder either cites or abstains, never both / neither.
  - a CITED output MUST carry a non-empty cited_span AND a source_url (you cannot show a
    receipt you do not have). A cited output missing either is rejected at construction.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from veriscrape import Verdict

from citeproof.eval.models import BinderOutput, Bucket, ClaimSourcePair, Fold


def _valid_pair() -> ClaimSourcePair:
    return ClaimSourcePair(
        id="t1",
        bucket=Bucket.CLEAN_ENTAILED,
        fold=Fold.TEST,
        claim="The sky is blue.",
        source_url="https://example.test/a",
        source_text="On a clear day the sky is blue.",
        verdict=Verdict.OK,
        gold_span="the sky is blue",
        entailed=True,
        answerable=True,
    )


def test_valid_pair_round_trips() -> None:
    pair = _valid_pair()
    assert pair.bucket is Bucket.CLEAN_ENTAILED
    assert pair.fold is Fold.TEST
    assert pair.verdict is Verdict.OK
    assert pair.annotator2_entailed is None
    assert pair.notes is None


def test_cited_output_requires_span_and_url() -> None:
    # A clean citation is accepted.
    out = BinderOutput(
        pair_id="t1",
        cited=True,
        abstained=False,
        cited_span="the sky is blue",
        source_url="https://example.test/a",
        entailment_prob=0.99,
        symbolic_ok=True,
    )
    assert out.cited is True
    assert out.abstained is False


def test_cited_without_span_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BinderOutput(
            pair_id="t1",
            cited=True,
            abstained=False,
            cited_span=None,
            source_url="https://example.test/a",
        )


def test_cited_with_empty_span_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BinderOutput(
            pair_id="t1",
            cited=True,
            abstained=False,
            cited_span="   ",
            source_url="https://example.test/a",
        )


def test_cited_without_source_url_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BinderOutput(
            pair_id="t1",
            cited=True,
            abstained=False,
            cited_span="the sky is blue",
            source_url=None,
        )


def test_cited_must_not_equal_abstained_both_true() -> None:
    with pytest.raises(ValidationError):
        BinderOutput(
            pair_id="t1",
            cited=True,
            abstained=True,
            cited_span="the sky is blue",
            source_url="https://example.test/a",
        )


def test_cited_must_not_equal_abstained_both_false() -> None:
    with pytest.raises(ValidationError):
        BinderOutput(pair_id="t1", cited=False, abstained=False)


def test_abstention_is_valid_without_span() -> None:
    out = BinderOutput(pair_id="t1", cited=False, abstained=True)
    assert out.abstained is True
    assert out.cited_span is None
    assert out.source_url is None
