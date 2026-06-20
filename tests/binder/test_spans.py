"""Tests for candidate-span splitting and verbatim anchoring with positional disambiguation.

The red-team fix this encodes: a value that REPEATS on the page is uniquely RE-LOCATABLE by
offset, not a reason to abstain. find_anchor returns char offsets; a repeat is disambiguated by
the `near` hint, never by giving up. is_uniquely_locatable is a diagnostic only.
"""

from __future__ import annotations

from citeproof.binder.spans import Anchor, Span, candidate_spans, find_anchor, is_uniquely_locatable


def test_candidate_spans_keep_char_offsets() -> None:
    text = "Mercury is the closest planet. It orbits the Sun every 88 days."
    spans = candidate_spans(text)
    assert len(spans) >= 2
    for sp in spans:
        assert isinstance(sp, Span)
        # The recorded offsets must reproduce the span text exactly.
        assert text[sp.start : sp.end] == sp.text


def test_candidate_spans_nonempty_and_stripped() -> None:
    text = "  First sentence here.   Second one follows.  "
    spans = candidate_spans(text)
    assert spans
    assert all(sp.text.strip() == sp.text for sp in spans)
    assert all(sp.text for sp in spans)


def test_find_anchor_locates_a_unique_span() -> None:
    page = "Intro line. The deluxe plan costs forty dollars per month. Outro."
    target = "The deluxe plan costs forty dollars per month."
    anchor = find_anchor(target, page)
    assert anchor is not None
    assert isinstance(anchor, Anchor)
    assert page[anchor.start : anchor.end] == target


def test_find_anchor_returns_none_when_absent() -> None:
    page = "Nothing about pricing here at all."
    assert find_anchor("a span that does not occur", page) is None


def test_repeated_value_is_relocatable_by_offset_not_abstained() -> None:
    # The SAME sentence appears twice. Anchoring must still succeed (positional), and the `near`
    # hint must pick the intended occurrence - repetition is NOT a reason to fail.
    repeated = "Net margin was 12%."
    page = f"Q1 summary. {repeated} More text in between here. Q2 summary. {repeated} The end."
    first = page.index(repeated)
    second = page.index(repeated, first + 1)
    assert first != second

    # No hint: anchors the first occurrence, still a valid offset (not None).
    a0 = find_anchor(repeated, page)
    assert a0 is not None
    assert page[a0.start : a0.end] == repeated

    # With a `near` hint close to the second occurrence, we get the second offset.
    a2 = find_anchor(repeated, page, near=second)
    assert a2 is not None
    assert a2.start == second
    assert page[a2.start : a2.end] == repeated

    # With a `near` hint close to the first occurrence, we get the first offset.
    a1 = find_anchor(repeated, page, near=first)
    assert a1 is not None
    assert a1.start == first


def test_is_uniquely_locatable_is_diagnostic_only() -> None:
    page = "alpha beta. alpha beta. gamma."
    assert is_uniquely_locatable("gamma.", page) is True
    # Repeats -> not globally unique, but this is a DIAGNOSTIC, not an abstain trigger.
    assert is_uniquely_locatable("alpha beta.", page) is False
    # And it can still be anchored despite not being unique.
    assert find_anchor("alpha beta.", page) is not None
