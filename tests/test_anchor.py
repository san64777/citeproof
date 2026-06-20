"""Anchor re-find: exact, normalized, fuzzy - and abstain when the quote is not really there."""

from citeproof.anchor import anchor_quote, proportional_near

_PAGE = (
    "Background section text comes first.\n\n"
    "The Alpha solar farm in Nevada produces 690 megawatts of power. "
    "It was completed in 2021 and covers about 3,000 acres.\n\n"
    "Unrelated trailing text closes the page."
)


def test_exact_match_with_textquote_context() -> None:
    q = "The Alpha solar farm in Nevada produces 690 megawatts of power."
    a = anchor_quote(q, _PAGE)
    assert a is not None
    assert a.strategy == "exact"
    assert _PAGE[a.start:a.end] == q
    assert a.exact == q
    assert a.prefix.endswith("first.\n\n")
    assert a.suffix.startswith(" It was completed")


def test_repeated_quote_disambiguated_by_near() -> None:
    page = "The cat sat. Filler text in the middle here. The cat sat. More text."
    first = anchor_quote("The cat sat.", page, near=0)
    second = anchor_quote("The cat sat.", page, near=50)
    assert first is not None and second is not None
    assert first.start == 0
    assert second.start == page.find("The cat sat.", 1)


def test_proportional_near_disambiguates_a_repeated_span() -> None:
    # The mis-highlight blocker: a span that repeats must anchor to the occurrence NEAREST the one
    # the binder verified (mapped from the source offset), not always the first.
    page = "The cat sat. " * 5  # repeats at 0, 13, 26, 39, 52
    near = proportional_near(39, len(page), len(page))  # binder verified the 4th occurrence
    a = anchor_quote("The cat sat.", page, near=near)
    assert a is not None
    assert a.start == 39  # the nearer occurrence, NOT 0


def test_proportional_near_scales_when_extraction_kept_most_text() -> None:
    assert proportional_near(50, 190, 200) == 53  # extracted ~95% of visible -> trusted
    assert proportional_near(0, 190, 200) == 0
    assert proportional_near(None, 190, 200) is None  # nothing to scale from
    assert proportional_near(50, 0, 200) is None  # empty source guarded


def test_proportional_near_abstains_when_extraction_stripped_heavy_chrome() -> None:
    # The "strictly worse than None" corner: extracted is much shorter than visible (heavy trailing
    # chrome), so the linear map is untrustworthy and must fall back to first-occurrence (None).
    assert proportional_near(6, 50, 500) is None  # kept only 10% -> do not trust the hint


def test_normalized_match_survives_whitespace_differences() -> None:
    # Extracted text has clean single spaces; the artifact has newlines + double spaces.
    quote = "The Alpha solar farm in Nevada produces 690 megawatts of power."
    artifact = "Intro.\nThe Alpha solar farm\nin Nevada  produces 690\nmegawatts of power. Tail."
    a = anchor_quote(quote, artifact)
    assert a is not None
    assert a.strategy == "normalized"
    # offsets point INTO THE ARTIFACT and span the real (newline-broken) occurrence
    assert artifact[a.start:a.end].startswith("The Alpha solar farm")
    assert artifact[a.start:a.end].endswith("megawatts of power.")


def test_soft_hyphen_in_artifact_is_bridged() -> None:
    quote = "It covers about 3,000 acres of desert."
    artifact = "It covers about 3,000 ac­res of desert."  # soft hyphen mid-word
    a = anchor_quote(quote, artifact)
    assert a is not None
    assert a.strategy in ("normalized", "fuzzy")


def test_fuzzy_match_tolerates_a_tiny_difference() -> None:
    # One small token differs (entity-decoded apostrophe variant) - fuzzy should still locate it.
    quote = "the operator's plant supplies electricity to roughly 180,000 homes across the state"
    artifact = "Filler. Indeed the operator’s plant supplies electricity to roughly 180,000 homes across the state. End."
    a = anchor_quote(quote, artifact)
    assert a is not None
    assert a.score >= 0.90
    assert "180,000 homes" in artifact[a.start:a.end]


def test_absent_quote_abstains() -> None:
    assert anchor_quote("Completely unrelated sentence about volcanoes.", _PAGE) is None


def test_similar_but_wrong_quote_abstains() -> None:
    # Same topic, materially different content - must NOT highlight the near-neighbor.
    q = "The Alpha solar farm in Nevada produces 950 megawatts of geothermal energy and opened in 2019."
    assert anchor_quote(q, _PAGE) is None


def test_empty_inputs_abstain() -> None:
    assert anchor_quote("", _PAGE) is None
    assert anchor_quote("anything", "") is None
