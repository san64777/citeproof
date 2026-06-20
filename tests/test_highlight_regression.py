"""Regression guards for the two real mis-highlight bugs found by clicking real receipts.

A receipt that highlights the WRONG passage is the cardinal sin (it makes a citation lie), so both
fixes get a permanent test:

  1. SEPARATOR bug: _visible_text must join text nodes WITHOUT inserting separators, the same way the
     in-browser highlight script (a TreeWalker doing `text += nodeValue`) does. Joining with " "
     inserted phantom spaces around inline links ("blast fishing , cyanide fishing"), so a passage
     crossing an <a> could never be re-located in the live DOM.
  2. NON-PROSE bug: the binder must drop navigation / reference / table chunks (a Wikipedia "See also"
     list out-scored the real sentence on an NLI and got cited, then highlighted).
"""

from veriscrape import Verdict

from citeproof.anchor import anchor_quote
from citeproof.binder.binder import EntailmentBinder, _is_citable_prose
from citeproof.binder.entailment import FakeEntailment
from citeproof.eval.models import Bucket, ClaimSourcePair, Fold
from citeproof.spine import _visible_text


def test_visible_text_does_not_insert_phantom_spaces_around_inline_elements() -> None:
    # A sentence whose middle word is a link must stay contiguous, so it matches the browser's
    # separator-less text-node walk and the receipt can re-anchor it.
    html = '<html><body><p>Reefs are threatened by <a href="/x">blast fishing</a>, cyanide fishing.</p></body></html>'
    visible = _visible_text(html)
    assert "blast fishing, cyanide fishing" in visible  # NOT "blast fishing , cyanide fishing"
    assert "blast fishing , cyanide" not in visible


def test_link_crossing_passage_is_reanchorable_in_visible_text() -> None:
    html = (
        "<html><body><p>The Alpha solar farm produces "
        '<a href="/mw">690 megawatts</a> of power.</p></body></html>'
    )
    visible = _visible_text(html)
    anchor = anchor_quote("The Alpha solar farm produces 690 megawatts of power.", visible)
    assert anchor is not None
    assert anchor.strategy in ("exact", "normalized")  # found cleanly, no risky fuzzy guess


def test_non_prose_sections_are_not_citable_prose() -> None:
    assert _is_citable_prose("Coral reefs are built from coral polyps.") is True
    assert _is_citable_prose("See also\n- List of prime ministers of India") is False
    assert _is_citable_prose("References\n1. Smith, J. (2020).") is False
    assert _is_citable_prose("External links\n- Official website") is False
    assert _is_citable_prose("- ^ Sources discussing the controversy[20][21]") is False
    assert _is_citable_prose("| Name | Born | Office | | Modi | 1950 | PM |") is False  # infobox


def test_binder_cites_the_prose_sentence_over_a_higher_scored_nav_list() -> None:
    claim = "Narendra Modi was born on 17 September 1950."
    nav = "See also\n- List of prime ministers of India\n- Premiership of Narendra Modi"
    prose = "Narendra Modi was born on 17 September 1950 in Vadnagar."
    page = f"{prose}\n\n{nav}"
    # the NLI scores the nav list HIGHER, yet the binder must cite the prose
    binder = EntailmentBinder(FakeEntailment(scores={(claim, nav): 0.99, (claim, prose): 0.80}), tau_mc=0.7)
    out = binder.bind(ClaimSourcePair(
        id="r", bucket=Bucket.CLEAN_ENTAILED, fold=Fold.TEST, claim=claim, source_url="u",
        source_text=page, verdict=Verdict.OK, gold_span=None, entailed=False, answerable=False))
    assert out.cited is True
    assert "See also" not in (out.cited_span or "")
    assert out.cited_span == prose
