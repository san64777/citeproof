"""Receipt rendering: banner + a REAL highlight in a real browser engine."""

from pathlib import Path

import pytest
from veriscrape import FetchRecord, Verdict

import citeproof.spine as spine_mod
from citeproof.binder.binder import EntailmentBinder
from citeproof.binder.entailment import FakeEntailment
from citeproof.receipt import render_receipt_html
from citeproof.spine import run_spine

_ARTICLE = """<html><head><title>Alpha solar farm</title></head><body>
<nav><a href="/">Home</a></nav>
<main><article><h1>Alpha solar farm</h1>
<p>The Alpha solar farm in Nevada produces 690 megawatts of power.
It was completed in 2021 and covers about 3,000 acres of desert land.
The plant supplies electricity to roughly 180,000 homes across the state.</p></article></main>
<footer>Copyright 2026.</footer></body></html>"""

_CLAIM = "The Alpha solar farm in Nevada produces 690 megawatts of power."


def _receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    rec = FetchRecord(url="https://example.test/solar", verdict=Verdict.OK, status=200, text=_ARTICLE)
    monkeypatch.setattr(spine_mod, "fetch", lambda url, timeout=20.0: rec)
    result = run_spine("https://example.test/solar", _CLAIM, EntailmentBinder(FakeEntailment(), tau_mc=0.7), tmp_path)
    assert result.stage == "cited" and result.receipt is not None
    return result.receipt


def test_receipt_html_carries_banner_and_anchor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    receipt = _receipt(monkeypatch, tmp_path)
    html = render_receipt_html(_ARTICLE, receipt)
    assert "citeproof-banner" in html
    assert _CLAIM in html  # the claim is stated on the receipt
    assert receipt.artifact_sha256[:16] in html  # the digest is disclosed
    assert "CSS.highlights" in html
    # injection lands inside <body>, so the original document structure is preserved
    assert html.index("<body") < html.index("citeproof-banner")


def test_receipt_escapes_hostile_claim_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    receipt = _receipt(monkeypatch, tmp_path)
    hostile = receipt.model_copy(update={"claim": '<script>alert(1)</script>'})
    html = render_receipt_html(_ARTICLE, hostile)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_hostile_page_text_cannot_break_out_of_the_anchor_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The anchor text comes from the FETCHED PAGE: a page containing "</script><script>..." must not
    # escape the receipt's script block (XSS). The "</" is neutralized to "<\/" in the JSON.
    receipt = _receipt(monkeypatch, tmp_path)
    hostile = receipt.model_copy(
        update={"anchor_exact": 'quote ends </script><script>alert(document.cookie)</script>'}
    )
    html = render_receipt_html(_ARTICLE, hostile)
    assert "</script><script>alert(document.cookie)</script>" not in html
    assert "<\\/script><script>alert" in html  # neutralized inside the JS string, still valid JSON


def _chromium_usable() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(headless=True)
            except Exception:
                b = pw.chromium.launch(headless=True, channel="chrome")
            b.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _chromium_usable(), reason="playwright/chromium not available")
def test_highlight_actually_registers_in_a_real_browser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The M1 exit concern in miniature: the receipt's script must re-find the passage in a LIVE DOM
    # and register a CSS Custom Highlight - verified in a real engine, not assumed.
    from playwright.sync_api import sync_playwright

    receipt = _receipt(monkeypatch, tmp_path)
    html = render_receipt_html(_ARTICLE, receipt)
    page_path = tmp_path / "receipt.html"
    page_path.write_text(html, encoding="utf-8")

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=True)
        except Exception:
            browser = pw.chromium.launch(headless=True, channel="chrome")
        try:
            page = browser.new_page()
            page.goto(page_path.as_uri())
            page.wait_for_function("document.getElementById('citeproof-status').textContent.includes('highlighted')")
            assert page.evaluate("CSS.highlights.has('citeproof')") is True
            highlighted = page.evaluate(
                "Array.from(CSS.highlights.get('citeproof')).map(r => r.toString()).join(' ')"
            )
            assert "690 megawatts" in highlighted
            status = page.text_content("#citeproof-status") or ""
            assert "highlighted" in status
        finally:
            browser.close()
