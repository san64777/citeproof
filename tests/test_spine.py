"""The spine end to end (offline): every outcome is an honest, named stage."""

from pathlib import Path

import pytest
from veriscrape import FetchRecord, Verdict

import citeproof.spine as spine_mod
from citeproof.binder.binder import EntailmentBinder
from citeproof.binder.entailment import FakeEntailment
from citeproof.extract import read_artifact
from citeproof.snapshot import SnapshotArtifact
from citeproof.spine import run_spine

_ARTICLE = """<html><head><title>Alpha solar farm</title></head><body>
<nav><a href="/">Home</a></nav>
<main><article><h1>Alpha solar farm</h1>
<p>The Alpha solar farm in Nevada produces 690 megawatts of power.
It was completed in 2021 and covers about 3,000 acres of desert land.
The plant supplies electricity to roughly 180,000 homes across the state.</p></article></main>
<footer>Copyright 2026.</footer></body></html>"""

_CLAIM = "The Alpha solar farm in Nevada produces 690 megawatts of power."


def _binder() -> EntailmentBinder:
    return EntailmentBinder(FakeEntailment(), tau_mc=0.7)


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, verdict: Verdict, body: str = _ARTICLE) -> None:
    rec = FetchRecord(url="https://example.test/solar", verdict=verdict, status=200, text=body)
    monkeypatch.setattr(spine_mod, "fetch", lambda url, timeout=20.0: rec)


def test_junk_page_is_excluded_at_the_cite_gate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # THE demo moment: a login wall is excluded with the verdict as the reason - never cited.
    _patch_fetch(monkeypatch, Verdict.LOGIN_WALL)
    r = run_spine("https://example.test/solar", _CLAIM, _binder(), tmp_path)
    assert r.stage == "excluded"
    assert r.receipt is None
    assert "LOGIN_WALL" in (r.reason or "")


def test_unverified_is_excluded_like_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_fetch(monkeypatch, Verdict.UNVERIFIED)
    r = run_spine("https://example.test/solar", _CLAIM, _binder(), tmp_path)
    assert r.stage == "excluded"


def test_supported_claim_on_ok_page_yields_a_full_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_fetch(monkeypatch, Verdict.OK)
    r = run_spine("https://example.test/solar", _CLAIM, _binder(), tmp_path)
    assert r.stage == "cited", r.reason
    rcpt = r.receipt
    assert rcpt is not None
    # the artifact exists and its recorded hash still matches the bytes on disk
    art_html = read_artifact(
        SnapshotArtifact(
            url=rcpt.url, path=rcpt.artifact_path, sha256=rcpt.artifact_sha256,
            tool="raw-fetch", created_at=0.0,
        )
    )
    assert "690 megawatts" in art_html
    # the anchor's exact text is genuinely present in the artifact's visible text
    assert "690 megawatts" in rcpt.anchor_exact
    assert rcpt.anchor_strategy in ("exact", "normalized", "fuzzy")
    assert rcpt.quote  # the binder's verbatim cited span


def test_unsupported_claim_abstains(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_fetch(monkeypatch, Verdict.OK)
    r = run_spine(
        "https://example.test/solar",
        "The Beta wind farm in Texas generates entirely unrelated geothermal energy.",
        _binder(),
        tmp_path,
    )
    assert r.stage == "abstained"
    assert r.receipt is None


def test_empty_page_is_no_content(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_fetch(monkeypatch, Verdict.OK, body="<html><head><title>x</title></head><body></body></html>")
    r = run_spine("https://example.test/solar", _CLAIM, _binder(), tmp_path)
    assert r.stage == "no_content"


def test_empty_shell_escalates_then_cites(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # fetch sees a JS skeleton; the (mocked) render reveals the real article; the spine cites it
    # and discloses tactic="rendered" on the receipt.
    _patch_fetch(monkeypatch, Verdict.EMPTY_SHELL, body="<html><body><div id='root'></div></body></html>")
    rendered = FetchRecord(
        url="https://example.test/solar", verdict=Verdict.OK, status=200, text=_ARTICLE, tactic="rendered"
    )
    monkeypatch.setattr(spine_mod, "render_and_reclassify", lambda url, timeout=30.0: rendered)
    r = run_spine("https://example.test/solar", _CLAIM, _binder(), tmp_path)
    assert r.stage == "cited", r.reason
    assert r.receipt is not None
    assert r.receipt.tactic == "rendered"


def test_unanchorable_span_is_dropped_not_mishighlighted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_fetch(monkeypatch, Verdict.OK)
    monkeypatch.setattr(spine_mod, "anchor_quote", lambda quote, target, **kw: None)
    r = run_spine("https://example.test/solar", _CLAIM, _binder(), tmp_path)
    assert r.stage == "unanchored"
    assert r.receipt is None
