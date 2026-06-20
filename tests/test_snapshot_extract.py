"""Snapshot + extract: the one-artifact invariant, enforced by hash."""

from pathlib import Path

import pytest

from citeproof.extract import ArtifactIntegrityError, extract_text, read_artifact
from citeproof.fetch import UnsafeURLError
from citeproof.snapshot import SnapshotError, snapshot_raw, snapshot_url

_ARTICLE = """<html><head><title>Solar farms</title></head><body>
<nav><a href="/">Home</a> <a href="/about">About</a></nav>
<main><article><h1>Alpha solar farm</h1>
<p>The Alpha solar farm in Nevada produces 690 megawatts of power. It was completed in 2021
and covers about 3,000 acres of desert land. The plant supplies electricity to roughly
180,000 homes across the state, the operator said.</p></article></main>
<footer>Copyright 2026. Subscribe to our newsletter.</footer></body></html>"""


def test_snapshot_raw_writes_and_pins_the_bytes(tmp_path: Path) -> None:
    art = snapshot_raw(_ARTICLE, "https://example.test/solar", tmp_path)
    assert art.tool == "raw-fetch"
    assert Path(art.path).exists()
    assert len(art.sha256) == 64
    # the recorded digest pins the exact bytes on disk
    assert read_artifact(art) == _ARTICLE


def test_extract_returns_main_content_not_chrome(tmp_path: Path) -> None:
    art = snapshot_raw(_ARTICLE, "https://example.test/solar", tmp_path)
    text = extract_text(art)
    assert text is not None
    assert "690 megawatts" in text
    assert "Subscribe to our newsletter" not in text  # footer chrome stripped
    assert "Home" not in text  # nav stripped


def test_tampered_artifact_is_refused(tmp_path: Path) -> None:
    # THE one-artifact invariant: if the snapshot file changes after recording, extraction must
    # hard-fail - a receipt built from different bytes than the snapshot is a lie.
    art = snapshot_raw(_ARTICLE, "https://example.test/solar", tmp_path)
    Path(art.path).write_text(_ARTICLE.replace("690 megawatts", "900 megawatts"), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"):
        extract_text(art)


def test_missing_artifact_is_refused(tmp_path: Path) -> None:
    art = snapshot_raw(_ARTICLE, "https://example.test/solar", tmp_path)
    Path(art.path).unlink()
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        extract_text(art)


def test_extract_empty_page_returns_none(tmp_path: Path) -> None:
    art = snapshot_raw("<html><head><title>x</title></head><body></body></html>", "https://e.test/", tmp_path)
    assert extract_text(art) is None


def test_snapshot_url_is_ssrf_guarded(tmp_path: Path) -> None:
    # SingleFile performs its own network fetch, so the metadata endpoint must be refused
    # BEFORE any subprocess runs.
    with pytest.raises(UnsafeURLError):
        snapshot_url("http://169.254.169.254/latest/meta-data/", tmp_path)


def test_snapshot_url_clean_error_without_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import citeproof.snapshot as snap

    monkeypatch.setattr(snap, "singlefile_available", lambda: False)
    with pytest.raises(SnapshotError, match="npx"):
        snapshot_url("https://example.com/", tmp_path)
