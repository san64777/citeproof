"""Snapshot: persist ONE artifact per cited page, the file every downstream step works from.

The receipt promise depends on a single invariant: the text we verify against, the span we highlight,
and the page the user opens are all THE SAME BYTES. So a snapshot is recorded as (path, sha256), and
the extractor re-hashes the file before reading it - a mismatch is a hard error, never a silent
re-fetch. That makes "one artifact" a checked property instead of a convention.

Two producers, one artifact shape:
  - snapshot_url: SingleFile CLI as a SUBPROCESS (AGPL-3.0 - it must NEVER enter the Python import
    tree; the license gate stays green because we only exec the binary). Produces a self-contained
    .html (resources inlined, scripts stripped) that renders offline in any webview.
  - snapshot_raw: writes an already-fetched body verbatim. The degraded path (no resource inlining)
    for offline/unit-test use and for when the CLI is unavailable; labeled `tool="raw-fetch"` so a
    receipt can say which fidelity it carries.

Both producers SSRF-guard nothing themselves except snapshot_url's target (SingleFile performs its
own network fetch, so the same pre-flight check as fetch.py applies).
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from citeproof.fetch import assert_safe_url

_SHA_PREFIX_LEN = 16  # filename prefix; the full digest lives in the artifact record


class SnapshotError(RuntimeError):
    """The snapshot could not be produced (CLI missing, subprocess failed, empty output)."""


class SnapshotArtifact(BaseModel):
    """One snapshotted page: the file plus the digest that pins every later read to these bytes."""

    url: str
    path: str
    sha256: str
    tool: Literal["singlefile", "raw-fetch"]
    created_at: float  # unix epoch seconds


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_for(path: Path, url: str, tool: Literal["singlefile", "raw-fetch"]) -> SnapshotArtifact:
    return SnapshotArtifact(
        url=url, path=str(path), sha256=_sha256_file(path), tool=tool, created_at=time.time()
    )


def snapshot_raw(body: str, url: str, out_dir: Path) -> SnapshotArtifact:
    """Write an already-fetched HTML body verbatim as the artifact (degraded fidelity: no resource
    inlining, so images/styles may not render offline - but the TEXT is byte-pinned all the same).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    path = out_dir / f"{digest[:_SHA_PREFIX_LEN]}.html"
    path.write_text(body, encoding="utf-8")
    return _artifact_for(path, url, "raw-fetch")


def singlefile_available() -> bool:
    """True if the SingleFile CLI can be invoked (npx on PATH; the package fetches on first use)."""
    return shutil.which("npx") is not None


def snapshot_url(
    url: str,
    out_dir: Path,
    *,
    browser_path: str | None = None,
    timeout: float = 90.0,
) -> SnapshotArtifact:
    """Snapshot a live URL with the SingleFile CLI (subprocess; AGPL stays out of the import tree).

    SingleFile drives its own headless Chromium fetch of `url`, so the same SSRF pre-flight as
    fetch.py applies. Produces one self-contained, script-stripped .html.
    """
    assert_safe_url(url)
    if not singlefile_available():
        raise SnapshotError("SingleFile CLI unavailable: npx not on PATH (install Node.js)")
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"snap-{int(time.time() * 1000)}.html"
    cmd = ["npx", "--yes", "single-file-cli", url, str(tmp), "--remove-scripts=true"]
    if browser_path:
        cmd.append(f"--browser-executable-path={browser_path}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise SnapshotError(f"SingleFile timed out after {timeout}s on {url}") from exc
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-400:]
        raise SnapshotError(f"SingleFile failed for {url} (rc={proc.returncode}): {detail}")
    # Rename to the content digest so the filename itself is stable and collision-free.
    digest = _sha256_file(tmp)
    final = out_dir / f"{digest[:_SHA_PREFIX_LEN]}.html"
    tmp.replace(final)
    return _artifact_for(final, url, "singlefile")
