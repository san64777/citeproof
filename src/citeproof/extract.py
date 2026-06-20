"""Extract main-content text FROM THE SNAPSHOT - never from a re-fetch.

This is the enforcement point of the one-artifact invariant: before reading, the artifact file is
re-hashed and compared to the digest recorded at snapshot time. A mismatch (file edited, replaced,
corrupted) raises ArtifactIntegrityError instead of silently extracting from different bytes - because
a receipt whose highlighted text came from anything other than the snapshotted page is a lie.

Extraction is trafilatura (Apache-2.0, license verified 2026-06-10): best-in-class main-content
F1, strips nav/chrome/boilerplate. CAVEAT: extractors return NO source offsets, so downstream
anchoring is always by VERBATIM QUOTE re-found in the artifact, never by char offset.
"""

from __future__ import annotations

from pathlib import Path

import trafilatura

from citeproof.snapshot import SnapshotArtifact, _sha256_file


class ArtifactIntegrityError(RuntimeError):
    """The artifact's bytes no longer match its recorded sha256 - refuse to extract."""


def read_artifact(artifact: SnapshotArtifact) -> str:
    """Return the artifact's HTML after verifying its digest (the one-artifact invariant)."""
    path = Path(artifact.path)
    if not path.exists():
        raise ArtifactIntegrityError(f"artifact missing: {artifact.path}")
    actual = _sha256_file(path)
    if actual != artifact.sha256:
        raise ArtifactIntegrityError(
            f"artifact {artifact.path} hash mismatch: recorded {artifact.sha256[:12]}..., "
            f"file is {actual[:12]}... - the snapshot was modified; refusing to extract"
        )
    return path.read_text(encoding="utf-8", errors="replace")


def extract_text(artifact: SnapshotArtifact) -> str | None:
    """Main-content text of the snapshot, or None when no main content is found (an empty/junk
    page - the caller treats that as nothing-to-cite, mirroring the abstain-first posture).
    """
    html = read_artifact(artifact)
    return trafilatura.extract(html, url=artifact.url, include_comments=False)
