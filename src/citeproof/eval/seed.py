"""Loader for the SYNTHETIC M0 seed pairs (bundled next to this module as seed_pairs.json).

The seed set is hand-written, clearly synthetic, and carries NO statistical weight toward the
M0 gate; the real >=50-per-bucket TEST fold runs through the real veriscrape. The seed exists
only to exercise the harness and the cite-gate coverage check (it includes non-OK pages whose
text entails their claim). The JSON path is resolved relative to the repo root, with an env
override (CITEPROOF_SEED_PAIRS) so tests and tools can point at a different file if needed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from citeproof.eval.models import ClaimSourcePair

# The synthetic seed ships WITH the package, next to this loader (a small test/demo fixture).
_DEFAULT_SEED_PATH = Path(__file__).resolve().parent / "seed_pairs.json"


def seed_path() -> Path:
    """Return the seed JSON path, honoring the CITEPROOF_SEED_PAIRS override."""
    override = os.environ.get("CITEPROOF_SEED_PAIRS")
    return Path(override) if override else _DEFAULT_SEED_PATH


def load_seed_pairs(path: Path | None = None) -> list[ClaimSourcePair]:
    """Load and validate the synthetic seed pairs.

    Args:
        path: optional explicit path; defaults to seed_path().

    Returns:
        The validated list of ClaimSourcePair (in file order).
    """
    target = path if path is not None else seed_path()
    raw = json.loads(target.read_text(encoding="utf-8"))
    return [ClaimSourcePair.model_validate(item) for item in raw["pairs"]]
