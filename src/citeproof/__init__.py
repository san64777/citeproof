"""citeproof: local Deep Research that won't cite a source it can't verify is real.

Every cited claim is grounded to a verified-OK source span (the veriscrape verdict gate
plus a local entailment binder) and links to a highlighted page snapshot, or it abstains.
Pre-alpha: the package skeleton is in place; the M0 binder spike gates everything past it.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
