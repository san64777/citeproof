"""Smoke test: the package imports and exposes its version.

Keeps the pytest gate green from the first scaffolded commit; the real binder/eval tests
arrive with M0.
"""

import citeproof


def test_version_exposed() -> None:
    assert citeproof.__version__ == "0.1.0"
