"""citeproof M0 eval foundation: the deterministic stats core, data model, baselines, harness.

Everything here is test-first and offline. No model weights are loaded by this package; the
real binder (MiniCheck plus the symbolic check) plugs into the Binder protocol later. The point
of M0 is the eval that gates the whole build, so the bulletproof core lives here first.
"""

from __future__ import annotations

__all__: list[str] = []
