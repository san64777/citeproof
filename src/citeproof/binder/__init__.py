"""citeproof binder core: the M0 VERIFY + ATTACH-OR-FLAG unit.

This package is the (claim, source) verification core the eval harness scores: the orthogonal
symbolic contradiction check, the entailment-model seam (FakeEntailment for tests, MiniCheck as a
lazy adapter), candidate-span splitting with positional anchoring, and the EntailmentBinder that
ANDs the two signals behind a strict cite-gate.

Out of scope here (operates above the (claim, source) unit, at the draft/corpus level): DECOMPOSE
(VeriScore decontextualized atomics), RETRIEVE (top-k over OK snapshots), and the ALCE leave-one-out
PRUNE. Those are separate chunks; this chunk is VERIFY + ATTACH-OR-FLAG.
"""

from __future__ import annotations

from citeproof.binder.binder import EntailmentBinder
from citeproof.binder.decompose import (
    AtomicClaim,
    Decomposer,
    DecompositionResult,
    FakeDecomposer,
    OllamaDecomposer,
    decompose_and_filter,
    is_faithful,
)
from citeproof.binder.entailment import (
    DebertaMnliEntailment,
    EntailmentModel,
    FakeEntailment,
    MiniCheckEntailment,
)
from citeproof.binder.spans import Anchor, Span, candidate_spans, find_anchor, is_uniquely_locatable
from citeproof.binder.symbolic import SymbolicResult, symbolic_consistency

__all__ = [
    "Anchor",
    "AtomicClaim",
    "DebertaMnliEntailment",
    "Decomposer",
    "DecompositionResult",
    "EntailmentBinder",
    "EntailmentModel",
    "FakeDecomposer",
    "FakeEntailment",
    "MiniCheckEntailment",
    "OllamaDecomposer",
    "Span",
    "SymbolicResult",
    "candidate_spans",
    "decompose_and_filter",
    "find_anchor",
    "is_faithful",
    "is_uniquely_locatable",
    "symbolic_consistency",
]
