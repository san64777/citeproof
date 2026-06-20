"""Tests for DECOMPOSE: the deterministic FakeDecomposer, the lazy Ollama adapter, and - the
spec-critical core - the faithfulness ROUND-TRIP (symbolic veto AND bidirectional entailment by a
NON-MiniCheck arbiter).

FakeDecomposer and FakeEntailment keep every assertion deterministic with NO LLM and NO new
dependency. The load-bearing assertions: (a) a number/entity/quantifier-drifted atomic is dropped
by the ORTHOGONAL symbolic check even when the arbiter would entail it (proving the round-trip is
genuinely non-MiniCheck), (b) an atomic the arbiter does not entail is dropped, and (c) a plainly
faithful atomic is NOT over-dropped (recall safety).
"""

from __future__ import annotations

import inspect

import pytest

from citeproof.binder.decompose import (
    AtomicClaim,
    Decomposer,
    DecompositionResult,
    FakeDecomposer,
    OllamaDecomposer,
    decompose_and_filter,
    is_faithful,
)
from citeproof.binder.entailment import FakeEntailment


def _entails_both_ways(*pairs: tuple[str, str]) -> FakeEntailment:
    """A FakeEntailment that scores 1.0 for both directions of each (a, b) pair, 0.0 otherwise.

    This isolates the SYMBOLIC veto: the arbiter is made to entail bidirectionally, so any drop is
    attributable to the symbolic check, not the NLI score.
    """
    scores: dict[tuple[str, str], float] = {}
    for a, b in pairs:
        scores[(a, b)] = 1.0
        scores[(b, a)] = 1.0
    return FakeEntailment(scores=scores)


# --- FakeDecomposer ------------------------------------------------------------------------------


def test_fake_decomposer_is_a_decomposer_protocol() -> None:
    dec: Decomposer = FakeDecomposer()
    assert isinstance(dec, Decomposer)


def test_fake_decomposer_scripted_mapping_is_deterministic() -> None:
    atomics = [
        AtomicClaim(text="Paris is the capital of France.", source_sentence="It is the capital.", index=0),
    ]
    dec = FakeDecomposer({"draft text": atomics})
    out1 = dec.decompose("draft text")
    out2 = dec.decompose("draft text")
    assert out1 == atomics
    assert out1 == out2


def test_fake_decomposer_callable_is_used() -> None:
    def fn(draft: str) -> list[AtomicClaim]:
        return [AtomicClaim(text=draft.upper(), source_sentence=draft, index=0)]

    dec = FakeDecomposer(fn=fn)
    [atomic] = dec.decompose("hello")
    assert atomic.text == "HELLO"


def test_fake_decomposer_naive_split_fallback_for_uncovered_draft() -> None:
    # The fallback is a baseline/test convenience only (see the FakeDecomposer docstring), and it is
    # a NAIVE split: one atomic per sentence, text == source_sentence (no decontextualization).
    dec = FakeDecomposer()
    out = dec.decompose("Revenue grew. Costs fell.")
    assert [a.text for a in out] == ["Revenue grew.", "Costs fell."]
    assert all(a.text == a.source_sentence for a in out)
    assert [a.index for a in out] == [0, 1]


def test_fake_decomposer_rejects_both_scripted_and_fn() -> None:
    with pytest.raises(ValueError):
        FakeDecomposer({"x": []}, fn=lambda d: [])


# --- is_faithful: the round-trip core ------------------------------------------------------------


def test_faithful_atomic_passes_round_trip() -> None:
    # No symbolic contradiction, arbiter preserves the claim and the context supports the atomic
    # -> faithful. (context == source here, and _entails_both_ways wires score(source, atomic)=1.0.)
    source = "The company's revenue grew in 2025."
    atomic = "The company's revenue grew in 2025."
    arbiter = _entails_both_ways((atomic, source))
    assert is_faithful(atomic, source, source, arbiter) is True


def test_number_drift_dropped_by_symbolic_veto_even_when_arbiter_entails() -> None:
    # The arbiter is RIGGED to entail both ways (1.0/1.0), so if the atomic is still dropped it can
    # ONLY be the orthogonal symbolic check firing - proving the round-trip is genuinely non-MiniCheck
    # and not just an NLI judging an NLI. Source says 12%, atomic says 21% -> a number drift.
    source = "Revenue grew 12% in 2025."
    atomic = "Revenue grew 21% in 2025."
    arbiter = _entails_both_ways((atomic, source))
    # Sanity: the arbiter alone WOULD bless it both ways (this is the trap the symbolic veto closes).
    assert arbiter.score(atomic, source) == 1.0
    assert arbiter.score(source, atomic) == 1.0
    # But the symbolic number-contradiction veto drops it.
    assert is_faithful(atomic, source, source, arbiter) is False


def test_year_drift_dropped_by_symbolic_veto() -> None:
    source = "The treaty was signed in 1994."
    atomic = "The treaty was signed in 1999."
    arbiter = _entails_both_ways((atomic, source))
    assert is_faithful(atomic, source, source, arbiter) is False


def test_quantifier_drift_dropped_by_symbolic_veto() -> None:
    # "all studies" -> "some studies": a quantifier flip on the same head noun. The arbiter is rigged
    # to entail both ways, so the drop is the symbolic quantifier veto, not the NLI.
    source = "All studies found a benefit."
    atomic = "Some studies found a benefit."
    arbiter = _entails_both_ways((atomic, source))
    assert is_faithful(atomic, source, source, arbiter) is False


def test_negation_drift_dropped_by_symbolic_veto() -> None:
    # A dropped "not" is the classic decontextualization drift; the symbolic polarity check catches it.
    source = "Sales did not fall in the quarter."
    atomic = "Sales fell in the quarter."
    arbiter = _entails_both_ways((atomic, source))
    assert is_faithful(atomic, source, source, arbiter) is False


def test_atomic_arbiter_does_not_entail_is_dropped() -> None:
    # No symbolic contradiction, but the arbiter returns a LOW score for this pair -> dropped on the
    # entailment leg of the round-trip (not the symbolic leg). context == source here.
    source = "The report covered several topics."
    atomic = "The report covered several topics in great depth."
    arbiter = FakeEntailment(scores={(atomic, source): 0.2, (source, atomic): 0.2})
    assert is_faithful(atomic, source, source, arbiter) is False


def test_round_trip_requires_does_not_add_against_context() -> None:
    # The CONTEXT does not support the atomic (context-support leg < tau) -> dropped. The
    # no-hallucination guard: an atomic the draft context cannot support is unfaithful.
    source = "The drug reduced symptoms."
    atomic = "The drug reduced symptoms."
    context = "The weather was sunny that day."
    arbiter = FakeEntailment(scores={(atomic, context): 0.3})  # context does NOT support the atomic
    assert is_faithful(atomic, source, context, arbiter) is False


def test_faithful_atomic_not_over_dropped_recall_safety() -> None:
    # A plainly faithful, clearly entailed atomic with no symbolic contradiction must survive: the
    # round-trip must not false-reject good atomics (recall safety).
    source = "The Eiffel Tower is in Paris."
    atomic = "The Eiffel Tower is in Paris."
    arbiter = _entails_both_ways((atomic, source))
    assert is_faithful(atomic, source, source, arbiter) is True


def test_faithful_pronoun_resolution_is_kept() -> None:
    # OVER-DROP REGRESSION GUARD (the recall-killing bug). A faithful decontextualizer RESOLVES
    # pronouns, so the resolved atomic ADDS antecedent tokens ("European Union", "GDPR") the bare
    # source sentence lacks. The OLD bidirectional-against-bare-sentence logic dropped this faithful
    # atomic (backward source -> atomic < tau), inflating the drift rate. The corrected does-not-add
    # leg checks the DRAFT CONTEXT, which DOES contain the antecedents, so the atomic is KEPT.
    source = "It enforced it in 2018."
    atomic = "The European Union enforced the GDPR in 2018."
    context = "The European Union passed the GDPR. It enforced it in 2018."
    arbiter = FakeEntailment(
        scores={
            (atomic, context): 0.9,  # context-support: the draft context supports the resolved atomic
            (atomic, source): 0.02,  # bare source does NOT entail the atomic (must NOT be consulted)
        }
    )
    assert is_faithful(atomic, source, context, arbiter) is True


def test_hallucinated_atomic_still_dropped_does_not_add_protection() -> None:
    # CONTEXT-SUPPORT protection: an atomic NOT supported by the draft context (context -> atomic <
    # tau) is a hallucination / added fact and must be dropped, even with no symbolic contradiction.
    source = "The agency issued a report."
    atomic = "The agency issued a 400-page report in March."
    context = "The agency issued a report."
    arbiter = FakeEntailment(
        scores={(atomic, context): 0.3}  # context does NOT support the added "400-page"/"March"
    )
    assert is_faithful(atomic, source, context, arbiter) is False


def test_multi_fact_source_atomic_is_kept_no_preserves_leg() -> None:
    # REGRESSION (over-drop found in the real-model dry run: drift_rate 0.67 on a clean draft). A
    # source SENTENCE often carries several facts, and a single atomic cannot entail the whole
    # sentence - the removed "preserves" leg (atomic |= source_sentence) scored ~0 and DROPPED every
    # faithful atomic of a multi-fact sentence. With only symbolic-veto + context-support, a faithful
    # atomic the CONTEXT supports is KEPT even though it does not entail the full source sentence.
    source = "It enforced it in 2018, and regulators fined several large firms."
    atomic = "The European Union began enforcing the GDPR in 2018."
    context = "The European Union passed the GDPR. It enforced it in 2018, and regulators fined firms."
    arbiter = FakeEntailment(
        scores={
            (atomic, context): 0.95,  # context-support: the draft supports the atomic -> KEEP
            (source, atomic): 0.01,  # the removed preserves leg WOULD score ~0 here; must be ignored
        }
    )
    assert is_faithful(atomic, source, context, arbiter) is True


def test_minicheck_arbiter_is_rejected_circularity_guard() -> None:
    # The round-trip arbiter must be the NON-MiniCheck orthogonal model (the pre-registered rule).
    # Passing any arbiter whose class name contains "MiniCheck" is the forbidden circularity and must
    # raise ValueError at runtime (not just a docstring ban).
    class MiniCheckEntailment:  # name deliberately matches the real adapter to trip the guard
        def score(self, claim: str, span: str) -> float:
            return 1.0

    arbiter = MiniCheckEntailment()
    with pytest.raises(ValueError, match="circularity"):
        is_faithful("a", "a", "a", arbiter)
    with pytest.raises(ValueError, match="circularity"):
        decompose_and_filter("draft", FakeDecomposer({"draft": []}), arbiter)


def test_tau_threshold_is_respected() -> None:
    # At exactly tau the atomic is kept (>= is inclusive); just below tau it is dropped.
    source = "Output increased."
    atomic = "Output increased."
    # context == source; both legs (preserves-the-claim and does-not-add) read the same value here.
    at_tau = FakeEntailment(scores={(atomic, source): 0.7, (source, atomic): 0.7})
    below = FakeEntailment(scores={(atomic, source): 0.69, (source, atomic): 0.69})
    assert is_faithful(atomic, source, source, at_tau, tau=0.7) is True
    assert is_faithful(atomic, source, source, below, tau=0.7) is False


# --- decompose_and_filter: partition + drift_rate ------------------------------------------------


def test_decompose_and_filter_partitions_and_computes_drift_rate() -> None:
    # Two atomics from a drafted pair: one faithful, one number-drifted. The drifted one is dropped
    # by the symbolic veto; drift_rate = 1/2 = 0.5; kept and dropped are disjoint and partition all.
    draft = "Revenue grew 12% in 2025. Costs rose 8% in 2025."
    good = AtomicClaim(
        text="Revenue grew 12% in 2025.",
        source_sentence="Revenue grew 12% in 2025.",
        index=0,
    )
    drifted = AtomicClaim(
        text="Costs rose 30% in 2025.",
        source_sentence="Costs rose 8% in 2025.",
        index=1,
    )
    decomposer = FakeDecomposer({draft: [good, drifted]})
    # Wire BOTH the preserves-the-claim leg (atomic -> source) and the does-not-add leg
    # (draft context -> atomic) so the kept atomic survives both legs of the round-trip.
    arbiter = _entails_both_ways(
        (good.text, good.source_sentence),
        (good.text, draft),
        (drifted.text, drifted.source_sentence),
        (drifted.text, draft),
    )

    result = decompose_and_filter(draft, decomposer, arbiter)

    assert isinstance(result, DecompositionResult)
    assert result.kept == [good]
    assert result.dropped == [drifted]
    assert result.drift_rate == pytest.approx(0.5)
    # kept and dropped are disjoint and together cover every emitted atomic.
    assert not (set(id(a) for a in result.kept) & set(id(a) for a in result.dropped))
    assert len(result.kept) + len(result.dropped) == 2


def test_decompose_and_filter_all_faithful_zero_drift() -> None:
    draft = "Water boils at 100 degrees. Ice melts at 0 degrees."
    a0 = AtomicClaim(text="Water boils at 100 degrees.", source_sentence="Water boils at 100 degrees.", index=0)
    a1 = AtomicClaim(text="Ice melts at 0 degrees.", source_sentence="Ice melts at 0 degrees.", index=1)
    decomposer = FakeDecomposer({draft: [a0, a1]})
    arbiter = _entails_both_ways(
        (a0.text, a0.source_sentence),
        (a0.text, draft),
        (a1.text, a1.source_sentence),
        (a1.text, draft),
    )
    result = decompose_and_filter(draft, decomposer, arbiter)
    assert result.kept == [a0, a1]
    assert result.dropped == []
    assert result.drift_rate == 0.0


def test_decompose_and_filter_empty_draft_zero_drift_rate() -> None:
    # No atomics -> drift_rate is 0.0 (no division by zero), kept and dropped both empty.
    decomposer = FakeDecomposer({"": []})
    result = decompose_and_filter("", decomposer, FakeEntailment())
    assert result.kept == []
    assert result.dropped == []
    assert result.drift_rate == 0.0


# --- OllamaDecomposer: lazy contract -------------------------------------------------------------


def test_ollama_decomposer_lazy_import_raises_actionable_error() -> None:
    # When the optional 'ollama' client is NOT installed, constructing the adapter must raise a clear
    # ImportError, never crash at module import. Skipped when ollama IS installed (binder extra
    # present), since the ImportError path then does not fire.
    import importlib.util

    if importlib.util.find_spec("ollama") is not None:
        pytest.skip("ollama present (binder extra); ImportError path not exercised")
    with pytest.raises(ImportError) as excinfo:
        OllamaDecomposer()
    msg = str(excinfo.value)
    assert "binder" in msg  # mentions the optional extra to install
    assert "uv sync" in msg


def test_decompose_module_has_no_top_level_heavy_import() -> None:
    # Importing decompose.py must not pull in minicheck/transformers/torch/ollama. The faithfulness
    # round-trip is forbidden from touching MiniCheck (the circularity the red-team forbids), and the
    # Ollama client is lazy. Inspect the source for any top-level (column-0) import of those names.
    import citeproof.binder.decompose as dec

    src = inspect.getsource(dec)
    forbidden = ("minicheck", "transformers", "torch", "ollama")
    for line in src.splitlines():
        stripped = line
        if stripped.startswith("import ") or stripped.startswith("from "):
            head = stripped.split()[1].split(".")[0]
            assert head not in forbidden, f"{head} must not be imported at module top level"


def test_real_ollama_decomposer_produces_decontextualized_atomics() -> None:
    # Opt-in real-model integration: needs a running Ollama daemon + a pulled chat model. Skipped by
    # default so check.sh / CI stay fast and model-free. Run with (after `ollama pull qwen3:4b`):
    #   CITEPROOF_RUN_MODEL_TESTS=1 CITEPROOF_OLLAMA_MODEL=qwen3:4b .venv/bin/python -m pytest -k real_ollama
    import os

    if not os.environ.get("CITEPROOF_RUN_MODEL_TESTS"):
        pytest.skip("set CITEPROOF_RUN_MODEL_TESTS=1 to run the real Ollama decomposer test")
    pytest.importorskip("ollama")
    model = os.environ.get("CITEPROOF_OLLAMA_MODEL", "qwen3:4b")
    dec = OllamaDecomposer(model=model)
    try:
        atomics = dec.decompose(
            "The European Union passed the GDPR in 2016. It began enforcing the law in 2018."
        )
    except Exception as exc:  # daemon down or model not pulled -> skip, not fail
        pytest.skip(f"Ollama not usable (daemon down or model {model!r} missing?): {exc}")
    assert len(atomics) >= 2  # at least one atomic fact per sentence
    assert all(a.text.strip() and a.source_sentence.strip() for a in atomics)
    # Decontextualization: the second sentence's pronoun "It" is resolved, so the subject entity
    # (European Union / EU) recurs across the atomics rather than staying a bare pronoun.
    joined = " ".join(a.text.lower() for a in atomics)
    assert "european union" in joined or "the eu" in joined
