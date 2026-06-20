"""Tests for the entailment seam: the Protocol, FakeEntailment, and the lazy MiniCheck adapter.

FakeEntailment is the deterministic stand-in used by every binder test (no model weights, no new
dependency). MiniCheckEntailment must NOT import minicheck/transformers at module top level; it
imports lazily and raises a clear, actionable error when the optional 'binder' extra is absent.
"""

from __future__ import annotations

import pytest

from citeproof.binder.entailment import (
    DebertaMnliEntailment,
    EntailmentModel,
    FakeEntailment,
    MiniCheckEntailment,
)


def test_fake_entailment_is_an_entailment_model() -> None:
    fake: EntailmentModel = FakeEntailment()
    score = fake.score("a b c", "a b c d")
    assert 0.0 <= score <= 1.0


def test_fake_entailment_jaccard_default_is_deterministic() -> None:
    fake = FakeEntailment()
    s1 = fake.score("the cat sat", "the cat sat on the mat")
    s2 = fake.score("the cat sat", "the cat sat on the mat")
    assert s1 == s2
    # Identical strings -> full overlap -> 1.0.
    assert fake.score("same text", "same text") == 1.0
    # Disjoint -> 0.0.
    assert fake.score("alpha beta", "gamma delta") == 0.0


def test_fake_entailment_scripted_dict_overrides_jaccard() -> None:
    # A caller-supplied dict keyed by (claim, span) gives exact control in tests.
    scripted = {("claim x", "span y"): 0.97}
    fake = FakeEntailment(scores=scripted)
    assert fake.score("claim x", "span y") == 0.97
    # A pair not in the dict falls back to the deterministic Jaccard default.
    assert fake.score("alpha beta", "gamma delta") == 0.0


def test_minicheck_adapter_lazy_import_raises_actionable_error() -> None:
    # When the optional 'binder' extra (minicheck) is NOT installed, constructing the adapter must
    # raise a clear, actionable ImportError, never crash at module import. Skipped when minicheck IS
    # installed (the binder extra is present), since the ImportError path then does not fire.
    import importlib.util

    if importlib.util.find_spec("minicheck") is not None:
        pytest.skip("minicheck installed (binder extra present); ImportError path not exercised")
    with pytest.raises(ImportError) as excinfo:
        MiniCheckEntailment()
    msg = str(excinfo.value)
    assert "binder" in msg  # mentions the optional extra to install
    assert "uv sync" in msg


def test_minicheck_module_has_no_top_level_heavy_import() -> None:
    # Importing the module must not pull in minicheck/transformers. Inspect the source for a
    # top-level import of those names (defensive: the lazy contract is load-bearing).
    import inspect

    import citeproof.binder.entailment as ent

    src = inspect.getsource(ent)
    # No top-level (column-0) import of the heavy packages.
    for line in src.splitlines():
        if line.startswith("import minicheck") or line.startswith("from minicheck"):
            raise AssertionError("minicheck must not be imported at module top level")
        if line.startswith("import transformers") or line.startswith("from transformers"):
            raise AssertionError("transformers must not be imported at module top level")


def test_minicheck_entailment_real_inference() -> None:
    # Opt-in real-model integration: downloads the roberta-large checkpoint and runs inference.
    # Skipped by default so check.sh / CI stay fast and model-free. Run it with:
    #   CITEPROOF_RUN_MODEL_TESTS=1 .venv/bin/python -m pytest tests/binder/test_entailment.py -k real_inference
    # (requires the binder extra + minicheck installed; see pyproject [project.optional-dependencies]).
    import os

    if not os.environ.get("CITEPROOF_RUN_MODEL_TESTS"):
        pytest.skip("set CITEPROOF_RUN_MODEL_TESTS=1 to run the real MiniCheck integration test")
    pytest.importorskip("minicheck")
    model = MiniCheckEntailment()
    supported = model.score("The Eiffel Tower is in Paris.", "The Eiffel Tower is located in Paris, France.")
    unrelated = model.score("The Eiffel Tower is in Paris.", "Bananas are a good source of potassium.")
    not_entailed = model.score("The drug is effective.", "The drug showed no benefit in the trial.")
    for s in (supported, unrelated, not_entailed):
        assert 0.0 <= s <= 1.0
    assert supported > 0.5  # a clearly-supported claim scores high
    assert unrelated < 0.5 and not_entailed < 0.5  # unsupported claims score low
    assert supported > unrelated


def test_deberta_arbiter_lazy_import_raises_actionable_error() -> None:
    # When the optional 'binder' extra (transformers) is NOT installed, constructing the arbiter must
    # raise a clear ImportError. Skipped when transformers IS installed (binder extra present).
    import importlib.util

    if importlib.util.find_spec("transformers") is not None:
        pytest.skip("transformers present (binder extra); ImportError path not exercised")
    with pytest.raises(ImportError) as excinfo:
        DebertaMnliEntailment()
    msg = str(excinfo.value)
    assert "binder" in msg
    assert "uv sync" in msg


def test_deberta_arbiter_real_entailment_directions() -> None:
    # Opt-in real-model integration: downloads a DeBERTa-v3 MNLI checkpoint. Skipped by default so
    # check.sh / CI stay fast. Run: CITEPROOF_RUN_MODEL_TESTS=1 .venv/bin/python -m pytest -k deberta_arbiter_real
    import os

    if not os.environ.get("CITEPROOF_RUN_MODEL_TESTS"):
        pytest.skip("set CITEPROOF_RUN_MODEL_TESTS=1 to run the real DeBERTa arbiter test")
    pytest.importorskip("transformers")
    arb = DebertaMnliEntailment()
    # score(claim, span) = P(span entails claim): a paraphrase scores high, an unrelated span low.
    high = arb.score("Paris is the capital of France.", "The capital of France is Paris.")
    low = arb.score("Paris is the capital of France.", "Bananas are a good source of potassium.")
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0
    assert high > 0.5 and low < 0.5
    assert high > low
