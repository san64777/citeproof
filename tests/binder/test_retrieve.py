"""Tests for RETRIEVE: the deterministic embedder, the cite-gated Retriever, and the metrics.

FakeEmbedder is a deterministic hashed bag-of-words, so cosine ranks lexically-overlapping text
higher and the retrieval-recall assertions are genuinely meaningful with no model weights and no
new runtime dependency. The load-bearing cite-gate assertion: a non-OK candidate is NEVER returned
when ok_only is True, even when it is the best lexical match.
"""

from __future__ import annotations

import pytest
from veriscrape import Verdict

from citeproof.binder.retrieve import (
    Candidate,
    Embedder,
    FakeEmbedder,
    OracleRetrieval,
    Retriever,
    SentenceTransformerEmbedder,
    retrieval_hit,
)


def _ok(span_text: str, url: str = "https://example.com/a", start: int = 0) -> Candidate:
    return Candidate(span_text=span_text, source_url=url, verdict=Verdict.OK, start=start)


# --- FakeEmbedder -------------------------------------------------------------------------------


def test_fake_embedder_is_an_embedder_and_deterministic() -> None:
    emb: Embedder = FakeEmbedder()
    v1 = emb.embed(["the cat sat on the mat"])
    v2 = emb.embed(["the cat sat on the mat"])
    assert v1 == v2  # same input -> same vector, every run
    assert len(v1) == 1
    assert len(v1[0]) == FakeEmbedder().dim


def test_fake_embedder_l2_normalizes_nonempty_text() -> None:
    [vec] = FakeEmbedder().embed(["alpha beta gamma"])
    norm = sum(x * x for x in vec) ** 0.5
    assert norm == pytest.approx(1.0)


def test_fake_embedder_zero_vector_for_no_word_tokens() -> None:
    # Punctuation-only text has no word tokens -> all-zero vector, not a crash.
    [vec] = FakeEmbedder().embed(["...!!!"])
    assert all(x == 0.0 for x in vec)


# --- Retriever: recall over a small corpus ------------------------------------------------------


def test_claim_retrieves_its_lexically_overlapping_gold_span_in_topk() -> None:
    gold = "The Eiffel Tower was completed in 1889 in Paris."
    candidates = [
        _ok("Bananas are a good source of potassium for athletes."),
        _ok("The stock market closed lower on Tuesday afternoon."),
        _ok(gold),
        _ok("Photosynthesis converts sunlight into chemical energy in plants."),
        _ok("A recipe for sourdough bread needs flour, water, and salt."),
    ]
    retriever = Retriever(FakeEmbedder(), candidates)
    claim = "The Eiffel Tower in Paris was completed in 1889."
    top = retriever.retrieve(claim, k=2)
    assert any(c.span_text == gold for c in top)
    # The gold span, sharing the most words with the claim, should rank first.
    assert top[0].span_text == gold


def test_retrieval_hit_helper_matches_normalized() -> None:
    gold = "Net margin was 12 percent."
    retrieved = [_ok("Revenue grew."), _ok("  net   MARGIN was 12 percent.  ")]
    assert retrieval_hit(retrieved, gold) is True
    assert retrieval_hit(retrieved, "a totally different sentence") is False
    # Empty gold never hits.
    assert retrieval_hit(retrieved, "") is False


# --- Retriever: the cite-gate (the load-bearing assertion) --------------------------------------


def test_ok_only_excludes_a_non_ok_candidate_even_when_best_match() -> None:
    # The BLOCKED candidate is a verbatim copy of the claim - the single best lexical match. With
    # ok_only=True it must be EXCLUDED from the index and therefore NEVER returned. This is the
    # cite-gate, defense in depth: a non-OK snapshot is never even a retrieval candidate.
    claim = "The treaty was signed in Vienna in 1815 by the great powers."
    blocked_best = Candidate(
        span_text=claim,  # identical text -> cosine 1.0, the best possible match
        source_url="https://blocked.example/x",
        verdict=Verdict.BLOCKED,
    )
    ok_weaker = _ok("A treaty signed in Vienna in 1815 ended the conflict.")
    ok_unrelated = _ok("The weather in spring is mild and pleasant.")
    candidates = [blocked_best, ok_weaker, ok_unrelated]

    retriever = Retriever(FakeEmbedder(), candidates, ok_only=True)
    assert retriever.size == 2  # the BLOCKED candidate was never indexed
    top = retriever.retrieve(claim, k=5)
    # The non-OK span must NEVER appear, no matter how strong its lexical match.
    assert all(c.verdict is Verdict.OK for c in top)
    assert all(c.source_url != "https://blocked.example/x" for c in top)


def test_ok_only_excludes_unverified_like_blocked() -> None:
    # UNVERIFIED is excluded exactly like BLOCKED.
    claim = "Quarterly revenue rose sharply."
    unverified = Candidate(
        span_text=claim,
        source_url="https://unverified.example/y",
        verdict=Verdict.UNVERIFIED,
    )
    ok = _ok("Quarterly revenue rose sharply this year.")
    retriever = Retriever(FakeEmbedder(), [unverified, ok], ok_only=True)
    top = retriever.retrieve(claim, k=5)
    assert all(c.verdict is Verdict.OK for c in top)
    assert all("unverified.example" not in c.source_url for c in top)


def test_ok_only_false_indexes_everything_for_diagnostics() -> None:
    blocked = Candidate(
        span_text="blocked text here", source_url="https://b/x", verdict=Verdict.BLOCKED
    )
    retriever = Retriever(FakeEmbedder(), [blocked, _ok("ok text here")], ok_only=False)
    assert retriever.size == 2


# --- Retriever: safety edges --------------------------------------------------------------------


def test_k_larger_than_corpus_is_safe() -> None:
    candidates = [_ok("one sentence here"), _ok("another sentence there")]
    retriever = Retriever(FakeEmbedder(), candidates)
    top = retriever.retrieve("one sentence", k=50)
    assert len(top) == 2  # returns all, ranked, never raises


def test_empty_corpus_returns_empty() -> None:
    retriever = Retriever(FakeEmbedder(), [])
    assert retriever.size == 0
    assert retriever.retrieve("any claim", k=5) == []


def test_all_ok_filtered_out_yields_empty_index() -> None:
    only_blocked = Candidate(
        span_text="blocked only", source_url="https://b/x", verdict=Verdict.BLOCKED
    )
    retriever = Retriever(FakeEmbedder(), [only_blocked], ok_only=True)
    assert retriever.size == 0
    assert retriever.retrieve("blocked only", k=5) == []


def test_zero_vector_claim_does_not_crash() -> None:
    # A claim with no word tokens embeds to a zero vector; cosine must treat it as similarity 0 to
    # everything, returning candidates (ranked, tie-broken by index) without a division error.
    candidates = [_ok("first candidate span"), _ok("second candidate span")]
    retriever = Retriever(FakeEmbedder(), candidates)
    top = retriever.retrieve("!!!", k=2)
    assert len(top) == 2


def test_k_zero_returns_empty() -> None:
    retriever = Retriever(FakeEmbedder(), [_ok("a span")])
    assert retriever.retrieve("a span", k=0) == []


# --- Oracle-retrieval ablation ------------------------------------------------------------------


def test_oracle_retrieval_always_includes_gold_span() -> None:
    # The corpus does NOT contain the gold span; a real pass would miss it. The oracle ablation
    # guarantees the gold span is present so the verifier-only ceiling can be measured.
    gold = "The melting point of gold is 1064 degrees Celsius."
    candidates = [
        _ok("Iron rusts when exposed to oxygen and water."),
        _ok("Copper is an excellent conductor of electricity."),
    ]
    base = Retriever(FakeEmbedder(), candidates)
    claim = "Gold melts at 1064 degrees Celsius."
    assert not any(c.span_text == gold for c in base.retrieve(claim, k=5))

    oracle = OracleRetrieval(base, gold_span=gold)
    top = oracle.retrieve(claim, k=3)
    assert any(c.span_text == gold for c in top)
    assert len(top) == 3  # length held at k, weakest real tail dropped


def test_oracle_retrieval_does_not_duplicate_when_already_found() -> None:
    gold = "The capital of France is Paris."
    candidates = [_ok(gold), _ok("France is in western Europe.")]
    base = Retriever(FakeEmbedder(), candidates)
    oracle = OracleRetrieval(base, gold_span=gold)
    top = oracle.retrieve("What is the capital of France? Paris.", k=5)
    gold_count = sum(1 for c in top if c.span_text == gold)
    assert gold_count == 1  # not duplicated; the real pass already had it


# --- Lazy adapter -------------------------------------------------------------------------------


def test_sentence_transformer_adapter_lazy_import_raises_actionable_error() -> None:
    # When the optional 'sentence-transformers' dep is NOT installed, constructing the adapter must
    # raise a clear ImportError, never crash at module import. Skipped when it IS installed (binder
    # extra present), since the ImportError path then does not fire.
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is not None:
        pytest.skip("sentence-transformers present (binder extra); ImportError path not exercised")
    with pytest.raises(ImportError) as excinfo:
        SentenceTransformerEmbedder()
    msg = str(excinfo.value)
    assert "binder" in msg
    assert "uv sync" in msg


def test_real_embedder_retrieval_ranks_relevant_first() -> None:
    # Opt-in real-model integration: downloads Qwen3-Embedding-0.6B and embeds. Skipped by default so
    # check.sh / CI stay fast and model-free. Run with:
    #   CITEPROOF_RUN_MODEL_TESTS=1 .venv/bin/python -m pytest tests/binder/test_retrieve.py -k real_embedder
    import os

    if not os.environ.get("CITEPROOF_RUN_MODEL_TESTS"):
        pytest.skip("set CITEPROOF_RUN_MODEL_TESTS=1 to run the real embedder integration test")
    pytest.importorskip("sentence_transformers")
    emb = SentenceTransformerEmbedder()
    relevant = Candidate("The Eiffel Tower is a landmark located in Paris, France.", "https://a", Verdict.OK)
    unrelated = Candidate("Bananas are a good source of potassium.", "https://b", Verdict.OK)
    top = Retriever(emb, [unrelated, relevant]).retrieve("The Eiffel Tower is in Paris.", k=1)
    assert len(top) == 1
    assert top[0].span_text == relevant.span_text  # real embeddings rank the relevant span first


def test_retrieve_module_has_no_top_level_heavy_import() -> None:
    # Importing the module must not pull in sentence-transformers. The lazy contract is load-bearing.
    import inspect

    import citeproof.binder.retrieve as ret

    src = inspect.getsource(ret)
    for line in src.splitlines():
        if line.startswith("import sentence_transformers") or line.startswith(
            "from sentence_transformers"
        ):
            raise AssertionError("sentence_transformers must not be imported at module top level")
