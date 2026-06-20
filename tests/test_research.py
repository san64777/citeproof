"""The research orchestrator: ledger honesty, the exclude-the-block-page demo, no mis-highlight."""

from pathlib import Path

import pytest
from veriscrape import FetchRecord, Verdict

import citeproof.spine as spine_mod
from citeproof.binder.binder import EntailmentBinder
from citeproof.binder.entailment import FakeEntailment
from citeproof.brain import FakeBrain, SourceContext
from citeproof.research import MemoryReceiptStore, focus_source, run_research
from citeproof.search import SearchResult

_ARTICLE = """<html><head><title>Solar</title></head><body><main><article><h1>Alpha</h1>
<p>The Alpha solar farm in Nevada produces 690 megawatts of power. It was completed in 2021
and covers about 3,000 acres. The plant supplies roughly 180,000 homes.</p></article></main></body></html>"""


class _Provider:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.last_query: str | None = None

    def search(self, query: str, k: int = 6) -> list[SearchResult]:
        self.last_query = query
        return self._results


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, table: dict[str, FetchRecord]) -> None:
    def fake(url: str, timeout: float = 20.0) -> FetchRecord:
        for key, rec in table.items():
            if key in url:
                return rec
        return FetchRecord(url=url, verdict=Verdict.UNVERIFIED, status=200, text="")
    monkeypatch.setattr(spine_mod, "fetch", fake)


def _binder() -> EntailmentBinder:
    return EntailmentBinder(FakeEntailment(), tau_mc=0.5)


def test_block_page_is_excluded_and_ok_page_is_cited(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # THE M2 demo: one OK source -> cited with a receipt; one login wall -> excluded with verdict.
    _patch_fetch(monkeypatch, {
        "ok.test": FetchRecord(url="https://ok.test/a", verdict=Verdict.OK, status=200, text=_ARTICLE),
        "wall.test": FetchRecord(url="https://wall.test/b", verdict=Verdict.LOGIN_WALL, status=200,
                                 text="<html><body>please log in</body></html>"),
    })
    provider = _Provider([SearchResult(title="a", url="https://ok.test/a"),
                          SearchResult(title="b", url="https://wall.test/b")])
    store = MemoryReceiptStore()
    rep = run_research(
        "How much power does the Alpha solar farm produce?",
        binder=_binder(), brain=FakeBrain("The Alpha solar farm in Nevada produces 690 megawatts of power."),
        provider=provider, store=store, out_dir=tmp_path,
    )
    assert rep.ledger.cited == 1
    assert rep.ledger.excluded == 1
    cited = [c for c in rep.claims if c.status == "cited"]
    assert len(cited) == 1
    # the receipt is real and resolvable from the store
    assert store.get(cited[0].receipt_id) is not None
    excluded = [s for s in rep.sources if s.status == "excluded"]
    assert excluded[0].verdict == "LOGIN_WALL"


def test_hallucinated_claim_stays_unverified_never_falsely_cited(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The brain invents a fact absent from the source - it must land in the ledger as UNVERIFIED,
    # never as a citation. This is the product's core guarantee.
    _patch_fetch(monkeypatch, {"ok.test": FetchRecord(url="https://ok.test/a", verdict=Verdict.OK, status=200, text=_ARTICLE)})
    provider = _Provider([SearchResult(title="a", url="https://ok.test/a")])
    rep = run_research(
        "Tell me about the Alpha solar farm.",
        binder=_binder(),
        brain=FakeBrain("The Alpha solar farm was personally opened by the Emperor of Mars in 3024."),
        provider=provider, store=MemoryReceiptStore(), out_dir=tmp_path,
    )
    assert rep.ledger.cited == 0
    assert rep.ledger.unverified >= 1
    assert all(c.status == "unverified" for c in rep.claims)


def test_no_ok_sources_yields_empty_draft_and_honest_ledger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_fetch(monkeypatch, {
        "wall.test": FetchRecord(url="https://wall.test/b", verdict=Verdict.LOGIN_WALL, status=200, text="login"),
        "block.test": FetchRecord(url="https://block.test/c", verdict=Verdict.BLOCKED, status=403, text="blocked"),
    })
    provider = _Provider([SearchResult(title="b", url="https://wall.test/b"),
                          SearchResult(title="c", url="https://block.test/c")])
    brain = FakeBrain("should never be called")
    rep = run_research("anything", binder=_binder(), brain=brain, provider=provider,
                       store=MemoryReceiptStore(), out_dir=tmp_path)
    assert rep.draft == ""
    assert rep.ledger.cited == 0 and rep.ledger.excluded == 2
    assert brain.last_question is None  # the brain never saw an unverified source


def test_focus_source_surfaces_a_deep_answer_not_the_head() -> None:
    # The Q9 fix: a long document's answer is often DEEP past a head-truncation window. focus_source
    # must keep the QUESTION-RELEVANT passage, not the boilerplate head.
    head = "Title block. Authors. Table of contents. " + ("filler boilerplate line. " * 300)
    answer = "The 404 status code means Not Found and 200 means OK."
    tail = ("more unrelated appendix text. " * 300)
    doc = head + "\n" + answer + "\n" + tail
    focused = focus_source("What status code means Not Found and what does 200 mean?", doc, 1000)
    assert len(focused) <= 1000
    assert "404" in focused and "Not Found" in focused  # the deep answer survived
    assert "Table of contents" not in focused  # the irrelevant head was dropped


def test_focus_source_passes_short_text_through() -> None:
    short = "A short source under budget."
    assert focus_source("anything", short, 4000) == short


def test_brain_abstention_sentinel_is_never_cited(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # When the brain abstains with its EXACT sentinel, that sentence is a control signal, not a
    # claim - it must produce ZERO claims and never be verified/cited (caught by the live demo).
    from citeproof.brain import ABSTENTION_SENTINEL
    _patch_fetch(monkeypatch, {"ok.test": FetchRecord(url="https://ok.test/a", verdict=Verdict.OK, status=200, text=_ARTICLE)})
    provider = _Provider([SearchResult(title="a", url="https://ok.test/a")])
    rep = run_research(
        "What is the box office of Titanic?", binder=_binder(),
        brain=FakeBrain(ABSTENTION_SENTINEL + "."), provider=provider,
        store=MemoryReceiptStore(), out_dir=tmp_path,
    )
    assert rep.claims == []
    assert rep.ledger.cited == 0 and rep.ledger.unverified == 0
    assert ABSTENTION_SENTINEL in rep.draft  # the abstention is shown, just not verified


def test_explicit_urls_bypass_search(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_fetch(monkeypatch, {"ok.test": FetchRecord(url="https://ok.test/a", verdict=Verdict.OK, status=200, text=_ARTICLE)})
    provider = _Provider([SearchResult(title="should-not-be-used", url="https://other.test/z")])
    rep = run_research(
        "How much power?", binder=_binder(),
        brain=FakeBrain("The Alpha solar farm in Nevada produces 690 megawatts of power."),
        provider=provider, store=MemoryReceiptStore(), out_dir=tmp_path,
        urls=["https://ok.test/a"],
    )
    assert provider.last_query is None  # search was skipped
    assert any(s.url == "https://ok.test/a" for s in rep.sources)


def test_brain_only_sees_verified_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_fetch(monkeypatch, {
        "ok.test": FetchRecord(url="https://ok.test/a", verdict=Verdict.OK, status=200, text=_ARTICLE),
        "wall.test": FetchRecord(url="https://wall.test/b", verdict=Verdict.LOGIN_WALL, status=200, text="login"),
    })
    provider = _Provider([SearchResult(title="a", url="https://ok.test/a"),
                          SearchResult(title="b", url="https://wall.test/b")])
    brain = FakeBrain("The Alpha solar farm in Nevada produces 690 megawatts of power.")
    run_research("q", binder=_binder(), brain=brain, provider=provider,
                 store=MemoryReceiptStore(), out_dir=tmp_path)
    seen_urls = [s.url for s in brain.last_sources]
    assert seen_urls == ["https://ok.test/a"]
    assert all(isinstance(s, SourceContext) for s in brain.last_sources)
