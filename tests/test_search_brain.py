"""Search providers + brain: parsing, keyless fallback, reasoning-strip, no-source honesty."""

import pytest

import citeproof.search as search_mod
from citeproof.brain import FakeBrain, SourceContext, _strip_reasoning
from citeproof.search import (
    CombinedSearch,
    SearXNGSearch,
    SearchResult,
    WikipediaSearch,
    default_provider,
    run_search,
    search_queries,
)


def test_entity_extraction_surfaces_the_proper_noun() -> None:
    # The Modi bug: a conversational question must also search the bare entity, which is what
    # returns the authoritative page. The most-specific (longest) entity wins.
    assert search_queries("What is the birth date of PM Narendra Modi?")[0] == "Narendra Modi"
    assert search_queries("Who is the CEO of OpenAI?")[0] == "OpenAI"  # not "CEO"
    # a question with no proper noun falls back to the question itself (no spurious entity)
    assert search_queries("What are coral reefs built from?") == ["What are coral reefs built from?"]


class _StubProvider:
    def __init__(self, by_query: dict[str, list[str]]) -> None:
        self._by_query = by_query
        self.queries: list[str] = []

    def search(self, query: str, k: int = 6) -> list[SearchResult]:
        self.queries.append(query)
        return [SearchResult(title=u, url=u) for u in self._by_query.get(query, [])][:k]


def test_run_search_merges_entity_then_question_results() -> None:
    prov = _StubProvider({
        "Narendra Modi": ["https://en.wikipedia.org/wiki/Narendra_Modi"],
        "When was Narendra Modi born?": ["https://en.wikipedia.org/wiki/Premiership_of_Narendra_Modi"],
    })
    urls = [r.url for r in run_search(prov, "When was Narendra Modi born?", k=3)]
    assert urls[0].endswith("/Narendra_Modi")  # the entity query's authoritative hit comes first
    assert "Narendra Modi" in prov.queries  # the entity query was actually issued


def test_run_search_diversifies_by_domain_and_drops_clones() -> None:
    # Wikipedia returns many same-site hits that crowd out good web sources; the merge must cap per
    # domain and drop known clones, so the result spans multiple sites.
    prov = _StubProvider({
        "SpaceX": [
            "https://en.wikipedia.org/wiki/SpaceX",
            "https://en.wikipedia.org/wiki/SpaceX_Starship",
            "https://en.wikipedia.org/wiki/Elon_Musk",  # 3rd wikipedia.org -> dropped by domain cap
            "https://grokipedia.com/page/SpaceX",        # clone -> dropped
            "https://www.spacex.com/",                   # the authoritative web source kept
        ],
    })
    urls = [r.url for r in run_search(prov, "SpaceX", k=5)]
    assert urls.count("https://en.wikipedia.org/wiki/Elon_Musk") == 0  # 3rd same-domain dropped
    assert not any("grokipedia" in u for u in urls)  # clone dropped
    assert "https://www.spacex.com/" in urls  # the non-Wikipedia source surfaced


def test_combined_search_round_robins_and_tolerates_a_failing_provider() -> None:
    good = _StubProvider({"q": ["https://a.test", "https://b.test"]})

    class _Broken:
        def search(self, query: str, k: int = 6) -> list[SearchResult]:
            raise RuntimeError("rate limited")

    merged = CombinedSearch([good, _Broken()]).search("q", k=3)
    assert [r.url for r in merged] == ["https://a.test", "https://b.test"]  # broken one ignored


def test_combined_search_interleaves_providers() -> None:
    p1 = _StubProvider({"q": ["https://wiki/1", "https://wiki/2"]})
    p2 = _StubProvider({"q": ["https://web/1", "https://web/2"]})
    merged = [r.url for r in CombinedSearch([p1, p2]).search("q", k=4)]
    assert merged == ["https://wiki/1", "https://web/1", "https://wiki/2", "https://web/2"]


def test_default_provider_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    monkeypatch.setenv("CITEPROOF_SEARCH", "wikipedia")
    assert isinstance(default_provider(), WikipediaSearch)
    monkeypatch.delenv("CITEPROOF_SEARCH", raising=False)
    assert isinstance(default_provider(), CombinedSearch)  # default = Wikipedia + web


def test_wikipedia_fulltext_search_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    # full-text API (action=query&list=search): handles natural-language questions, unlike the
    # title-prefix opensearch API. The url is built from the title (the API omits it).
    payload = {"query": {"search": [
        {"title": "Antikythera mechanism", "snippet": "An <span class=\"m\">ancient</span> device"},
        {"title": "Solar power", "snippet": "energy from the sun"},
    ]}}
    monkeypatch.setattr(search_mod, "_get_json", lambda url, timeout=15.0: payload)
    results = WikipediaSearch().search("what is the antikythera mechanism?", k=2)
    assert results[0].title == "Antikythera mechanism"
    assert results[0].url == "https://en.wikipedia.org/wiki/Antikythera_mechanism"
    assert "<span" not in results[0].snippet  # search-match markup stripped


def test_wikipedia_ignores_malformed_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(search_mod, "_get_json", lambda url, timeout=15.0: {"unexpected": "shape"})
    assert WikipediaSearch().search("x") == []


def test_searxng_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"results": [{"url": "https://a.test", "title": "A", "content": "snip"},
                           {"url": "ftp://skip", "title": "bad"}]}
    monkeypatch.setattr(search_mod, "_get_json", lambda url, timeout=15.0: payload)
    results = SearXNGSearch("http://localhost:8888").search("q", k=5)
    assert len(results) == 1  # the ftp result is dropped
    assert results[0].url == "https://a.test"


def test_searxng_rejects_bad_base_url() -> None:
    with pytest.raises(ValueError):
        SearXNGSearch("not-a-url")


def test_default_provider_is_searxng_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEARXNG_URL", "http://localhost:8888")
    assert isinstance(default_provider(), SearXNGSearch)


def test_default_provider_is_combined_web_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    monkeypatch.delenv("CITEPROOF_SEARCH", raising=False)
    assert isinstance(default_provider(), CombinedSearch)  # Wikipedia + keyless whole-web, merged


def test_strip_reasoning_removes_think_block() -> None:
    raw = "<think>let me reason about this</think>The answer is 42."
    assert _strip_reasoning(raw).strip() == "The answer is 42."


def test_strip_reasoning_handles_unclosed_think() -> None:
    # A truncated generation can leave an UNCLOSED <think>; the raw reasoning must not leak into the
    # draft (where it would be split into claims and verified as if it were prose).
    raw = "The answer is 42.<think>now let me second-guess myself and the output got cut off"
    assert _strip_reasoning(raw).strip() == "The answer is 42."


def test_fake_brain_records_inputs() -> None:
    brain = FakeBrain("draft text")
    out = brain.draft("q", [SourceContext(url="u", title="t", text="body")])
    assert out == "draft text"
    assert brain.last_question == "q"
    assert brain.last_sources[0].url == "u"


def test_search_result_is_typed() -> None:
    r = SearchResult(title="t", url="https://x.test")
    assert r.snippet == ""
