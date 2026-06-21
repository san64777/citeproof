"""Search providers: where candidate URLs come from. Pluggable, local-first, keyless by default.

The search step only PROPOSES urls - every proposal still goes through the SSRF guard, the
veriscrape verdict, and the binder before anything is cited, so a bad search result costs recall,
never trust. Providers:

  - SearXNGSearch: the fully-local default WHEN a SearXNG instance is configured (SEARXNG_URL env or
    explicit base_url). Keeps the whole pipeline offline-capable.
  - WikipediaSearch: a keyless, stable, JSON-API fallback (api.php opensearch) so the v0 demo works
    with zero infrastructure. Encyclopedic-only by nature; fine for benchmark questions.

The UI also accepts pasted URLs directly (bring-your-own-sources), which bypasses search entirely -
that is how the "excluded the block page" demo mixes a known wall into the source list.
"""

from __future__ import annotations

import json
import os
import re
from typing import Protocol
from urllib.parse import quote, urlparse

from curl_cffi import requests as curl_requests
from pydantic import BaseModel
from tld import get_fld

from citeproof.lexical import content_words, idf_overlap_scores


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str = ""


class SearchProvider(Protocol):
    def search(self, query: str, k: int = 6) -> list[SearchResult]: ...


# A conversational question ("What is the birth date of PM Narendra Modi?") ranks search engines
# toward tangential pages (articles ABOUT his birth/premiership), not the authoritative biography -
# verified live on both Wikipedia and the web. The fix: also search the bare ENTITY ("Narendra Modi"),
# which returns the biography first on every engine tested. So we derive query variants from the
# question (entity phrases first, the raw question as a breadth fallback) and merge their results.
_Q_WORDS = frozenset(
    "what when where who whom whose why how which is are was were do does did can could will would "
    "tell give list explain name".split()
)
_HONORIFICS = frozenset("pm mr mrs ms dr prof sir prime minister president king queen saint st".split())
_CAP_RUN = re.compile(r"[A-Za-z][A-Za-z'’.-]*|\d+")


def _entity_phrases(question: str) -> list[str]:
    """Maximal runs of Capitalized words (proper-noun phrases), with a leading question word or
    honorific dropped: 'What is ... PM Narendra Modi?' -> 'Narendra Modi'."""
    phrases: list[list[str]] = []
    current: list[str] = []
    for m in _CAP_RUN.finditer(question):
        w = m.group(0)
        if w[:1].isupper():
            current.append(w)
        elif current:
            phrases.append(current)
            current = []
    if current:
        phrases.append(current)
    out: list[str] = []
    for ph in phrases:
        toks = list(ph)
        while toks and toks[0].lower().strip(".") in (_Q_WORDS | _HONORIFICS):
            toks.pop(0)
        if toks:
            out.append(" ".join(toks))
    return out


def search_queries(question: str) -> list[str]:
    """Query variants for a question: the most specific entity phrase first (precise), then the raw
    question (breadth). Two queries at most, to keep search fast and avoid rate-limits. The entity is
    the LONGEST proper-noun phrase, so 'Who is the CEO of OpenAI?' searches 'OpenAI', not 'CEO'."""
    entities = _entity_phrases(question)
    out: list[str] = []
    if entities:
        out.append(max(entities, key=len))
    q = question.strip()
    if q and q.lower() not in {x.lower() for x in out}:
        out.append(q)
    return out


# Search engines return MANY hits from the same site (Wikipedia alone yields 5+ articles for an
# entity), which crowd out the authoritative non-Wikipedia sources (spacex.com, toureiffel.paris,
# un.org - all verified OK) that rank just below them. Capping results PER DOMAIN forces a diverse
# set so the answer is drawn from across the web, not five Wikipedia pages.
_MAX_PER_DOMAIN = 2
# Low-value mirrors/clones that only echo another source (no independent verification value).
_JUNK_DOMAINS = frozenset({"grokipedia.com"})
# Platforms that are not citable PROSE sources - JS apps, video, social feeds, forums, link
# aggregators. The veriscrape gate already excludes most of them (they render as shells/login walls),
# so a top-of-list social hit just wastes a fetch slot a real article could have used. We DEMOTE them
# (sort to the back), not drop them: if nothing else fills the slots they remain available (recall
# safety). Validated: adding this demotion lifted precision@6 from 0.71 to 0.76 on the labeled set.
_DEMOTE_DOMAINS = frozenset({
    "youtube.com", "m.youtube.com", "tiktok.com", "twitter.com", "x.com", "facebook.com",
    "instagram.com", "pinterest.com", "reddit.com", "quora.com", "linkedin.com",
})


def _domain_key(url: str) -> str:
    """The registrable domain (public-suffix aware) so en.wikipedia.org and de.wikipedia.org count as
    one site, AND bbc.co.uk does not collapse to 'co.uk' (which would merge every UK site into one
    diversity slot - the bug a naive last-two-labels split has). tld bundles a public-suffix snapshot,
    so this is offline. Falls back to the bare host on a parse miss (e.g. a bare-IP or unknown TLD)."""
    fld = get_fld(url, fail_silently=True)
    if fld:
        return fld
    return urlparse(url).netloc.lower().split(":")[0]


# Reciprocal Rank Fusion constant: contribution of a result at 0-based rank r in a list is
# 1/(_RRF_K + r + 1). k=60 is the empirically robust default; large k flattens the rank discount so a
# url found across MANY lists (provider/query consensus = the authoritative signal) beats one ranked
# slightly higher in a single list.
_RRF_K = 60


def _rrf_fuse(ranked_lists: list[list[SearchResult]]) -> list[SearchResult]:
    """Fuse several ranked result lists by Reciprocal Rank Fusion. Rank-based, so it sidesteps the
    fact that Wikipedia's and ddgs's relevance scores are on incomparable scales (and citeproof never
    sees raw scores). A url that ranks high across providers AND query variants rises - consensus is
    the authoritative signal. A url absent from a list simply contributes nothing (never penalized)."""
    score: dict[str, float] = {}
    first: dict[str, SearchResult] = {}
    for lst in ranked_lists:
        for rank, r in enumerate(lst):
            score[r.url] = score.get(r.url, 0.0) + 1.0 / (_RRF_K + rank + 1)
            first.setdefault(r.url, r)
    return sorted(first.values(), key=lambda r: score[r.url], reverse=True)


def _relevance_rerank(question: str, results: list[SearchResult]) -> list[SearchResult]:
    """Stable re-order by IDF-weighted overlap of the QUESTION's content words against each result's
    title + snippet - so the snippet (captured but otherwise unused) finally decides relevance, and a
    tangential title match ('Blue Lights (2023 TV series)' for 'northern lights') sinks below the page
    that is actually on topic. A pure re-order, never a filter: a result that shares nothing keeps its
    incoming (RRF) position via the stable sort, so worst case equals the fusion order."""
    qwords = content_words(question)
    if not qwords:
        return results
    scores = idf_overlap_scores(qwords, [content_words(f"{r.title} {r.snippet}") for r in results])
    order = sorted(range(len(results)), key=lambda i: scores[i], reverse=True)
    return [results[i] for i in order]


def run_search(provider: SearchProvider, question: str, k: int = 6) -> list[SearchResult]:
    """Rank the candidate urls against the QUESTION before the fetch cut, so the few sources that pay
    the expensive fetch+verify+bind cost are the most relevant, not whichever the providers happened
    to return first. Pipeline: search each query variant -> RRF-fuse the variant lists (consensus =
    authoritative) -> relevance re-rank by the snippet against the question -> diversify by domain.
    Ranking is microseconds on snippets already in hand; only the kept top-k are ever fetched.
    """
    variant_lists: list[list[SearchResult]] = []
    for q in search_queries(question):
        try:
            variant_lists.append(provider.search(q, k=max(k, 10)))
        except Exception:
            variant_lists.append([])  # one variant failing must not sink the search
    ranked = _relevance_rerank(question, _rrf_fuse(variant_lists))
    # Demote non-prose platforms to the back (stable, so relevance order holds within each group), so
    # a real article wins a fetch slot over a video/social hit that the gate would exclude anyway.
    ranked = sorted(ranked, key=lambda r: _domain_key(r.url) in _DEMOTE_DOMAINS)

    out: list[SearchResult] = []
    per_domain: dict[str, int] = {}
    for r in ranked:
        domain = _domain_key(r.url)
        if domain in _JUNK_DOMAINS or per_domain.get(domain, 0) >= _MAX_PER_DOMAIN:
            continue
        per_domain[domain] = per_domain.get(domain, 0) + 1
        out.append(r)
        if len(out) >= k:
            break
    return out


def _get_json(url: str, timeout: float = 15.0) -> object:
    resp = curl_requests.get(url, impersonate="chrome", timeout=timeout)
    resp.raise_for_status()
    return json.loads(resp.text)


class WikipediaSearch:
    """Keyless JSON search via the MediaWiki FULL-TEXT search API - a stable public API, not scraping.

    Uses `action=query&list=search` (full-text, relevance-ranked), NOT `opensearch` (which only
    autocompletes title PREFIXES and so returns nothing for a natural-language question like
    "What is the Antikythera mechanism?"). Encyclopedic-only by nature; fine for the v0 demo when no
    SearXNG instance is configured.
    """

    def __init__(self, lang: str = "en") -> None:
        self._lang = lang
        self._base = f"https://{lang}.wikipedia.org/w/api.php"

    def search(self, query: str, k: int = 6) -> list[SearchResult]:
        url = (
            f"{self._base}?action=query&list=search&srsearch={quote(query)}"
            f"&srlimit={int(k)}&srnamespace=0&format=json"
        )
        data = _get_json(url)
        hits = data.get("query", {}).get("search", []) if isinstance(data, dict) else []
        out: list[SearchResult] = []
        for hit in hits:
            title = hit.get("title")
            if not isinstance(title, str) or not title:
                continue
            # build the canonical article URL from the title (full-text search omits the url)
            page = quote(title.replace(" ", "_"))
            snippet = re.sub(r"<[^>]+>", "", hit.get("snippet", "") or "")  # strip search-match markup
            out.append(SearchResult(title=title, url=f"https://{self._lang}.wikipedia.org/wiki/{page}",
                                    snippet=snippet))
        return out


class SearXNGSearch:
    """A self-hosted SearXNG instance (fully local posture). Needs `format=json` enabled there."""

    def __init__(self, base_url: str) -> None:
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"invalid SearXNG base url: {base_url!r}")
        self._base = base_url.rstrip("/")

    def search(self, query: str, k: int = 6) -> list[SearchResult]:
        data = _get_json(f"{self._base}/search?q={quote(query)}&format=json")
        results = data.get("results", []) if isinstance(data, dict) else []
        out: list[SearchResult] = []
        for r in results[: int(k)]:
            u = r.get("url", "")
            if isinstance(u, str) and u.startswith("http"):
                out.append(SearchResult(title=r.get("title", u), url=u, snippet=r.get("content", "")))
        return out


class DDGSearch:
    """Keyless WHOLE-WEB search via the `ddgs` metasearch library (MIT). No API key, no service to
    run - the zero-setup whole-web default. ddgs is a SEPARATE library; the verification gate still
    excludes whatever junk a web search surfaces, so a noisy result list costs recall, never trust.
    """

    def __init__(self, region: str = "wt-wt") -> None:
        self._region = region

    def search(self, query: str, k: int = 6) -> list[SearchResult]:
        from ddgs import DDGS  # lazy: only the web-search path needs it

        out: list[SearchResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, region=self._region, max_results=int(k)):
                url = r.get("href") or r.get("url") or ""
                if isinstance(url, str) and url.startswith("http"):
                    out.append(SearchResult(title=r.get("title", url), url=url, snippet=r.get("body", "")))
        return out


class CombinedSearch:
    """Query several providers and ROUND-ROBIN merge their results, tolerating any one failing.

    Whole-web search (ddgs) is broad but noisy - for a factual question it often surfaces clones and
    social media over the authoritative page, and it can rate-limit. Wikipedia is narrow but reliably
    returns the canonical article. Interleaving them gives both: Wikipedia anchors the answer, the
    web adds breadth, and if the web search errors the Wikipedia results still come through. The
    verification gate then excludes whatever junk either surfaced.
    """

    def __init__(self, providers: list[SearchProvider]) -> None:
        self._providers = providers

    def search(self, query: str, k: int = 6) -> list[SearchResult]:
        per: list[list[SearchResult]] = []
        for p in self._providers:
            try:
                per.append(p.search(query, k=k))
            except Exception:
                per.append([])  # one provider down must not sink the search
        # RRF-fuse rather than round-robin: a page returned by BOTH Wikipedia and the web search rises
        # (consensus = authoritative), instead of being deduped to a single arbitrary position.
        return _rrf_fuse(per)[:k]


def default_provider() -> SearchProvider:
    """Search backend:
      - SEARXNG_URL set -> SearXNG (self-hosted, fully local, whole web).
      - CITEPROOF_SEARCH=wikipedia -> encyclopedic-only (most reliable, zero noise).
      - default -> Wikipedia (authoritative) + keyless whole-web DuckDuckGo, merged. Out of the box
        this covers the whole web with no setup, while Wikipedia keeps factual lookups reliable.
    """
    mode = os.environ.get("CITEPROOF_SEARCH", "").lower()
    if mode == "wikipedia":
        return WikipediaSearch()
    searx = os.environ.get("SEARXNG_URL")
    if searx:
        return SearXNGSearch(searx)
    return CombinedSearch([WikipediaSearch(), DDGSearch()])
