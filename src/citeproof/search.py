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


def _domain_key(url: str) -> str:
    """A coarse registrable-domain key (last two labels) so en.wikipedia.org and de.wikipedia.org
    count as one site. Good enough for diversity capping; not a public-suffix-perfect parse."""
    host = urlparse(url).netloc.lower().split(":")[0]
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def run_search(provider: SearchProvider, question: str, k: int = 6) -> list[SearchResult]:
    """Search each query variant, MERGE (entity-query hits first), then DIVERSIFY by domain so the
    result set spans multiple sites instead of one site's first k pages."""
    collected: list[SearchResult] = []
    seen_urls: set[str] = set()
    for q in search_queries(question):
        for r in provider.search(q, k=max(k, 6)):
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                collected.append(r)
    out: list[SearchResult] = []
    per_domain: dict[str, int] = {}
    for r in collected:
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
        merged: list[SearchResult] = []
        seen: set[str] = set()
        for i in range(k):
            for results in per:
                if i < len(results) and results[i].url not in seen:
                    seen.add(results[i].url)
                    merged.append(results[i])
                    if len(merged) >= k:
                        return merged
        return merged


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
