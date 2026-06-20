"""EMPTY_SHELL escalation: render a JS app skeleton once in headless Chromium, re-classify the DOM.

Rendering, NOT circumvention - the bright line:
  - ONLY an EMPTY_SHELL verdict escalates. A JS app skeleton is a page that WANTS to show content
    but needs JavaScript to do it; rendering it is what a normal browser does.
  - BLOCKED / CHALLENGE / HONEYPOT / LOGIN_WALL / SOFT_404 verdicts are NEVER escalated: those pages
    said no, and citeproof's answer to "no" is to exclude the page, not to try harder.

The rendered DOM goes back through the SAME veriscrape classifier (classify on the rendered HTML),
so escalation can only ever change the verdict by revealing real content - the junk detectors run
again on whatever rendering produced. Playwright is an optional extra (`uv sync --extra render`),
lazy-imported so the default install and CI never need it.
"""

from __future__ import annotations

from veriscrape import FetchRecord, Verdict, classify

from citeproof.fetch import assert_safe_url

# The ONLY verdict that may escalate to rendering. Everything else is final.
_ESCALATABLE = frozenset({Verdict.EMPTY_SHELL})


class RenderUnavailableError(RuntimeError):
    """Playwright is not installed; install with `uv sync --extra render` + `playwright install chromium`."""


def should_escalate(record: FetchRecord) -> bool:
    """True only for an EMPTY_SHELL verdict: a JS skeleton worth one headless render. A blocked,
    challenged, gated, or junk page is final - escalating those would be circumvention, not rendering.
    """
    return record.verdict in _ESCALATABLE


def render_and_reclassify(url: str, *, timeout: float = 30.0) -> FetchRecord:
    """Render `url` once in headless Chromium, then run the rendered DOM back through veriscrape's
    classifier. Returns a new FetchRecord whose verdict reflects the RENDERED page; `tactic` is set
    to "rendered" so a receipt can disclose how the content was obtained.
    """
    assert_safe_url(url)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RenderUnavailableError(
            "playwright is not installed: uv sync --extra render && playwright install chromium"
        ) from exc

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(headless=True)
        except Exception:
            # The bundled headless shell can fail to start on some hosts (observed on WSL2: clean
            # launch then instant exit). The system Chrome channel is the reliable fallback there.
            browser = pw.chromium.launch(headless=True, channel="chrome")
        try:
            page = browser.new_page()
            response = page.goto(url, timeout=timeout * 1000, wait_until="networkidle")
            status = response.status if response is not None else None
            headers = dict(response.headers) if response is not None else {}
            html = page.content()
        finally:
            browser.close()

    verdict, cause, confidence, evidence = classify(status=status, headers=headers, body=html)
    return FetchRecord(
        url=url,
        status=status,
        verdict=verdict,
        cause=cause,
        tactic="rendered",
        confidence=confidence,
        evidence=evidence,
        headers=headers,
        text=html,
        elapsed_ms=0,
    )
