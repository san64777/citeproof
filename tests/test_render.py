"""EMPTY_SHELL escalation policy: render is for JS skeletons ONLY - never a workaround for a no."""

import pytest

from veriscrape import FetchRecord, Verdict

from citeproof.fetch import UnsafeURLError
from citeproof.render import render_and_reclassify, should_escalate


def _rec(verdict: Verdict) -> FetchRecord:
    return FetchRecord(url="https://example.test/", verdict=verdict)


def test_empty_shell_escalates() -> None:
    assert should_escalate(_rec(Verdict.EMPTY_SHELL)) is True


@pytest.mark.parametrize(
    "verdict",
    [v for v in Verdict if v is not Verdict.EMPTY_SHELL],
)
def test_every_other_verdict_is_final(verdict: Verdict) -> None:
    # The bright line (rendering, not circumvention): BLOCKED/CHALLENGE/HONEYPOT/LOGIN_WALL/SOFT_404 said
    # no - citeproof excludes them. OK needs no render. UNVERIFIED is an abstention, not a skeleton.
    assert should_escalate(_rec(verdict)) is False


def test_render_is_ssrf_guarded_before_any_browser() -> None:
    with pytest.raises(UnsafeURLError):
        render_and_reclassify("http://169.254.169.254/latest/meta-data/")


def _chromium_usable() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    try:
        with sync_playwright() as pw:
            try:
                b = pw.chromium.launch(headless=True)
            except Exception:
                b = pw.chromium.launch(headless=True, channel="chrome")
            b.close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _chromium_usable(), reason="playwright/chromium not available")
def test_live_render_flips_a_js_shell_to_content(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A JS app skeleton: empty mount + script that injects a real article AFTER load. The raw fetch
    # verdicts EMPTY_SHELL; the rendered DOM must re-classify as real content (OK).
    import http.server
    import socketserver
    import threading

    import citeproof.render as render_mod

    shell = (
        "<html><head><title>Quarterly results</title></head><body>"
        '<div id="root"></div>'
        "<script>document.getElementById('root').innerHTML = "
        "'<main><article><h1>Quarterly results</h1>' + "
        "'<p>Revenue for the quarter rose to 4.2 billion dollars, the company reported on Tuesday. "
        "Operating margin improved to 21 percent, and the firm raised its full-year outlook. "
        "Analysts had expected 3.9 billion dollars in revenue for the period. "
        "Shares gained about 6 percent in extended trading after the announcement was published. "
        "The company also announced a 2 billion dollar buyback program for next year.</p>' + "
        "'<p>Executives attributed the growth to stronger demand in the data center segment, where "
        "sales nearly doubled compared with the same period a year earlier. The consumer division "
        "remained flat, while the services arm grew at a steady 11 percent. Hiring will continue in "
        "the engineering organization, the chief executive said on the earnings call, though the "
        "company plans to slow expansion in administrative functions through the rest of the year. "
        "The board also approved a modest increase to the quarterly dividend.</p>' + "
        "'</article></main>';</script>"
        + "<script>// filler so the shell clears the min-size floor: "
        + "x".ljust(3000, "x")
        + "</script></body></html>"
    )
    (tmp_path / "shell.html").write_text(shell, encoding="utf-8")

    handler = type(
        "H", (http.server.SimpleHTTPRequestHandler,), {"directory": str(tmp_path), "log_message": lambda *a: None}
    )
    with socketserver.TCPServer(("127.0.0.1", 0), lambda *a, **k: handler(*a, directory=str(tmp_path), **k)) as srv:
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            url = f"http://127.0.0.1:{port}/shell.html"
            # the production SSRF guard rightly blocks loopback; tests disable it explicitly
            monkeypatch.setattr(render_mod, "assert_safe_url", lambda u: None)

            import veriscrape

            raw = veriscrape.get(url, timeout=15)
            assert raw.verdict is Verdict.EMPTY_SHELL, f"precondition: raw fetch should be EMPTY_SHELL, got {raw.verdict}"
            assert should_escalate(raw)

            rendered = render_and_reclassify(url, timeout=20.0)
            assert rendered.verdict is Verdict.OK, f"rendered DOM should classify OK, got {rendered.verdict}"
            assert rendered.tactic == "rendered"
            assert "4.2 billion" in (rendered.text or "")
        finally:
            srv.shutdown()
