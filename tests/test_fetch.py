"""SSRF guard on the fetch layer: a planner-generated URL must never reach an internal address."""

import socket

import pytest

from citeproof.fetch import UnsafeURLError, assert_safe_url, fetch


def _addrinfo(*ips: str):
    """Build a socket.getaddrinfo-shaped result for the given literal IPs."""
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0)) for ip in ips]


@pytest.mark.parametrize(
    "url",
    ["ftp://example.com/x", "file:///etc/passwd", "gopher://example.com/", "data:text/html,hi", "javascript:alert(1)"],
)
def test_non_http_schemes_blocked(url: str) -> None:
    with pytest.raises(UnsafeURLError):
        assert_safe_url(url)


def test_missing_host_blocked() -> None:
    with pytest.raises(UnsafeURLError):
        assert_safe_url("http:///just-a-path")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "https://127.0.0.1:8080/admin",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",  # ipv4-mapped loopback must not smuggle past
    ],
)
def test_internal_literal_ips_blocked(url: str) -> None:
    with pytest.raises(UnsafeURLError):
        assert_safe_url(url)


@pytest.mark.parametrize("url", ["https://8.8.8.8/", "http://1.1.1.1/path"])
def test_public_literal_ips_allowed(url: str) -> None:
    assert_safe_url(url)  # must not raise


def test_localhost_blocked() -> None:
    # localhost always resolves locally to a loopback address.
    with pytest.raises(UnsafeURLError):
        assert_safe_url("http://localhost/")


def test_resolution_to_private_ip_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo("10.1.2.3"))
    with pytest.raises(UnsafeURLError):
        assert_safe_url("http://internal.evil.example/")


def test_resolution_to_public_ip_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34"))
    assert_safe_url("http://example.com/")  # must not raise


def test_resolution_mixed_public_and_internal_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    # A host pointing at BOTH a public and an internal address must be rejected (the internal one wins).
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34", "127.0.0.1"))
    with pytest.raises(UnsafeURLError):
        assert_safe_url("http://rebind.evil.example/")


def test_unresolvable_host_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*a: object, **k: object):
        raise socket.gaierror("name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", _boom)
    with pytest.raises(UnsafeURLError):
        assert_safe_url("http://nope.invalid/")


def test_fetch_rejects_unsafe_url_before_any_network(monkeypatch: pytest.MonkeyPatch) -> None:
    import citeproof.fetch as fetch_module

    calls = {"n": 0}
    monkeypatch.setattr(fetch_module.veriscrape, "get", lambda *a, **k: calls.__setitem__("n", calls["n"] + 1))
    with pytest.raises(UnsafeURLError):
        fetch("http://169.254.169.254/latest/meta-data/")
    assert calls["n"] == 0  # never reached the network


# --- redirect-SSRF: the guard must re-validate every hop, not just the initial URL ---

def test_redirect_to_internal_target_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    # A public page that 302s to an internal address must NOT be followed (the confirmed finding).
    from veriscrape import FetchRecord

    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34"))  # public initial
    fetched: list[str] = []

    def fake_get(url: str, **kw: object) -> FetchRecord:
        fetched.append(url)
        return FetchRecord(url=url, status=302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

    monkeypatch.setattr("citeproof.fetch.veriscrape.get", fake_get)
    with pytest.raises(UnsafeURLError):
        fetch("https://evil.test/")
    assert fetched == ["https://evil.test/"]  # the internal redirect target was NEVER requested


def test_redirect_to_public_target_is_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    from veriscrape import FetchRecord, Verdict

    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34"))

    def fake_get(url: str, **kw: object) -> FetchRecord:
        if url == "https://a.test/":
            return FetchRecord(url=url, status=301, headers={"location": "https://b.test/"})
        return FetchRecord(url=url, status=200, verdict=Verdict.OK, text="final-body")

    monkeypatch.setattr("citeproof.fetch.veriscrape.get", fake_get)
    rec = fetch("https://a.test/")
    assert rec.text == "final-body"
    assert rec.url == "https://b.test/"  # the final hop's record is returned


def test_too_many_redirects_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from veriscrape import FetchRecord

    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: _addrinfo("93.184.216.34"))
    n = {"i": 0}

    def fake_get(url: str, **kw: object) -> FetchRecord:
        n["i"] += 1
        return FetchRecord(url=url, status=302, headers={"location": f"https://hop{n['i']}.test/"})

    monkeypatch.setattr("citeproof.fetch.veriscrape.get", fake_get)
    with pytest.raises(UnsafeURLError, match="too many redirects"):
        fetch("https://start.test/")
