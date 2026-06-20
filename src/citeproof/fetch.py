"""SSRF-guarded fetch: the M1 spine's entry point.

Every URL citeproof fetches is planner- or search-generated, so the fetch layer is a classic SSRF
surface: a crafted link (``http://169.254.169.254/...``, ``http://localhost/admin``,
``http://10.0.0.1``) could make the agent hit cloud metadata or an internal service. The guard refuses
any URL whose host is, or resolves to, an internal / loopback / link-local / reserved address, and any
non-http(s) scheme.

A validated initial URL is not enough: an attacker-controlled public page can answer with a 30x
redirect to an internal address, and curl_cffi follows redirects by default (the CVE-2026-33752 fix
restricts redirect PROTOCOLS, not destination IPs). So fetch() disables auto-redirect and follows
redirects ITSELF, re-running assert_safe_url on every hop, with a hop cap and loop detection.

The actual fetch + trust verdict is veriscrape.get (curl_cffi + the hardened classifier). This module
adds the pre-flight + per-hop safety checks and returns veriscrape's FetchRecord for the final hop.

KNOWN LIMIT (follow-up): assert_safe_url resolves the host and checks the result, but veriscrape re-
resolves at fetch time, so a DNS-rebinding attacker could pass the check then serve an internal IP. The
robust fix is to pin the validated IP and fetch by-IP with a Host header; tracked for the hardening pass.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import veriscrape
from veriscrape import FetchRecord

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_MAX_REDIRECTS = 5


class UnsafeURLError(ValueError):
    """The URL must not be fetched: a non-http(s) scheme, no host, or an internal/reserved address."""


def _ip_is_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True for any address an outbound research fetch must never reach: RFC1918 private, loopback,
    link-local (169.254/16 incl the 169.254.169.254 cloud-metadata endpoint, and fe80::/10), reserved,
    multicast, or unspecified (0.0.0.0 / ::). IPv4-mapped IPv6 is unwrapped first so ::ffff:127.0.0.1
    cannot smuggle a loopback address past the check.
    """
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _resolve(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Resolve a hostname to every address it maps to (so a host pointing at a mix of public and
    internal addresses is still rejected). Raises socket.gaierror on failure.
    """
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return [ipaddress.ip_address(info[4][0]) for info in infos]


def assert_safe_url(url: str) -> None:
    """Raise UnsafeURLError unless `url` is an http(s) URL whose host neither is nor resolves to an
    internal / reserved address. Call this before any fetch of an untrusted (planner-generated) URL.
    """
    parsed = urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeURLError(f"scheme {parsed.scheme!r} is not allowed (http/https only)")
    host = parsed.hostname
    if not host:
        raise UnsafeURLError(f"no host in URL {url!r}")

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _ip_is_blocked(literal):
            raise UnsafeURLError(f"host {host} is an internal/reserved address")
        return

    try:
        addresses = _resolve(host)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"could not resolve {host!r}: {exc}") from exc
    if not addresses:
        raise UnsafeURLError(f"{host!r} resolved to no addresses")
    for ip in addresses:
        if _ip_is_blocked(ip):
            raise UnsafeURLError(f"{host!r} resolves to internal/reserved address {ip}")


def _location_header(headers: dict[str, str] | None) -> str | None:
    """The Location header value (case-insensitive), or None."""
    if not headers:
        return None
    for key, value in headers.items():
        if key.lower() == "location":
            return value
    return None


def fetch(url: str, *, timeout: float = 20.0, impersonate: str = "chrome") -> FetchRecord:
    """SSRF-guard `url` AND every redirect hop, then return veriscrape's trust verdict + body.

    Redirects are followed manually (auto-redirect disabled) so a 30x to an internal address is caught:
    each hop's target is re-validated with assert_safe_url before it is fetched. The caller cites only
    when `record.ok`. Raises UnsafeURLError for an unsafe initial URL, an unsafe redirect target, a
    redirect loop, or too many hops (before the offending request is ever made).
    """
    assert_safe_url(url)
    seen = {url}
    for _ in range(_MAX_REDIRECTS + 1):
        record = veriscrape.get(url, timeout=timeout, impersonate=impersonate, allow_redirects=False)
        status = record.status or 0
        location = _location_header(record.headers) if 300 <= status < 400 else None
        if not location:
            return record  # final (non-redirect) response, with veriscrape's verdict
        target = urljoin(url, location)
        assert_safe_url(target)  # re-validate EVERY hop - this is what closes the redirect-SSRF gap
        if target in seen:
            raise UnsafeURLError(f"redirect loop following {url!r}")
        seen.add(target)
        url = target
    raise UnsafeURLError(f"too many redirects (> {_MAX_REDIRECTS})")
