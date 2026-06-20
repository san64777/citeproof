"""SSRF-guarded fetch: the M1 spine's entry point.

Every URL citeproof fetches is planner- or search-generated, so the fetch layer is a classic SSRF
surface: a crafted link (``http://169.254.169.254/...``, ``http://localhost/admin``,
``http://10.0.0.1``) could make the agent hit cloud metadata or an internal service. curl_cffi >= 0.15.0
fixes the redirect-SSRF CVE (CVE-2026-33752); this guard blocks the OTHER half - the INITIAL target -
by refusing any URL whose host is, or resolves to, an internal / loopback / link-local / reserved
address, and any non-http(s) scheme.

The actual fetch + trust verdict is veriscrape.get (curl_cffi + the hardened classifier). This module
only adds the pre-flight safety check and returns veriscrape's FetchRecord unchanged.

KNOWN LIMIT (follow-up): assert_safe_url resolves the host and checks the result, but veriscrape re-
resolves at fetch time, so a DNS-rebinding attacker could pass the check then serve an internal IP. The
robust fix is to pin the validated IP and fetch by-IP with a Host header; tracked for the hardening pass.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import veriscrape
from veriscrape import FetchRecord

_ALLOWED_SCHEMES = frozenset({"http", "https"})


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


def fetch(url: str, *, timeout: float = 20.0, impersonate: str = "chrome") -> FetchRecord:
    """SSRF-guard `url`, then fetch it and return veriscrape's trust verdict + body (FetchRecord).

    The caller cites only when `record.ok` (verdict is OK); UNVERIFIED and every block/junk verdict are
    excluded. Raises UnsafeURLError (before any network call) for an unsafe URL.
    """
    assert_safe_url(url)
    return veriscrape.get(url, timeout=timeout, impersonate=impersonate)
