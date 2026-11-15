"""Small, dependency-free SSRF guards shared by the download / scrape paths.

The app fetches user-supplied URLs server-side (yt-dlp downloads, SaaS landing
page scraping, actor-image download). Without a guard, a caller can point those
at internal services or the cloud metadata endpoint (169.254.169.254) to read
instance credentials. ``assert_public_url`` rejects non-HTTP(S) schemes and any
host that resolves to a private / loopback / link-local / reserved address.
"""
from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse


class UnsafeURLError(ValueError):
    """Raised when a URL is not safe to fetch server-side."""


def _ip_is_public(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def normalize_media_url(raw_url: str) -> str:
    """Normalize user input URLs, short-links, and bare video IDs into canonical URLs."""
    if not raw_url or not isinstance(raw_url, str):
        return ""
    s = raw_url.strip()
    if not s:
        return ""

    # Check for bare YouTube video ID (11 chars), possibly with query params (?si=... etc)
    # e.g. BgeNDG5jD60?si=zSjxkinw2zPaaAPH or BgeNDG5jD60
    bare_match = re.match(r"^([a-zA-Z0-9_-]{11})(?:\?.*)?$", s)
    if bare_match:
        return f"https://www.youtube.com/watch?v={bare_match.group(1)}"

    # If scheme is missing (e.g. youtube.com/watch?v=... or youtu.be/...)
    if not s.startswith(("http://", "https://")):
        if any(s.startswith(prefix) for prefix in (
            "youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be",
            "twitch.tv", "www.twitch.tv", "kick.com", "www.kick.com"
        )):
            s = "https://" + s

    # Canonicalize any youtu.be, /shorts/, /live/, /embed/, /v/
    yt_match = re.search(r"(?:youtu\.be/|youtube\.com/(?:embed/|v/|shorts/|live/|(?:watch\?(?:.*&)?v=)))([a-zA-Z0-9_-]{11})", s)
    if yt_match:
        return f"https://www.youtube.com/watch?v={yt_match.group(1)}"

    return s


def assert_public_url(url: str) -> str:
    """Return ``url`` if it is safe to fetch, else raise ``UnsafeURLError``.

    Blocks non-http(s) schemes and hosts that resolve to any non-public IP.
    Resolves every A/AAAA record so a hostname that maps to a private range
    (or to 169.254.169.254) is rejected rather than silently fetched.
    """
    if not url or not isinstance(url, str):
        raise UnsafeURLError("Empty URL")

    url = normalize_media_url(url)
    parsed = urlparse(url.strip())
    if parsed.scheme.lower() not in ("http", "https"):
        raise UnsafeURLError(f"Unsupported URL scheme: {parsed.scheme!r}")

    host = parsed.hostname
    if not host:
        raise UnsafeURLError("URL has no host")

    # If the host is a literal IP, validate it directly (no DNS).
    try:
        ipaddress.ip_address(host)
        is_ip_literal = True
    except ValueError:
        is_ip_literal = False

    if is_ip_literal:
        if not _ip_is_public(host):
            raise UnsafeURLError(f"URL host is not a public address: {host}")
        return url

    try:
        infos = socket.getaddrinfo(host, parsed.port or None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise UnsafeURLError(f"Could not resolve host {host!r}: {e}")

    resolved = {info[4][0] for info in infos}
    if not resolved:
        raise UnsafeURLError(f"Host {host!r} did not resolve")
    for ip in resolved:
        if not _ip_is_public(ip):
            raise UnsafeURLError(f"Host {host!r} resolves to a non-public address: {ip}")
    return url
