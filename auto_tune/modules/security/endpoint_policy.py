"""Endpoint policy for Studio model calls.

Validates and normalizes an OpenAI-compatible chat completions endpoint before
any network request.  Public endpoints require HTTPS and must not resolve to
loopback/private/link-local/multicast/unspecified/reserved addresses.  Local
and private HTTP(S) endpoints are allowed only when the caller explicitly
enables the private-endpoint switch.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlsplit

DEFAULT_DEEPSEEK_ENDPOINT = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_QWEN_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"


class EndpointPolicyError(ValueError):
    """Endpoint failed validation and must not be called."""


def _is_public_address(ip: ipaddress._BaseAddress) -> bool:
    return not (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        or ip.is_reserved
    )


def _is_allowed_private(ip: ipaddress._BaseAddress) -> bool:
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _resolve_addresses(hostname: str, port: int | None, resolver) -> list[str]:
    try:
        infos = resolver(hostname, port)
    except Exception:
        raise EndpointPolicyError("endpoint DNS resolution failed")
    addresses: list[str] = []
    for info in infos or []:
        sockaddr = info[4] if isinstance(info, tuple) and len(info) >= 5 else info
        if isinstance(sockaddr, (list, tuple)) and sockaddr:
            addresses.append(str(sockaddr[0]))
    if not addresses:
        raise EndpointPolicyError("endpoint resolved to no addresses")
    return addresses


def validate_endpoint(
    endpoint: object,
    allow_private_endpoint: bool,
    resolver=socket.getaddrinfo,
) -> str:
    """Validate and normalize an endpoint, raising EndpointPolicyError on any
    rejection.  Resolves the hostname immediately and rejects the whole
    endpoint if any resolved address violates the policy.
    """
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise EndpointPolicyError("endpoint must be a non-empty string")
    endpoint = endpoint.strip()

    parts = urlsplit(endpoint)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise EndpointPolicyError("endpoint must use http or https")
    if parts.username is not None or parts.password is not None:
        raise EndpointPolicyError("endpoint must not embed credentials")
    if parts.query:
        raise EndpointPolicyError("endpoint must not contain a query string")
    if parts.fragment:
        raise EndpointPolicyError("endpoint must not contain a fragment")
    hostname = parts.hostname
    if not hostname:
        raise EndpointPolicyError("endpoint must contain a hostname")
    try:
        port = parts.port
    except ValueError:
        raise EndpointPolicyError("endpoint has an invalid port")
    if port is not None and not (1 <= port <= 65535):
        raise EndpointPolicyError("endpoint has an invalid port")

    addresses = _resolve_addresses(hostname, port, resolver)
    is_http = scheme == "http"

    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise EndpointPolicyError("endpoint resolves to an invalid address")
        if _is_public_address(ip):
            # Public address is always reachable; plain HTTP to it is never a
            # safe public endpoint even when the private switch is enabled.
            if is_http:
                raise EndpointPolicyError("public http endpoint is not allowed")
            continue
        if not allow_private_endpoint:
            raise EndpointPolicyError("endpoint resolves to a private or local address")
        if not _is_allowed_private(ip):
            raise EndpointPolicyError("endpoint resolves to a disallowed address")

    # Reconstruct a normalized URL (no userinfo/query/fragment were permitted).
    if ":" in hostname and not hostname.startswith("["):
        normalized_host = f"[{hostname}]"
    else:
        normalized_host = hostname
    netloc = normalized_host
    if port is not None:
        default_port = 443 if scheme == "https" else 80
        if port != default_port:
            netloc = f"{normalized_host}:{port}"
    return f"{scheme}://{netloc}{parts.path}"
