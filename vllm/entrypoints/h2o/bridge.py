# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drive a shim's ``BaseHTTPRequestHandler`` on an in-memory request.

The released package shims (``h2o_lightning_shim.py``, ``lightning_31b_shim.py``)
are standard-library HTTP servers: a ``BaseHTTPRequestHandler`` subclass whose
``do_GET`` / ``do_POST`` parse the request, run the decision and write the JSON
reply. Running that handler on a request held in memory keeps every HTTP-level
detail of the shim (status codes, bodies, multipart parsing, the ``/health``
query flags) byte for byte, with no socket, no port and no second process.
"""

from __future__ import annotations

import http.client
import io
import json
from collections.abc import Iterable
from typing import Any, NamedTuple


class BridgeResponse(NamedTuple):
    status: int
    headers: list[tuple[str, str]]
    body: bytes

    def header(self, name: str, default: str | None = None) -> str | None:
        name = name.lower()
        for k, v in self.headers:
            if k.lower() == name:
                return v
        return default

    @property
    def content_type(self) -> str:
        return self.header("Content-Type", "application/octet-stream") or ""

    def json(self) -> Any:
        return json.loads(self.body)


def build_request_bytes(
    method: str,
    target: str,
    headers: Iterable[tuple[str, str]] | None = None,
    body: bytes = b"",
) -> bytes:
    """The wire form of one HTTP/1.1 request (the request line, the headers
    given plus Host and Content-Length, a blank line, the body)."""
    if "\r" in target or "\n" in target:
        raise ValueError("request target must not contain line breaks")
    lines = [f"{method.upper()} {target} HTTP/1.1", "Host: shim"]
    seen = set()
    for k, v in headers or ():
        if "\r" in k or "\n" in k or "\r" in v or "\n" in v:
            raise ValueError("request headers must not contain line breaks")
        if k.lower() in ("host", "content-length"):
            continue
        lines.append(f"{k}: {v}")
        seen.add(k.lower())
    if body or method.upper() in ("POST", "PUT", "PATCH"):
        lines.append(f"Content-Length: {len(body)}")
    head = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")
    return head + body


class _FakeSocket:
    """What ``http.client.HTTPResponse`` needs of a socket: ``makefile``."""

    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def makefile(self, mode: str, *args: Any, **kwargs: Any) -> io.BytesIO:
        return io.BytesIO(self.raw)


def parse_response_bytes(raw: bytes, method: str = "GET") -> BridgeResponse:
    """A handler's raw reply bytes as (status, headers, body)."""
    resp = http.client.HTTPResponse(_FakeSocket(raw), method=method.upper())  # type: ignore[arg-type]
    resp.begin()
    body = resp.read()
    return BridgeResponse(resp.status, resp.getheaders(), body)


class HandlerBridge:
    """Runs a ``BaseHTTPRequestHandler`` subclass on in-memory requests.

    Each call builds a fresh handler instance without a socket: ``rfile`` holds
    the request, ``wfile`` collects the reply, and ``handle_one_request`` does
    what it does on a real connection (request line, headers, ``do_<METHOD>``).
    """

    def __init__(self, handler_cls: type, server: Any = None) -> None:
        self.handler_cls = handler_cls
        self.server = server

    def respond(
        self,
        method: str,
        target: str,
        headers: Iterable[tuple[str, str]] | None = None,
        body: bytes = b"",
    ) -> BridgeResponse:
        handler = self.handler_cls.__new__(self.handler_cls)
        handler.rfile = io.BytesIO(build_request_bytes(method, target, headers, body))
        handler.wfile = io.BytesIO()
        handler.client_address = ("127.0.0.1", 0)
        handler.server = self.server
        handler.request = None
        handler.connection = None
        handler.close_connection = True
        handler.handle_one_request()
        return parse_response_bytes(handler.wfile.getvalue(), method)

    def respond_json(
        self,
        method: str,
        target: str,
        obj: Any,
        headers: Iterable[tuple[str, str]] | None = None,
    ) -> BridgeResponse:
        """A JSON request (``Content-Type: application/json``)."""
        body = json.dumps(obj).encode()
        return self.respond(
            method,
            target,
            [("Content-Type", "application/json"), *(headers or ())],
            body,
        )
