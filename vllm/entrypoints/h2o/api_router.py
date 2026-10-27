# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decision routes: the package shim's endpoints on this server.

Every route hands the raw request (method, path with its query, the headers the
shim reads, the body bytes) to the shim's own handler through the bridge and
returns its status, body and content type unchanged. ``/health`` is the shim's
(503 with ``warming`` until the warm-up passed; ``?live=1`` meanwhile), so it
goes before the engine's health route; the engine's own check stays at
``/health/engine``.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()

# the request headers the shims read (plus Content-Length, which the bridge sets from the body)
PASS_HEADERS = ("content-type",)


def _layer(request: Request):
    return getattr(request.app.state, "h2o_decisions", None)


async def _through_shim(raw_request: Request) -> Response:
    layer = _layer(raw_request)
    if layer is None:
        return JSONResponse(
            {
                "ok": False,
                "warming": True,
                "error": "the decisions layer is still starting",
            },
            status_code=503,
        )
    body = await raw_request.body()
    target = raw_request.url.path
    if raw_request.url.query:
        target += "?" + raw_request.url.query
    headers = [
        (k, v) for k, v in raw_request.headers.items() if k.lower() in PASS_HEADERS
    ]
    r = await layer.respond_async(
        asyncio.get_running_loop(), raw_request.method, target, headers, body
    )
    return Response(content=r.body, status_code=r.status, media_type=r.content_type)


@router.get("/health")
async def health(raw_request: Request) -> Response:
    """The shim's health: the backend checks, the mode probe, the warm-up gate."""
    return await _through_shim(raw_request)


@router.get("/health/engine", response_class=Response)
async def engine_health(raw_request: Request) -> Response:
    """The engine's own health check (what ``/health`` is without the layer)."""
    from vllm.entrypoints.serve.instrumentator.health import (
        health as engine_health_route,
    )

    return await engine_health_route(raw_request)


@router.post("/v1/systemone")
async def systemone(raw_request: Request) -> Response:
    return await _through_shim(raw_request)


@router.post("/v1/decisions")
async def decisions(raw_request: Request) -> Response:
    return await _through_shim(raw_request)


@router.post("/api/alpha/decisions")
async def alpha_decisions(raw_request: Request) -> Response:
    return await _through_shim(raw_request)


@router.get("/h2o/decisions")
async def describe(raw_request: Request) -> Response:
    """The layer's own state: the package, the pinned shim, the warm-up."""
    layer = _layer(raw_request)
    if layer is None:
        return JSONResponse({"ok": False, "warming": True}, status_code=503)
    return JSONResponse(layer.describe())


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
