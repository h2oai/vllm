# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shim's engine calls, served in-process.

A released shim talks to "vLLM" through one class, ``VLLM``, whose
``call(method, path, body)`` returns ``(status, parsed JSON)`` from a keep-alive
HTTP connection; every other method of that class (``post``, ``refuse``,
``tokenize``, ``label_logprobs``, ``chat_generate``, ...) builds on ``call``.
``make_inprocess_vllm`` subclasses the shim's own ``VLLM`` and replaces only
``call``: the request goes to this server's serving handlers, the reply is the
JSON the matching route would have sent. The shim's logic above ``call`` is the
shipped code, untouched.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from http import HTTPStatus
from types import ModuleType
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# what the shim is told its engine is (its URL parser accepts any scheme)
INPROCESS_URL = "inprocess://engine"


def wire_json(content: Any) -> Any:
    """What a route's ``JSONResponse(content)`` puts on the wire, parsed back:
    the same floats, strings and nesting the shim would read from a socket
    (Starlette renders with ``allow_nan=False`` and compact separators)."""
    return json.loads(
        json.dumps(content, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    )


def error_body(message: str, status: int, param: str | None = None) -> dict[str, Any]:
    """vLLM's ``ErrorResponse`` shape, as its exception handlers send it."""
    return {
        "error": {
            "message": message,
            "type": HTTPStatus(status).phrase,
            "param": param,
            "code": status,
        }
    }


class EngineDispatcher:
    """Routes a shim's ``(method, path, body)`` to this server's serving handlers.

    ``call`` runs from the shim's worker threads: it submits the coroutine to
    the API server's event loop and waits. The same routes as the HTTP API,
    the same request models, the same response and error bodies.
    """

    def __init__(
        self, state: Any, loop: asyncio.AbstractEventLoop, timeout: float | None = None
    ):
        self.state = state
        self.loop = loop
        self.timeout = timeout

    # ---- the shim's side (a worker thread)
    def call(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self.loop:
            raise RuntimeError(
                "EngineDispatcher.call blocks: it must not run on the server's event loop"
            )
        future = asyncio.run_coroutine_threadsafe(
            self.dispatch(method, path, body), self.loop
        )
        return future.result(self.timeout)

    # ---- the server's side (the event loop)
    async def dispatch(
        self, method: str, path: str, body: Any = None
    ) -> tuple[int, Any]:
        route = path.split("?", 1)[0]
        handler = self._routes().get((method.upper(), route))
        if handler is None:
            return 404, {"detail": "Not Found"}
        try:
            return await handler(body)
        except Exception as e:  # the routes' exception handlers, in order
            return self._error(e)

    def _routes(self):
        return {
            ("GET", "/v1/models"): self._models,
            ("POST", "/tokenize"): self._tokenize,
            ("POST", "/detokenize"): self._detokenize,
            ("POST", "/v1/completions"): self._completions,
            ("POST", "/v1/chat/completions"): self._chat_completions,
        }

    @staticmethod
    def _error(e: Exception) -> tuple[int, Any]:
        from pydantic import ValidationError

        from vllm.entrypoints.serve.exception_handling.error_response import (
            create_error_response,
        )

        if isinstance(e, ValidationError):
            # the request-validation handler: HTTP 400 with the errors in the message
            message = f"{e.error_count()} validation error(s): " + "; ".join(
                f"{'.'.join(str(x) for x in err.get('loc', ()))}: {err.get('msg')}"
                for err in e.errors()[:10]
            )
            return 400, error_body(message[:1000], 400)
        try:
            from fastapi import HTTPException

            if isinstance(e, HTTPException):
                return e.status_code, error_body(str(e.detail), e.status_code)
        except ImportError:  # pragma: no cover
            pass
        err = create_error_response(e)
        return err.error.code, wire_json(err.model_dump())

    @staticmethod
    async def _finish(result: Any) -> tuple[int, Any]:
        from pydantic import BaseModel

        from vllm.entrypoints.serve.engine.protocol import ErrorResponse

        if isinstance(result, ErrorResponse):
            return result.error.code, wire_json(result.model_dump())
        if isinstance(result, BaseModel):
            return 200, wire_json(result.model_dump())
        if inspect.isasyncgen(result):
            await result.aclose()
            return 400, error_body("a streaming response is not served in-process", 400)
        return 500, error_body(
            f"unexpected handler result {type(result).__name__}", 500
        )

    async def _models(self, body: Any) -> tuple[int, Any]:
        return await self._finish(
            await self.state.openai_serving_models.show_available_models()
        )

    async def _tokenize(self, body: Any) -> tuple[int, Any]:
        from pydantic import TypeAdapter

        from vllm.entrypoints.serve.tokenize.protocol import TokenizeRequest

        request = TypeAdapter(TokenizeRequest).validate_python(body)
        return await self._finish(
            await self.state.serving_tokenization.create_tokenize(request, None)
        )

    async def _detokenize(self, body: Any) -> tuple[int, Any]:
        from vllm.entrypoints.serve.tokenize.protocol import DetokenizeRequest

        request = DetokenizeRequest.model_validate(body)
        return await self._finish(
            await self.state.serving_tokenization.create_detokenize(request, None)
        )

    async def _completions(self, body: Any) -> tuple[int, Any]:
        from vllm.entrypoints.openai.completion.protocol import CompletionRequest

        handler = getattr(self.state, "openai_serving_completion", None)
        if handler is None:
            return 501, error_body("The model does not support Completions API", 501)
        request = CompletionRequest.model_validate(body)
        if request.stream:
            return 400, error_body("a streaming response is not served in-process", 400)
        return await self._finish(await handler.create_completion(request, None))

    async def _chat_completions(self, body: Any) -> tuple[int, Any]:
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        handler = getattr(self.state, "openai_serving_chat", None)
        if handler is None:
            return 501, error_body(
                "The model does not support Chat Completions API", 501
            )
        request = ChatCompletionRequest.model_validate(body)
        if request.stream:
            return 400, error_body("a streaming response is not served in-process", 400)
        return await self._finish(await handler.create_chat_completion(request, None))


class BridgeDispatcher:
    """A dispatcher over an in-memory HTTP handler (a stand-in engine in the
    tests): the same ``call`` contract as ``EngineDispatcher``."""

    def __init__(self, handler_cls: type) -> None:
        from vllm.entrypoints.h2o.bridge import HandlerBridge

        self.bridge = HandlerBridge(handler_cls)

    def call(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        if body is None:
            r = self.bridge.respond(method, path)
        else:
            r = self.bridge.respond_json(method, path, body)
        try:
            return r.status, json.loads(r.body)
        except ValueError:
            return r.status, r.body.decode("utf-8", "replace")


def make_inprocess_vllm(shim_module: ModuleType, dispatcher: Any) -> type:
    """A subclass of the shim's ``VLLM`` whose ``call`` goes through
    ``dispatcher`` (no socket). Failures of the dispatcher itself are the
    shim's ``UpstreamError`` (a 502), as an unreachable engine would be."""
    base = shim_module.VLLM
    upstream_error = shim_module.UpstreamError

    class InProcessVLLM(base):  # type: ignore[misc,valid-type]
        def __init__(self, base_url: str = INPROCESS_URL, model: str = "") -> None:
            super().__init__(INPROCESS_URL, model)

        def _conn(self, fresh: bool = False) -> Any:
            raise RuntimeError("the in-process transport has no connection")

        def call(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
            try:
                return dispatcher.call(method, path, body)
            except upstream_error:
                raise
            except Exception as e:
                raise upstream_error(
                    f"vLLM in-process {method} {path} failed: {type(e).__name__}: {e}"
                ) from e

    InProcessVLLM.__name__ = "InProcessVLLM"
    InProcessVLLM.__qualname__ = "InProcessVLLM"
    return InProcessVLLM
