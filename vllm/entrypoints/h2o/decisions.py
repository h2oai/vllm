# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The decisions layer: the package's shim, started by its own ``main()``,
served in-process.

``start_layer`` imports the shim module from the package directory, swaps its
``VLLM`` class for the in-process transport and its ``Server`` class for a
stand-in that only keeps the request handler, and runs the shim's ``main()``
with the arguments ``serve.sh`` passes. Everything ``main()`` does (the
contract from ``serve_config*.json``, the profile checks of the 31B, the
one-engine two-way flag, the start-up engine check) happens as in the released
package. The handler then answers in-memory requests (``bridge.py``) behind the
server's routes (``api_router.py``), and the warm-up ``serve.sh`` runs before it
reports ready is replayed here before ``/health`` turns 200.
"""

from __future__ import annotations

import asyncio
import functools
import importlib.util
import json
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

from vllm.logger import init_logger

from .bridge import BridgeResponse, HandlerBridge
from .package import Package, locate_package
from .transport import INPROCESS_URL, EngineDispatcher, make_inprocess_vllm

logger = init_logger(__name__)

READY_FILE_ENV = "SHIM_READY_FILE"

# the 4B's warm-up image (serve.sh): a 64x64 PNG
WARMUP_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAT0lEQVR42u3PQQ0AAAgEILV/"
    "sItlCh9u0IBOUp9NPScgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBwbwFY4wLYRBP0hgAAAABJRU5ErkJggg=="
)


@dataclass
class DecisionsLayer:
    package: Package
    module: ModuleType
    handler_cls: type
    bridge: HandlerBridge
    executor: ThreadPoolExecutor
    ready_file: str | None
    dispatcher: Any
    http_vllm_cls: type  # the shim's own HTTP transport class (kept for tests)
    two_way: bool
    profile: str | None
    warmup: dict[str, Any] = field(
        default_factory=lambda: {"state": "pending", "steps": [], "error": None}
    )

    # ---- requests
    def respond(
        self, method: str, target: str, headers: Any = None, body: bytes = b""
    ) -> BridgeResponse:
        return self.bridge.respond(method, target, headers, body)

    async def respond_async(
        self,
        loop: asyncio.AbstractEventLoop,
        method: str,
        target: str,
        headers: Any = None,
        body: bytes = b"",
    ) -> BridgeResponse:
        return await loop.run_in_executor(
            self.executor, self.respond, method, target, headers, body
        )

    def post_json(self, path: str, obj: Any) -> BridgeResponse:
        return self.bridge.respond_json("POST", path, obj)

    def health(self, live: bool = False) -> BridgeResponse:
        return self.bridge.respond("GET", "/health?live=1" if live else "/health")

    # ---- readiness (the shim's own gate: SHIM_READY_FILE, as serve.sh uses it)
    def mark_ready(self) -> None:
        if self.ready_file:
            with open(self.ready_file, "w", encoding="utf-8"):
                pass

    def is_ready(self) -> bool:
        return not self.ready_file or os.path.exists(self.ready_file)

    def describe(self) -> dict[str, Any]:
        return {
            "package": self.package.describe(),
            "two_way": self.two_way,
            "profile": self.profile,
            "ready": self.is_ready(),
            "warmup": self.warmup,
        }


# ---------------------------------------------------------------------- what serve.sh passes
def shim_argv(
    package: Package, two_way: bool = True, profile: str | None = None
) -> list[str]:
    """The shim's command line, as the package's ``serve.sh`` builds it (the
    port is unused in-process; the engine URL is the in-process marker)."""
    argv = ["--config", package.config_path, "--port", "0"]
    if package.family == "4b":
        # serve.sh: --config serve_config.json --model-dir $MODEL --port $PORT --vllm http://127.0.0.1:$VLLM_PORT
        argv += ["--model-dir", package.dir, "--vllm", INPROCESS_URL]
        return argv
    # 31B serve.sh: --config $CONFIG --vllm URL --port $PORT --tokenizer $MODEL/tokenizer.json [--config-two-way
    # serve_config_two_way.json --two-way-reading | --profile NAME]
    argv += ["--vllm", INPROCESS_URL]
    tokenizer = package.file("tokenizer.json")
    if tokenizer:
        argv += ["--tokenizer", tokenizer]
    if profile:
        argv += ["--profile", profile]
    elif two_way:
        two_way_config = package.file("serve_config_two_way.json")
        if two_way_config is None:
            raise RuntimeError(
                "the two-way reading needs serve_config_two_way.json beside the 31B shim"
            )
        argv += ["--config-two-way", two_way_config, "--two-way-reading"]
    return argv


def shim_environment(
    package: Package,
    two_way: bool,
    ready_file: str | None,
    profile: str | None = None,
    two_way_window: int | None = None,
) -> dict[str, str]:
    """The environment the package's ``serve.sh`` exports for its shim."""
    env: dict[str, str] = {}
    if ready_file:
        env[READY_FILE_ENV] = ready_file
    if package.family == "4b":
        # one engine of the per-request attention build: the flag serves both readings
        env["SHIM_TWO_WAY_FLAG"] = "1"
        if not two_way:
            env["SHIM_ATTENTION_MODE"] = (
                "causal"  # TWO_WAY=0: the one engine answering one-way by default
            )
    elif two_way and not profile and two_way_window:
        env["SHIM_TWO_WAY_WINDOW"] = str(int(two_way_window))
    return env


# ---------------------------------------------------------------------- the shim module
def load_shim_module(package: Package) -> ModuleType:
    """Import the package's shim from its file, under the file's own module
    name, as ``python3 <shim>.py`` would see it: ``__file__`` beside its
    siblings, the package directory first on ``sys.path`` (the 31B shim imports
    ``pooled_readout`` by name)."""
    name = os.path.splitext(package.entry)[0]
    if package.dir not in sys.path:
        sys.path.insert(0, package.dir)
    spec = importlib.util.spec_from_file_location(name, package.entry_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {package.entry_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for attr in ("VLLM", "Server", "main", "make_handler", "UpstreamError"):
        if not hasattr(module, attr):
            raise RuntimeError(
                f"{package.entry} has no {attr}: not a shim this layer can serve"
            )
    return module


def _capture_server(captured: list[Any]) -> type:
    class CaptureServer:
        """Stands in for the shim's ``Server`` (a ``ThreadingHTTPServer``):
        ``main()`` builds it with the request handler and calls
        ``serve_forever``, which here returns at once."""

        daemon_threads = True
        request_queue_size = 512

        def __init__(self, address: Any, handler_cls: type) -> None:
            self.server_address = address
            self.handler_cls = handler_cls
            captured.append(self)

        def serve_forever(self, poll_interval: float = 0.5) -> None:
            return

        def shutdown(self) -> None:
            return

        def server_close(self) -> None:
            return

    return CaptureServer


def start_layer(
    package: Package,
    dispatcher: Any,
    *,
    two_way: bool = True,
    profile: str | None = None,
    ready_file: str | None = "",
    two_way_window: int | None = None,
    argv: list[str] | None = None,
    request_threads: int | None = None,
) -> DecisionsLayer:
    """Build the shim through its own ``main()`` and return the layer.

    ``ready_file``: the path of the shim's readiness marker (``""`` makes a
    temporary one; None serves ready at once, which is the test setting).
    """
    if ready_file == "":
        fd, ready_file = tempfile.mkstemp(prefix="h2o-decisions-ready.")
        os.close(fd)
        os.unlink(ready_file)  # the shim is ready once the file exists
    os.environ.update(
        shim_environment(package, two_way, ready_file, profile, two_way_window)
    )
    module = load_shim_module(package)
    http_vllm_cls = module.VLLM
    module.VLLM = make_inprocess_vllm(module, dispatcher)
    captured: list[Any] = []
    module.Server = _capture_server(captured)
    try:
        module.main(argv if argv is not None else shim_argv(package, two_way, profile))
    except SystemExit as e:
        raise RuntimeError(
            f"the package shim refused to start (exit {e.code}); its message is in the log above"
        ) from e
    if not captured:
        raise RuntimeError("the shim's main() returned without building its server")
    handler_cls = captured[-1].handler_cls
    threads = request_threads or int(os.environ.get("SHIM_REQUEST_THREADS", "64"))
    return DecisionsLayer(
        package=package,
        module=module,
        handler_cls=handler_cls,
        bridge=HandlerBridge(handler_cls),
        executor=ThreadPoolExecutor(
            max_workers=threads, thread_name_prefix="h2o-decisions"
        ),
        ready_file=ready_file,
        dispatcher=dispatcher,
        http_vllm_cls=http_vllm_cls,
        two_way=two_way,
        profile=profile,
    )


# ---------------------------------------------------------------------- the warm-up (serve.sh)
def warm_up_bodies_4b(
    max_len: int, images: bool, two_way: bool
) -> list[tuple[str, dict[str, Any]]]:
    """The 4B ``serve.sh`` warm-up, in order: in each mode the engine serves
    (two-way and one-way on the one engine; one-way alone with TWO_WAY=0) a
    short yes/no question, a three-question request, a long record (about
    MAX_LEN/6, at most 30,000 tokens) and, when the model reads images, a small
    image. The box question is not warmed, as in serve.sh."""
    n = min(max_len // 24, 7500)
    long_record = "warm-up record line " * n
    modes: list[str | None] = ["bidirectional", "causal"] if two_way else [None]
    out: list[tuple[str, dict[str, Any]]] = []
    for mode in modes:
        tag = {"attention_mode": mode} if mode else {}
        label = mode or "default mode"
        out.append(
            (
                f"yes/no ({label})",
                {
                    **tag,
                    "state": "warm-up",
                    "questions": {
                        "d": {
                            "type": "noul",
                            "instructions": "Is this a warm-up?",
                            "criteria": {"true": "yes", "false": "no"},
                        }
                    },
                },
            )
        )
        out.append(
            (
                f"three questions ({label})",
                {
                    **tag,
                    "state": {"ticket": "warm-up"},
                    "questions": {
                        "c": {
                            "type": "choice",
                            "instructions": "Which team?",
                            "criteria": {"a": "billing", "b": "support", "c": "sales"},
                        },
                        "n": {
                            "type": "noul",
                            "instructions": "Is it urgent?",
                            "criteria": {"true": "yes", "false": "no"},
                        },
                        "s": {
                            "type": "score",
                            "instructions": "How urgent?",
                            "criteria": ["low", "medium", "high"],
                        },
                    },
                },
            )
        )
        out.append(
            (
                f"long record ({label})",
                {
                    **tag,
                    "state": long_record,
                    "questions": {
                        "d": {
                            "type": "choice",
                            "instructions": "Which?",
                            "criteria": {"a": "first", "b": "second"},
                        }
                    },
                },
            )
        )
        if images:
            out.append(
                (
                    f"image ({label})",
                    {
                        **tag,
                        "state": {"images": [WARMUP_PNG], "note": "warm-up"},
                        "questions": {
                            "d": {
                                "type": "noul",
                                "instructions": "Is this a warm-up?",
                                "criteria": {"true": "yes", "false": "no"},
                            }
                        },
                    },
                )
            )
    return out


def _wait_live(layer: DecisionsLayer, deadline_s: float) -> None:
    """Until ``/health?live=1`` answers 200 (the shim's backend checks pass)."""
    deadline = time.monotonic() + deadline_s
    last_log = 0.0
    while True:
        r = layer.health(live=True)
        if r.status == 200:
            return
        if time.monotonic() - last_log > 30:
            logger.warning(
                "h2o decisions: not live yet: HTTP %d %s",
                r.status,
                r.body[:300].decode("utf-8", "replace"),
            )
            last_log = time.monotonic()
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"the shim did not turn live within {deadline_s:.0f} s: {r.body[:300]!r}"
            )
        time.sleep(5)


def run_warm_up(
    layer: DecisionsLayer, max_len: int, live_deadline_s: float = 3600.0
) -> None:
    """Replay the package's warm-up through the shim, then mark it ready.
    A failed text warm-up leaves the layer not ready (``/health`` 503,
    ``warming``), with the error in the log and in ``warmup``; a failed image
    warm-up is a warning, as in ``serve.sh``."""
    layer.warmup["state"] = "running"
    try:
        _wait_live(layer, live_deadline_s)
        if layer.package.family == "4b":
            c = layer.package.serve_config
            for name, body in warm_up_bodies_4b(
                max_len, bool(c.get("image")), layer.two_way
            ):
                t0 = time.monotonic()
                r = layer.post_json("/v1/systemone", body)
                step = {
                    "name": name,
                    "status": r.status,
                    "seconds": round(time.monotonic() - t0, 2),
                }
                layer.warmup["steps"].append(step)
                logger.info(
                    "h2o decisions: warm-up %s: HTTP %d, %.1f s",
                    name,
                    r.status,
                    step["seconds"],
                )
                if r.status != 200:
                    if name.startswith("image"):
                        logger.warning(
                            "h2o decisions: the image warm-up failed (%s); serving anyway",
                            r.body[:300],
                        )
                        continue
                    raise RuntimeError(
                        f"the warm-up request {name!r} failed: HTTP {r.status} {r.body[:300]!r}"
                    )
        else:
            _run_warmup_31b(layer, max_len)
        layer.mark_ready()
        layer.warmup["state"] = "done"
        logger.info("h2o decisions: ready (%s)", layer.package.model)
    except Exception as e:
        layer.warmup["state"] = "failed"
        layer.warmup["error"] = f"{type(e).__name__}: {e}"
        logger.error("h2o decisions: WARM-UP FAILED, /health stays 503: %s", e)


def _run_warmup_31b(layer: DecisionsLayer, max_len: int) -> None:
    """The 31B package's ``warmup.py``, with its ``post`` routed in-process
    (the shim through the bridge, "straight to vLLM" through the dispatcher)."""
    path = layer.package.file("warmup.py")
    if path is None:
        logger.warning(
            "h2o decisions: the 31B package has no warmup.py; the warm-up is skipped"
        )
        return
    spec = importlib.util.spec_from_file_location("h2o_lightning_warmup", path)
    assert spec is not None and spec.loader is not None
    warm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warm)
    shim_url, vllm_url = "http://shim", "http://engine"

    def post(url: str, body: Any, timeout: float = 900) -> tuple[int, Any, float]:
        t = time.monotonic()
        if url.startswith(shim_url):
            r = layer.post_json(url[len(shim_url) :], body)
            try:
                out = json.loads(r.body)
            except ValueError:
                out = {"error": r.body[:300].decode("utf-8", "replace")}
            return r.status, out, time.monotonic() - t
        status, out = layer.dispatcher.call("POST", url[len(vllm_url) :], body)
        return status, out, time.monotonic() - t

    warm.post = post
    env = dict(os.environ)
    env.setdefault("MAX_LEN", str(max_len))
    env["READING"] = "two-way" if layer.two_way else "one-way"
    env["IMAGES"] = "1" if layer.package.serve_config.get("image") else "0"
    try:
        warm.main(shim_url, vllm_url, env)
    except SystemExit as e:
        raise RuntimeError(f"warmup.py stopped: {e.code}") from e


# ---------------------------------------------------------------------- the server hook
def engine_serves_two_way(vllm_config: Any) -> tuple[bool, str]:
    """Whether this engine applies the per-request two-way flag (the frontend's
    own check for a flagged request), and the reason when it does not."""
    try:
        from vllm.sampling_params import SamplingParams
        from vllm.v1.request import check_bidirectional_request

        check_bidirectional_request(
            SamplingParams(
                max_tokens=1, extra_args={"attention_mode": "bidirectional"}
            ),
            vllm_config,
        )
        return True, ""
    except Exception as e:  # a stock backend, a KV connector, an unsupported model
        return False, str(e)


async def init_decisions_state(state: Any, args: Any) -> DecisionsLayer:
    """Called from ``init_app_state`` once the engine serves: build the layer
    (the shim's ``main()`` in a worker thread) and start the warm-up."""
    package = getattr(args, "_h2o_package", None) or locate_package(
        args.model,
        getattr(args, "revision", None),
        getattr(args, "h2o_decisions_package", None),
    )
    vllm_config = state.vllm_config
    two_way_ok, reason = engine_serves_two_way(vllm_config)
    want_two_way = getattr(args, "h2o_decisions_two_way", True)
    two_way = bool(want_two_way and two_way_ok)
    if want_two_way and not two_way_ok:
        logger.warning(
            "h2o decisions: this engine does not serve the two-way reading (%s); "
            "the decision API answers one-way by default",
            reason,
        )
    profile = getattr(args, "h2o_decisions_profile", None) or None
    loop = asyncio.get_running_loop()
    timeout = float(os.environ.get("SHIM_TIMEOUT", "180"))
    dispatcher = EngineDispatcher(state, loop, timeout=timeout)
    window = getattr(args, "h2o_decisions_two_way_window", None)
    start = functools.partial(
        start_layer,
        package,
        dispatcher,
        two_way=two_way,
        profile=profile,
        ready_file="",
        two_way_window=window,
    )
    # main() runs in a worker thread: under a 31B profile it starts the shim's own
    # engine check, which calls the dispatcher and must not block this loop
    layer = await loop.run_in_executor(None, start)
    state.h2o_decisions = layer
    max_len = int(vllm_config.model_config.max_model_len)
    threading.Thread(
        target=run_warm_up,
        args=(layer, max_len),
        name="h2o-decisions-warmup",
        daemon=True,
    ).start()
    logger.info(
        "h2o decisions: %s %s (%s) served at /v1/decisions, /v1/systemone, /api/alpha/decisions; "
        "/health is the shim's (503 until the warm-up passes), /health/engine the engine's",
        package.model,
        package.known["tag"] if package.known else "unpinned",
        package.entry,
    )
    return layer
