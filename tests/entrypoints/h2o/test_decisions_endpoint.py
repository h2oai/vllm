# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The H2O decisions layer, without a GPU: the in-memory HTTP bridge, the
in-process transport over the serving handlers, the package loader and its
pinning, the package engine defaults, and parity with the released shims run
standalone against each package's own stand-in engine (those tests need the
package files of h2oai/h2o-lightning-4b v1.4.1 and h2oai/h2o-lightning-31b
v1.2.2, from the Hub cache or fetched; they are skipped when unavailable)."""

import asyncio
import importlib.util
import json
import os
import sys
import threading
import types
import urllib.error
import urllib.request
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vllm.entrypoints.h2o import decisions as D
from vllm.entrypoints.h2o.bridge import (
    HandlerBridge,
    build_request_bytes,
    parse_response_bytes,
)
from vllm.entrypoints.h2o.engine_defaults import (
    apply_engine_defaults,
    package_engine_defaults,
    parse_vllm_argv,
)
from vllm.entrypoints.h2o.package import (
    KNOWN_SHIMS,
    Package,
    PackageError,
    load_package,
    locate_package,
    sha256_file,
)
from vllm.entrypoints.h2o.transport import (
    INPROCESS_URL,
    BridgeDispatcher,
    EngineDispatcher,
    error_body,
    make_inprocess_vllm,
    wire_json,
)

pytestmark = pytest.mark.cpu_test


# ---------------------------------------------------------------- the bridge
class _Echo(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code, obj):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.startswith("/ping"):
            return self._send(200, {"ok": True, "path": self.path})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        self._send(
            201,
            {
                "ctype": self.headers.get("Content-Type"),
                "len": n,
                "raw": raw.decode("latin-1"),
            },
        )


def _serve(handler):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _http(url, method="GET", body=None, headers=None):
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read(), r.headers.get("Content-Type")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type")


def test_bridge_runs_the_handler_in_memory():
    bridge = HandlerBridge(_Echo)
    r = bridge.respond("GET", "/ping?live=1")
    assert r.status == 200
    assert r.json() == {"ok": True, "path": "/ping?live=1"}
    assert r.content_type == "application/json"
    assert bridge.respond("GET", "/nothing").status == 404
    r = bridge.respond_json("POST", "/x", {"a": 1})
    assert r.status == 201
    assert r.json() == {"ctype": "application/json", "len": 8, "raw": '{"a": 1}'}
    raw = b"\xff\x00binary\r\n--zz--"
    r = bridge.respond(
        "POST", "/x", [("Content-Type", "multipart/form-data; boundary=zz")], raw
    )
    assert r.json()["len"] == len(raw) and r.json()["raw"] == raw.decode("latin-1")
    assert r.json()["ctype"].startswith("multipart/form-data")
    # the same bytes as over a socket
    srv, url = _serve(_Echo)
    try:
        status, wire, ctype = _http(url + "/ping?live=1")
        assert (status, wire, ctype) == (
            200,
            bridge.respond("GET", "/ping?live=1").body,
            "application/json",
        )
        status, wire, _ = _http(
            url + "/x", "POST", b'{"a": 1}', {"Content-Type": "application/json"}
        )
        assert (status, wire) == (201, bridge.respond_json("POST", "/x", {"a": 1}).body)
    finally:
        srv.shutdown()
    with pytest.raises(ValueError):
        build_request_bytes("GET", "/a\r\nX: y")
    r = parse_response_bytes(
        b"HTTP/1.1 503 Service Unavailable\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\nno"
    )
    assert (r.status, r.body, r.content_type) == (503, b"no", "text/plain")


def test_wire_json_is_what_the_routes_send():
    from fastapi.responses import JSONResponse

    content = {
        "a": [1.5, 2, "é"],
        "b": {"c": None, "d": True},
        "e": 1e-7,
        "f": -0.30000000000000004,
    }
    assert wire_json(content) == json.loads(JSONResponse(content=content).body)
    with pytest.raises(ValueError):
        wire_json({"x": float("nan")})
    assert error_body("m", 404)["error"] == {
        "message": "m",
        "type": "Not Found",
        "param": None,
        "code": 404,
    }


# ---------------------------------------------------------------- the engine dispatcher
@pytest.fixture
def loop_thread():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


def _fake_state():
    from vllm.entrypoints.openai.completion.protocol import (
        CompletionLogProbs,
        CompletionResponse,
        CompletionResponseChoice,
    )
    from vllm.entrypoints.serve.engine.protocol import (
        ErrorInfo,
        ErrorResponse,
        ModelCard,
        ModelList,
        UsageInfo,
    )
    from vllm.entrypoints.serve.tokenize.protocol import (
        DetokenizeResponse,
        TokenizeResponse,
    )

    class Models:
        async def show_available_models(self):
            return ModelList(data=[ModelCard(id="m", root="m", max_model_len=4096)])

    class Tokenization:
        def __init__(self):
            self.seen = []

        async def create_tokenize(self, request, raw_request):
            assert raw_request is None
            self.seen.append(request)
            return TokenizeResponse(count=3, max_model_len=4096, tokens=[1, 2, 3])

        async def create_detokenize(self, request, raw_request):
            return DetokenizeResponse(prompt="abc")

    class Completion:
        async def create_completion(self, request, raw_request):
            assert raw_request is None
            if request.prompt == "err":
                return ErrorResponse(
                    error=ErrorInfo(
                        message="This model's maximum context length is 10 tokens.",
                        type="BadRequestError",
                        code=400,
                    )
                )
            if request.prompt == "boom":
                raise ValueError("boom")
            choice = CompletionResponseChoice(
                index=0,
                text="",
                finish_reason="length",
                logprobs=CompletionLogProbs(
                    tokens=["token_id:7"],
                    token_logprobs=[-0.5],
                    top_logprobs=[{"token_id:1000": -0.1, "token_id:1001": -0.4}],
                ),
                prompt_token_ids=[1, 2] if request.return_token_ids else None,
            )
            return CompletionResponse(
                model="m",
                choices=[choice],
                usage=UsageInfo(prompt_tokens=2, completion_tokens=1, total_tokens=3),
            )

    state = types.SimpleNamespace()
    state.openai_serving_models = Models()
    state.serving_tokenization = Tokenization()
    state.openai_serving_completion = Completion()
    state.openai_serving_chat = None
    return state


def test_engine_dispatcher_routes_to_the_serving_handlers(loop_thread):
    state = _fake_state()
    d = EngineDispatcher(state, loop_thread, timeout=30)
    status, out = d.call("GET", "/v1/models")
    assert (
        status == 200
        and out["data"][0]["id"] == "m"
        and out["data"][0]["max_model_len"] == 4096
    )
    status, out = d.call(
        "POST", "/tokenize", {"model": "m", "prompt": "x", "add_special_tokens": False}
    )
    assert (status, out["tokens"]) == (200, [1, 2, 3])
    assert state.serving_tokenization.seen[0].prompt == "x"
    status, out = d.call("POST", "/detokenize", {"model": "m", "tokens": [1, 2]})
    assert (status, out) == (200, {"prompt": "abc"})
    body = {
        "model": "m",
        "prompt": "the prompt",
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": 2,
        "logprob_token_ids": [1000, 1001],
        "return_tokens_as_token_ids": True,
        "add_special_tokens": False,
        "return_token_ids": True,
    }
    status, out = d.call("POST", "/v1/completions", body)
    assert status == 200
    assert out["choices"][0]["logprobs"]["top_logprobs"][0] == {
        "token_id:1000": -0.1,
        "token_id:1001": -0.4,
    }
    assert (
        out["choices"][0]["prompt_token_ids"] == [1, 2]
        and out["usage"]["prompt_tokens"] == 2
    )
    # an ErrorResponse from the handler: its code and body, as the route sends them
    status, out = d.call("POST", "/v1/completions", dict(body, prompt="err"))
    assert status == 400 and "maximum context length" in out["error"]["message"]
    # an exception in the handler: the route's fallback handler
    from vllm.entrypoints.serve.exception_handling.error_response import (
        create_error_response,
    )

    status, out = d.call("POST", "/v1/completions", dict(body, prompt="boom"))
    assert (
        status == create_error_response(ValueError("boom")).error.code
        and "boom" in out["error"]["message"]
    )
    # a body the request model refuses: 400 with an error object
    status, out = d.call("POST", "/v1/completions", {"model": "m", "max_tokens": "x"})
    assert status == 400 and out["error"]["code"] == 400
    # streaming is never served in-process; unknown routes are 404; no chat handler is 501
    status, out = d.call("POST", "/v1/completions", dict(body, stream=True))
    assert status == 400 and "streaming" in out["error"]["message"]
    assert d.call("POST", "/v1/embeddings", {})[0] == 404
    assert (
        d.call("POST", "/v1/chat/completions", {"model": "m", "messages": []})[0] == 501
    )

    # a call from the loop's own thread would deadlock: refused
    async def on_loop():
        d.call("GET", "/v1/models")

    with pytest.raises(RuntimeError):
        asyncio.run_coroutine_threadsafe(on_loop(), loop_thread).result(5)


def _fake_shim_module():
    """The seam of a shim: VLLM (an HTTP client), UpstreamError, and a method built on call()."""
    mod = types.ModuleType("fake_shim")
    code = """
import urllib.parse
class UpstreamError(RuntimeError):
    def __init__(self, message, status=502):
        super().__init__(message); self.status = status
class VLLM:
    def __init__(self, base, model):
        u = urllib.parse.urlsplit(base if "://" in base else "http://" + base)
        self.base = f"{u.scheme}://{u.netloc}"; self.https = u.scheme == "https"; self.host = u.netloc; self.model = model
    def _conn(self, fresh=False):
        raise AssertionError("a socket was opened")
    def call(self, method, path, body=None):
        raise AssertionError("the HTTP call ran")
    def post(self, path, body):
        status, out = self.call("POST", path, body)
        if status // 100 != 2:
            raise UpstreamError(f"HTTP {status}", status)
        return out
"""
    exec(code, mod.__dict__)
    return mod


def test_inprocess_vllm_is_the_shims_class_over_a_dispatcher():
    mod = _fake_shim_module()
    cls = make_inprocess_vllm(mod, BridgeDispatcher(_Echo))
    assert issubclass(cls, mod.VLLM)
    v = cls("http://127.0.0.1:1", "m")
    assert v.base == INPROCESS_URL and v.model == "m"
    assert v.post("/x", {"a": 1}) == {
        "ctype": "application/json",
        "len": 8,
        "raw": '{"a": 1}',
    }
    assert v.call("GET", "/ping?live=1") == (200, {"ok": True, "path": "/ping?live=1"})
    assert v.call("GET", "/nothing")[0] == 404

    class Exploding:
        def call(self, method, path, body=None):
            raise OSError("no engine")

    bad = make_inprocess_vllm(mod, Exploding())("x", "m")
    with pytest.raises(mod.UpstreamError, match="no engine"):
        bad.call("GET", "/v1/models")


# ---------------------------------------------------------------- the package
def _write_package(
    tmp_path, entry="h2o_lightning_shim.py", model="h2oai/x", extra=None
):
    (tmp_path / entry).write_text("VLLM = Server = main = None\n")
    cfg = {"model": model, "labels": ["A", "B"], "temperature": 1.0}
    if extra:
        cfg.update(extra)
    (tmp_path / "serve_config.json").write_text(json.dumps(cfg))
    return tmp_path


def test_package_loading_pins_the_shim(tmp_path, monkeypatch):
    monkeypatch.delenv("VLLM_H2O_DECISIONS_ALLOW_UNPINNED", raising=False)
    d = _write_package(tmp_path)
    with pytest.raises(PackageError, match="not a shim this build was checked against"):
        load_package(str(d))
    pkg = load_package(str(d), allow_unpinned=True)
    assert pkg.known is None and pkg.family == "4b" and pkg.model == "h2oai/x"
    assert pkg.entry_sha256 == sha256_file(str(d / "h2o_lightning_shim.py"))
    monkeypatch.setenv("VLLM_H2O_DECISIONS_ALLOW_UNPINNED", "1")
    assert load_package(str(d)).entry == "h2o_lightning_shim.py"
    assert locate_package(str(d)).dir == str(d)
    (d / "serve_config.json").unlink()
    with pytest.raises(PackageError, match="no serve_config.json"):
        load_package(str(d))
    with pytest.raises(PackageError, match="no decision shim"):
        load_package(str(tmp_path / "nothing")) if (
            tmp_path / "nothing"
        ).mkdir() is None else None
    for digest, info in KNOWN_SHIMS.items():
        assert len(digest) == 64 and info["entry"] in (
            "h2o_lightning_shim.py",
            "lightning_31b_shim.py",
        )
    with pytest.raises(PackageError, match="not a directory"):
        locate_package("h2oai/x", explicit=str(tmp_path / "missing"))


# ---------------------------------------------------------------- argv, environment, engine defaults
def _pkg(tmp_path, entry, cfg, files=()):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / entry).write_text("x = 1\n")
    (tmp_path / "serve_config.json").write_text(json.dumps(cfg))
    for f in files:
        p = tmp_path / f
        if f.endswith("/"):
            p.mkdir()
        else:
            p.write_text("{}")
    return Package(
        dir=str(tmp_path),
        entry=entry,
        entry_sha256="0" * 64,
        serve_config=cfg,
        known=None,
    )


CFG_4B = {
    "model": "h2oai/h2o-lightning-4b",
    "labels": ["A", "B"],
    "temperature": {"choice": 0.75},
    "image": {"max_images": 4, "max_pixels": 1048576, "temperature": 0.65},
    "two_way": {"label_rows": "label_rows.lightning", "default": "bidirectional"},
}
CFG_31B = {
    "model": "h2oai/h2o-lightning-31b",
    "temperature": 0.7,
    "image": {"max_images": 4, "temperature": 0.59},
}


def test_shim_argv_and_environment_follow_serve_sh(tmp_path):
    p4 = _pkg(tmp_path / "a", "h2o_lightning_shim.py", CFG_4B)
    (tmp_path / "a").mkdir(exist_ok=True)
    assert D.shim_argv(p4) == [
        "--config",
        p4.config_path,
        "--port",
        "0",
        "--model-dir",
        p4.dir,
        "--vllm",
        INPROCESS_URL,
    ]
    assert D.shim_environment(p4, True, "/tmp/r") == {
        "SHIM_READY_FILE": "/tmp/r",
        "SHIM_TWO_WAY_FLAG": "1",
    }
    assert D.shim_environment(p4, False, None) == {
        "SHIM_TWO_WAY_FLAG": "1",
        "SHIM_ATTENTION_MODE": "causal",
    }
    (tmp_path / "b").mkdir()
    p31 = _pkg(
        tmp_path / "b",
        "lightning_31b_shim.py",
        CFG_31B,
        files=("tokenizer.json", "serve_config_two_way.json"),
    )
    tok = os.path.join(p31.dir, "tokenizer.json")
    assert D.shim_argv(p31, two_way=False) == [
        "--config",
        p31.config_path,
        "--port",
        "0",
        "--vllm",
        INPROCESS_URL,
        "--tokenizer",
        tok,
    ]
    assert D.shim_argv(p31, two_way=True)[-3:] == [
        "--config-two-way",
        os.path.join(p31.dir, "serve_config_two_way.json"),
        "--two-way-reading",
    ]
    assert D.shim_argv(p31, profile="two-way")[-2:] == ["--profile", "two-way"]
    assert D.shim_environment(p31, True, "/tmp/r", two_way_window=16384) == {
        "SHIM_READY_FILE": "/tmp/r",
        "SHIM_TWO_WAY_WINDOW": "16384",
    }
    assert (
        D.shim_environment(p31, True, None, profile="two-way", two_way_window=16384)
        == {}
    )
    os.unlink(os.path.join(p31.dir, "serve_config_two_way.json"))
    with pytest.raises(RuntimeError, match="serve_config_two_way.json"):
        D.shim_argv(p31, two_way=True)


def test_engine_defaults_4b(tmp_path):
    pkg = _pkg(tmp_path, "h2o_lightning_shim.py", CFG_4B, files=("tokenizer/",))
    d = package_engine_defaults(pkg)
    assert d.args == {
        "served_model_name": ["h2oai/h2o-lightning-4b"],
        "tokenizer": os.path.join(pkg.dir, "tokenizer"),
        "max_model_len": 40960,
        "attention_backend": "TRITON_ATTN",
        "max_num_batched_tokens": 40960,
        "limit_mm_per_prompt": {"image": 4, "video": 0},
        "mm_processor_kwargs": {"max_pixels": 1048576},
        "mm_processor_cache_gb": 0,
    }
    assert d.env == {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_GDN_PREFILL_GRAPH_PRECAPTURE": "1",
    }
    assert (
        package_engine_defaults(pkg, two_way=False).args["max_num_batched_tokens"]
        == 8192
    )
    # applied to the arguments left at vLLM's default; the command line wins
    args = Namespace(
        served_model_name=None,
        tokenizer=None,
        max_model_len=None,
        attention_backend=None,
        max_num_batched_tokens=16384,
        limit_mm_per_prompt={},
        mm_processor_kwargs=None,
        mm_processor_cache_gb=4,
    )
    env_before = os.environ.pop("VLLM_USE_FLASHINFER_SAMPLER", None)
    try:
        applied = apply_engine_defaults(args, d)
    finally:
        os.environ.pop("VLLM_USE_FLASHINFER_SAMPLER", None)
        os.environ.pop("VLLM_GDN_PREFILL_GRAPH_PRECAPTURE", None)
        if env_before is not None:
            os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = env_before
    assert (
        args.max_num_batched_tokens == 16384 and "max_num_batched_tokens" not in applied
    )
    assert (
        args.tokenizer == os.path.join(pkg.dir, "tokenizer")
        and args.attention_backend == "TRITON_ATTN"
    )
    assert (
        args.limit_mm_per_prompt == {"image": 4, "video": 0}
        and args.mm_processor_cache_gb == 0
    )
    assert applied["env:VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_engine_defaults_31b(tmp_path):
    pkg = _pkg(tmp_path, "lightning_31b_shim.py", CFG_31B)
    d = package_engine_defaults(pkg, two_way=False)
    assert d.args == {
        "served_model_name": ["h2oai/h2o-lightning-31b"],
        "max_logprobs": 20,
        "dtype": "bfloat16",
        "quantization": "fp8",
        "limit_mm_per_prompt": {"image": 4, "video": 0},
        "mm_processor_kwargs": {"max_soft_tokens": 560},
        "max_model_len": 131072,
    }
    d = package_engine_defaults(pkg, two_way=True, gpu_mib=81559)
    assert (
        d.args["attention_backend"] == "TRITON_ATTN" and d.args["max_num_seqs"] == 1024
    )
    assert (
        d.args["max_model_len"],
        d.args["max_num_batched_tokens"],
        d.args["h2o_decisions_two_way_window"],
    ) == (81920, 16384 + 1024, 16384)
    d = package_engine_defaults(pkg, two_way=True, gpu_mib=97887)
    assert (d.args["max_model_len"], d.args["h2o_decisions_two_way_window"]) == (
        131072,
        8192,
    )
    d = package_engine_defaults(pkg, two_way=True, gpu_mib=None)
    assert "max_model_len" not in d.args and any(
        "no measured sizes" in n for n in d.notes
    )
    # a profile: the shim's own printed arguments
    shim = types.SimpleNamespace(
        profile_engine=lambda name: (
            {
                "gpu": "one RTX PRO 6000",
                "quantization": "fp8",
                "max_len": 131072,
                "gpu_util": 0.92,
                "max_images": 4,
                "max_soft_tokens": 560,
                "attention_backend": "TRITON_ATTN",
                "max_num_seqs": 1024,
                "two_way_window": 8192,
            },
            {},
        ),
        profile_vllm_args=lambda e, served: [
            "--max-logprobs",
            "20",
            "--served-model-name",
            served,
            "--dtype",
            "bfloat16",
            "--quantization",
            e["quantization"],
            "--max-model-len",
            str(e["max_len"]),
            "--gpu-memory-utilization",
            "0.92",
            "--limit-mm-per-prompt",
            json.dumps({"image": 4, "video": 0}),
            "--mm-processor-kwargs",
            json.dumps({"max_soft_tokens": 560}),
            "--attention-backend",
            "TRITON_ATTN",
            "--max-num-seqs",
            "1024",
            "--max-num-batched-tokens",
            "9216",
        ],
    )
    d = package_engine_defaults(pkg, profile="two-way", shim_module=shim)
    assert (
        d.args["max_model_len"] == 131072 and d.args["max_num_batched_tokens"] == 9216
    )
    assert d.args["gpu_memory_utilization"] == 0.92 and d.args["served_model_name"] == [
        "h2oai/h2o-lightning-31b"
    ]
    assert parse_vllm_argv(["--a-b", "1", "--flag", "--c", "x"]) == {
        "a_b": 1,
        "flag": True,
        "c": "x",
    }
    with pytest.raises(ValueError):
        package_engine_defaults(pkg, profile="two-way")


def test_warm_up_bodies_follow_serve_sh():
    bodies = D.warm_up_bodies_4b(40960, images=True, two_way=True)
    assert [n for n, _ in bodies] == [
        "yes/no (bidirectional)",
        "three questions (bidirectional)",
        "long record (bidirectional)",
        "image (bidirectional)",
        "yes/no (causal)",
        "three questions (causal)",
        "long record (causal)",
        "image (causal)",
    ]
    assert (
        bodies[0][1]["attention_mode"] == "bidirectional"
        and bodies[4][1]["attention_mode"] == "causal"
    )
    assert bodies[2][1]["state"].count("warm-up record line ") == 40960 // 24
    assert (
        D.warm_up_bodies_4b(1_000_000, images=False, two_way=False)[2][1][
            "state"
        ].count("warm-up")
        == 7500
    )
    assert "attention_mode" not in D.warm_up_bodies_4b(4096, False, False)[0][1]


def test_decision_routes_go_before_the_engines_health():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm.entrypoints.h2o.api_router import attach_router
    from vllm.entrypoints.serve.instrumentator import (
        register_instrumentator_api_routers,
    )

    app = FastAPI()
    attach_router(app)
    register_instrumentator_api_routers(app)
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 503 and r.json()["warming"] is True
    assert client.post("/v1/systemone", json={}).status_code == 503
    assert client.get("/h2o/decisions").status_code == 503


# ---------------------------------------------------------------- parity with the released shims
def _package_or_skip(repo, revision):
    """The package files: a directory named by H2O_TEST_PACKAGE_4B / _31B, else
    the Hub cache, else a fetch (the 31B needs a token); skipped when none."""
    explicit = os.environ.get("H2O_TEST_PACKAGE_" + ("4B" if "-4b" in repo else "31B"))
    if explicit:
        return locate_package(repo, revision, explicit=explicit)
    try:
        return locate_package(repo, revision, download=False)
    except PackageError:
        pass
    try:
        return locate_package(repo, revision, download=True)
    except PackageError as e:
        pytest.skip(f"{repo} {revision} package files unavailable: {e}")


def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _LateDispatcher:
    """The dispatcher the layer is started with before the stand-in engine exists."""

    target = None

    def call(self, method, path, body=None):
        return self.target.call(method, path, body)


@pytest.fixture
def clean_env():
    saved = dict(os.environ)
    for k in list(os.environ):
        if k.startswith(("SHIM_", "LIGHTNING_")):
            del os.environ[k]
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _app_client(layer):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from vllm.entrypoints.h2o.api_router import attach_router

    app = FastAPI()
    attach_router(app)
    app.state.h2o_decisions = layer
    return TestClient(app)


# the stand-in engine's socket URL: the shim prints its engine URL in /health, and
# in-process that is the INPROCESS_URL marker (the one difference between the paths)
_FAKE_URL: list[str] = []


def _without_latency(body: bytes) -> bytes:
    """The body with usage.latency_ms removed (re-serialised as the shim does)."""
    try:
        obj = json.loads(body)
    except ValueError:
        return body
    if isinstance(obj, dict) and isinstance(obj.get("usage"), dict):
        obj["usage"].pop("latency_ms", None)
        return json.dumps(obj).encode()
    return body


def _compare(client, shim_url, method, path, body=None, ctype="application/json"):
    """The same request to the standalone shim and through the layer: identical status and body."""
    headers = {"Content-Type": ctype} if body is not None else {}
    sa_status, sa_body, sa_ctype = _http(shim_url + path, method, body, headers)
    for url in _FAKE_URL:
        sa_body = sa_body.replace(url.encode(), INPROCESS_URL.encode())
    r = client.request(method, path, content=body, headers=headers)
    # usage.latency_ms is the shim's wall clock: the one field that differs between
    # the two paths (the in-process one is the faster); everything else is compared
    # byte for byte
    in_body = _without_latency(r.content)
    sa_body = _without_latency(sa_body)
    if (r.status_code, in_body) != (sa_status, sa_body):
        i = next(
            (k for k, (a, b) in enumerate(zip(r.content, sa_body)) if a != b),
            min(len(r.content), len(sa_body)),
        )
        pytest.fail(
            f"{method} {path}: in-process HTTP {r.status_code} vs standalone {sa_status}; "
            f"first difference at byte {i}:\n  in-process: {r.content[max(0, i - 120) : i + 120]!r}\n"
            f"  standalone: {sa_body[max(0, i - 120) : i + 120]!r}"
        )
    assert r.headers.get("content-type") == sa_ctype
    return sa_status, (json.loads(sa_body) if sa_body else None)


def _decisions(include_images):
    noul = {
        "type": "noul",
        "instructions": "Is it urgent?",
        "criteria": {"true": "yes", "false": "no"},
    }
    out = [
        (
            "POST",
            "/v1/systemone",
            {
                "state": {"ticket": "the printer on floor 3 is down"},
                "questions": {
                    "c": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": {"a": "billing", "b": "support", "c": "sales"},
                    },
                    "n": noul,
                    "s": {
                        "type": "score",
                        "instructions": "How urgent?",
                        "criteria": ["low", "medium", "high"],
                    },
                },
            },
        ),
        (
            "POST",
            "/v1/systemone",
            {
                "state": "MOCK_LP=[-0.1,-2.3,-0.7] a record",
                "questions": {
                    "c": {
                        "type": "choice",
                        "instructions": "Which?",
                        "criteria": {"x": "1", "y": "2", "z": "3"},
                    }
                },
            },
        ),
        (
            "POST",
            "/v1/systemone",
            {"state": "a record " * 2000, "questions": {"d": noul}},
        ),
        ("POST", "/v1/systemone", {"state": "MOCK_OVERFLOW", "questions": {"d": noul}}),
        (
            "POST",
            "/v1/systemone",
            {"state": "x", "questions": {"d": {"type": "wat", "instructions": "?"}}},
        ),
        ("POST", "/v1/systemone", {"questions": {}}),
        ("POST", "/api/alpha/decisions", {"state": "x", "questions": {"d": noul}}),
        (
            "POST",
            "/v1/decisions",
            {
                "model": "h2oai/h2o-lightning-4b",
                "input": "the printer is down",
                "questions": [
                    {"type": "predicate", "instructions": "Is it urgent?"},
                    {
                        "type": "choice",
                        "instructions": "Which team?",
                        "choices": [{"value": "billing"}, {"value": "support"}],
                    },
                    {
                        "type": "score",
                        "instructions": "How urgent?",
                        "levels": [{"label": "low"}, {"label": "high"}],
                    },
                ],
            },
        ),
        (
            "POST",
            "/v1/systemone",
            {"attention_mode": "bidirectional", "state": "x", "questions": {"d": noul}},
        ),
    ]
    if include_images:
        out.append(
            (
                "POST",
                "/v1/systemone",
                {
                    "state": {"images": [D.WARMUP_PNG], "note": "see the image"},
                    "questions": {"d": noul},
                },
            )
        )
    return out


def test_4b_parity_with_the_released_shim(clean_env, tmp_path):
    pkg = _package_or_skip("h2oai/h2o-lightning-4b", "v1.4.1")
    assert pkg.known and pkg.known["tag"] == "v1.4.1"
    ready = tmp_path / "ready"
    late = _LateDispatcher()
    # the layer first: its module is the shim file under its own name; the package's test suite then
    # imports that module and turns its tokenizer contract off (the stand-in has a toy vocabulary)
    layer = D.start_layer(pkg, late, two_way=False, ready_file=str(ready))
    assert layer.module is sys.modules["h2o_lightning_shim"]
    T = _import_file("h2o_pkg_test_shim_4b", os.path.join(pkg.dir, "test_shim.py"))
    assert T.S is layer.module and layer.module.TOKENIZER_CONTRACT is None
    T.FakeVLLM.reset()
    late.target = BridgeDispatcher(T.FakeVLLM)
    S = layer.module
    fake_srv, fake_url = _serve(T.FakeVLLM)
    _FAKE_URL.append(fake_url)
    with open(pkg.config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_dir"] = pkg.dir
    standalone = S.Shim(
        S.Contract(cfg),
        layer.http_vllm_cls(fake_url, cfg["model"]),
        4096,
        None,
        pkg.dir,
    )
    shim_srv, shim_url = _serve(S.make_handler(standalone))
    client = _app_client(layer)
    try:
        status, out = _compare(client, shim_url, "GET", "/health?live=1")
        assert status == 200 and out["ok"] is True
        status, out = _compare(client, shim_url, "GET", "/health")
        assert status == 503 and out["warming"] is True
        seen = []
        for method, path, body in _decisions(include_images=bool(cfg.get("image"))):
            status, out = _compare(
                client, shim_url, method, path, json.dumps(body).encode()
            )
            seen.append((path, status))
        assert (
            seen[0] == ("/v1/systemone", 200)
            and seen[1][1] == 200
            and seen[2][1] == 200
        )
        assert seen[3][1] == 422 and seen[4][1] == 422 and seen[5][1] in (400, 422)
        assert seen[6][1] == 200 and seen[7][1] == 200
        assert (
            seen[8][1] == 503
        )  # two-way asked: the release's probe answers are not the stand-in's
        _compare(client, shim_url, "POST", "/v1/systemone", b"not json")
        # the warm-up serve.sh runs, through the layer, then ready on both sides (the same ready file)
        D.run_warm_up(layer, 4096, live_deadline_s=30)
        assert layer.warmup["state"] == "done", layer.warmup
        assert ready.exists()
        status, out = _compare(client, shim_url, "GET", "/health")
        assert status == 200 and out["ok"] is True and "warming" not in out
        assert client.get("/h2o/decisions").json()["ready"] is True
    finally:
        shim_srv.shutdown()
        fake_srv.shutdown()
        _FAKE_URL.clear()


def test_31b_parity_with_the_released_shim(clean_env, tmp_path):
    pytest.importorskip("tokenizers")
    pkg = _package_or_skip("h2oai/h2o-lightning-31b", "v1.2.2")
    assert pkg.known and pkg.known["tag"] == "v1.2.2"
    ready = tmp_path / "ready"
    late = _LateDispatcher()
    # the package's own test suite first: its stand-in engine cannot serve the shipped
    # wide-label pool, the readout block or the image block, so its narrow config (what
    # that suite tests the shim with) is served on both sides; it patches nothing on
    # the shim module (the 4B suite does, which is why that test loads the layer first)
    T = _import_file("h2o_pkg_test_shim_31b", os.path.join(pkg.dir, "test_shim.py"))
    config_path = str(tmp_path / "serve_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(T.CFG, f)
    # no --tokenizer either: the shim checks a local tokenizer against the engine's
    layer = D.start_layer(
        pkg,
        late,
        two_way=False,
        ready_file=str(ready),
        argv=["--config", config_path, "--port", "0", "--vllm", INPROCESS_URL],
    )
    assert layer.module is sys.modules["lightning_31b_shim"]
    T.FakeVLLM.reset()
    late.target = BridgeDispatcher(T.FakeVLLM)
    S = layer.module
    fake_srv, fake_url = _serve(T.FakeVLLM)
    _FAKE_URL.append(fake_url)
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_dir"] = str(tmp_path)
    contract = S.Contract(cfg, dict(os.environ))
    standalone = S.Shim(contract, layer.http_vllm_cls(fake_url, contract.model), 4096)
    shim_srv, shim_url = _serve(S.make_handler(standalone))
    client = _app_client(layer)
    noul = {
        "type": "noul",
        "instructions": "Is it urgent?",
        "criteria": {"true": "yes", "false": "no"},
    }
    try:
        status, out = _compare(client, shim_url, "GET", "/health?live=1")
        assert status == 200, out
        status, out = _compare(client, shim_url, "GET", "/health")
        assert status == 503 and out["warming"] is True
        for body in (
            {
                "state": {"ticket": "the printer on floor 3 is down"},
                "questions": {
                    "c": {
                        "type": "choice",
                        "instructions": "Which team?",
                        "criteria": {
                            "billing": "money",
                            "support": "help",
                            "sales": "deals",
                        },
                    },
                    "n": noul,
                    "s": {
                        "type": "score",
                        "instructions": "How urgent?",
                        "criteria": ["low", "medium", "high"],
                    },
                },
            },
            {
                "state": "MOCK_LOGITS=[-0.1,-2.3,-0.7] a record",
                "questions": {
                    "c": {
                        "type": "choice",
                        "instructions": "Which?",
                        "criteria": {"x": "1", "y": "2", "z": "3"},
                    }
                },
            },
            {"state": "MOCK_OVERFLOW", "questions": {"d": noul}},
            {"state": "x", "questions": {"d": {"type": "wat", "instructions": "?"}}},
            {
                "model": "h2oai/h2o-lightning-31b",
                "input": "the printer is down",
                "questions": [
                    {"type": "predicate", "instructions": "Is it urgent?"},
                    {
                        "type": "choice",
                        "instructions": "Which team?",
                        "choices": [{"value": "billing"}, {"value": "support"}],
                    },
                ],
            },
        ):
            path = "/v1/decisions" if "input" in body else "/v1/systemone"
            _compare(client, shim_url, "POST", path, json.dumps(body).encode())
        _compare(client, shim_url, "POST", "/v1/systemone", b"not json")
        ready.write_text("")
        status, out = _compare(client, shim_url, "GET", "/health")
        assert status == 200
    finally:
        shim_srv.shutdown()
        fake_srv.shutdown()
        _FAKE_URL.clear()
