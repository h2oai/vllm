# The decision API inside the fork: `vllm serve <model> --h2o-decisions`

Status: branch `h2o/decisions-endpoint` (off `b05cc29ab8`, h2o-v0.30.0.3). CPU-tested, and GPU parity with the released
shims measured on one RTX PRO 6000 96 GB (section 4.4). The goal (2026-10-26): "ideally our vllm fork can handle all shim stuff and the shim could go away in the
limit of using it for on prem or as endpoint, while for cloud one might want still gateway to handle things."

## 1. What it does

`vllm serve h2oai/h2o-lightning-4b --h2o-decisions` (and `h2oai/h2o-lightning-31b`) serves the package's decision
API on the vLLM port, with the released shim's behaviour:

| route | what answers |
|---|---|
| `POST /v1/systemone`, `POST /api/alpha/decisions` | the shim's decision handler (the wire the cards document) |
| `POST /v1/decisions` | the OpenAI Decisions shape, mapped by the shim onto the same decision |
| `GET /health` | the shim's health: the backend checks (served model and context, every label one token at the answer slot, the tokenizer contract), the two-way mode probe, and 503 `"warming": true` until the package's warm-up has passed (`?live=1` answers meanwhile; the 31B's `?probe=1` re-runs its mode probe) |
| `GET /health/engine` | the engine's own health check (what `/health` is on a stock server) |
| `GET /h2o/decisions` | the layer's state: the package, the pinned shim, the warm-up steps |
| everything else | vLLM's own API, unchanged (`/v1/completions`, `/v1/chat/completions`, `/tokenize`, `/v1/models`, metrics) |

Prompt render, label rows and pooled readout, temperatures and the yes/no floor, the two-way per-request attention with
its window and one-way fallback, the start-up mode probes, Box (guided JSON), images (data URIs, multipart, safe URL
fetch), the BM25 state cut and the never-fail head+tail cut, `usage` = the full input tokens, and every `SHIM_*` /
`LIGHTNING_*` control: all of it is the shipped shim code running in this process. Nothing is re-derived.

## 2. Architecture

```
client ──HTTP──> uvicorn/FastAPI (vllm serve)
                   ├── /v1/decisions, /v1/systemone, /api/alpha/decisions, /health   (vllm/entrypoints/h2o/api_router.py)
                   │       │  raw request (method, path+query, Content-Type, body bytes)
                   │       ▼
                   │   HandlerBridge  (bridge.py): the shim's BaseHTTPRequestHandler on an in-memory request,
                   │       │           run in a worker thread; its reply bytes become the HTTP response
                   │       ▼
                   │   the released shim module, imported from the model package, built by its own main()
                   │       │  (decisions.py: VLLM and Server swapped, serve.sh's argv and environment)
                   │       ▼
                   │   InProcessVLLM.call(method, path, body)  (transport.py: a subclass of the shim's VLLM)
                   │       │
                   │       ▼  asyncio.run_coroutine_threadsafe(..., the server's loop)
                   │   EngineDispatcher: the same serving handlers the HTTP routes call
                   │       (openai_serving_completion / _chat, serving_tokenization, openai_serving_models),
                   │       the same request models, the same JSON bodies and error shapes
                   └── vLLM's own routes
```

Three pieces, each small and testable without a GPU:

1. **bridge.py.** A shim is a standard-library `BaseHTTPRequestHandler`. The bridge builds the request bytes (request
   line, `Content-Type`, `Content-Length`, body), runs `handle_one_request()` on a handler instance whose `rfile`/`wfile`
   are in-memory buffers, and parses the reply with `http.client`. Status codes, bodies, multipart parsing, the
   `/health` query flags and the error mapping are the shim's own, byte for byte (test: the same bytes as over a
   socket).
2. **transport.py.** The shim reaches "vLLM" through one class, `VLLM`, whose `call(method, path, body)` returns
   `(status, parsed JSON)`; `post`, `refuse`, `tokenize`, `label_logprobs`, `label_logprobs_chat`, `chat_generate`
   and `embed` build on it. `make_inprocess_vllm` subclasses the shim's own `VLLM` and overrides only `call`.
   `EngineDispatcher` routes `GET /v1/models`, `POST /tokenize`, `/detokenize`, `/v1/completions` and
   `/v1/chat/completions` to the serving handlers, validates the body with the same pydantic request models,
   and returns what the route's `JSONResponse` would have sent (`model_dump()` rendered and parsed back with
   Starlette's settings), including `ErrorResponse` bodies and the exception handlers' statuses. Streaming is refused
   (the shims never stream). `/v1/embeddings` (the retired two-engine pooling path) is a 404; the one-engine
   two-way reading does not use it.
3. **decisions.py.** `start_layer` imports the shim file under its own module name (the package directory first on
   `sys.path`, `__file__` beside the siblings, so the shim's own sha256 checks of `box.py` / `assets.py` /
   `pooled_readout.py` / `tokenizer_contract.json` run as shipped), swaps `VLLM` for the in-process class and `Server`
   for a stand-in that only keeps the request handler, sets the environment `serve.sh` exports (4B: `SHIM_TWO_WAY_FLAG=1`,
   `SHIM_ATTENTION_MODE=causal` for one-way; 31B: `SHIM_TWO_WAY_WINDOW`; both: `SHIM_READY_FILE`) and runs the shim's
   `main()` with the arguments `serve.sh` passes (4B: `--config serve_config.json --model-dir <dir> --vllm <engine>`;
   31B: `--config serve_config.json --vllm <engine> --tokenizer tokenizer.json` plus `--config-two-way
   serve_config_two_way.json --two-way-reading`, or `--profile NAME`). So the contract, the 31B's profile checks and
   environment rules, the one-engine flag and the 31B's start-up engine check are the released code paths. The
   warm-up `serve.sh` runs before it reports ready is replayed through the bridge (4B: the four bodies of `warm()`
   in each mode served; 31B: the package's own `warmup.py` with its `post` routed in-process), then the readiness
   file is created and `/health` turns 200. A failed text warm-up leaves `/health` at 503 with the error in the
   log and in `GET /h2o/decisions`.

**Engine settings.** `vllm serve` is started with the package's engine settings (what its `serve.sh` passes) for every
argument the command line leaves at vLLM's default (`engine_defaults.py`, applied before the engine is built;
`--no-h2o-package-defaults` turns it off). 4B: `--served-model-name`, `--tokenizer <package>/tokenizer`,
`--max-model-len 40960`, `--attention-backend TRITON_ATTN`, `--max-num-batched-tokens 40960` (a two-way prompt is
prefilled whole), `--max-cudagraph-capture-size 512` (2048 one-way), `--allowed-media-domains media.invalid`,
`--limit-mm-per-prompt`, `--mm-processor-kwargs max_pixels`, `--mm-processor-cache-gb 0`, and the environment `VLLM_USE_FLASHINFER_SAMPLER=0`, `VLLM_GDN_PREFILL_GRAPH_PRECAPTURE=1`.
`--no-h2o-decisions-two-way` is serve.sh's `ONE_ENGINE=1 TWO_WAY=0`: the same one engine (TRITON, a step budget of the
whole context), answering one-way by default; only the step graphs grow to 2048. The step-graph size is explicit
because vLLM's own default is 1024 on compute capability 10.x. `--allowed-media-domains media.invalid` is serve.sh's
setting; it matters more here because vLLM's own routes share the public port, so its chat route must not fetch URLs
for a caller (the shim fetches image links with its own checks; the 31B gets the flag too). 31B: `--max-logprobs
20`, `--dtype bfloat16`, `--quantization fp8`, the image limits with `max_soft_tokens 560`, and for the two-way reading
`--attention-backend TRITON_ATTN`, `--max-num-seqs 1024` and the measured sizes by GPU memory class (80 GB: context
81920, window 16384; 96 GB: 131072 / 8192; another GPU: pass `--max-model-len` and `--h2o-decisions-two-way-window`),
or the profile's arguments as the shim prints them (`--h2o-decisions-profile`). `--gpu-memory-utilization` is never
set here (serve.sh uses 0.90 for the 4B and 0.92 for the 31B two-way). `VLLM_CACHE_ROOT` is not set either: keep one
compile cache per engine configuration as the packages do.

**Pinning.** The shim is run only when the sha256 of its entry file is in `KNOWN_SHIMS` (`package.py`: the 4B v1.4.1
`h2o_lightning_shim.py`, the 31B v1.2.2 `lightning_31b_shim.py`); another revision is refused with the sha256, unless
`VLLM_H2O_DECISIONS_ALLOW_UNPINNED=1`. The shim runs code from the model repository and the transport relies on its
`VLLM`, `Server`, `main` names, so a new package tag is a one-line change here after its parity run.

**Threads.** A decision runs on the layer's thread pool (`SHIM_REQUEST_THREADS`, 64); the shim fans its questions out on
its own pool (`SHIM_QUESTION_THREADS`, 64); every engine call is a coroutine on the server's loop, so concurrent
questions batch in the engine as they do today. `SHIM_TIMEOUT` (180 s) bounds an engine call as in the shim; a call
that reaches it is also cancelled in the engine (the shim has given up on it).

## 3. What moves into the fork, what stays in the gateway

| function | package shim today | in the fork (this branch) | hosted gateway (cloud) |
|---|---|---|---|
| prompt render, label ids, temperatures, floor, readout, two-way flag and probes, Box, assets, images, state cut, usage | the shim beside the weights, one process per engine | the same shim, in the server process; one artifact `fork + package` | the gateway's `native_core` copy becomes unnecessary: it forwards `/v1/systemone` to the host's decision route and bills from `usage` |
| warm-up and readiness | `serve.sh` | the layer, same bodies, `/health` 503 until passed | the prober reads the host's `/health` |
| engine flags | `serve.sh` | package defaults applied to `vllm serve` | hosted lanes keep their own flags (`hosted_engine_differences.json`); `package_flags_test` compares them |
| keys, rate limits, admission and queueing across hosts, routing by length / region / model, failover, spend and billing records, analytics, retention | - | not here | stays in the gateway (the 2026-10-10 decision: the gateway's workers and hooks carry load and analytics; multi-model logic lives in one place) |
| cross-model cascade (4B then 31B), second reads across models, degraded cross-model answers | - | not here (a same-model second read is the shim's) | the gateway orchestrates two host calls |
| image URL fetch | the shim's safe fetch (off on hosts) | the shim's (`LIGHTNING_IMAGE_URLS=off` on hosts) | the gateway fetches and passes data URIs |
| per-site live defaults (floor, state budget) | `SHIM_*` env, `kwargs` per request | the same env and request fields | request fields set by the gateway |

On-prem and the endpoint image: one process, one port, the card's configuration. Cloud: the gateway in front of the
same process, thinner by the readout copy and the Gemma profiles (DEPLOYMENT_LAYERS.md section 5.2).

## 4. How parity with the released shim is proven

Three layers of evidence, from CPU to GPU:

1. **Same code.** The shim file is the released one by sha256; its siblings are checked by the shim itself; the
   contract comes from the package's `serve_config*.json` through the shim's own `main()`. The in-process class
   overrides one method (`call`).
2. **Same bytes on the same engine replies (CPU, `tests/entrypoints/h2o/test_decisions_endpoint.py`).** The released
   shim is run two ways against one stand-in engine (the package's own `test_shim.FakeVLLM`): standalone, as
   `serve.sh` runs it (its `ThreadingHTTPServer`, its HTTP client, a socket to the stand-in) and through this layer
   (FastAPI test client, the bridge, the in-process transport, the stand-in driven in memory). For `/health` (live,
   warming, ready), `/v1/systemone` (choice, yes/no, score, several questions, a long record, an image, an
   over-context record, a malformed body, an unknown type), `/v1/decisions` and `/api/alpha/decisions`, the status
   and the body bytes must be identical. The engine dispatcher is tested separately against fake serving handlers
   (request models, `model_dump()` bodies, error shapes, the streaming refusal).
3. **Same answers on the GPU** (result: section 4.4). With one engine started once: (a) the released
   `serve.sh` path (engine + shim process) and (b) `vllm serve --h2o-decisions` must give identical JSON bodies on
   the kit's 600-question slice and on JB231 (two-way and one-way, batched and one at a time), with the KV size
   pinned as in `fork_exit_design.md` sections 9-11; the 4B card's JB231 206 and the 31B's 215 (gateway = shim
   today) are the numbers to reproduce; the Box check and an image decision in both paths; `/health` 503 until
   the warm-up, then 200 with the probe fields; latency c=1 and c=16 against `serve.sh` (the loopback hop goes
   away; the handler runs in-process).

### 4.4 GPU parity result (2026-10-27, one RTX PRO 6000 Blackwell 96 GB)

Setup:
- One engine, two paths. B = `vllm serve <package> --h2o-decisions` with the package defaults only. A = the released shim
  run standalone as `serve.sh` runs it, pointed at B's engine.
- 833 bodies per run: 600 decision-index kit questions (seeded shuffle), JevBench's 231 public items, one image decision
  and one Box question.
- Runs in the order A, B, A again (A2), one at a time and at 16 in flight.
- Compared: the full response JSON, ignoring only `usage.latency_ms` (wall time).

The 4B rows ran with B as the on-prem image (the h2o-v0.30.0.3 release image with this branch over it, the package at
its tag, `vllm serve ... --h2o-decisions`), and with A on the host. The 31B rows ran with B in the release image with
this branch over it.

| model, reading | one at a time: B vs A2 | 16 in flight: B vs A, max abs dp / flips (floor A vs A2) | JB231 A = B |
|---|---|---|---|
| 4B v1.4.1, two-way (default) | 715/833 byte-identical; the other 118 differ only in the last bits of the floats (max abs dp < 1e-6; the host's Python 3.10 vs the image's 3.12 in the shim's own arithmetic, the engine's log-probabilities equal) | 0.025 / 0 (0.025 / 0) | 205 = 205 (card 206) |
| 4B v1.4.1, one-way (`"attention_mode": "causal"`) | 713/833, the same last-bit kind | 0.027 / 0 (0.020 / 1) | 201 = 201 (card 202) |
| 31B v1.2.2, two-way (default; 96 GB class: context 131072, window 8192, util 0.92) | 832/833 (one row at dp 0) | 1e-5 / 0 (8e-6 / 0) | 214 = 214 (card 214) |
| 31B v1.2.2, one-way deployment (`--no-h2o-decisions-two-way` vs `READING=one-way`; measured one build earlier, before the cancellation and load changes) | 833/833 | 0.085 / 0 (0.122 / 0) | 214 = 214 |

The released 31B shim takes no per-request reading (its responses carry no `attention_mode`), hence the separate
one-way deployment.

Latency (A vs B) is unchanged:

| model, reading | one at a time, p50 | 16 in flight |
|---|---|---|
| 4B two-way | 21.5 vs 21.4 ms | 47.4 vs 47.3 requests/s |
| 4B one-way | | 130.3 vs 130.3 requests/s |
| 31B two-way | 50 ms either way | 13.7 vs 13.7 requests/s |

Startup: `/health` answered 503 `warming` from the layer's start until its warm-up passed, then 200 (4B image with a
fresh compile cache: 152 s and 168 s after the container start).

Client disconnect:
- 4B: a 48-question decision on a long record (6.9 s to complete) was dropped 0.3 s into the engine's work. The layer
  aborted its 41 in-flight engine reads, and vLLM's running and waiting counts were 0 within 58 ms.
- 31B two-way: a 64-question decision (24-27 s to complete) was dropped 50 ms into the engine's work, three times. All
  63-64 reads were aborted each time, and the counts were 0 within 1.15-1.17 s. That is one engine step at this load:
  9,216-token chunked-prefill steps at the measured 7,300-7,500 prompt tokens/s are about 1.2 s each, and an aborted
  request leaves the batch when the step in flight ends.

## 5. Open risks and what is not done

- **GPU coverage.** Section 4.4 covers the 4B and the 31B (default reading and one-way deployment) on one 96 GB GPU.
  Not run on a GPU: the 31B profiles (`--h2o-decisions-profile`), the 80 GB sizing, `--api-server-count > 1`.
- **`raw_request=None`.** The serving handlers get no Starlette request: request ids are generated and trace headers
  and `X-Request-Id` are absent. The decision routes themselves run under `with_cancellation` and `load_aware_call`
  (in vLLM's order), so a client disconnect cancels the decision's engine reads and, with
  `--enable-server-load-tracking`, decisions count in the server-load metric. The shim's own `/v1/models` body
  (`"readout": ...`) is not served; vLLM's is.
- **Validation errors** are rendered as `400` with an `error` object; the message text differs from the HTTP
  validation handler's. The shims only pattern-match context-limit errors (`_CONTEXT_ERROR`), which come from the
  serving layer with the same text.
- **Engine defaults** cannot see what the command line set when the value equals vLLM's default; `--gpu-memory-
  utilization` and `VLLM_CACHE_ROOT` are left to the user; the 31B two-way sizing needs an NVML read of the first GPU.
- **Process model.** The shim's thread pools live in the API server process (with `--api-server-count > 1` each
  server builds its own layer and runs its own warm-up; the readiness files are per process). Data-parallel engines
  were not considered.
- **31B profiles** refuse stray `SHIM_*` variables; the layer sets only `SHIM_READY_FILE` (allowed) and, for the 4B,
  `SHIM_TWO_WAY_FLAG` / `SHIM_ATTENTION_MODE`.
- **The gateway side** (forwarding, billing from `usage`, the prober on the host's `/health`, cascade over two host
  calls) is a separate change in the private gateway repository: this branch only gives it a host to talk to.
- **Shutdown.** The layer's threads are daemon threads; an in-flight decision at shutdown is dropped like an
  in-flight request of the shim is.

## 6. Place in the migration (ARCHITECTURE_LAYERS.md section 6, open-system-one, 2026-10-26)

This branch is **M1** of the migration M0-M4: the fork's native decisions endpoint at parity with the released package
shim, model by model. M1's gate is JB231 item by item identical to the package shim on the same engine in both read
modes (4B 206 two-way / 202 one-way on v1.4.1; 31B 215 two-way), the package's `test_shim.py` cases carried by the
fork's tests, and Box, images, never-fail and `usage.input_tokens` = the full input. Section 4 above is how the CPU
part of that gate is met today (the package's own stand-in engine, the released shim standalone vs in-process, byte
for byte); the GPU part is section 4.3. M2 (the package `serve.sh` gains the native path), M3 (the gateway forwards
decisions to the host's endpoint and bills from `usage`) and M4 (the gateway's readout code deleted) build on it.

The three open items of that plan, as this layer answers them:

**Lanes by length (hosted US: a short and a long engine on one GPU).** The layer serves one engine, like the package
shim; a lane is one `vllm serve --h2o-decisions` with its own `--max-model-len`. Routing stays the gateway's, by
request size, as today. What the endpoint gives it: `/health` reports `vllm_max_model_len` (the shim's field) and
`GET /h2o/decisions` the package's state budget, so the gateway knows each lane's capacity; a request over the lane's
context is answered as the shim answers it, HTTP 422 with the shim's "over the server's context limit" text (the
`_CONTEXT_ERROR` match on vLLM's error), which the gateway can re-route to the long lane at the cost of one failed
prefill attempt, or avoid with its size estimate. On the short lane the package's state budget (32,000 tokens on the
4B) is left as shipped: a long record must reach the long lane whole, not be cut on the short one (the cut is the
long lane's job and changes the answer). One engine per GPU per model, as the package runs, needs no routing at all.

**Cascade cancel (the 2026-10-16 form: start the 4B and the 31B together, drop the 31B when the 4B is confident).**
Implemented in this layer. A decision runs under a `DecisionContext`; every engine call the shim makes for it, from
whichever of its question threads (the shims' `ThreadPoolExecutor` is swapped for one that carries the context),
is registered on that context. The routes read the body and then run under vLLM's `with_cancellation`, which
listens for the client's disconnect; when it fires, the route's task is cancelled, the layer cancels the context,
every in-flight engine read is cancelled (vLLM's handler sees `CancelledError`, answers "Client disconnected" and the
engine aborts the request at the next step), later reads of that decision are refused before they reach the engine,
and the shim's threads finish with a 502 nobody reads. So the gateway cancels the 31B by closing its HTTP request,
which is what LiteLLM does when its own caller is done. Test: `test_a_cancelled_decision_aborts_its_engine_reads`.
To measure on the GPU (M3): a long 31B read cancelled after 50 ms must leave `vllm:num_requests_running` at 0 within
a step and cost no more than that step.

**API-server capacity.** The shim's CPU work (render, BM25, JSON) now shares the API server's process and its GIL
with vLLM's own front end (tokenisation, response building), where the standalone shim had its own process; the
loopback HTTP hop, its JSON round trip and the N per-question socket calls go away. Which wins is a measurement:
decisions/s and p50 at c = 1 / 8 / 32 for 1- and 3-question requests, `vllm serve --h2o-decisions` vs `serve.sh`
(engine + shim process) on one engine, and the API-server process CPU. If one process binds, vLLM 0.30's
`--api-server-count N` runs N API server processes over one engine: each process builds its own layer, runs its own
warm-up (N times the warm-up requests, once per process) and keeps its own readiness file, which is correct but
should be known; the engine batches across them as across any clients. That is the knob to try before any
restructuring; the M3 measurement of ARCHITECTURE_LAYERS.md section 5.3 (c=1 and 3-question p50, saturation, CPU on
the host) is the gate.

## 7. Files

- `vllm/entrypoints/h2o/{__init__,bridge,package,transport,decisions,api_router,engine_defaults}.py`
- `vllm/entrypoints/launchers/cli_args.py` (`--h2o-decisions`, `--h2o-decisions-package`, `--[no-]h2o-package-defaults`,
  `--[no-]h2o-decisions-two-way`, `--h2o-decisions-profile`, `--h2o-decisions-two-way-window`)
- `vllm/entrypoints/launchers/api_server/{routers,app_state,entry}.py` (the three hooks)
- `tests/entrypoints/h2o/test_decisions_endpoint.py`; `tools/h2o/smoke.sh` runs it.
