# Serving with h2oai/vllm: every fork feature and every recommended engine setting

This page is for anyone who serves a model on this fork, H2O-Lightning-4B and H2O-Lightning-31B in particular. Part 1
lists what the fork changes in vLLM 0.30.0: each feature, how you turn it on, and its default. Part 2 lists the engine
settings we recommend per model and reading mode. Most of those are stock vLLM arguments rather than fork code, but each
one was measured, and the numbers are given with their source. Part 3 lists what was tried and what is still open.

It describes the release **h2o-v0.30.0.3** (commit `b05cc29ab8`). The packages pin their own commit (see Part 2).
Install and build: [README.md](README.md).

Words used below:
- **one-way**: a stock causal read.
- **two-way**: the model's full-attention layers read the prompt both ways (`attention_mode: bidirectional`). The
  released checkpoints were trained for both readings.
- **package**: the `serve.sh` and shim shipped in a model repository.
- **hosted**: our own API, which runs this release.

## Part 1. Fork features

| Feature | What it does | Switch | Default | Commits |
|---|---|---|---|---|
| Per-request attention (two-way) | A request sent with `vllm_xargs {"attention_mode": "bidirectional"}` is prefilled with the full-attention layers reading both ways. It runs on the same generate-mode engine as ordinary causal requests, in the same batch. Sliding-window and chunked-local layers stay causal: Gemma-4's 50 sliding layers keep the trained causal reading. Aliases: `two_way`, `one_way`, `causal`. | Per request: `vllm_xargs` `attention_mode`. Needs `max_tokens=1`, because the readout is the prefill's next-token distribution. | causal | `236568a780` .. `a5251b4edb` |
| Admission rules for two-way | A two-way prompt is prefilled **whole** in one step, so a bigger prompt can never fit. Any of these is finished with `finish_reason "error"`, never left hanging or cut: a prompt above the step budget (`--max-num-batched-tokens`, less any draft slots); a prompt the encoder budget or cache can never hold. A prompt that only misses this step's remaining budget is deferred. After `BIDIRECTIONAL_MAX_DEFERRALS` (8) deferrals, nothing queued behind it is admitted until it runs. Two-way requests never write the prefix cache (their KV is not causal), and get no Mamba block-aligned split. | Automatic for flagged requests | | `30d5e0f77d`, `3fd477eebb`, `d4c4dfa4b0`, `89ff4133fc`, `056623112f` |
| Backend guard for two-way | A flagged request is refused at the frontend (HTTP 400, naming the reason) when the engine cannot apply the flag: any backend but `TRITON_ATTN`, or `FLASH_ATTN` with FlashAttention 4 (see the next row); a `backend_per_kind` override to another backend; a KV connector; `max_tokens != 1`; a model with parametric attention sinks or a value head size of its own (DiffKV). An unflagged request is never checked. | `--attention-backend TRITON_ATTN` (recommended), or FA4 as below | | `4a712a2cd4`, `bfb47cf311`, `216efae81c`, `a5251b4edb` |
| FlashAttention 4 for two-way (Hopper only) | `FLASH_ATTN` applies the per-sequence causal flag through FA4's `dynamic_causal`, with its own JIT cache. **Hopper (SM 9.x) only:** the vendored FA4 asserts SM90. Blackwell and Ampere are refused by the guard. Every head size the model declares must resolve to FA4: Gemma-4's head 512 resolves to FA2 on Blackwell. Status: in this release, opt-in only. Triton stays the default. The FA2 split variant is parked and not in this release. | `--attention-config '{"backend":"FLASH_ATTN","flash_attn_version":4}'` (the version must be set explicitly) | off | `56d487a24a`, `8373fdc012`, `216efae81c`, `ba6033e842`, `d81d9354e2`, `773da0fc2f` |
| Gemma-4 KV-sharing fast prefill with two-way (E-series, for example E4B) | The KV-shared full-attention layers get the per-sequence flag over the logits rows; the sliding ones get plain causal. Cross-decoder buffers are sized to the logits rows, not to the whole step. | Automatic when the model uses KV sharing and fast prefill | | `05e5ed9964`, `17d5ec2393`, `6270012a96` |
| CUDA graphs for two-way batches (V2 runner) | A batch that holds a two-way request replays PIECEWISE graphs. Attention is a splitting op and reads the step's flag. FULL graphs stay off for such a batch, because a captured FULL graph holds the causal kernel. | Automatic | | `614f0ab4bd`, `cf31284480`, `0a4e237cdf` |
| **GDN prefill CUDA graphs** (Qwen3.5 / Gated DeltaNet hybrids, the 4B) | The GDN core (conv1d + gated delta rule + gated norm) is a splitting op of vLLM's piecewise graphs, so stock vLLM runs it eagerly: 24 layers x ~18 launches per step. For a step whose GDN batch is prefills only (no decodes, no speculative tokens), each layer replays a graph captured per (sequence count, token bucket). Sequence counts round up to 1/2/4/8/16; buckets go from 64 up to `MAX_TOKENS`. The result is bit-exact against eager. FlashInfer's SM90 kernel is used only when its host-side split for the bucket equals the eager one. Since `b05cc29ab8`, graphs captured against the memory profiler's throwaway KV cache are dropped when the real cache is bound, together with their private pool. h2o-v0.30.0.2 crashed a lane with `--max-num-seqs` 8 or 16 at start-up. | `VLLM_GDN_PREFILL_GRAPH_MAX_TOKENS` (largest bucket; 0 = off); `VLLM_GDN_PREFILL_GRAPH_MAX_SEQS` (most prefills per step that still replay; 1 = single-sequence only); `VLLM_GDN_PREFILL_GRAPH_PRECAPTURE=1` (capture every shape at a layer's first prefill, which a server warm-up absorbs) | MAX_TOKENS **2048**, MAX_SEQS **16**, PRECAPTURE **0** | `1555071802`, `b2d2b32090`, `81a5008f94`, `a874119ca6`, `c34405a635`, `b05cc29ab8` |
| Patch A: Mamba-align same-step skip | With prefix caching in Mamba "align" mode, stock vLLM defers a request whose cached prefix ends on a state block being written **in this step**, and then stops scanning the waiting queue. Under load that runs about one decision per step. The fork defers only that request; the scan goes on. Scheduling only, no numerics. | Always on | on | `c45f47de63` |
| Patch B: GDN initial state without a GPU sync | Rows without a cached state are zeroed with `torch.where` instead of a boolean-mask assignment. The mask called `nonzero()` and synchronised the GPU in each of the 24 GDN layers per step. Identical values. | Always on | on | in `1555071802` |
| Per-request `prefix_cache_write` | `vllm_xargs {"prefix_cache_write": false}` keeps a causal request out of the prefix cache as a writer. It still reads cached blocks but commits none, so with Mamba align caching its prefill is not split at the last state-block boundary and runs in one step. Stock vLLM ignores the field. | Per request | true (stock behaviour) | `a0acf966cb` |
| Compile (AOT) cache key includes zero modality limits | Stock 0.30.0 hashes the multimodal config without the per-prompt limits. A cache compiled with images off is then reused for an images-on start, which fails with `'NoneType' object has no attribute 'size'`. The fork puts a modality with a zero limit in the model hash. Other differences are still not in the key (sizes, build, reading mode), so keep one `VLLM_CACHE_ROOT` per engine configuration (Part 2). | Always on | on | `28c7087589` |
| Early exits / layer skipping | Not in any release. Research and prototype only. On the 4B, a prototype exit after layer 24 took 93.7 % of items at 99.5 % agreement with full depth and saved 4-23 ms per read (125-2,160 tokens). No package or hosted deployment ships it. | none | | (none) |

Everything else is upstream vLLM 0.30.0. The fork changes Python files only; `verify_wheel.py` checks the wheel file by
file against upstream.

## Part 2. Recommended engine settings, per model and reading mode

Sources:
- the packages' `serve.sh` at the release tags: `h2oai/h2o-lightning-4b` v1.4.0.2 (`2f49932376`) and
  `h2oai/h2o-lightning-31b` v1.2.2 (`bde62882`, profile `two-way` in `lightning_31b_shim.py`);
- the model cards;
- our measurement reports: GDN prefill speed (2026-10-25, one H100); engine parity (2026-10-25); the hosted switch to
  this release (2026-10-26).

"H100" is one H100 80 GB; "RTX PRO 6000" is one RTX PRO 6000 Blackwell 96 GB.

### H2O-Lightning-4B (Qwen3.5-4B hybrid: 24 GDN + 8 full-attention layers)

| Setting | Two-way (package default) | One-way (on this fork) | Measured effect, source |
|---|---|---|---|
| Fork commit | v1.4.0.2 pins `a874119ca6` (its `engine/vllm_per_request_attention.SHA256SUMS`); this release is newer and is a superset (adds patch A, the cache-key fix, `prefix_cache_write`, the graph fixes) | same | engine parity: the newer engine with the release's flags gave 600/600 bit-identical answers to the release pin, in both modes |
| `--attention-backend` | `TRITON_ATTN` (required for two-way: the guard) | package: `TRITON_ATTN` on the one engine; stock fallback: vLLM's default. Hosted: vLLM's default (FLASH_ATTN on H100) | FA4 two-way gains nothing on the 4B (host-bound); 4B package |
| `--max-num-batched-tokens` | `= MAX_LEN` (a two-way prompt is prefilled whole; the package passes it last so nothing overrides it) | `8192` (chunked prefill) | card "Serving at scale"; engine parity row 4 |
| `--max-cudagraph-capture-size` | vLLM default (512). The next package (v1.4.1) keeps 512 for two-way | `2048` | engine parity (H100): 2048 is exact; one-way 1k tokens 38.5 -> 34.6 ms, +7/+4/-1 % at c = 1/8/32; two-way flat; start +50 s, KV -1.2 %. A sparse list was not exact and was rejected. Hosted: on |
| GDN prefill graphs | `VLLM_GDN_PREFILL_GRAPH_PRECAPTURE=1`, MAX_TOKENS 2048 (default) | same | GDN speed (H100): two-way served p50 at 125 / 512 / 2,160 tokens 40.7 / 42.3 / 51.3 -> 14.8 / 20.6 / 52.2 ms; c = 1/8/32 at 125 tokens 24.7 / 95.1 / 277 -> 67.8 / 162 / 339 req/s; graph vs eager max \|dp\| 0.0 on 600 items; JB231 206 = 206. Cost: +1.1 GB after KV sizing at 2048 (+0.5 GB at 1024); a 24 GB GPU can set MAX_TOKENS=1024. Hosted, 2026-10-26 (H100, live traffic, c = 1): p50 46.2 -> 20.9 ms, singles 26.4 -> 42.7/s, parity vs a two-read live floor 0 flips |
| `--max-num-seqs` | vLLM default | vLLM default; with two lanes on one card: 128 (short) / 8 (long) | engine tuning (gateway lanes) |
| `--quantization` | none (BF16; the temperatures in `serve_config.json` were fitted on BF16 output) | none; `fp8` is an option (Ada/Hopper/Blackwell only) | FP8 at c = 64: 81.6 -> 99.2/s; 21/687 dev flips, 19 of them at a stock margin under 0.1; McNemar p = 0.63; median \|dp\| 0.009, max 0.246. Hosted: fp8 |
| `--mamba-ssm-cache-dtype` | model default | model default; `bfloat16` is an option | c = 16: 78.1 -> 81.7/s (+3-5 %); prefix block 528 -> 272 tokens; 0 dev flips with patches A+B, max \|dp\| 0.022. Not yet parity-checked together with the GDN graphs. Hosted: bfloat16 |
| `--max-model-len` | `MAX_LEN`, default 40960 | same; hosted short lane 16384, long lane auto (131072 on an H100) | 4B package |
| `--gpu-memory-utilization` | 0.90 | 0.90 (hosted: 0.45 + 0.40 for two lanes) | 4B package |
| `--tokenizer` | `<model>/tokenizer` (the training `Qwen2Tokenizer`) | same | From transformers 5.18 the default load is `Qwen3_5Tokenizer`, which splits 8 of 24 Devanagari/Arabic probes differently. JB231 206 with either tokenizer; DI 0.3: 196 of 158,028 rows tokenize differently, hit rate 0.851 vs 0.877 on those rows (4B card, "Tokenizer") |
| Images | `--limit-mm-per-prompt '{"image": 4, "video": 0}'`, `--mm-processor-kwargs '{"max_pixels": 1048576}'` (= `serve_config.json` image.max_pixels; when the engine cap differs from the cap the shim sends, the first image request is very slow) | same | 4B package; image limit 16 costs no KV (engine tuning, A6000) |
| `--mm-processor-cache-gb 0` | from v1.4.1 | from v1.4.1; hosted: on | The front-end image cache drifted from the engine's on the first image after a start (a 500). Off: answers exact, an image 3-question decision at c = 8 up to ~12 % slower (engine parity row 7) |
| `--max-logprobs` | not needed (exact label-id reads) | not needed | |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | yes (reads never sample; FlashInfer's JIT sampler fails with older system CUDA toolkits) | yes | 4B package |
| `VLLM_CACHE_ROOT` | one folder per engine configuration (v1.4.1) | same; hosted: per fork commit, port and arguments | the cache-key note in Part 1 |
| Patches A/B as files | not with this fork (it carries them; the package refuses `VLLM_PATCHES=1` on the one engine) | stock fallback only | engine tuning: stock 29.0/s vs A+B 78-81/s at c = 16/64 (one-way siblings); patch A on this fork: 600/600 exact, +7 % at c = 16, +10 % at c = 32 (engine parity) |

### H2O-Lightning-31B (Gemma-4 31B: 50 sliding + 10 full-attention layers)

| Setting | Two-way, profile `two-way` (package default, RTX PRO 6000) | Two-way on an H100 | One-way (stock fallback; hosted) | Measured effect, source |
|---|---|---|---|---|
| Fork commit | v1.2.2 pins `a874119ca6` | same | stock 0.30.0 is enough; hosted runs this release | 31B package |
| `--quantization` | `fp8` (W8A8 dynamic; the benchmarked setting), `--dtype bfloat16` | same | same | 31B package |
| `--attention-backend` | `TRITON_ATTN`; FA4 is an opt-in on Hopper only | `TRITON_ATTN`, or FA4 (`--attention-config '{"backend":"FLASH_ATTN","flash_attn_version":4}'`) | vLLM default | FA4 on an H100: JB231 214 vs 213 (Triton), decisions/s +21-24 %, c = 1 224 vs 265 ms, start +40 s (engine parity row 10) |
| `--max-model-len` | 131072 | 81920 (one H100 in FP8; the Decision Index's longest text request is ~75,100 tokens) | 131072 (hosted: MAXLEN auto, 98304 on the RTX PRO 6000 at util 0.90) | 31B package; hosted 2026-10-26: M = 106,059 KV tokens -> 98304 |
| Two-way window (largest prompt read two-way; must be one prefill step) | 8192 | 16384 | n/a | Keep it small and the context long. With `--max-num-batched-tokens = --max-model-len` a 64k step needed 55 GiB of KV: sliding layers hold window + one step per request. A flagged 16k read costs 3.4 s on an H100 (8k 1.2 s), so the shim's latency cap (4,096 summed tokens) decides which requests are read two-way |
| `--max-num-batched-tokens` | window + `--max-num-seqs` = 9216 | 16384 + seqs | vLLM default (8192 at 70 GB and above); hosted 8192 | 31B package: a window-sized two-way prompt runs beside a full decode batch |
| `--max-num-seqs` | 1024 | 1024 | vLLM default; hosted 256 | 31B package |
| `--gpu-memory-utilization` | 0.92 | 0.92 | 0.90 | RTX PRO 6000: 51.31 GiB KV, 270,589 tokens at step 9,216; H100: 36.1 GiB, 92,318 tokens (31B package) |
| Images | 4 per prompt, `--mm-processor-kwargs '{"max_soft_tokens": 560}'`, no video | same | same | 31B package; a 1120-token re-read of unsure image questions is a shim option of the `decision-index` profile |
| `--max-logprobs` | 20 (letters read from the top 20; 21+ options by token id) | same | same | 31B package |
| `--mm-processor-cache-gb 0` | from the next package | same | hosted: on | engine parity row 7 |
| `VLLM_CACHE_ROOT` | one folder per fork commit, reading mode, MAX_LEN, window, image setting and quantization | same | same | 31B package (vLLM's AOT key misses these) |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | yes | yes | yes | 31B package |
| transformers | 5.20.0 (the measured set) | 5.20.0 | package 5.20.0; hosted pins 5.19.0 | 31B package `PROFILE_ENGINE` |

### Any other model on this fork

- Two-way needs `--attention-backend TRITON_ATTN` (or FA4 on Hopper) and a step budget that holds the longest
  prompt you want read two-way.
- The GDN graphs apply to Gated DeltaNet layers only. Other models are unaffected.
- Keep one `VLLM_CACHE_ROOT` per engine configuration.
- The flags above are measured on the two models named; re-measure before you copy a number to another model.

## Part 3. Tried and rejected, and open

- Rejected:
  - a capture list up to 4,096 for the GDN graphs: 2,160-token steps got slower (52.5 ms) and start-up took about
    2.5 minutes longer;
  - a sparse capture list: not exact;
  - the fine-grained Mamba prefix cache: -12 % throughput, +23 ms at c = 1;
  - vLLM 0.31: still one decision per step under load;
  - pinned GPU clocks: no measurable gain on an H100.
- Open:
  - an H100 run of the GDN graphs with `--mamba-ssm-cache-dtype bfloat16` together;
  - an H100 proof of h2o-v0.30.0.3 at the hosted shapes (the live US switch passed its gates);
  - FA4 for the 31B on a Hopper host as a package default (the decision so far: Triton stays the default);
  - early exits (Part 1).
