# h2oai/vllm builds

This fork is vLLM 0.30.0 plus per-request attention: a request sent with
`vllm_xargs {"attention_mode": "bidirectional"}` is prefilled with the model's full-attention layers reading both
ways, on the same generate-mode engine that serves ordinary (causal) requests. It covers the Triton attention backend,
FlashAttention 4 on Hopper GPUs (the `FLASH_ATTN` backend), and the Gemma-4 KV-sharing fast prefill. Unflagged
requests run exactly as in upstream vLLM.

Every fork feature with its switch and default, and the engine settings we recommend per model and reading mode
(with their measured effects): [SERVING.md](SERVING.md).

The fork changes Python files only (no C++, CUDA or Rust), so its wheels are built without a CUDA compiler: the
fork's Python over the compiled parts of the upstream v0.30.0 wheel (`VLLM_USE_PRECOMPILED=1`), on an ordinary CPU
machine in a few minutes. `verify_wheel.py` then checks every file of the result against the upstream wheel: all
compiled and data files identical, each fork-changed file equal to the fork commit, the same dependency pins.

## Install

Pick the CUDA variant of your driver: `cu129` (CUDA 12.9) or `cu130` (CUDA 13.0). Install torch for that variant
first, then the wheel from the [Releases](https://github.com/h2oai/vllm/releases) page:

```bash
pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 torchcodec==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu129
pip install vllm-0.30.0+h2o.<commit>.cu129-cp38-abi3-manylinux_2_28_x86_64.whl --extra-index-url https://flashinfer.ai/whl/
```

Or use the image, which is the upstream `vllm/vllm-openai` image of the same release with the wheel installed and
torch pinned to the variant's 2.13.0 build:

```bash
docker run --gpus all -p 8000:8000 ghcr.io/h2oai/vllm:<version>-cu129 <model> --attention-backend TRITON_ATTN
```

## Build

```bash
tools/h2o/build_wheel.sh cu129 dist   # or cu130; needs git, curl, python3 (3.10-3.14) and ~8 GB of disk
tools/h2o/smoke.sh cu129 dist/vllm-*.cu129-*.whl
docker build -f docker/Dockerfile.h2o --build-arg VARIANT=cu129 \
  --build-arg BASE_IMAGE=vllm/vllm-openai:v0.30.0-cu129 -t vllm-h2o:cu129 dist/
```

`build_wheel.sh` refuses a commit that changes compiled sources relative to upstream v0.30.0 (`csrc/`, `cmake/`,
`rust/`, `setup.py`, `requirements/`, ...); such a change needs a full source build instead.

## CI

`.github/workflows/h2o-build.yml` runs on GitHub-hosted runners (no GPU):

- every push to an `h2o/**` branch: both wheels, built and verified, then installed in a fresh environment with the
  variant's torch, imported, and the fork's CPU tests run against the installed package;
- a tag `h2o-v<version>` (for example `h2o-v0.30.0.1`): the same, then a GitHub Release with both wheels and their
  sha256, and images `ghcr.io/h2oai/vllm:<version>-cu129` / `-cu130`;
- manual run (`workflow_dispatch`, `image: true`): the images without a release.

On a runner without an NVIDIA driver the smoke test leaves out the CPU tests listed in `smoke_needs_driver.txt`
(they need vLLM to detect a platform); with a driver it runs them all. The GPU tests of the fork (`tests/models/**/test_gemma4_*`, marked to skip without a CUDA GPU) are not run in CI.
