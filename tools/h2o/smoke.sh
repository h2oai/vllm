#!/usr/bin/env bash
# Install a built h2oai/vllm wheel into a fresh virtual environment and check it without a GPU: the installed version
# is the wheel's, torch is the pinned CUDA build for the variant, the per-request attention code imports, and the
# fork's CPU unit tests pass against the installed package (not the source tree).
#
#   tools/h2o/smoke.sh cu129|cu130 WHEEL
set -euo pipefail

VARIANT="${1:?usage: smoke.sh cu129|cu130 WHEEL}"
WHEEL="$(cd "$(dirname "${2:?usage: smoke.sh cu129|cu130 WHEEL}")" && pwd)/$(basename "$2")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
# shellcheck source=upstream.env
. "$HERE/upstream.env"
PYTHON="${PYTHON:-python3}"
export CUDA_VISIBLE_DEVICES=""

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
"$PYTHON" -m venv "$WORK/venv"; VPY="$WORK/venv/bin/python"
"$VPY" -m pip install -q --upgrade pip
# torch (and torchcodec, whose PyPI build is CUDA 13 only) from the variant's index first, so the wheel's pins are
# already met by the right CUDA builds
"$VPY" -m pip install -q "torch==${TORCH_VERSION}" "torchvision==0.28.0" "torchaudio==2.11.0" "torchcodec==0.16.0" \
  --index-url "https://download.pytorch.org/whl/${VARIANT}"
"$VPY" -m pip install -q "$WHEEL" --extra-index-url https://flashinfer.ai/whl/
"$VPY" -m pip install -q pytest pytest-asyncio tblib

WANT="$(basename "$WHEEL" | cut -d- -f2)"
cd "$WORK"  # never import the source tree
"$VPY" -I - "$WANT" "$VARIANT" <<'EOF'
import importlib.metadata, sys
import torch
want, variant = sys.argv[1], sys.argv[2]
got = importlib.metadata.version("vllm")
assert got == want, (got, want)
cuda = {"cu129": "12.9", "cu130": "13.0"}[variant]
assert torch.__version__.startswith("2.13.0") and torch.version.cuda == cuda, (torch.__version__, torch.version.cuda)
import vllm
from vllm.v1.request import parse_attention_mode  # the per-request attention build
assert vllm.__version__ == want, vllm.__version__
print(f"smoke: vllm {got}, torch {torch.__version__} (CUDA {torch.version.cuda}), per-request attention present")
EOF
"$VPY" -m pip check || true

# the fork's CPU tests, run from a copy of tests/ so `import vllm` resolves to the installed wheel
# (pyproject.toml for the pytest markers)
cp -r "$REPO/tests" "$WORK/tests"; cp "$REPO/pyproject.toml" "$WORK/"
TESTS=()  # the ones this commit has (older release lines have fewer)
for t in tests/v1/core/test_per_request_attention.py tests/v1/worker/test_per_request_attention_groups.py \
    tests/v1/attention/test_kv_sharing_fast_prefill_per_request.py tests/models/test_gemma4_noncausal.py \
    tests/models/test_gemma4_noncausal_inert.py tests/kernels/mamba/test_gdn_prefill_graph.py \
    tests/kernels/mamba/test_gdn_prefill_graph_rebind.py tests/entrypoints/h2o/test_decisions_endpoint.py; do
  [ -f "$WORK/$t" ] && TESTS+=("$t")
done
[ "${#TESTS[@]}" -gt 0 ] || { echo "no fork tests found"; exit 1; }
# without an NVIDIA driver vLLM detects no platform, and the tests that build a scheduler or a model config need one:
# leave out the ones listed in smoke_needs_driver.txt there (with a driver, even with no GPU visible, all run)
DESELECT=()
PLATFORM="$("$VPY" -I -c 'from vllm.platforms import current_platform as p; print(p.device_type or "")' 2>/dev/null)" || true
if [ -z "$PLATFORM" ]; then
  while read -r id; do
    case "$id" in ''|'#'*) continue;; esac
    [ -f "$WORK/${id%%::*}" ] && DESELECT+=(--deselect "$id")
  done < "$HERE/smoke_needs_driver.txt"
  echo "smoke: no platform detected (no NVIDIA driver): deselecting $(( ${#DESELECT[@]} / 2 )) tests from smoke_needs_driver.txt"
else
  echo "smoke: platform ${PLATFORM}: running every test"
fi
"$VPY" -m pytest -q -rs -p no:cacheprovider "${DESELECT[@]}" "${TESTS[@]}"
