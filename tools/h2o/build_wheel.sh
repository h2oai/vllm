#!/usr/bin/env bash
# Build the h2oai/vllm wheel for one CUDA variant: this checkout's Python on top of the compiled parts of the
# matching upstream vLLM release wheel (VLLM_USE_PRECOMPILED=1; no CUDA compiler, no GPU, a few minutes on a CPU).
#
#   tools/h2o/build_wheel.sh cu129|cu130 OUT_DIR
#
# The wheel is named vllm-<upstream>+h2o.<10-char commit>.<variant>-cp38-abi3-manylinux_2_28_x86_64.whl.
# The build refuses a checkout that changes anything compiled (csrc/, cmake/, rust/, setup.py, requirements/, ...)
# relative to the upstream release, since the reused binaries would then not match the source. After the build,
# verify_wheel.py checks the result file by file against the upstream wheel.
# Env: H2O_WHEEL_CACHE (where upstream wheels are kept; default OUT_DIR/.upstream), PYTHON (default python3).
set -euo pipefail
umask 022  # file modes in the wheel do not depend on the machine's umask

VARIANT="${1:?usage: build_wheel.sh cu129|cu130 OUT_DIR}"
OUT="${2:?usage: build_wheel.sh cu129|cu130 OUT_DIR}"
case "$VARIANT" in cu129|cu130) ;; *) echo "variant must be cu129 or cu130"; exit 2;; esac
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(git -C "$HERE" rev-parse --show-toplevel)"
# shellcheck source=upstream.env
. "$HERE/upstream.env"
PYTHON="${PYTHON:-python3}"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
CACHE="${H2O_WHEEL_CACHE:-$OUT/.upstream}"; mkdir -p "$CACHE"

[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || { echo "the checkout has uncommitted changes"; exit 1; }
SHA="$(git -C "$REPO" rev-parse HEAD)"; SHORT="${SHA:0:10}"
# zip timestamps from the commit time, so a rebuild of the same commit gives the same bytes
SOURCE_DATE_EPOCH="$(git -C "$REPO" log -1 --format=%ct "$SHA")"; export SOURCE_DATE_EPOCH
VERSION="${UPSTREAM_VERSION}+h2o.${SHORT}.${VARIANT}"

# the fork must differ from the upstream release only where no compiled code comes from
git -C "$REPO" cat-file -e "${UPSTREAM_COMMIT}^{commit}" 2>/dev/null \
  || git -C "$REPO" fetch -q --depth 1 https://github.com/vllm-project/vllm.git "$UPSTREAM_COMMIT"
CHANGED="$OUT/changed_files_${VARIANT}.txt"
git -C "$REPO" diff --name-only --no-renames "$UPSTREAM_COMMIT" "$SHA" > "$CHANGED"
BAD="$(grep -v -E '^(vllm/|tests/|docs/|examples/|\.github/|tools/h2o/|docker/Dockerfile\.h2o$|README\.md$)' "$CHANGED"
  grep -E '^vllm/.*\.(so|cu|cuh|cpp|cc|h|hpp|rs)$' "$CHANGED")" || true
[ -z "$BAD" ] || { echo "these changes need a source build, not the precompiled one:"; echo "$BAD"; exit 1; }
echo "fork vs upstream ${UPSTREAM_TAG}: $(grep -c '^vllm/' "$CHANGED" || true) files under vllm/, $(wc -l < "$CHANGED") in all"

# the upstream wheel for this variant, checked by sha256
WHEEL_VAR="WHEEL_${VARIANT}"; SHA_VAR="SHA256_${VARIANT}"
UP="$CACHE/${!WHEEL_VAR}"
if ! echo "${!SHA_VAR}  $UP" | sha256sum -c --quiet - 2>/dev/null; then
  curl -sSfL --retry 3 -o "$UP.part" "${UPSTREAM_RELEASE_URL}/${!WHEEL_VAR//+/%2B}"
  mv "$UP.part" "$UP"
  echo "${!SHA_VAR}  $UP" | sha256sum -c --quiet - || { echo "sha256 mismatch: $UP"; exit 1; }
fi

# build in a fresh clone of the commit: the precompiled step copies binaries into the source tree, and the package
# data (configs, scripts) is collected from the git-tracked files, so the build needs a git checkout, not an export
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
git clone -q --shared --no-checkout "$REPO" "$WORK/src"
git -C "$WORK/src" checkout -q --detach "$SHA"
"$PYTHON" -m venv "$WORK/venv"
VPY="$WORK/venv/bin/python"
"$VPY" -m pip install -q --upgrade pip
# setup.py reads the CUDA build of torch to pick requirements/cuda.txt (nothing is compiled; no GPU needed)
"$VPY" -m pip install -q "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/${VARIANT}"
"$VPY" -m pip install -q -r <(grep -v -E '^\s*torch\s*==' "$WORK/src/requirements/build/cuda.txt")
(
  cd "$WORK/src"
  VLLM_TARGET_DEVICE=cuda VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_LOCATION="$UP" \
    VLLM_VERSION_OVERRIDE="$VERSION" MAX_JOBS=4 \
    "$VPY" -m pip wheel -q --no-deps --no-build-isolation -w "$WORK/dist" .
)
BUILT="$(ls "$WORK"/dist/vllm-*.whl)"
# setuptools tags the wheel linux_x86_64; the reused binaries are the upstream manylinux_2_28 ones
FINAL="$OUT/vllm-${VERSION}-cp38-abi3-manylinux_2_28_x86_64.whl"
"$VPY" -m pip install -q "wheel>=0.42"
"$VPY" -m wheel tags --python-tag cp38 --abi-tag abi3 --platform-tag manylinux_2_28_x86_64 --remove "$BUILT" > /dev/null
mv "$(ls "$WORK"/dist/vllm-*.whl)" "$FINAL"
"$VPY" -I "$HERE/verify_wheel.py" --upstream "$UP" --wheel "$FINAL" --repo "$REPO" --commit "$SHA" --changed "$CHANGED"
(cd "$OUT" && sha256sum "$(basename "$FINAL")" > "$(basename "$FINAL").sha256")
echo "built $FINAL"
