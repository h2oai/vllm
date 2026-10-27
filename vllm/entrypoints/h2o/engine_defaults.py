# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The package's engine settings, as its ``serve.sh`` passes them.

``vllm serve <model> --h2o-decisions`` should give the card's configuration
without the user retyping ``serve.sh``. ``package_engine_defaults`` lists what
the package's launcher passes to ``vllm serve`` (the 4B's ``serve.sh``, the 31B's
``serve.sh`` or its profile as the shim prints it), and
``apply_package_defaults_to_args`` fills in the arguments the command line
left at vLLM's default, before the engine is built. An argument given on the
command line always wins (a value equal to vLLM's default counts as not
given). GPU sizing (``--gpu-memory-utilization``) is never set here.
"""

from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger

from .package import Package, locate_package

logger = init_logger(__name__)


@dataclass
class EngineDefaults:
    args: dict[str, Any] = field(default_factory=dict)  # Namespace attribute -> value
    env: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# the 31B serve.sh sizes by GPU memory class (MiB), measured values only
GPU_CLASSES_31B = (
    (76000, 86000, {"max_model_len": 81920, "two_way_window": 16384}),  # one H100 80 GB
    (
        92000,
        100000,
        {"max_model_len": 131072, "two_way_window": 8192},
    ),  # one RTX PRO 6000 96 GB
)


def package_engine_defaults(
    package: Package,
    two_way: bool = True,
    profile: str | None = None,
    gpu_mib: int | None = None,
    shim_module: Any = None,
) -> EngineDefaults:
    cfg = package.serve_config
    d = EngineDefaults()
    d.env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"  # the reads never sample (both serve.sh)
    if package.family == "4b":
        return _defaults_4b(package, cfg, two_way, d)
    return _defaults_31b(package, cfg, two_way, profile, gpu_mib, shim_module, d)


def _defaults_4b(
    package: Package, cfg: dict[str, Any], two_way: bool, d: EngineDefaults
) -> EngineDefaults:
    # serve.sh: --served-model-name h2oai/h2o-lightning-4b --max-model-len ${MAX_LEN:-40960}
    #   [--tokenizer $HERE/tokenizer] --attention-backend TRITON_ATTN (one engine) --max-num-batched-tokens MAX_LEN
    #   [--limit-mm-per-prompt {"image": N, "video": 0} --mm-processor-kwargs {"max_pixels": P} --mm-processor-cache-gb 0]
    #   VLLM_GDN_PREFILL_GRAPH_PRECAPTURE=1 (one engine)
    if cfg.get("model"):
        d.args["served_model_name"] = [cfg["model"]]
    if package.file("tokenizer"):
        d.args["tokenizer"] = os.path.join(package.dir, "tokenizer")
        d.notes.append(
            "tokenizer: <package>/tokenizer (the training Qwen2Tokenizer; the root loads another)"
        )
    d.args["max_model_len"] = 40960
    # the one engine of the per-request attention build: two-way prompts are prefilled whole
    d.args["attention_backend"] = "TRITON_ATTN"
    d.args["max_num_batched_tokens"] = 40960
    d.env["VLLM_GDN_PREFILL_GRAPH_PRECAPTURE"] = "1"
    # serve.sh CUDA_GRAPH_MAX_TOKENS: 512 when two-way is the default reading, 2048
    # with TWO_WAY=0; explicit, so a GPU whose vLLM default differs (1024 on compute
    # capability 10.x) still runs the card's step graphs
    d.args["max_cudagraph_capture_size"] = 512
    if not two_way:
        # serve.sh ONE_ENGINE=1 TWO_WAY=0: the same one engine (TRITON, a step
        # budget of the whole context), answering one-way by default; only the
        # step graphs grow to 2048
        d.args["max_cudagraph_capture_size"] = 2048
        d.notes.append("one-way: serve.sh ONE_ENGINE=1 TWO_WAY=0 (one-way by default)")
    # serve.sh: --allowed-media-domains media.invalid (vLLM never fetches a URL; the
    # shim fetches image links with its own checks). Here vLLM's own routes share the
    # public port, so this also keeps /v1/chat/completions from fetching URLs.
    d.args["allowed_media_domains"] = ["media.invalid"]
    image = cfg.get("image")
    if isinstance(image, dict):
        d.args["limit_mm_per_prompt"] = {
            "image": int(image.get("max_images", 4)),
            "video": 0,
        }
        if image.get("max_pixels"):
            d.args["mm_processor_kwargs"] = {"max_pixels": int(image["max_pixels"])}
        d.args["mm_processor_cache_gb"] = 0
    return d


def _defaults_31b(
    package: Package,
    cfg: dict[str, Any],
    two_way: bool,
    profile: str | None,
    gpu_mib: int | None,
    shim_module: Any,
    d: EngineDefaults,
) -> EngineDefaults:
    if cfg.get("model"):
        d.args["served_model_name"] = [cfg["model"]]
    if profile:
        # the profile's engine, as the shim prints it (lightning_31b_shim.py --profile NAME --print-vllm-args)
        if shim_module is None:
            raise ValueError(
                "the 31B profile defaults need the shim module (profile_vllm_args)"
            )
        engine, _over = shim_module.profile_engine(profile)
        argv = shim_module.profile_vllm_args(engine, cfg["model"])
        d.args.update(parse_vllm_argv(argv))
        d.notes.append(f"profile {profile}: sized for {engine['gpu']}")
        return d
    # serve.sh without a profile: --max-logprobs 20 --served-model-name ... --dtype bfloat16 --quantization fp8
    #   --max-model-len ${MAX_LEN} [--limit-mm-per-prompt {"image": 4, "video": 0} --mm-processor-kwargs
    #   {"max_soft_tokens": 560}] [--attention-backend TRITON_ATTN --max-num-seqs 1024
    #   --max-num-batched-tokens window+seqs (two-way)]
    d.args["max_logprobs"] = 20
    d.args["dtype"] = "bfloat16"
    # not in the 31B serve.sh (its vLLM port is the shim's private backend); here
    # vLLM's routes are on the public port, so its chat route must not fetch URLs
    # for a caller (the shim fetches image links with its own checks)
    d.args["allowed_media_domains"] = ["media.invalid"]
    d.args["quantization"] = "fp8"
    image = cfg.get("image")
    if isinstance(image, dict):
        d.args["limit_mm_per_prompt"] = {
            "image": int(image.get("max_images", 4)),
            "video": 0,
        }
        d.args["mm_processor_kwargs"] = {"max_soft_tokens": 560}
    else:
        d.args["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    if two_way:
        d.args["attention_backend"] = "TRITON_ATTN"
        d.args["max_num_seqs"] = 1024
        sizes = next(
            (
                s
                for lo, hi, s in GPU_CLASSES_31B
                if gpu_mib is not None and lo <= gpu_mib <= hi
            ),
            None,
        )
        if sizes is None:
            d.notes.append(
                "two-way: no measured sizes for this GPU (80 GB class: --max-model-len 81920, window 16384; "
                "96 GB class: 131072 / 8192; GPU util 0.92): pass --max-model-len and "
                "--h2o-decisions-two-way-window yourself"
            )
        else:
            d.args["max_model_len"] = sizes["max_model_len"]
            d.args["max_num_batched_tokens"] = sizes["two_way_window"] + 1024
            d.args["h2o_decisions_two_way_window"] = sizes["two_way_window"]
            d.notes.append(
                f"two-way sizes for a {gpu_mib} MiB GPU: --max-model-len {sizes['max_model_len']}, "
                f"window {sizes['two_way_window']}, serve.sh uses GPU util 0.92"
            )
    else:
        d.args["max_model_len"] = 131072
    return d


def parse_vllm_argv(argv: list[str]) -> dict[str, Any]:
    """``--flag value`` pairs as Namespace attributes (JSON values parsed)."""
    out: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        flag = argv[i]
        if not flag.startswith("--"):
            raise ValueError(
                f"unexpected token {flag!r} in the profile's vLLM arguments"
            )
        key = flag[2:].replace("-", "_")
        value: Any = (
            argv[i + 1]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--")
            else True
        )
        if isinstance(value, str):
            with contextlib.suppress(ValueError):
                value = json.loads(value)
        if key == "served_model_name":
            value = [value]
        out[key] = value
        i += 1 if value is True else 2
    return out


# vLLM's own defaults for the arguments a package may fill: a value equal to its
# default counts as not given on the command line
_VLLM_DEFAULTS: dict[str, Any] = {
    "served_model_name": None,
    "tokenizer": None,
    "max_model_len": None,
    "attention_backend": None,
    "max_num_batched_tokens": None,
    "max_num_seqs": None,
    "limit_mm_per_prompt": ({}, None),
    "mm_processor_kwargs": None,
    "mm_processor_cache_gb": 4,
    "max_logprobs": 20,
    "dtype": "auto",
    "quantization": None,
    "h2o_decisions_two_way_window": None,
    "max_cudagraph_capture_size": None,
    "allowed_media_domains": None,
}


def _is_default(name: str, value: Any) -> bool:
    default = _VLLM_DEFAULTS.get(name)
    if isinstance(default, tuple):
        return any(value == d for d in default)
    return value == default


def apply_engine_defaults(args: Any, defaults: EngineDefaults) -> dict[str, Any]:
    """Fill the arguments left at their default; set the environment the
    launcher exports (``os.environ.setdefault``). Returns what was applied."""
    applied: dict[str, Any] = {}
    for name, value in defaults.args.items():
        current = getattr(args, name, None)
        if not hasattr(args, name) or _is_default(name, current):
            setattr(args, name, value)
            applied[name] = value
    for name, value in defaults.env.items():
        if name not in os.environ:
            os.environ[name] = value
            applied[f"env:{name}"] = value
    return applied


def apply_package_defaults_to_args(args: Any) -> dict[str, Any]:
    """The ``--h2o-decisions`` start: locate the package (fetching its files
    when needed), keep it on ``args`` for the server, and apply the package's
    engine defaults unless ``--no-h2o-package-defaults``."""
    package = locate_package(
        args.model,
        getattr(args, "revision", None),
        getattr(args, "h2o_decisions_package", None),
    )
    args._h2o_package = package
    if not getattr(args, "h2o_package_defaults", True):
        logger.info(
            "h2o decisions: package engine defaults not applied (--no-h2o-package-defaults)"
        )
        return {}
    profile = getattr(args, "h2o_decisions_profile", None) or None
    two_way = getattr(args, "h2o_decisions_two_way", True)
    shim_module = None
    if package.family == "31b" and profile:
        from .decisions import load_shim_module

        shim_module = load_shim_module(package)
    gpu_mib = (
        _gpu_memory_mib()
        if package.family == "31b" and two_way and not profile
        else None
    )
    defaults = package_engine_defaults(package, two_way, profile, gpu_mib, shim_module)
    applied = apply_engine_defaults(args, defaults)
    for note in defaults.notes:
        logger.info("h2o decisions: %s", note)
    logger.info(
        "h2o decisions: package engine defaults applied: %s",
        json.dumps(applied, default=str),
    )
    return applied


def _gpu_memory_mib() -> int | None:
    """The first visible GPU's memory in MiB (NVML, no CUDA context), or None."""
    try:
        from vllm.platforms import current_platform

        total = current_platform.get_device_total_memory(0)
        return int(total // (1024 * 1024)) if total else None
    except Exception:
        return None
