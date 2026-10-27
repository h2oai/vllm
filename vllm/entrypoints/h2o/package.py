# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model package: the released shim and its siblings beside the weights.

A package is a model repository (``h2oai/h2o-lightning-4b``,
``h2oai/h2o-lightning-31b``) at a tag: besides the weights it ships the
decision shim (``h2o_lightning_shim.py`` / ``lightning_31b_shim.py``), its
siblings (``pooled_readout.py``, ``box.py``, ``assets.py``, ...),
``serve_config*.json``, ``tokenizer_contract.json`` / ``tokenizer/``,
``label_rows.lightning`` and ``serve.sh``. This module finds those files
(a local directory, or the repository's package files fetched from the Hub
without the weights) and pins the shim by sha256.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# the package files besides the weights (the Hub fetch takes only these)
PACKAGE_PATTERNS: tuple[str, ...] = (
    "*.py",
    "*.json",
    "*.jinja",
    "serve.sh",
    "label_rows.lightning",
    "tokenizer/*",
    "engine/*",
    "*.SHA256SUMS",
)

# the shim entry files, in the order they are looked for
SHIM_ENTRIES: tuple[str, ...] = ("h2o_lightning_shim.py", "lightning_31b_shim.py")

# the released shims this layer was written and tested against, by the sha256 of
# the entry file. A package whose shim is not listed is refused, unless
# VLLM_H2O_DECISIONS_ALLOW_UNPINNED=1 lets it run (with a warning naming the
# sha256): the shim runs arbitrary code from the model repository, and the
# in-process transport relies on its VLLM / Server / main names.
KNOWN_SHIMS: dict[str, dict[str, str]] = {
    "bc386d26f4669a7d1d8b87507378c6156d0f4e8b451a3be4ac7c4ef5ded59bb7": {
        "repo": "h2oai/h2o-lightning-4b",
        "tag": "v1.4.1",
        "entry": "h2o_lightning_shim.py",
    },
    "dfe8dbe489ec0206f4fd75e0057a8acf751438f6d7cf2945992463fc6ae7caf2": {
        "repo": "h2oai/h2o-lightning-31b",
        "tag": "v1.2.2",
        "entry": "lightning_31b_shim.py",
    },
}

ALLOW_UNPINNED_ENV = "VLLM_H2O_DECISIONS_ALLOW_UNPINNED"


class PackageError(RuntimeError):
    pass


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Package:
    dir: str
    entry: str  # the shim file name
    entry_sha256: str
    serve_config: dict[str, Any]
    known: dict[str, str] | None  # the KNOWN_SHIMS entry, None when unpinned

    @property
    def entry_path(self) -> str:
        return os.path.join(self.dir, self.entry)

    @property
    def family(self) -> str:
        """``4b`` (Qwen hybrid, ``h2o_lightning_shim.py``) or ``31b`` (Gemma,
        ``lightning_31b_shim.py``)."""
        return "4b" if self.entry == "h2o_lightning_shim.py" else "31b"

    @property
    def model(self) -> str:
        return str(self.serve_config.get("model") or "")

    @property
    def config_path(self) -> str:
        return os.path.join(self.dir, "serve_config.json")

    def file(self, name: str) -> str | None:
        """The path of a package file, or None when the package lacks it."""
        path = os.path.join(self.dir, name)
        return path if os.path.exists(path) else None

    def describe(self) -> dict[str, Any]:
        return {
            "dir": self.dir,
            "entry": self.entry,
            "entry_sha256": self.entry_sha256,
            "known": self.known,
            "model": self.model,
        }


def find_package_dir(
    model: str,
    revision: str | None = None,
    explicit: str | None = None,
    download: bool = True,
) -> str:
    """The directory holding the package files: ``explicit`` when given, the
    model path when it is a directory, else the repository's package files
    from the Hub (cached beside the weights' snapshot; the weights are not
    fetched here)."""
    if explicit:
        if not os.path.isdir(explicit):
            raise PackageError(
                f"--h2o-decisions-package {explicit!r} is not a directory"
            )
        return explicit
    if os.path.isdir(model):
        return model
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - huggingface_hub is a vLLM dependency
        raise PackageError(
            "huggingface_hub is needed to fetch the package files"
        ) from e
    try:
        return snapshot_download(
            repo_id=model,
            revision=revision,
            allow_patterns=list(PACKAGE_PATTERNS),
            local_files_only=not download,
        )
    except Exception as e:
        raise PackageError(
            f"could not fetch the package files of {model!r}"
            f"{' at ' + revision if revision else ''}: {type(e).__name__}: {e}"
        ) from e


def load_package(directory: str, allow_unpinned: bool | None = None) -> Package:
    """The package in ``directory``: its shim entry (pinned by sha256) and its
    ``serve_config.json``."""
    import json

    entry = next(
        (e for e in SHIM_ENTRIES if os.path.isfile(os.path.join(directory, e))), None
    )
    if entry is None:
        raise PackageError(
            f"no decision shim in {directory}: expected one of {', '.join(SHIM_ENTRIES)} "
            "(is this an H2O-Lightning model repository?)"
        )
    config_path = os.path.join(directory, "serve_config.json")
    if not os.path.isfile(config_path):
        raise PackageError(f"{directory} has {entry} but no serve_config.json")
    with open(config_path, encoding="utf-8") as f:
        serve_config = json.load(f)
    digest = sha256_file(os.path.join(directory, entry))
    known = KNOWN_SHIMS.get(digest)
    if known is not None and known["entry"] != entry:
        known = None
    if known is None:
        if allow_unpinned is None:
            allow_unpinned = os.environ.get(ALLOW_UNPINNED_ENV, "").strip() == "1"
        if not allow_unpinned:
            raise PackageError(
                f"{entry} in {directory} (sha256 {digest[:16]}) is not a shim this build was "
                f"checked against ({', '.join(v['repo'] + ' ' + v['tag'] for v in KNOWN_SHIMS.values())}); "
                f"set {ALLOW_UNPINNED_ENV}=1 to run it anyway"
            )
        logger.warning(
            "h2o decisions: %s (sha256 %s) is not a pinned shim; running it because %s=1",
            entry,
            digest[:16],
            ALLOW_UNPINNED_ENV,
        )
    return Package(
        dir=os.path.abspath(directory),
        entry=entry,
        entry_sha256=digest,
        serve_config=serve_config,
        known=known,
    )


def locate_package(
    model: str,
    revision: str | None = None,
    explicit: str | None = None,
    download: bool = True,
    allow_unpinned: bool | None = None,
) -> Package:
    return load_package(
        find_package_dir(model, revision, explicit, download),
        allow_unpinned=allow_unpinned,
    )
