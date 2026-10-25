"""Check an h2oai/vllm wheel against the upstream wheel it was built on.

Every file of the upstream wheel must be in ours with the same bytes, except the
files the fork changes (which must match the fork commit byte for byte) and
vllm/_version.py; files the wheel has beyond upstream must be files the fork adds.
Requires-Dist must equal upstream's, so the fork installs exactly the dependencies
vLLM pins.

  python -I verify_wheel.py --upstream UP.whl --wheel OURS.whl --repo REPO \
      --commit SHA --changed CHANGED.txt
"""

import argparse
import email.parser
import hashlib
import subprocess
import sys
import zipfile


def contents(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            out[info.filename] = hashlib.sha256(z.read(info)).hexdigest()
    return out


def metadata(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
        return email.parser.Parser().parsestr(z.read(name).decode())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--wheel", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--changed", required=True)
    a = ap.parse_args()

    with open(a.changed) as f:
        changed = {line.strip() for line in f if line.strip().startswith("vllm/")}
    up, ours = contents(a.upstream), contents(a.wheel)
    up = {k: v for k, v in up.items() if ".dist-info/" not in k}
    ours = {k: v for k, v in ours.items() if ".dist-info/" not in k}
    errors = []

    def git_sha(path):
        blob = subprocess.run(
            ["git", "-C", a.repo, "show", f"{a.commit}:{path}"],
            capture_output=True,
            check=True,
        ).stdout
        return hashlib.sha256(blob).hexdigest()

    same = fork = 0
    for path, digest in sorted(up.items()):
        if path == "vllm/_version.py":
            continue
        if path not in ours:
            if path in changed:  # deleted by the fork
                continue
            errors.append(f"missing from the wheel: {path}")
        elif path in changed:
            fork += 1
            if ours[path] != git_sha(path):
                errors.append(f"differs from the fork commit: {path}")
        elif ours[path] != digest:
            errors.append(f"differs from upstream: {path}")
        else:
            same += 1
    added = sorted(set(ours) - set(up) - {"vllm/_version.py"})
    for path in added:
        if path not in changed:
            errors.append(f"not in upstream and not added by the fork: {path}")
        elif ours[path] != git_sha(path):
            errors.append(f"differs from the fork commit: {path}")
    # every fork-changed Python file that exists at the commit must be packaged
    at_commit = subprocess.run(
        ["git", "-C", a.repo, "ls-tree", "-r", "--name-only", a.commit, "--", "vllm"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.split()
    errors += [
        f"fork file not in the wheel: {p}"
        for p in sorted(changed & set(at_commit))
        if p.endswith(".py") and p not in ours
    ]

    m_up, m_ours = metadata(a.upstream), metadata(a.wheel)
    if sorted(m_up.get_all("Requires-Dist") or []) != sorted(
        m_ours.get_all("Requires-Dist") or []
    ):
        errors.append("Requires-Dist differs from upstream")
    if m_up.get("Requires-Python") != m_ours.get("Requires-Python"):
        errors.append("Requires-Python differs from upstream")

    print(
        f"verify: {same} files identical to upstream, {fork} changed and "
        f"{len(added)} added by the fork (each equal to {a.commit[:10]}), "
        f"version {m_ours['Version']}"
    )
    if errors:
        print("\n".join(errors[:50]))
        print(f"FAIL: {len(errors)} problems")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
