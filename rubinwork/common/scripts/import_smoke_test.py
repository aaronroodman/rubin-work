#!/usr/bin/env python
"""Import-only smoke test over every repo module that imports a moved library.

Phase 1 of the products/libraries/studies reorganization moves ``common/``,
``aos_state``, ``open_loop`` and three ``smatrix`` modules into the
``rubinwork`` package, leaving shims at the old paths. This script records the
import outcome of every affected module so a run before the move can be
compared with a run after it: any module whose outcome changes is a shim that
does not work.

Each module is imported in a fresh subprocess, in script mode (its own
directory first on ``sys.path``), which is how these files are actually run.
Modules that fail for reasons unrelated to the move (a missing optional
dependency, no Butler, a hardcoded path) fail identically before and after, so
the comparison stays meaningful without needing them to pass.

Usage
-----
    python common/scripts/import_smoke_test.py --out /tmp/before.json
    # ... do the move ...
    python common/scripts/import_smoke_test.py --out /tmp/after.json
    python common/scripts/import_smoke_test.py --compare /tmp/before.json /tmp/after.json
"""

import argparse
import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

# Modules whose imports are rewritten or shimmed by phase 1.
MOVED_NAMES = [
    "common",
    "aos_state",
    "open_loop",
    "compute_smatrix",
    "normalization_weights",
    "regularized_inversion",
]

# Directories with no importable repo code.
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", "output", "logs"}


def affected_modules():
    """Return repo-relative paths of `.py` files importing a moved library.

    Returns
    -------
    paths : `list` [`pathlib.Path`]
        Sorted repo-relative paths, one per module.
    """
    pattern = r"^\s*(from|import)\s+(" + "|".join(MOVED_NAMES) + r")\b"
    out = subprocess.run(
        ["rg", "-l", "--no-heading", "-g", "*.py", pattern, "."],
        cwd=REPO_ROOT, capture_output=True, text=True,
    ).stdout
    paths = []
    for line in out.splitlines():
        p = pathlib.Path(line.lstrip("./"))
        if SKIP_DIRS & set(p.parts):
            continue
        paths.append(p)
    return sorted(set(paths))


def import_one(path, timeout=120):
    """Import one module in a fresh subprocess and classify the outcome.

    Parameters
    ----------
    path : `pathlib.Path`
        Repo-relative path of the module to import.
    timeout : `int`, optional
        Seconds to wait before recording a timeout.

    Returns
    -------
    result : `dict`
        ``outcome`` is ``"ok"``, ``"timeout"`` or the exception class name;
        ``detail`` is the first line of the exception message, or ``""``.
    """
    full = REPO_ROOT / path
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('_smoke', r'{full}')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['_smoke'] = mod\n"
        "spec.loader.exec_module(mod)\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=full.parent, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"outcome": "timeout", "detail": ""}
    if proc.returncode == 0:
        return {"outcome": "ok", "detail": ""}
    lines = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
    last = lines[-1] if lines else "unknown failure"
    exc = last.split(":", 1)[0].split(".")[-1].strip()
    detail = last.split(":", 1)[1].strip() if ":" in last else ""
    return {"outcome": exc, "detail": detail[:200]}


def run(out_path):
    """Import every affected module and write the results as JSON."""
    mods = affected_modules()
    print(f"n_modules = {len(mods)} (count, dimensionless)")
    results = {}
    for i, p in enumerate(mods, 1):
        r = import_one(p)
        results[str(p)] = r
        print(f"[{i:3d}/{len(mods)}] {r['outcome']:22s} {p}")
    pathlib.Path(out_path).write_text(json.dumps(results, indent=1, sort_keys=True))
    n_ok = sum(1 for r in results.values() if r["outcome"] == "ok")
    print(f"\nn_ok = {n_ok} / {len(results)} (count, dimensionless) -> {out_path}")


def compare(before_path, after_path):
    """Print modules whose import outcome differs between two runs.

    Returns
    -------
    n_changed : `int`
        Number of modules with a changed outcome (count, dimensionless).
    """
    before = json.loads(pathlib.Path(before_path).read_text())
    after = json.loads(pathlib.Path(after_path).read_text())
    changed = []
    for mod in sorted(set(before) | set(after)):
        b = before.get(mod, {}).get("outcome", "<absent>")
        a = after.get(mod, {}).get("outcome", "<absent>")
        if b != a:
            changed.append((mod, b, a))
    if not changed:
        print(f"no change: all {len(before)} module import outcomes identical "
              f"(count, dimensionless)")
    else:
        print(f"n_changed = {len(changed)} (count, dimensionless)")
        for mod, b, a in changed:
            print(f"  {mod}\n      before: {b}\n      after:  {a}")
            if a not in ("ok", b):
                print(f"      {after[mod]['detail']}")
    return len(changed)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="write results JSON here")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="compare two results files")
    args = ap.parse_args()
    if args.compare:
        sys.exit(1 if compare(*args.compare) else 0)
    if not args.out:
        ap.error("one of --out or --compare is required")
    run(args.out)


if __name__ == "__main__":
    main()
