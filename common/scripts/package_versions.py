#!/usr/bin/env python
"""Record and compare the versions of the git-cloned packages an analysis used.

The ``ts_XXX`` and other packages under ``$LSST_PACKAGES`` are set up with
``eups setup -r``, so they are whatever the working tree happened to be at the
time -- a ``git pull`` silently changes the code under a finished analysis. This
module writes a *lockfile*: one line per package giving branch, commit and
``git describe``, in the spirit of ``ts_cycle_build``'s ``cycle.env``.

Two modes:

``snapshot``
    Write the current state to a lockfile. Do this at the start of a long build
    and commit the lockfile next to the output it describes.

``diff``
    Compare a lockfile against the current working trees (or against a second
    lockfile) and print the packages whose commit moved, with the log of what
    landed in between. This is the "what changed under me?" question.

Examples
--------
Snapshot before a rebuild, and keep it with the output::

    python common/scripts/package_versions.py snapshot \\
        -o value_added/output/aos_efd.versions.txt

What has changed since that snapshot::

    python common/scripts/package_versions.py diff \\
        value_added/output/aos_efd.versions.txt --log

Restore the exact code a snapshot describes (prints the commands, runs with
``--apply``)::

    python common/scripts/package_versions.py checkout \\
        value_added/output/aos_efd.versions.txt
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import socket
import subprocess
import sys
from pathlib import Path

__all__ = ["DEFAULT_PACKAGES_DIR", "read_lockfile", "scan", "snapshot_text"]

# Where the hand-cloned (non-DM-stack) packages live. Overridable so this works
# on the summit RSP as well as at USDF.
DEFAULT_PACKAGES_DIR = Path(
    os.environ.get("LSST_PACKAGES_DIR", "/sdf/data/rubin/user/roodman/LSST/packages")
)


def _git(repo: Path, *args: str) -> str:
    """Run a git command in ``repo`` and return stripped stdout ('' on error)."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def scan(packages_dir: Path = DEFAULT_PACKAGES_DIR) -> dict[str, dict[str, str]]:
    """Collect git state for every repository directly under ``packages_dir``.

    Returns
    -------
    `dict`
        Package name -> dict with ``branch``, ``commit``, ``describe``,
        ``date``, ``dirty`` (count of modified files, as a string) and
        ``origin``. Non-git directories are skipped.
    """
    packages_dir = Path(packages_dir)
    state: dict[str, dict[str, str]] = {}
    for path in sorted(p for p in packages_dir.iterdir() if p.is_dir()):
        if not (path / ".git").exists():
            continue
        branch = _git(path, "rev-parse", "--abbrev-ref", "HEAD")
        state[path.name] = {
            # A detached HEAD reports "HEAD"; record that rather than a lie.
            "branch": branch or "unknown",
            "commit": _git(path, "rev-parse", "HEAD"),
            "describe": _git(path, "describe", "--tags", "--always", "--dirty") or "-",
            "date": _git(path, "log", "-1", "--format=%cs"),
            # Uncommitted work is not reproducible from the commit alone, so
            # flag it loudly rather than recording a clean-looking hash.
            "dirty": str(len([l for l in _git(path, "status", "--porcelain").splitlines() if l])),
            "origin": _git(path, "config", "--get", "remote.origin.url"),
        }
    return state


def snapshot_text(
    state: dict[str, dict[str, str]], packages_dir: Path = DEFAULT_PACKAGES_DIR
) -> str:
    """Render ``state`` as a lockfile: comment header plus one line per package."""
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "# rubin-work package lockfile -- versions of the non-DM-stack packages in use.",
        "# Written by common/scripts/package_versions.py; see its docstring.",
        f"# generated: {now}",
        f"# host:      {socket.gethostname()}",
        f"# packages:  {packages_dir}",
        "#",
        "# name branch commit describe last_commit_date dirty_files",
    ]
    for name, info in state.items():
        flag = "" if info["dirty"] == "0" else f"  # DIRTY: {info['dirty']} file(s)"
        lines.append(
            f"{name} {info['branch']} {info['commit']} {info['describe']} "
            f"{info['date']} {info['dirty']}{flag}"
        )
    return "\n".join(lines) + "\n"


def read_lockfile(path: Path) -> dict[str, dict[str, str]]:
    """Parse a lockfile written by `snapshot_text`."""
    state: dict[str, dict[str, str]] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        name, branch, commit, describe = parts[:4]
        state[name] = {
            "branch": branch,
            "commit": commit,
            "describe": describe,
            "date": parts[4] if len(parts) > 4 else "",
            "dirty": parts[5] if len(parts) > 5 else "0",
        }
    return state


def _cmd_snapshot(args: argparse.Namespace) -> int:
    text = snapshot_text(scan(args.packages_dir), args.packages_dir)
    if args.output:
        Path(args.output).write_text(text)
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(text)
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    old = read_lockfile(args.lockfile)
    new = read_lockfile(args.against) if args.against else scan(args.packages_dir)
    label = str(args.against) if args.against else "working tree"

    moved, added, removed, dirty = [], [], [], []
    for name in sorted(set(old) | set(new)):
        if name not in new:
            removed.append(name)
            continue
        if name not in old:
            added.append(name)
            continue
        if old[name]["commit"] != new[name]["commit"]:
            moved.append(name)
        if new[name].get("dirty", "0") not in ("0", ""):
            dirty.append(name)

    print(f"comparing {args.lockfile} -> {label}\n")
    if not moved:
        print("No package commits moved.")
    for name in moved:
        o, n = old[name], new[name]
        print(f"=== {name}: {o['describe']} -> {n['describe']}")
        print(f"    {o['commit'][:10]} ({o.get('date','?')})  ->  {n['commit'][:10]} ({n.get('date','?')})")
        if o["branch"] != n["branch"]:
            print(f"    BRANCH CHANGED: {o['branch']} -> {n['branch']}")
        if args.log and not args.against:
            repo = Path(args.packages_dir) / name
            # Only meaningful when the old commit is still reachable locally.
            log = _git(repo, "log", "--oneline", "--no-merges", f"{o['commit']}..{n['commit']}")
            if log:
                shown = log.splitlines()
                for entry in shown[: args.max_log]:
                    print(f"      {entry}")
                if len(shown) > args.max_log:
                    print(f"      ... {len(shown) - args.max_log} more commit(s)")
            else:
                print("      (old commit not reachable locally; run 'git fetch' in the repo)")
        print()

    for name in added:
        print(f"+++ {name}: new since lockfile ({new[name]['describe']})")
    for name in removed:
        print(f"--- {name}: present in lockfile but missing now")
    if dirty:
        print(f"\nUncommitted changes in: {', '.join(dirty)}")
    # Exit non-zero when anything moved, so a build script can gate on this.
    return 1 if (moved or added or removed) else 0


def _cmd_checkout(args: argparse.Namespace) -> int:
    old = read_lockfile(args.lockfile)
    cur = scan(args.packages_dir)
    for name, info in old.items():
        if name not in cur or cur[name]["commit"] == info["commit"]:
            continue
        repo = Path(args.packages_dir) / name
        cmd = ["git", "-C", str(repo), "checkout", info["commit"]]
        print(" ".join(cmd))
        if args.apply:
            subprocess.run(cmd, check=False)
    if not args.apply:
        print("\n(dry run -- re-run with --apply to execute; "
              "return to your branch later with 'git checkout <branch>')")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--packages-dir", type=Path, default=DEFAULT_PACKAGES_DIR,
        help="directory holding the cloned packages (default: %(default)s)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_snap = sub.add_parser("snapshot", help="write a lockfile of current versions")
    p_snap.add_argument("-o", "--output", type=Path, help="lockfile to write (default: stdout)")
    p_snap.set_defaults(func=_cmd_snapshot)

    p_diff = sub.add_parser("diff", help="compare a lockfile against now")
    p_diff.add_argument("lockfile", type=Path)
    p_diff.add_argument("--against", type=Path, help="compare to a second lockfile instead of the working tree")
    p_diff.add_argument("--log", action="store_true", help="show the commits that landed in between")
    p_diff.add_argument("--max-log", type=int, default=25, help="max commits to print per package")
    p_diff.set_defaults(func=_cmd_diff)

    p_co = sub.add_parser("checkout", help="restore the commits a lockfile names")
    p_co.add_argument("lockfile", type=Path)
    p_co.add_argument("--apply", action="store_true", help="actually run the checkouts")
    p_co.set_defaults(func=_cmd_checkout)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
