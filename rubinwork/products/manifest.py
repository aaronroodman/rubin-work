"""Write the ``manifest.json`` that makes a build directory a catalog entry.

Every product build calls `write` as its last step. The manifest records what
the build is, what configuration produced it, which upstream builds it read and
what it wrote, so two builds of one variant that disagree can be told apart
without rerunning either. The ``stack`` key records the environment as well —
the eups ``lsst_distrib`` weekly tag, the conda environment, and the branch and
commit of each locally set up package — so a difference that comes from a stack
move rather than from a config change can be attributed to it.

A build directory with no manifest, or one whose ``status`` is not
``"complete"``, is not in the catalog: that is what makes an interrupted build
invisible rather than silently half-read.

Writing a manifest does not move the variant's ``current`` symlink. That is
`rubinwork.products.catalog.set_current`, called by hand once the build has been
looked at.

Examples
--------
At the end of a build::

    from rubinwork.products import manifest
    manifest.write(build_dir, product="fam_tables", variant="danish_1_2",
                   build="20261005", config=variant_config,
                   inputs={"cwfs_tables": "danish_1_2@20260930"})
"""

import copy
import datetime
import functools
import json
import os
import pathlib
import re
import subprocess
import tempfile

__all__ = ["write", "git_state", "stack_state", "file_stats",
           "set_current_build"]

MANIFEST_NAME = "manifest.json"
CURRENT = "current"

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# A dated weekly, "w_2026_41". Excludes the aliases "w_latest" and "current".
_WEEKLY_RE = re.compile(r"w_\d{4}_\d{2}")

# The locally set up packages whose version changes a product's numbers. Others
# appear in `eups list -s` too; these are the ones the AOS products compute
# with, so a build that differs is worth attributing to one of them first.
STACK_PACKAGES = (
    "ts_intrinsic_wavefront",
    "ts_wep",
    "ts_ofc",
    "summit_utils",
    "ts_config_mttcs",
    "ts_aos_analysis",
)


def git_state(repo=None):
    """Commit and dirty flag of the repository the build ran from.

    Parameters
    ----------
    repo : `str` or `pathlib.Path`, optional
        Repository to inspect. The rubin-work checkout holding this module by
        default.

    Returns
    -------
    commit : `str` or `None`
        Abbreviated commit hash, or `None` if git is unavailable.
    dirty : `bool` or `None`
        Whether the working tree had uncommitted tracked changes, or `None` if
        git is unavailable.
    """
    repo = pathlib.Path(repo) if repo else _REPO_ROOT
    def _git(*args):
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, timeout=30)
    try:
        head = _git("rev-parse", "--short", "HEAD")
        if head.returncode != 0:
            return None, None
        status = _git("status", "--porcelain", "--untracked-files=no")
        return head.stdout.strip(), bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def _git_out(repo, *args):
    """One git command's stdout, or `None` if it cannot be run.

    Parameters
    ----------
    repo : `str` or `pathlib.Path`
        Repository to run in.
    *args
        Arguments after ``git -C <repo>``.

    Returns
    -------
    out : `str` or `None`
        Stripped stdout, or `None` on any failure or empty output.
    """
    try:
        done = subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip() or None


@functools.lru_cache(maxsize=None)
def stack_state(packages=STACK_PACKAGES):
    """The stack and locally set up packages the build ran against.

    Read from the eups environment, so it records what was actually set up
    rather than what a config says should have been. A build made on one weekly
    and a build made on another are otherwise indistinguishable from their
    manifests.

    Cached for the life of the process: it costs one ``eups list`` and four git
    commands per package, and a build cannot change its own environment
    mid-run. Call ``stack_state.cache_clear()`` in a test that patches what it
    reads.

    Parameters
    ----------
    packages : `tuple` [`str`], optional
        eups package names to look for among the locally set up packages.
        `STACK_PACKAGES` by default. Must be hashable, for the cache.

    Returns
    -------
    state : `dict`
        ``eups_tag`` is the ``lsst_distrib`` tag (``"w_2026_41"``), or `None` if
        eups is unavailable or ``lsst_distrib`` is not set up; ``conda_env`` is
        the conda environment name; ``packages`` maps each package found to
        ``{"path", "branch", "commit", "dirty", "describe"}``, with `None` for
        any field git could not supply. A package that is not set up is absent
        rather than present-and-null, so its absence is visible.
    """
    state = {
        "eups_tag": None,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": {},
    }
    try:
        done = subprocess.run(["eups", "list", "-s"],
                              capture_output=True, text=True, timeout=120)
        listing = done.stdout if done.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        listing = ""

    wanted = set(packages)
    for line in listing.splitlines():
        fields = line.split()
        if not fields:
            continue
        name = fields[0]
        if name == "lsst_distrib":
            # The dated weekly, not the build hash and not a moving alias. The
            # tag list is unordered and carries aliases: w_2026_39 lists as
            # "<hash> w_2026_39 w_latest current setup" but w_2026_41 lists as
            # "<hash> current w_latest w_2026_41 setup", so taking the first
            # w_-prefixed field records "w_latest" and names no weekly at all.
            tags = [f for f in fields[1:] if _WEEKLY_RE.fullmatch(f)]
            state["eups_tag"] = tags[0] if tags else None
            continue
        if name not in wanted:
            continue
        local = [f for f in fields[1:] if f.startswith("LOCAL:")]
        if not local:
            continue
        path = local[0][len("LOCAL:"):]
        commit, dirty = git_state(path)
        state["packages"][name] = {
            "path": path,
            "branch": _git_out(path, "rev-parse", "--abbrev-ref", "HEAD"),
            "commit": commit,
            "dirty": dirty,
            "describe": _git_out(path, "describe", "--tags", "--dirty"),
        }
    return state


def file_stats(build_dir, files=None):
    """Row count and size of each data file in a build.

    Parameters
    ----------
    build_dir : `str` or `pathlib.Path`
        The build directory.
    files : `list` [`str`], optional
        Names of the files to record, relative to `build_dir`. By default every
        regular file in `build_dir` except the manifest and anything whose name
        starts with ``.`` or ``_`` (the ``_work`` scratch directory).

    Returns
    -------
    stats : `dict`
        ``{name: {"rows": int or None, "bytes": int}}``. ``rows`` is the row
        count (count, dimensionless) for a parquet file and `None` otherwise;
        ``bytes`` is the file size in bytes.
    """
    build_dir = pathlib.Path(build_dir)
    if files is None:
        files = sorted(p.name for p in build_dir.iterdir()
                       if p.is_file() and p.name != MANIFEST_NAME
                       and not p.name.startswith((".", "_")))
    stats = {}
    for name in files:
        target = build_dir / name
        if not target.is_file():
            continue
        rows = None
        if target.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
                rows = pq.ParquetFile(target).metadata.num_rows
            except Exception:
                # A row count is a convenience, not a reason to fail a build
                # that has already written its data.
                rows = None
        stats[name] = {"rows": rows, "bytes": target.stat().st_size}
    return stats


def write(build_dir, product, variant, build, config=None, inputs=None,
          files=None, location=None, status="complete", extra=None):
    """Write ``manifest.json`` for one build.

    Parameters
    ----------
    build_dir : `str` or `pathlib.Path`
        The build directory. Created if it does not exist.
    product : `str`
        Product name.
    variant : `str`
        Variant name, as defined in the product's ``variants.yaml``.
    build : `str`
        Build name, normally the date (``"20261007"``, then ``"20261007b"`` for a
        second run the same day).
    config : `dict`, optional
        The full expanded configuration this build ran with.
    inputs : `dict`, optional
        Upstream builds read, as ``{product: "variant@build"}``.
    files : `list` [`str`], optional
        Names of the data files to record. Every data file in `build_dir` by
        default.
    location : `str` or `pathlib.Path`, optional
        For an external variant whose data lives elsewhere: the path or
        collection name. A build with a ``location`` records no files.
    status : `str`, optional
        ``"complete"`` by default. Any other value keeps the build out of the
        catalog.
    extra : `dict`, optional
        Additional top-level keys to merge into the manifest. Merged last, so a
        key given here overrides one the writer computed, ``stack`` included.

    Returns
    -------
    manifest_path : `pathlib.Path`
        The manifest that was written.
    """
    build_dir = pathlib.Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    commit, dirty = git_state()
    man = {
        "product": product,
        "variant": variant,
        "build": build,
        "status": status,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        "git_dirty": dirty,
        "config": config or {},
        "inputs": inputs or {},
        "files": {} if location else file_stats(build_dir, files),
        # Total over `files`, so PRODUCTS.md can show a size without walking the tree.
        "bytes": None,
        # Deep copy: `stack_state` is cached and hands back one shared dict.
        "stack": copy.deepcopy(stack_state()),
    }
    if location:
        man["location"] = str(location)
    else:
        man["bytes"] = sum(f["bytes"] for f in man["files"].values())
    if extra:
        man.update(extra)

    manifest_path = build_dir / MANIFEST_NAME
    text = json.dumps(man, indent=1, sort_keys=True) + "\n"
    # Unique temp name in the same directory: a fixed one lets two concurrent writers
    # each delete the other's temp file, and both lose the rename. Same directory, so
    # the rename stays atomic.
    fd, tmp = tempfile.mkstemp(dir=build_dir, prefix=f".{MANIFEST_NAME}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        # Rename, so a reader never sees a half-written manifest.
        os.replace(tmp, manifest_path)
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise
    return manifest_path


def set_current_build(variant_dir, build):
    """Repoint a variant's ``current`` symlink, atomically and with no checks.

    The mechanism only. `rubinwork.products.catalog.set_current` is the entry
    point, and it is what refuses a build that is not complete.

    Parameters
    ----------
    variant_dir : `str` or `pathlib.Path`
        The variant directory holding the builds.
    build : `str`
        Name of the build to make current.

    Returns
    -------
    link : `pathlib.Path`
        The ``current`` symlink.
    """
    variant_dir = pathlib.Path(variant_dir)
    link = variant_dir / CURRENT
    tmp = variant_dir / f".{CURRENT}.tmp"
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    # Relative target, so the tree survives being copied or re-rooted.
    tmp.symlink_to(build)
    os.replace(tmp, link)
    return link
