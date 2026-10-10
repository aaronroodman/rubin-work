"""Write the ``manifest.json`` that makes a build directory a catalog entry.

Every product build calls `write` as its last step. The manifest records what
the build is, what configuration produced it, which upstream builds it read and
what it wrote, so two builds of one variant that disagree can be told apart
without rerunning either.

A build directory with no manifest is not in the catalog: that is what makes an
interrupted build invisible rather than silently half-read.

Examples
--------
At the end of a build::

    from rubinwork.products import manifest
    manifest.write(build_dir, product="fam_tables", variant="danish_1_2",
                   build="20261005", config=variant_config,
                   inputs={"cwfs_tables": "danish_1_2@20260930"})
"""

import datetime
import json
import os
import pathlib
import subprocess

__all__ = ["write", "git_state", "file_stats", "set_current_build"]

MANIFEST_NAME = "manifest.json"
CURRENT = "current"

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


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
          files=None, location=None, status="complete", set_current=True,
          extra=None):
    """Write ``manifest.json`` for one build, and optionally point ``current`` at it.

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
        ``"complete"`` by default. Any other value marks the build as not for
        general use.
    set_current : `bool`, optional
        Whether to repoint the variant's ``current`` symlink at this build.
    extra : `dict`, optional
        Additional top-level keys to merge into the manifest.

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
    }
    if location:
        man["location"] = str(location)
    if extra:
        man.update(extra)

    manifest_path = build_dir / MANIFEST_NAME
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    # Rename, so a reader never sees a half-written manifest.
    os.replace(tmp, manifest_path)

    if set_current and status == "complete":
        set_current_build(build_dir.parent, build)
    return manifest_path


def set_current_build(variant_dir, build):
    """Point a variant's ``current`` symlink at one build.

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
