"""Dated study run directories and their ``run.json``.

A study writes its output into ``<data root>/studies/<topic>/<study>/<YYYYMMDD>_<label>/``.
`study_run` makes that directory and writes ``run.json`` into it: the git commit, the
command or notebook, and every product build the process has read. The product catalog
reports each build it resolves to `record_read`, so the list is what was actually read,
not what a config says should have been. ``/wrap`` copies it into the study's ``uses:``.

The catalog imports this module, not the reverse: a library never imports a product.

Examples
--------
At the top of a study script or notebook::

    from rubinwork.common import study_run
    run = study_run("aos/miw_hull", "ring_coverage")
    fig.savefig(run.path / "ring_coverage.pdf")
"""

import datetime
import json
import os
import pathlib
import re
import subprocess
import sys

__all__ = ["DATA_ROOT_DEFAULT", "data_root", "record_read", "builds_read",
           "StudyRun", "study_run", "RUN_FILE"]

DATA_ROOT_DEFAULT = "/sdf/group/rubin/u/roodman/LSST/rubin-work"
"""Default data root on S3DF, overridden by ``$RUBINWORK_DATA``."""

RUN_FILE = "run.json"

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LABEL_RE = re.compile(r"[a-z0-9][a-z0-9_]*")

_reads = {}
_active = []


def data_root():
    """Root of the data tree, from ``$RUBINWORK_DATA`` or the S3DF default.

    Returns
    -------
    root : `pathlib.Path`
        The data root. Not required to exist.
    """
    return pathlib.Path(os.environ.get("RUBINWORK_DATA", DATA_ROOT_DEFAULT))


def record_read(product, variant, build):
    """Note that this process read one product build.

    Called by the catalog for every build it resolves. Rewrites the ``run.json`` of
    any open `StudyRun` when the build is new to it.

    Parameters
    ----------
    product : `str`
        Product name.
    variant : `str`
        Variant name.
    build : `str`
        Build name, with ``current`` already resolved.
    """
    key = (product, variant, build)
    if key in _reads:
        return
    _reads[key] = True
    for run in _active:
        run.write()


def builds_read():
    """Every product build this process has read, as ``uses:`` would hold them.

    Returns
    -------
    builds : `dict` [`str`, `list` [`str`]]
        ``{product: ["variant@build", ...]}``, products and builds sorted.
    """
    grouped = {}
    for product, variant, build in _reads:
        grouped.setdefault(product, set()).add(f"{variant}@{build}")
    return {p: sorted(v) for p, v in sorted(grouped.items())}


def _git(*args):
    """One git command's stdout in this checkout, or `None` on failure."""
    try:
        done = subprocess.run(["git", "-C", str(_REPO_ROOT), *args],
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _command():
    """The command line, or the notebook when run under a Jupyter kernel."""
    if sys.argv and "ipykernel" in pathlib.Path(sys.argv[0]).name:
        # Set by Jupyter Server 2 for each kernel; absent elsewhere.
        return os.environ.get("JPY_SESSION_NAME") or "notebook (path unknown)"
    return " ".join([pathlib.Path(sys.executable).name, *sys.argv])


class StudyRun:
    """One study run: its directory and its ``run.json``.

    Parameters
    ----------
    topic : `str`
        Topic name.
    study : `str`
        Study name.
    label : `str`
        Run label, lowercase snake_case.
    date : `str`
        ``YYYYMMDD``.
    """

    def __init__(self, topic, study, label, date):
        self.topic = topic
        self.study = study
        self.label = label
        self.date = date
        self.path = data_root() / "studies" / topic / study / f"{date}_{label}"
        self.created = datetime.datetime.now().isoformat(timespec="seconds")
        self.command = _command()
        self.git_commit = _git("rev-parse", "--short", "HEAD")
        status = _git("status", "--porcelain", "--untracked-files=no")
        self.git_dirty = None if status is None else bool(status)

    def record(self):
        """The ``run.json`` content.

        Returns
        -------
        record : `dict`
        """
        return {
            "study": f"{self.topic}/{self.study}",
            "label": self.label,
            "created": self.created,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "command": self.command,
            "uses": builds_read(),
        }

    def write(self):
        """Write ``run.json``.

        Returns
        -------
        path : `pathlib.Path`
            The file written.
        """
        target = self.path / RUN_FILE
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.record(), indent=1, sort_keys=True) + "\n")
        os.replace(tmp, target)
        return target


def study_run(study, label, date=None):
    """Make a dated run directory for a study and write its ``run.json``.

    The record is rewritten each time the process reads a product build it had not
    read before, so it stays complete however the run ends. A rerun with the same
    label on the same day reuses the directory and replaces ``run.json``.

    Parameters
    ----------
    study : `str`
        ``"<topic>/<study>"``, such as ``"aos/miw_hull"``.
    label : `str`
        Short run label, lowercase snake_case, such as ``"ring_coverage"``.
    date : `str`, optional
        ``YYYYMMDD``; today by default.

    Returns
    -------
    run : `StudyRun`
        ``run.path`` is the run directory.

    Raises
    ------
    `ValueError`
        If `study` is not ``"<topic>/<study>"`` or `label` is not snake_case.
    """
    topic, sep, name = study.partition("/")
    if not sep or not topic or not name or "/" in name:
        raise ValueError(f"study must be '<topic>/<study>', not {study!r}")
    if not _LABEL_RE.fullmatch(label):
        raise ValueError(f"label must be lowercase snake_case, not {label!r}")
    date = date or datetime.date.today().strftime("%Y%m%d")
    if not re.fullmatch(r"\d{8}", date):
        raise ValueError(f"date must be YYYYMMDD, not {date!r}")
    run = StudyRun(topic, name, label, date)
    run.path.mkdir(parents=True, exist_ok=True)
    _active[:] = [r for r in _active if r.path != run.path] + [run]
    run.write()
    return run
