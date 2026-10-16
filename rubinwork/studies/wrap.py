"""Gather and apply a study's Log entry: the ``/wrap`` skill's mechanism.

``/wrap`` is two calls. `gather` collects what happened since the last Log entry: the
commits touching the study directory after the commit that last wrote ``study.md``,
uncommitted changes there, the run
directories under ``output/`` whose ``run.json`` it does not already name, the product
builds those runs read, and the newest PNGs. Claude writes the entry from that. `apply`
then inserts the entry at the top of the Log, updates ``status``, ``answer`` and
``uses:``, copies figures into ``figures/``, rewrites ``STUDIES.md`` and commits. It
never pushes.

For a study with no ``study.md`` (phase 5 backfill), `gather` takes the directory and
also lists the existing docs that mention the study by name.
"""

import dataclasses
import datetime
import json
import pathlib
import re
import shutil
import subprocess

from rubinwork.common.runs import RUN_FILE

from .core import (STATUSES, STUDY_FILE, UsesEntry, format_uses, load_study,
                   parse_uses, repo_root)
from .index import write_index
from .new import _yaml_scalar

__all__ = ["Commit", "Run", "Gathered", "gather", "insert_log_entry",
           "set_front_matter", "normalize", "apply"]

_N_PNGS = 6


@dataclasses.dataclass(frozen=True)
class Commit:
    """One commit touching the study directory."""

    sha: str
    date: str
    subject: str


@dataclasses.dataclass(frozen=True)
class Run:
    """One run directory under the study's ``output/``."""

    name: str
    created: str
    git_commit: str
    command: str
    uses: dict


@dataclasses.dataclass
class Gathered:
    """What `gather` found, for Claude to write the entry from.

    Attributes
    ----------
    key : `str`
        ``<topic>/<study>``.
    has_study_md : `bool`
    last_log_date : `str` or `None`
    commits : `list` [`Commit`]
        Newest first, not yet cited in ``study.md``.
    runs : `list` [`Run`]
        Run directories not yet named in ``study.md``, oldest first.
    uses_from_runs : `dict` [`str`, `str`]
        The builds those runs read, as a ``uses:`` mapping.
    uses_now : `dict` [`str`, `str`]
        Current ``uses:``.
    pngs : `list` [`str`]
        Newest PNGs under ``output/`` not already in ``figures/``, paths relative
        to the study directory.
    docs : `list` [`str`]
        Repo docs naming the study (backfill only).
    uncommitted : `list` [`str`]
        ``git status --porcelain`` lines for the study directory.
    invalid_yaml : `list` [`str`]
        Front-matter keys `apply` will re-quote.
    """

    key: str
    has_study_md: bool
    last_log_date: str = None
    commits: list = dataclasses.field(default_factory=list)
    runs: list = dataclasses.field(default_factory=list)
    uses_from_runs: dict = dataclasses.field(default_factory=dict)
    uses_now: dict = dataclasses.field(default_factory=dict)
    pngs: list = dataclasses.field(default_factory=list)
    docs: list = dataclasses.field(default_factory=list)
    uncommitted: list = dataclasses.field(default_factory=list)
    invalid_yaml: list = dataclasses.field(default_factory=list)

    @property
    def anything_new(self):
        """Whether there is work for a Log entry."""
        return bool(self.commits or self.runs or self.uncommitted)

    def as_dict(self):
        out = dataclasses.asdict(self)
        out["anything_new"] = self.anything_new
        return out



def _git(root, *args):
    done = subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True, timeout=60)
    if done.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout


def _commits(root, directory, after=None):
    """Commits touching `directory`, newest first, after commit `after` if given."""
    args = ["log", "--format=%h%x09%ad%x09%s", "--date=short"]
    if after:
        args.append(f"{after}..HEAD")
    out = _git(root, *args, "--", str(directory.relative_to(root)))
    commits = []
    for line in out.splitlines():
        sha, date, subject = line.split("\t", 2)
        commits.append(Commit(sha, date, subject))
    return commits


def _runs(directory):
    """Run directories with a ``run.json`` under ``output/``, oldest first."""
    output = directory / "output"
    runs = []
    if not output.is_dir():
        return runs
    for path in sorted(output.glob(f"*/{RUN_FILE}")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        runs.append(Run(path.parent.name, rec.get("created") or "",
                        rec.get("git_commit") or "", rec.get("command") or "",
                        rec.get("uses") or {}))
    return sorted(runs, key=lambda r: (r.created, r.name))


def _pngs(directory):
    """Newest PNGs under ``output/`` whose name is not already in ``figures/``."""
    output = directory / "output"
    if not output.is_dir():
        return []
    have = {p.name for p in (directory / "figures").glob("*.png")}
    found = [p for p in output.rglob("*.png") if p.name not in have]
    found.sort(key=lambda p: (-p.stat().st_mtime, str(p)))
    return [p.relative_to(directory).as_posix() for p in found[:_N_PNGS]]


def _docs_naming(root, topic, name):
    """Tracked ``.md`` files in the topic or ``notes/status/`` that name the study.

    The topic's ``docs/studies/<name>.md`` comes first when it exists: it is the
    usual source of a backfilled entry.
    """
    done = subprocess.run(["git", "-C", str(root), "grep", "-l", "-w", name, "--",
                           f"{topic}/*.md", "notes/status/*.md"],
                          capture_output=True, text=True, timeout=60)
    found = sorted(done.stdout.splitlines())
    first = f"{topic}/docs/studies/{name}.md"
    return [first] * (first in found) + [p for p in found if p != first]


def gather(study_dir, root=None):
    """Collect what happened in a study since its last Log entry.

    Parameters
    ----------
    study_dir : `str` or `pathlib.Path`
        ``<topic>/<study>``, relative to the repo root, or an absolute path.
    root : `str` or `pathlib.Path`, optional
        Repository root, `repo_root` by default.

    Returns
    -------
    gathered : `Gathered`
    """
    root = pathlib.Path(root) if root else repo_root()
    directory = pathlib.Path(study_dir)
    directory = directory if directory.is_absolute() else root / directory
    key = f"{directory.parent.name}/{directory.name}"
    study_md = directory / STUDY_FILE
    if not study_md.is_file():
        commits = _commits(root, directory) if directory.exists() else []
        return Gathered(key, False, commits=commits, runs=_runs(directory),
                        pngs=_pngs(directory), docs=_docs_naming(root, directory.parent.name,
                                                     directory.name))
    study = load_study(study_md)
    text = study_md.read_text()
    # The commit that last wrote study.md carried the last Log entry; work after it is new.
    last_md = _git(root, "log", "-1", "--format=%h", "--",
                   str(study_md.relative_to(root))).strip() or None
    commits = _commits(root, directory, after=last_md)
    runs = [r for r in _runs(directory) if r.name not in text]
    entries = [UsesEntry(p, *spec.split("@", 1))
               for r in runs for p, specs in r.uses.items() for spec in specs]
    return Gathered(key, True, study.last_log_date(), commits, runs,
                    format_uses(entries), study.meta.get("uses") or {},
                    _pngs(directory), uncommitted=_uncommitted(root, directory),
                    invalid_yaml=list(study.notes))


def _uncommitted(root, directory):
    """``git status --porcelain`` lines for the study directory."""
    out = _git(root, "status", "--porcelain", "--", str(directory.relative_to(root)))
    return [line for line in out.splitlines() if line.strip()]


def normalize(text):
    """Re-quote front-matter values that are not valid YAML on their own.

    Parameters
    ----------
    text : `str`
        The whole ``study.md``.

    Returns
    -------
    text : `str`
        Unchanged if every value already parses.
    """
    from .core import _split_front_matter, parse_front_matter
    meta, notes = parse_front_matter(_split_front_matter(text)[0])
    if not notes:
        return text
    return set_front_matter(text, **{k: meta[k] for k in notes})


def insert_log_entry(text, date, title, body):
    """Insert a dated entry at the top of the ``## Log`` section.

    Parameters
    ----------
    text : `str`
        The whole ``study.md``.
    date : `str`
        ISO date.
    title : `str`
        Entry title, after the date.
    body : `str`
        Entry text, Markdown.

    Returns
    -------
    text : `str`

    Raises
    ------
    `ValueError`
        If there is no ``## Log`` section.
    """
    lines = text.splitlines(keepends=True)
    try:
        at = next(i for i, line in enumerate(lines) if line.rstrip() == "## Log")
    except StopIteration:
        raise ValueError("study.md has no '## Log' section") from None
    entry = f"### {date} — {title}\n\n{body.strip()}\n\n"
    # After the heading and any intro text, before the first entry or next section.
    j = at + 1
    while j < len(lines) and not lines[j].startswith(("### ", "## ")):
        j += 1
    if j == len(lines) and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    if j > 0 and lines[j - 1].strip():
        entry = "\n" + entry
    if j == len(lines):
        entry = entry.rstrip("\n") + "\n"
    return "".join(lines[:j]) + entry + "".join(lines[j:])


def set_front_matter(text, **updates):
    """Replace top-level front-matter keys, leaving every other line as it was.

    Parameters
    ----------
    text : `str`
        The whole ``study.md``.
    **updates
        Key to new value. A `dict` value is written as a block mapping, anything
        else as one YAML scalar, quoted only if needed.

    Returns
    -------
    text : `str`
    """
    lines = text.splitlines(keepends=True)
    end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    head, rest = lines[1:end], lines[end:]
    key_re = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):")

    def render(key, value):
        if isinstance(value, dict):
            if not value:
                return [f"{key}: {{}}\n"]
            return [f"{key}:\n"] + [f"  {k}: {_yaml_scalar(v)}\n" for k, v in value.items()]
        return [f"{key}: {_yaml_scalar(value)}\n"]

    out, i, seen = [], 0, set()
    while i < len(head):
        match = key_re.match(head[i])
        j = i + 1
        while j < len(head) and not key_re.match(head[j]):
            j += 1
        if match and match.group(1) in updates:
            key = match.group(1)
            out += render(key, updates[key])
            seen.add(key)
        else:
            out += head[i:j]
        i = j
    for key, value in updates.items():
        if key not in seen:
            out += render(key, value)
    return "".join([lines[0], *out, *rest])


def apply(study_dir, title=None, body=None, date=None, status=None, answer=None,
          uses=None, figures=(), root=None, commit=True):
    """Write a Log entry and front-matter updates, copy figures, commit.

    Front-matter values that are not valid YAML are re-quoted on the way. With no
    `title` no Log entry is written, for a wrap that found nothing new.

    Parameters
    ----------
    study_dir : `str` or `pathlib.Path`
        ``<topic>/<study>``, relative to the repo root, or absolute.
    title : `str`, optional
        Entry title; no entry if omitted.
    body : `str`, optional
        Entry text; required with `title`.
    date : `str`, optional
        ISO date; today by default.
    status : `str`, optional
        New ``status``.
    answer : `str`, optional
        New ``answer`` (one or two sentences, with units).
    uses : `dict`, optional
        ``uses:`` entries to add, ``{product: "variant@build, ..."}``; merged
        with the existing ones.
    figures : iterable of `str`, optional
        PNGs to copy into ``figures/``, relative to the study directory.
    root : `str` or `pathlib.Path`, optional
        Repository root, `repo_root` by default.
    commit : `bool`, optional
        Commit the study's tracked files and ``STUDIES.md``. Never pushes.

    Returns
    -------
    sha : `str` or `None`
        The new commit, or `None` when not committing or nothing changed.

    Raises
    ------
    `ValueError`
        On an unknown status or a figure that is not a PNG.
    """
    root = pathlib.Path(root) if root else repo_root()
    directory = pathlib.Path(study_dir)
    directory = directory if directory.is_absolute() else root / directory
    study_md = directory / STUDY_FILE
    date = date or datetime.date.today().isoformat()
    if status is not None and status not in STATUSES:
        raise ValueError(f"status {status!r} is not one of {', '.join(STATUSES)}")

    text = study_md.read_text()
    updates = {}
    if status is not None:
        updates["status"] = status
    if answer is not None:
        updates["answer"] = " ".join(answer.split())
    if uses:
        old = load_study(study_md).meta.get("uses")
        updates["uses"] = format_uses(parse_uses(old) + parse_uses(uses))
    original = text
    text = normalize(text)
    if updates:
        text = set_front_matter(text, **updates)
    if title:
        if not body:
            raise ValueError("a Log entry needs a body")
        text = insert_log_entry(text, date, title, body)

    copied = []
    for rel in figures:
        src = directory / rel
        if src.suffix.lower() != ".png":
            raise ValueError(f"{rel} is not a PNG")
        dest = directory / "figures" / src.name
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(src, dest)
        copied.append(dest)
    if text == original and not copied:
        return None
    study_md.write_text(text)
    write_index(root)
    if not commit:
        return None
    paths = [study_md, root / "STUDIES.md", *copied]
    _git(root, "add", "--", *[str(p.relative_to(root)) for p in paths])
    key = f"{directory.parent.name}/{directory.name}"
    summary = title or "front matter re-quoted as valid YAML"
    message = (f"{directory.name}: {summary}"[:72] + f"\n\nWritten by /wrap for {key}."
               "\n\nGenerated with AI\n\nCo-Authored-By: SLAC AI\n")
    _git(root, "commit", "-q", "-m", message, "--",
         *[str(p.relative_to(root)) for p in paths])
    return _git(root, "rev-parse", "--short", "HEAD").strip()
