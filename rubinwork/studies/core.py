"""Parse and validate ``study.md`` files.

The front matter is YAML between two ``---`` lines. A value that is not valid YAML on
its own -- a long ``answer:`` holding ``": "`` unquoted, which an editor will happily
write -- is read as plain text and noted as a warning, so a study written by hand is
never rejected for punctuation.
"""

import dataclasses
import datetime
import pathlib
import re

import yaml

__all__ = [
    "STATUSES", "REQUIRED_KEYS", "SECTIONS", "Study", "Problem", "UsesEntry",
    "repo_root", "find_studies", "load_study", "parse_front_matter", "parse_uses",
    "format_uses", "validate",
]

STATUSES = ("active", "answered", "parked", "done", "dropped")
"""Allowed ``status`` values. ``answered``: the question is answered, the study may
continue."""

REQUIRED_KEYS = ("study", "topic", "status", "question", "answer", "uses", "started")

SECTIONS = ("Question", "Current answer", "Log")
"""Level-2 sections every ``study.md`` carries."""

STUDY_FILE = "study.md"

_TOP_KEY_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):(.*)$")
_LOG_ENTRY_RE = re.compile(r"^###\s+(\d{4}-\d{2}-\d{2})\b\s*(?:[—–-]+\s*)?(.*)$")


def repo_root():
    """The rubin-work checkout holding this package.

    Returns
    -------
    root : `pathlib.Path`
        Repository root.
    """
    return pathlib.Path(__file__).resolve().parents[2]


@dataclasses.dataclass(frozen=True)
class UsesEntry:
    """One product build a study reads.

    Attributes
    ----------
    product : `str`
        Product name.
    variant : `str`
        Variant name.
    build : `str`
        Build name, ``"current"`` if the entry gave no ``@build``.
    """

    product: str
    variant: str
    build: str

    def __str__(self):
        return f"{self.variant}@{self.build}"


@dataclasses.dataclass(frozen=True)
class Problem:
    """One thing wrong with a study.

    Attributes
    ----------
    study : `str`
        ``<topic>/<study>``.
    level : `str`
        ``"error"`` (the file breaks the schema) or ``"warning"``.
    message : `str`
        What is wrong.
    """

    study: str
    level: str
    message: str

    def __str__(self):
        return f"{self.level}: {self.study}: {self.message}"


@dataclasses.dataclass
class Study:
    """One parsed ``study.md``.

    Attributes
    ----------
    path : `pathlib.Path`
        The ``study.md`` file.
    meta : `dict`
        Front matter, with dates as ISO strings.
    body : `str`
        Everything after the front matter.
    notes : `list` [`str`]
        Keys read as plain text because their value was not valid YAML.
    """

    path: pathlib.Path
    meta: dict
    body: str
    notes: list = dataclasses.field(default_factory=list)

    @property
    def directory(self):
        return self.path.parent

    @property
    def name(self):
        """Study name from the directory, ``<study>``."""
        return self.directory.name

    @property
    def topic(self):
        """Topic name from the directory, ``<topic>``."""
        return self.directory.parent.name

    @property
    def key(self):
        """``<topic>/<study>``."""
        return f"{self.topic}/{self.name}"

    @property
    def status(self):
        return self.meta.get("status")

    @property
    def uses(self):
        """Parsed ``uses:`` entries; see `parse_uses`."""
        return parse_uses(self.meta.get("uses"))

    def sections(self):
        """Level-2 section titles, in file order.

        Returns
        -------
        titles : `list` [`str`]
        """
        return [line[3:].strip() for line in self.body.splitlines()
                if line.startswith("## ")]

    def log_entries(self):
        """Dated ``### YYYY-MM-DD — title`` entries under ``## Log``, in file order.

        Returns
        -------
        entries : `list` [`tuple` [`str`, `str`]]
            ``(date, title)`` pairs; the date is ISO ``YYYY-MM-DD``.
        """
        entries = []
        in_log = False
        for line in self.body.splitlines():
            if line.startswith("## "):
                in_log = line[3:].strip() == "Log"
                continue
            if in_log:
                match = _LOG_ENTRY_RE.match(line)
                if match:
                    entries.append((match.group(1), match.group(2).strip()))
        return entries

    def last_log_date(self):
        """Latest Log entry date, or `None` if the Log has no dated entry.

        Returns
        -------
        date : `str` or `None`
            ISO ``YYYY-MM-DD``.
        """
        dates = [d for d, _ in self.log_entries()]
        return max(dates) if dates else None


def _iso(value):
    """A date or datetime as an ISO string; anything else unchanged."""
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    return value


def _split_front_matter(text):
    """Split ``study.md`` text into front-matter text and body.

    Raises
    ------
    `ValueError`
        If the file does not open with a ``---`` line or the block is not closed.
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError("does not open with a '---' front-matter line")
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "".join(lines[1:i]), "".join(lines[i + 1:])
    raise ValueError("front matter has no closing '---' line")


def parse_front_matter(text):
    """Parse front-matter text into a dict.

    Each top-level key is parsed as YAML on its own. A key whose value fails is kept
    as plain text, its continuation lines joined with single spaces.

    Parameters
    ----------
    text : `str`
        The lines between the two ``---`` markers.

    Returns
    -------
    meta : `dict`
        Parsed keys, dates as ISO strings.
    notes : `list` [`str`]
        Keys read as plain text.
    """
    chunks = []
    for line in text.splitlines():
        if _TOP_KEY_RE.match(line):
            chunks.append([line])
        elif chunks:
            chunks[-1].append(line)
    meta, notes = {}, []
    for chunk in chunks:
        key = _TOP_KEY_RE.match(chunk[0]).group(1)
        try:
            parsed = yaml.safe_load("\n".join(chunk))
            value = parsed[key]
        except yaml.YAMLError:
            first = _TOP_KEY_RE.match(chunk[0]).group(2)
            value = " ".join(s.strip() for s in [first, *chunk[1:]] if s.strip())
            notes.append(key)
        if isinstance(value, dict):
            value = {k: _iso(v) for k, v in value.items()}
        meta[key] = _iso(value)
    return meta, notes


def load_study(path):
    """Read one ``study.md``.

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        The ``study.md`` file, or the study directory holding it.

    Returns
    -------
    study : `Study`

    Raises
    ------
    `ValueError`
        If the front matter is missing or unterminated.
    """
    path = pathlib.Path(path)
    if path.is_dir():
        path = path / STUDY_FILE
    front, body = _split_front_matter(path.read_text())
    meta, notes = parse_front_matter(front)
    return Study(path=path, meta=meta, body=body, notes=notes)


def find_studies(root=None, topic=None):
    """Every ``<topic>/<study>/study.md`` under the repository.

    Parameters
    ----------
    root : `str` or `pathlib.Path`, optional
        Repository root, `repo_root` by default.
    topic : `str`, optional
        Restrict to one topic.

    Returns
    -------
    paths : `list` [`pathlib.Path`]
        Sorted ``study.md`` paths.
    """
    root = pathlib.Path(root) if root else repo_root()
    pattern = f"{topic}/*/{STUDY_FILE}" if topic else f"*/*/{STUDY_FILE}"
    return sorted(p for p in root.glob(pattern)
                  if not any(part.startswith((".", "_"))
                             for part in p.relative_to(root).parts))


def parse_uses(uses):
    """Parse a ``uses:`` mapping into entries.

    Parameters
    ----------
    uses : `dict` or `None`
        ``{product: "variant@build"}``; several builds of one product are
        comma-separated. A missing ``@build`` means ``current``.

    Returns
    -------
    entries : `list` [`UsesEntry`]
        In mapping order, then string order.

    Raises
    ------
    `ValueError`
        If `uses` is not a mapping or an entry has no variant.
    """
    if not uses:
        return []
    if not isinstance(uses, dict):
        raise ValueError(f"uses must be a mapping of product to variant@build, "
                         f"not {type(uses).__name__}")
    entries = []
    for product, spec in uses.items():
        for item in str(spec).split(","):
            item = item.strip()
            if not item:
                continue
            variant, _, build = item.partition("@")
            if not variant:
                raise ValueError(f"uses entry {product}: {item!r} has no variant")
            entries.append(UsesEntry(str(product), variant, build or "current"))
    return entries


def format_uses(entries):
    """Format entries back into a ``uses:`` mapping.

    Parameters
    ----------
    entries : iterable of `UsesEntry`

    Returns
    -------
    uses : `dict` [`str`, `str`]
        Products sorted, builds comma-separated and sorted, duplicates dropped.
    """
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry.product, set()).add(str(entry))
    return {p: ", ".join(sorted(v)) for p, v in sorted(grouped.items())}


def _check_entry(entry):
    """Whether one ``uses:`` entry resolves in the catalog.

    Returns
    -------
    problem : `str` or `None`
        Why it does not resolve, or `None`.
    """
    from rubinwork.products import catalog
    try:
        catalog.path(entry.product, entry.variant, entry.build)
    except catalog.ProductNotFound as exc:
        # An incremental product (guider_moments) has partitions and no `current`; a
        # study that reads every partition names the variant alone.
        if entry.build == "current" and any(
                r["variant"] == entry.variant and "/" in r["build"]
                for r in catalog.list_products(entry.product)):
            return None
        return str(exc)
    return None


def validate(study, check_catalog=True):
    """Check one study against the schema.

    Parameters
    ----------
    study : `Study`
    check_catalog : `bool`, optional
        Also check that every ``uses:`` entry resolves through the catalog.

    Returns
    -------
    problems : `list` [`Problem`]
        Empty if the study is valid.
    """
    problems = []

    def add(level, message):
        problems.append(Problem(study.key, level, message))

    meta = study.meta
    for key in REQUIRED_KEYS:
        if key not in meta:
            add("error", f"front matter has no {key!r}")
    if "study" in meta and meta["study"] != study.name:
        add("error", f"study: {meta['study']!r} does not match its directory "
                     f"{study.name!r}")
    if "topic" in meta and meta["topic"] != study.topic:
        add("error", f"topic: {meta['topic']!r} does not match its directory "
                     f"{study.topic!r}")
    if "status" in meta and meta["status"] not in STATUSES:
        add("error", f"status: {meta['status']!r} is not one of {', '.join(STATUSES)}")
    started = meta.get("started")
    if started is not None:
        try:
            datetime.date.fromisoformat(str(started))
        except ValueError:
            add("error", f"started: {started!r} is not a YYYY-MM-DD date")
    for key in study.notes:
        add("warning", f"{key}: is not valid YAML (an unquoted ': ' or '#'?); "
                       "read as plain text")
    for section in SECTIONS:
        if section not in study.sections():
            add("warning", f"no '## {section}' section")
    try:
        entries = study.uses
    except ValueError as exc:
        add("error", str(exc))
        entries = []
    if check_catalog:
        for entry in entries:
            reason = _check_entry(entry)
            if reason:
                add("error", f"uses {entry.product}: {entry} does not resolve: "
                             f"{reason}")
    return problems
