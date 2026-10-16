"""Create a study: the ``/start-study`` skill's mechanism.

Makes ``<topic>/<study>/`` with ``study.md``, ``code/``, ``notebooks/``, ``figures/``,
the gitignored ``output`` symlink to ``<data root>/studies/<topic>/<study>`` and that
data directory, then rewrites ``STUDIES.md``::

    python -m rubinwork.studies new aos coadd_vs_miw \\
        --question "Why do per-block FAM coadds disagree with the pooled MIW?" \\
        --uses fam_tables=danish_1_2@20261014 --uses miw=d12-A_50_34_i_5rot
"""

import datetime
import pathlib
import re

import yaml

from rubinwork.common.runs import data_root

from .core import STUDY_FILE, UsesEntry, find_studies, format_uses, repo_root
from .index import write_index

__all__ = ["NAME_RE", "topics", "product_variants", "study_text", "create_study"]

NAME_RE = re.compile(r"[a-z][a-z0-9_]{1,47}")
"""A study or topic name: lowercase snake_case, 2 to 48 characters."""

SUBDIRS = ("code", "notebooks", "figures")

# Top-level directories that are not topics.
_NOT_TOPICS = {"rubinwork", "common", "notes", "output", "logs", "scratch", "data"}


def topics(root=None):
    """Topic directories at the repository root.

    Parameters
    ----------
    root : `str` or `pathlib.Path`, optional
        Repository root, `repo_root` by default.

    Returns
    -------
    names : `list` [`str`]
        Sorted topic names.
    """
    root = pathlib.Path(root) if root else repo_root()
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and not p.is_symlink() and NAME_RE.fullmatch(p.name)
                  and p.name not in _NOT_TOPICS and not p.name.endswith("egg-info"))


def product_variants(root=None):
    """Every product's ``variants.yaml``, and the variant names it defines.

    For ``/start-study`` to answer "is this a product change?": a request that changes
    a product's configuration is a new variant in this file, not a study. Names come
    from each product package's own ``variants()``, which knows its file's layout.

    Parameters
    ----------
    root : `str` or `pathlib.Path`, optional
        Repository root, `repo_root` by default.

    Returns
    -------
    variants : `dict` [`str`, `tuple` [`pathlib.Path`, `list` [`str`]]]
        ``{product: (variants.yaml path, sorted variant names)}``; the list is empty
        if the package cannot be imported here.
    """
    import importlib
    root = pathlib.Path(root) if root else repo_root()
    out = {}
    for path in sorted((root / "rubinwork" / "products").glob("*/variants.yaml")):
        product = path.parent.name
        try:
            module = importlib.import_module(f"rubinwork.products.{product}")
            names = sorted(str(v) for v in module.variants())
        except Exception:
            names = []
        out[product] = (path, names)
    return out


def _yaml_scalar(text):
    """One front-matter value, quoted only if plain YAML would misread it."""
    dumped = yaml.safe_dump(text, default_flow_style=True, width=10**6,
                            allow_unicode=True).strip()
    return dumped.removesuffix("\n...").removesuffix("...").strip()


def study_text(topic, name, question, uses=(), started=None, title=None):
    """The text of a new ``study.md``.

    Parameters
    ----------
    topic : `str`
        Topic name.
    name : `str`
        Study name.
    question : `str`
        The question, one sentence.
    uses : iterable of `UsesEntry`, optional
        Products the study reads.
    started : `str`, optional
        ISO date; today by default.
    title : `str`, optional
        H1 title text after the study name.

    Returns
    -------
    text : `str`
    """
    started = started or datetime.date.today().isoformat()
    uses_map = format_uses(uses)
    if uses_map:
        uses_lines = "uses:\n" + "".join(f"  {p}: {_yaml_scalar(v)}\n"
                                         for p, v in uses_map.items())
    else:
        uses_lines = "uses: {}\n"
    heading = f"# `{name}`" + (f": {title}" if title else "")
    return (
        "---\n"
        f"study: {name}\n"
        f"topic: {topic}\n"
        "status: active\n"
        f"question: {_yaml_scalar(question)}\n"
        "answer: ''\n"
        f"{uses_lines}"
        f"started: {started}\n"
        "---\n"
        "\n"
        f"{heading}\n"
        "\n"
        "## Question\n"
        "\n"
        f"{question}\n"
        "\n"
        "## Current answer\n"
        "\n"
        "None yet.\n"
        "\n"
        "## Log\n"
        "\n"
        f"### {started} — started\n"
        "\n"
        "Study created.\n"
    )


def create_study(topic, name, question, uses=(), root=None, started=None, title=None,
                 new_topic=False, check_catalog=True):
    """Create a study directory, its data directory and its ``study.md``.

    Parameters
    ----------
    topic : `str`
        Topic name; must exist unless `new_topic`.
    name : `str`
        Study name, lowercase snake_case, unique across topics.
    question : `str`
        The question, one sentence.
    uses : iterable of `UsesEntry`, optional
        Products the study reads.
    root : `str` or `pathlib.Path`, optional
        Repository root, `repo_root` by default.
    started : `str`, optional
        ISO date; today by default.
    title : `str`, optional
        H1 title text.
    new_topic : `bool`, optional
        Allow `topic` not to exist yet.
    check_catalog : `bool`, optional
        Refuse a ``uses`` entry that does not resolve through the catalog, and pin
        ``current`` to the build it names.

    Returns
    -------
    directory : `pathlib.Path`
        The new study directory.

    Raises
    ------
    `ValueError`
        If a name is malformed, the topic is unknown, the study exists (in any
        topic), the directory is not empty, or a ``uses`` entry does not resolve.
    """
    root = pathlib.Path(root) if root else repo_root()
    uses = list(uses)
    for label, value in (("topic", topic), ("study", name)):
        if not NAME_RE.fullmatch(value):
            raise ValueError(f"{label} name {value!r} is not lowercase snake_case "
                             "of 2 to 48 characters")
    if not new_topic and topic not in topics(root):
        raise ValueError(f"no topic {topic!r}; known topics: {', '.join(topics(root))}")
    clash = [p.parent for p in find_studies(root) if p.parent.name == name]
    if clash:
        raise ValueError(f"a study named {name!r} already exists: "
                         f"{clash[0].relative_to(root)}")
    directory = root / topic / name
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f"{directory} exists and is not empty")
    if check_catalog:
        from rubinwork.products import catalog
        # Pin: "current" moves, and a study's answer must name what it read.
        uses = [UsesEntry(e.product, e.variant,
                          catalog.path(e.product, e.variant, e.build).name)
                for e in uses]

    data_dir = data_root() / "studies" / topic / name
    data_dir.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (directory / sub).mkdir(parents=True, exist_ok=True)
    (directory / "output").symlink_to(data_dir)
    (directory / STUDY_FILE).write_text(
        study_text(topic, name, question, uses, started=started, title=title))
    write_index(root)
    return directory


def parse_uses_arg(text):
    """One ``--uses product=variant[@build]`` argument as a `UsesEntry`.

    Raises
    ------
    `ValueError`
        If the argument has no ``=`` or no variant.
    """
    product, sep, spec = text.partition("=")
    variant, _, build = spec.partition("@")
    if not sep or not product or not variant:
        raise ValueError(f"--uses wants product=variant[@build], not {text!r}")
    return UsesEntry(product, variant, build or "current")
