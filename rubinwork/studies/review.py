"""Review the repository's studies and dependency rules: the ``/review`` skill.

Rewrites ``STUDIES.md`` and ``PRODUCTS.md``, then reports:

- studies whose front matter breaks the schema;
- active studies untouched for more than `STALE_DAYS` days, counting from the later of
  the last Log entry and the last commit touching the study directory;
- answered or done studies with no answer;
- studies pinned to a build that is no longer ``current``;
- tracked code under a topic that no study covers, counted per topic;
- violations of the dependency rules in ``notes/status/organization_plan.md``
  section 2, split into new ones and the known cases in `ALLOWED`.

Report only: nothing here fails a test. A rule check reads source with `ast`, so a
path in a docstring or comment is not a violation.
"""

import ast
import dataclasses
import datetime
import json
import pathlib
import re
import subprocess
import warnings

from .core import find_studies, load_study, repo_root, validate
from .index import write_index as write_studies_index
from .new import topics

__all__ = ["STALE_DAYS", "ALLOWED", "Violation", "rule_violations", "review",
           "report_markdown"]

STALE_DAYS = 14

ALLOWED = (
    # The wrong-way imports and reads listed in organization_plan.md section 2, fixed
    # in phase 5. (path prefix, rule).
    # optatmo reads the aos/cwfs study's wfs_corner_compare.parquet; no product holds it.
    ("optatmo/code/extract_cwfs_sigma.py", "no-foreign-output"),
)
"""Known violations: reported, but as known rather than new."""

RULES = {
    "library-imports-product": "rule 2: a library imports a product",
    "no-study-import": "rule 2: code imports another study's module",
    "no-foreign-output": "rule 3: code reads another topic's or study's output",
    "hand-built-product-path": "rule 4: a product path built by hand, not through the "
                               "catalog",
}

_OUTPUT_RE = re.compile(r"(?:^|[/.])([a-z][a-z0-9_]*)/output(?:/|$)")
_STUDIES_RE = re.compile(r"(?:^|/)studies/([a-z][a-z0-9_]*)/([a-z][a-z0-9_]*)")
_PRODUCTS_RE = re.compile(r"(?:^|/)products/([a-z][a-z0-9_]*)(?:/|$)")


@dataclasses.dataclass(frozen=True)
class Violation:
    """One dependency-rule violation.

    Attributes
    ----------
    path : `str`
        File, relative to the repo root.
    line : `int`
    rule : `str`
        Key into `RULES`.
    detail : `str`
    allowed : `bool`
        Whether `ALLOWED` lists it.
    """

    path: str
    line: int
    rule: str
    detail: str
    allowed: bool = False


def _git(root, *args):
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                          text=True, timeout=120)
    return done.stdout if done.returncode == 0 else ""


def _tracked(root, *patterns):
    return [p for p in _git(root, "ls-files", "--", *patterns).splitlines() if p]


def _parses(src):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(src)
        return True
    except SyntaxError:
        return False


def _source(root, rel):
    """Python source of a tracked file; a notebook's code cells, magics dropped."""
    path = root / rel
    try:
        if rel.endswith(".ipynb"):
            cells = json.loads(path.read_text()).get("cells", [])
            lines = []
            for cell in cells:
                if cell.get("cell_type") == "code":
                    src = cell.get("source", "")
                    src = "".join(src) if isinstance(src, list) else src
                    if _parses(src):
                        # A line starting with % may be a continuation, not a magic.
                        lines += src.splitlines()
                    else:
                        lines += [ln for ln in src.splitlines()
                                  if not ln.lstrip().startswith(("%", "!"))]
            return "\n".join(lines)
        return path.read_text()
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _strings(tree):
    """String constants that are not docstrings, with line numbers."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(
                    getattr(body[0], "value", None), ast.Constant):
                docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            yield node.lineno, node.value


def _imports(tree):
    """Imported module names, with line numbers."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            yield node.lineno, node.module


def _module_homes(root, files, study_dirs=()):
    """Map a bare module name to the ``<topic>/code/<sub>`` or ``<topic>/<study>``
    directories defining it."""
    homes = {}
    studies = set(study_dirs)
    for rel in files:
        parts = pathlib.PurePosixPath(rel).parts
        if not rel.endswith(".py") or len(parts) != 4:
            continue
        if parts[1] == "code":
            home = "/".join(parts[:3])
        elif parts[2] == "code" and "/".join(parts[:2]) in studies:
            home = "/".join(parts[:2])
        else:
            continue
        homes.setdefault(pathlib.PurePosixPath(rel).stem, set()).add(home)
    return homes


def _unit(rel, study_dirs):
    """The study a file belongs to: a study directory, a ``code/<sub>`` dir, or None."""
    for d in study_dirs:
        if rel.startswith(d + "/"):
            return d
    parts = pathlib.PurePosixPath(rel).parts
    if len(parts) >= 4 and parts[1] in ("code", "notebooks"):
        return f"{parts[0]}/code/{parts[2]}"
    return None


def rule_violations(root=None):
    """Check every tracked ``.py`` and notebook against the section 2 rules.

    Parameters
    ----------
    root : `str` or `pathlib.Path`, optional
        Repository root, `repo_root` by default.

    Returns
    -------
    violations : `list` [`Violation`]
        Sorted by path and line.
    unparsed : `list` [`str`]
        Files that could not be parsed, and so were not checked.
    """
    root = pathlib.Path(root) if root else repo_root()
    files = _tracked(root, "*.py", "*.ipynb")
    topic_names = set(topics(root))
    study_dirs = [p.parent.relative_to(root).as_posix() for p in find_studies(root)]
    homes = _module_homes(root, files, study_dirs)
    found, unparsed = [], []

    def add(rel, line, rule, detail):
        allowed = any(rel.startswith(prefix) and rule == r for prefix, r in ALLOWED)
        found.append(Violation(rel, line, rule, detail, allowed))

    for rel in files:
        parts = pathlib.PurePosixPath(rel).parts
        if "tests" in parts or ".ipynb_checkpoints" in parts:
            continue
        in_products = rel.startswith("rubinwork/products/")
        is_library = (rel.startswith("rubinwork/") and not in_products
                      and not rel.startswith("rubinwork/studies/"))
        topic = parts[0] if parts[0] in topic_names else None
        if not (topic or is_library):
            continue
        source = _source(root, rel)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(source or "")
        except SyntaxError:
            unparsed.append(rel)
            continue
        unit = _unit(rel, study_dirs) if topic else None

        for line, name in _imports(tree):
            if is_library and (name == "rubinwork.products"
                               or name.startswith("rubinwork.products.")):
                add(rel, line, "library-imports-product", f"imports {name}")
            # Importer is a study, or flat topic-common code; a loose notebook in a
            # flat notebooks/ belongs to no study yet, so it cannot import "another".
            flat_common = len(parts) == 3 and parts[1] == "code"
            if topic and (unit or flat_common):
                where = homes.get(name.split(".")[0], set())
                if where and unit not in where and all(
                        w.split("/")[0] == topic for w in where):
                    add(rel, line, "no-study-import",
                        f"imports {name} from {', '.join(sorted(where))}")

        if in_products:
            continue
        for line, text in _strings(tree):
            match = _OUTPUT_RE.search(text)
            if topic and match and match.group(1) in topic_names and (
                    match.group(1) != topic):
                add(rel, line, "no-foreign-output", f"reads {text!r}")
            match = _STUDIES_RE.search(text)
            if match and unit and f"{match.group(1)}/{match.group(2)}" != unit:
                add(rel, line, "no-foreign-output", f"reads {text!r}")
            match = _PRODUCTS_RE.search(text)
            if match and " " not in text and not text.startswith("rubinwork/products"):
                add(rel, line, "hand-built-product-path", f"path {text!r}")
        if topic:
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and getattr(node.func, "attr", None) == "products_root"):
                    add(rel, node.lineno, "hand-built-product-path",
                        "calls catalog.products_root()")
    return sorted(found, key=lambda v: (v.path, v.line)), sorted(unparsed)


def _last_commit_date(root, rel_dir):
    out = _git(root, "log", "-1", "--format=%ad", "--date=short", "--", rel_dir).strip()
    return out or None


def _current_build(product, variant):
    from rubinwork.products import catalog
    try:
        return catalog.path(product, variant).name
    except catalog.ProductNotFound:
        return None


def review(topic=None, root=None, today=None, regenerate=True):
    """Run every check.

    Parameters
    ----------
    topic : `str`, optional
        Restrict the study checks and the uncovered-code count to one topic. The rule
        checks and the generators always cover the whole repo.
    root : `str` or `pathlib.Path`, optional
        Repository root, `repo_root` by default.
    today : `datetime.date`, optional
        For the staleness check; today by default.
    regenerate : `bool`, optional
        Rewrite ``STUDIES.md`` and ``PRODUCTS.md`` first.

    Returns
    -------
    result : `dict`
        Keys ``regenerated``, ``schema``, ``stale``, ``no_answer``, ``superseded``,
        ``uncovered``, ``violations``, ``unparsed``, ``n_studies``.
    """
    root = pathlib.Path(root) if root else repo_root()
    today = today or datetime.date.today()
    result = {"regenerated": {}}
    if regenerate:
        from rubinwork.products import catalog
        result["regenerated"]["STUDIES.md"] = write_studies_index(root)
        result["regenerated"]["PRODUCTS.md"] = catalog.write_index(root / "PRODUCTS.md")

    schema, stale, no_answer, superseded = [], [], [], []
    paths = find_studies(root, topic)
    for path in paths:
        study = load_study(path)
        schema += [p for p in validate(study)]
        rel_dir = study.directory.relative_to(root).as_posix()
        if study.status == "active":
            dates = [d for d in (study.last_log_date(), _last_commit_date(root, rel_dir))
                     if d]
            last = max(dates) if dates else None
            if last is None or (today - datetime.date.fromisoformat(last)).days > STALE_DAYS:
                age = (today - datetime.date.fromisoformat(last)).days if last else None
                stale.append((study.key, last, age))
        if study.status in ("answered", "done") and not str(
                study.meta.get("answer") or "").strip():
            no_answer.append(study.key)
        try:
            entries = study.uses
        except ValueError:
            entries = []
        for entry in entries:
            if entry.build == "current":
                continue
            current = _current_build(entry.product, entry.variant)
            if current and current != entry.build:
                superseded.append((study.key, f"{entry.product}: {entry}", current))
    result.update(schema=schema, stale=stale, no_answer=no_answer,
                  superseded=superseded, n_studies=len(paths))

    study_dirs = [p.parent.relative_to(root).as_posix() for p in find_studies(root)]
    uncovered = {}
    for rel in _tracked(root, "*.py", "*.ipynb"):
        first = rel.split("/")[0]
        if first not in topics(root) or (topic and first != topic):
            continue
        if not any(rel.startswith(d + "/") for d in study_dirs):
            uncovered[first] = uncovered.get(first, 0) + 1
    result["uncovered"] = dict(sorted(uncovered.items()))
    result["violations"], result["unparsed"] = rule_violations(root)
    return result


def report_markdown(result, topic=None):
    """The review as a short Markdown report.

    Parameters
    ----------
    result : `dict`
        From `review`.
    topic : `str`, optional
        The topic reviewed, for the heading.

    Returns
    -------
    text : `str`
    """
    out = [f"# /review{' ' + topic if topic else ''}", ""]
    regen = result["regenerated"]
    if regen:
        out.append("Generated: " + ", ".join(
            f"{k} {'rewritten' if v else 'unchanged'}" for k, v in regen.items()) + ".")
    out.append(f"Studies checked: {result['n_studies']} (count).")
    out.append("")

    def section(title, rows):
        out.append(f"## {title}")
        out.append("")
        out.extend(rows or ["None."])
        out.append("")

    section("Schema", [f"- {p}" for p in result["schema"]])
    section(f"Active, untouched for more than {STALE_DAYS} days", [
        f"- {k}: last touched {d or 'never'}"
        + (f", {a} days ago" if a is not None else "") for k, d, a in result["stale"]])
    section("Answered or done with no answer", [f"- {k}" for k in result["no_answer"]])
    section("Pinned to a superseded build", [
        f"- {k}: {u}; current is {c}" for k, u, c in result["superseded"]])
    unc = result["uncovered"]
    section("Code not covered by any study (tracked .py and .ipynb files, count)", [
        ", ".join(f"{t} {n}" for t, n in unc.items()) + f"; total {sum(unc.values())}."
    ] if unc else [])
    new = [v for v in result["violations"] if not v.allowed]
    known = [v for v in result["violations"] if v.allowed]
    section("Rule violations, new", [
        f"- `{v.path}:{v.line}` {RULES[v.rule]}: {v.detail}" for v in new])
    by_rule = {}
    for v in known:
        by_rule.setdefault(v.rule, set()).add(v.path)
    section("Rule violations, known (allow-listed, fixed in phase 5)", [
        f"- {RULES[r]}: {len(p)} files ({', '.join(sorted(p))})"
        for r, p in sorted(by_rule.items())])
    unused = [f"{prefix} ({RULES[rule]})" for prefix, rule in ALLOWED
              if not any(v.path.startswith(prefix) and v.rule == rule for v in known)]
    if unused:
        section("Allow-list entries that matched nothing", [f"- {u}" for u in unused])
    if result["unparsed"]:
        section("Not checked (did not parse)", [f"- {p}" for p in result["unparsed"]])
    return "\n".join(out).rstrip() + "\n"
