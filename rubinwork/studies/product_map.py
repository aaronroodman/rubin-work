"""Write ``PRODUCT_MAP.md``: how the products feed each other and which code reads each.

Three sources, each named in the output so a reader knows how much to trust a line:

- **manifest** — a build's ``inputs``, the product builds it was made from. Exact.
- **declared** — a study's ``uses:`` front matter. Exact, but most studies leave it empty.
- **scanned** — the tracked code that reaches a product: an import of
  ``rubinwork.products.<name>``, or a ``catalog.path|load|manifest`` call whose first
  argument is the product name as a literal. Read with `ast` for ``.py`` and notebooks and
  with a regular expression for Snakefiles, so a docstring or comment does not count. A
  Snakefile rule whose inputs come from a product resolver also counts as a read by every
  script that rule runs, since those scripts are handed the product path as an argument.
  An inference: a module imported for one helper counts as a read.

Code is attributed to a study when it sits in a study directory, to a product when it is
another product's builder, and otherwise to its topic or library.
"""

import ast
import collections
import pathlib
import re
import warnings

from .core import find_studies, load_study, repo_root
from .review import _source, _tracked

__all__ = ["scan_readers", "product_map_markdown", "write_map"]

_CALLS = ("path", "load", "manifest", "set_current", "register")
_SNAKE_RE = re.compile(
    r"rubinwork\.products(?:\.([a-z_]+)|\s+import\s+([a-z_, ]+))"
    r"|catalog\.(?:path|load|manifest)\(\s*['\"]([a-z_]+)['\"]")


def _product_names(root):
    d = root / "rubinwork" / "products"
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.is_dir() and (p / "__init__.py").is_file())


def _from_import(names, module, products):
    """Products named by one import statement."""
    found = set()
    if module == "rubinwork.products":
        found |= {n for n in names if n in products}
    elif module.startswith("rubinwork.products."):
        head = module.split(".")[2]
        if head in products:
            found.add(head)
    return found


def _scan_python(source, products):
    """``{product: set of literal variants}`` reached by one Python source."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source or "")
    except SyntaxError:
        return None
    hits = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for p in _from_import([a.name for a in node.names], node.module, products):
                hits.setdefault(p, set())
        elif isinstance(node, ast.Import):
            for alias in node.names:
                for p in _from_import([], alias.name, products):
                    hits.setdefault(p, set())
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr in _CALLS and node.args
              and isinstance(node.args[0], ast.Constant)
              and node.args[0].value in products):
            variants = hits.setdefault(node.args[0].value, set())
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) \
                    and isinstance(node.args[1].value, str):
                variants.add(node.args[1].value)
    return hits


def _scan_snakefile(source, products):
    hits = {}
    for m in _SNAKE_RE.finditer(source or ""):
        names = [m.group(1)] if m.group(1) else (
            [n.strip() for n in m.group(2).split(",")] if m.group(2) else [m.group(3)])
        for n in names:
            if n in products:
                hits.setdefault(n, set())
    return hits


_DEF_RE = re.compile(r"^def ([A-Za-z_][A-Za-z0-9_]*)\(", re.M)
_RULE_RE = re.compile(r"^rule ([A-Za-z_][A-Za-z0-9_]*):", re.M)
_RESOLVE_RE = re.compile(
    r"(?:_resolve|_?catalog\.(?:path|load|manifest))\(\s*['\"]([a-z_]+)['\"]")
_ALIAS_RE = re.compile(r"_?([a-z_]+)\.PRODUCT\b")
_SCRIPT_RE = re.compile(r"[A-Za-z0-9_./-]+\.(?:py|ipynb)\b")


def _blocks(source, pattern):
    """``{name: text}`` of each top-level block opened by `pattern`."""
    starts = [(m.start(), m.group(1)) for m in pattern.finditer(source)]
    ends = [m.start() for m in re.finditer(r"^\S", source, re.M)]
    out = {}
    for pos, name in starts:
        nxt = [e for e in ends if e > pos]
        out[name] = source[pos:nxt[0] if nxt else len(source)]
    return out


def _snakefile_rule_reads(rel, source, products):
    """``{script file: products}`` for the scripts a Snakefile's rules run.

    A helper ``def`` reaches a product if its body resolves one (``_resolve("miw", ...)``,
    ``catalog.path(...)``, ``<product>.PRODUCT``) or calls a helper that does. A rule
    reaches the products of the helpers it calls, and its scripts are the ``.py`` and
    ``.ipynb`` paths in its text, resolved against the Snakefile's directory.
    """
    defs = _blocks(source, _DEF_RE)
    direct = {}
    for name, body in defs.items():
        found = {m for m in _RESOLVE_RE.findall(body) if m in products}
        found |= {m for m in _ALIAS_RE.findall(body) if m in products}
        direct[name] = found
    reach = {n: set(v) for n, v in direct.items()}
    for _ in range(len(defs)):
        changed = False
        for name, body in defs.items():
            for other in defs:
                if other != name and re.search(rf"\b{other}\(", body):
                    new = reach[other] - reach[name]
                    if new:
                        reach[name] |= new
                        changed = True
        if not changed:
            break
    base = pathlib.PurePosixPath(rel).parent
    out = collections.defaultdict(set)
    for _, body in _blocks(source, _RULE_RE).items():
        used = set()
        for name, prods in reach.items():
            if prods and re.search(rf"\b{name}\(", body):
                used |= prods
        if not used:
            continue
        for path in _SCRIPT_RE.findall(body):
            if path.startswith(("{", "$")) or "{" in path:
                continue
            target = pathlib.PurePosixPath(*(
                p for p in (base / path).parts))
            parts = []
            for part in target.parts:
                if part == "..":
                    if parts:
                        parts.pop()
                elif part != ".":
                    parts.append(part)
            out["/".join(parts)] |= used
    return out


def _owner(rel, study_dirs, products):
    """``(kind, name)`` of the unit a tracked file belongs to."""
    for d in study_dirs:
        if rel.startswith(d + "/"):
            return "study", d
    parts = pathlib.PurePosixPath(rel).parts
    if parts[0] == "rubinwork":
        if len(parts) > 2 and parts[1] == "products" and parts[2] in products:
            return "product", parts[2]
        if len(parts) > 2 and parts[1] == "studies":
            return "tooling", "rubinwork.studies"
        name = parts[1][:-3] if parts[1].endswith(".py") else parts[1]
        return "library", f"rubinwork.{name}"
    return "topic", parts[0]


def scan_readers(root=None):
    """Every tracked file that reaches a product, by product.

    Parameters
    ----------
    root : `str` or `pathlib.Path`, optional
        Repository root, `repo_root` by default.

    Returns
    -------
    readers : `dict` [`str`, `list` [`tuple`]]
        ``{product: [(kind, owner, file, variants), ...]}``, where ``kind`` is
        ``study``, ``product``, ``library``, ``topic`` or ``tooling``, ``owner`` names the
        unit, and ``variants`` is the sorted literal variants seen (often empty). A
        product's own files are left out.
    """
    root = pathlib.Path(root) if root else repo_root()
    products = set(_product_names(root))
    study_dirs = sorted((p.parent.relative_to(root).as_posix()
                         for p in find_studies(root)), key=len, reverse=True)
    readers = collections.defaultdict(list)
    tracked = set(_tracked(root, "*.py", "*.ipynb", "*Snakefile"))
    via_rule = collections.defaultdict(set)
    for rel in sorted(tracked):
        if rel.endswith("Snakefile") and not rel.startswith("rubinwork/products/"):
            for script, prods in _snakefile_rule_reads(
                    rel, (root / rel).read_text(), products).items():
                if script in tracked:
                    via_rule[script] |= prods
    for rel in sorted(tracked):
        parts = pathlib.PurePosixPath(rel).parts
        if "tests" in parts or ".ipynb_checkpoints" in parts:
            continue
        if rel.endswith("Snakefile"):
            hits = _scan_snakefile((root / rel).read_text(), products)
        else:
            hits = _scan_python(_source(root, rel), products)
        hits = hits if hits is not None else {}
        for prod in via_rule.get(rel, ()):
            hits.setdefault(prod, set())
        if not hits:
            continue
        kind, owner = _owner(rel, study_dirs, products)
        for product, variants in hits.items():
            if kind == "product" and owner == product:
                continue
            readers[product].append((kind, owner, rel, sorted(variants)))
    return {p: sorted(v) for p, v in sorted(readers.items())}


def _cell(text):
    return " ".join(str(text or "").split()).replace("|", "\\|")


def product_map_markdown(root=None, builds=None, readers=None):
    """The text of ``PRODUCT_MAP.md``.

    Parameters
    ----------
    root : `str` or `pathlib.Path`, optional
        Repository root, `repo_root` by default.
    builds : `list` [`dict`], optional
        As from `rubinwork.products.catalog.list_products`; read by default.
    readers : `dict`, optional
        As from `scan_readers`; scanned by default.

    Returns
    -------
    text : `str`
        Markdown. Deterministic: it depends only on the manifests, the ``study.md``
        files and the tracked code.
    """
    from rubinwork.products import catalog

    root = pathlib.Path(root) if root else repo_root()
    builds = catalog.list_products() if builds is None else builds
    readers = scan_readers(root) if readers is None else readers
    studies = {s.key: s for s in (load_study(p) for p in find_studies(root))}

    variants = collections.defaultdict(list)
    for row in builds:
        variants[(row["product"], row["variant"])].append(row)
    shown = {}
    for key, rows in variants.items():
        cur = [r for r in rows if r["current"]]
        shown[key] = cur[0] if cur else max(
            rows, key=lambda r: (r.get("created") or "", r["build"]))
    inputs = {}
    for (product, variant), row in shown.items():
        try:
            man = catalog.manifest(product, variant, row["build"])
        except Exception:
            man = {}
        inputs[(product, variant)] = dict(man.get("inputs") or {})

    edges = collections.Counter()
    for (product, _), ins in inputs.items():
        for up in ins:
            edges[(up, product)] += 1
    declared = collections.defaultdict(list)
    for key, s in studies.items():
        for e in s.uses:
            declared[e.product].append((key, f"{e.variant}@{e.build}"))
    products = sorted({p for p, _ in variants} | set(readers) | set(declared))
    created = [r["created"] for r in builds if r.get("created")]
    updated = max(created)[:10] if created else "never"

    out = [
        "# Product map",
        "",
        f"> **Status:** generated · **Last updated:** {updated} · **Kind:** reference "
        "(index)",
        "",
        "How the products feed each other, and which studies and code read each one. "
        "Generated by `python -m rubinwork.studies map` (and by `/review`); do not edit by "
        "hand. What each product is: [`rubinwork/products/README.md`]"
        "(rubinwork/products/README.md). Builds: [`PRODUCTS.md`](PRODUCTS.md). Studies: "
        "[`STUDIES.md`](STUDIES.md).",
        "",
        "Sources: **manifest** — a build's recorded inputs (exact); **declared** — a "
        "study's `uses:` (exact, but most studies leave it empty); **scanned** — tracked "
        "code that imports `rubinwork.products.<name>` or calls `catalog.path|load|"
        "manifest` with the product name as a literal (an inference).",
        "",
        "## Product graph (manifest)",
        "",
        "Arrows point from an input product to the product built from it; the number is "
        "the count of variants whose shown build (current, else latest) records that "
        "input.",
        "",
        "```mermaid",
        "flowchart LR",
    ]
    for (up, down), n in sorted(edges.items()):
        out.append(f"    {up} -->|{n}| {down}")
    for p in products:
        if not any(p in e for e in edges):
            out.append(f"    {p}")
    out += ["```", "", "## Variants and their inputs (manifest)", "",
            "| product | variant | build shown | current | inputs |",
            "|---|---|---|---|---|"]
    for (product, variant), row in sorted(shown.items()):
        ins = "; ".join(f"{k}: {v}" for k, v in sorted(inputs[(product, variant)].items()))
        out.append(f"| {product} | `{variant}` | {row['build']} | "
                   f"{'yes' if row['current'] else 'no'} | {_cell(ins)} |")

    out += ["", "## Readers by product", ""]
    for p in products:
        out += [f"### {p}", ""]
        downs = sorted(d for (u, d) in edges if u == p)
        if downs:
            out.append("Built into (manifest): " + ", ".join(f"`{d}`" for d in downs) + ".")
            out.append("")
        if declared.get(p):
            out.append("Declared by studies (`uses:`): " + "; ".join(
                f"[{k}]({k}/study.md) {v}" for k, v in sorted(declared[p])) + ".")
            out.append("")
        rows = readers.get(p, [])
        if rows:
            out += ["| reader | kind | files (scanned) | literal variants |",
                    "|---|---|---|---|"]
            grouped = collections.defaultdict(lambda: ([], set()))
            for kind, owner, rel, vs in rows:
                grouped[(kind, owner)][0].append(rel)
                grouped[(kind, owner)][1].update(vs)
            order = {"study": 0, "product": 1, "library": 2, "topic": 3, "tooling": 4}
            for (kind, owner), (files, vs) in sorted(
                    grouped.items(), key=lambda kv: (order[kv[0][0]], kv[0][1])):
                name = (f"[{owner}]({owner}/study.md)" if kind == "study"
                        else f"`{owner}`")
                kind_txt = "topic code, no study" if kind == "topic" else kind
                out.append(f"| {name} | {kind_txt} | {len(files)}: "
                           f"{_cell(', '.join(f'`{f}`' for f in files))} | "
                           f"{_cell(', '.join(f'`{v}`' for v in sorted(vs)))} |")
            out.append("")
        if not downs and not declared.get(p) and not rows:
            out += ["No reader found.", ""]

    out += ["## Studies and the products they use", "",
            "A study listed here reads at least one product. **undeclared** marks a "
            "product its code reaches that its `uses:` does not name.", "",
            "| study | status | declared (`uses:`) | scanned | undeclared |",
            "|---|---|---|---|---|"]
    scanned = collections.defaultdict(set)
    for p, rows in readers.items():
        for kind, owner, _, _ in rows:
            if kind == "study":
                scanned[owner].add(p)
    for key in sorted(set(scanned) | {k for k, s in studies.items() if s.uses}):
        s = studies.get(key)
        decl = sorted({e.product for e in s.uses}) if s else []
        undecl = sorted(scanned[key] - set(decl))
        out.append(f"| [{key}]({key}/study.md) | {_cell(s.status if s else '')} | "
                   f"{', '.join(decl)} | {', '.join(sorted(scanned[key]))} | "
                   f"{', '.join(undecl)} |")
    return "\n".join(out) + "\n"


def write_map(root=None, target=None):
    """Write ``PRODUCT_MAP.md``, touching the file only if its text changes.

    Parameters
    ----------
    root : `str` or `pathlib.Path`, optional
        Repository root, `repo_root` by default.
    target : `str` or `pathlib.Path`, optional
        Output file, ``<root>/PRODUCT_MAP.md`` by default.

    Returns
    -------
    changed : `bool`
        Whether the file was written.
    """
    root = pathlib.Path(root) if root else repo_root()
    target = pathlib.Path(target) if target else root / "PRODUCT_MAP.md"
    text = product_map_markdown(root)
    if target.is_file() and target.read_text() == text:
        return False
    target.write_text(text)
    return True
