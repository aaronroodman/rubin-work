"""Build a static HTML site for leafing through the repository: products, studies, docs,
code, and the links between them.

Written outside the repository, to ``<data root>/site/`` (`write_site`), and opened in a
browser from ``index.html``. Every link is relative, so the tree works from ``file://``
and can be copied anywhere. The product graphs load Mermaid from a CDN; everything else is
self-contained.

Pages:

- ``index.html``: the product graph, products, studies by topic, libraries, and a search
  box over every page title.
- ``products/<name>.html``: the package docstring, every variant and build with its
  recorded inputs, the products upstream and downstream, and every reader.
- ``topics/<topic>.html``: the topic's studies, docs and the code no study covers.
- ``repo/<path>.html``: each tracked Markdown file, rendered, with front matter and the
  product links for a ``study.md``; and each tracked ``.py``, ``.ipynb`` (code cells),
  ``Snakefile`` and ``.yaml``, highlighted, with what it belongs to, which products it
  reaches and which libraries it imports.

The product reads come from `rubinwork.studies.product_map.scan_readers`; library imports
from the same kind of `ast` scan.

Run from the repository root::

    python -m rubinwork.studies site
"""

import ast
import collections
import html
import json
import os
import pathlib
import posixpath
import re
import shutil
import warnings

import markdown
import yaml
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_for_filename, PythonLexer, TextLexer

from .core import _split_front_matter, find_studies, load_study, repo_root
from .new import topics as list_topics
from .product_map import _owner, _product_names, scan_readers
from .review import _source, _tracked

__all__ = ["site_root", "build_site", "write_site"]

GITHUB = "https://github.com/aaronroodman/rubin-work/blob/main/"
MARKER = ".rubinwork_site"
MERMAID = ("<script type=\"module\">import mermaid from "
           "'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';"
           "mermaid.initialize({startOnLoad: true});</script>")
CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 0; color: #222; }
nav { background: #1f3a5f; padding: 0.5em 1em; }
nav a { color: #fff; margin-right: 1.2em; text-decoration: none; font-weight: 600; }
main { max-width: 1150px; margin: 1em auto; padding: 0 1em; line-height: 1.45; }
table { border-collapse: collapse; margin: 0.6em 0; font-size: 0.92em; }
th, td { border: 1px solid #ccc; padding: 0.25em 0.5em; vertical-align: top; }
th { background: #eef2f7; }
code { background: #f3f3f3; padding: 0 0.2em; font-size: 0.92em; }
pre { background: #f7f7f7; padding: 0.6em; overflow-x: auto; font-size: 0.85em; }
.meta { background: #f4f7fb; border-left: 4px solid #1f3a5f; padding: 0.5em 1em;
        margin: 0.8em 0; }
.tag { display: inline-block; background: #e3ebf5; border-radius: 3px; padding: 0 0.4em;
       margin: 0 0.2em 0.2em 0; font-size: 0.85em; }
.muted { color: #777; }
#q { width: 100%; font-size: 1.1em; padding: 0.4em; }
#hits li { margin: 0.15em 0; }
img { max-width: 100%; }
"""


def site_root():
    """Default output directory, ``<data root>/site``."""
    from rubinwork.products import catalog
    return catalog.data_root() / "site"


def _esc(text):
    return html.escape(str(text if text is not None else ""))


class _Site:
    """Everything one build needs: the file lists, the scans, and page paths."""

    def __init__(self, root, out):
        self.root = pathlib.Path(root)
        self.out = pathlib.Path(out)
        tracked = [p for p in _tracked(self.root, "*") if ".ipynb_checkpoints" not in p]
        self.md = sorted(p for p in tracked if p.endswith(".md")
                         and not p.startswith("notes/claude-memory/"))
        self.code = sorted(p for p in tracked if p.endswith(
            (".py", ".ipynb", "Snakefile", ".yaml", ".yml", ".sh", ".sbatch")))
        self.images = sorted(p for p in tracked if p.endswith((".png", ".jpg", ".svg")))
        self.md_set, self.code_set = set(self.md), set(self.code)
        self.img_set = set(self.images)
        self.products = _product_names(self.root)
        self.studies = {s.key: s for s in (load_study(p) for p in find_studies(self.root))}
        self.study_dirs = sorted(self.studies, key=len, reverse=True)
        self.topics = list_topics(self.root)
        self.readers = scan_readers(self.root)
        self.reaches = collections.defaultdict(set)
        for product, rows in self.readers.items():
            for _, _, rel, _ in rows:
                self.reaches[rel].add(product)
        self.libraries = self._libraries()
        self.lib_users = self._library_imports()

    # ---- paths ----------------------------------------------------------------------

    def repo_page(self, rel):
        return f"repo/{rel}.html"

    def link(self, here, target):
        """Relative URL from page `here` to page `target` (both site-relative)."""
        return posixpath.relpath(target, posixpath.dirname(here) or ".")

    def owner(self, rel):
        return _owner(rel, self.study_dirs, set(self.products))

    def owner_link(self, here, rel):
        kind, name = self.owner(rel)
        if kind == "study":
            return (f"study <a href='{self.link(here, self.repo_page(name + '/study.md'))}'>"
                    f"{_esc(name)}</a>")
        if kind == "product":
            return f"product <a href='{self.link(here, f'products/{name}.html')}'>{name}</a>"
        if kind == "topic" and name in self.topics:
            return (f"topic <a href='{self.link(here, f'topics/{name}.html')}'>{name}</a>"
                    ", in no study")
        if kind == "topic":
            return "repository root" if "/" not in rel else f"<code>{_esc(name)}/</code>"
        return f"{kind} <code>{_esc(name)}</code>"

    # ---- libraries --------------------------------------------------------------------

    def _libraries(self):
        """``{module name: file}`` for the libraries under ``rubinwork/``."""
        libs = {}
        for rel in self.code:
            parts = rel.split("/")
            if parts[0] != "rubinwork" or len(parts) < 2 or parts[1] in (
                    "products", "studies") or "tests" in parts:
                continue
            if not rel.endswith(".py"):
                continue
            mod = ".".join(parts)[:-3]
            if mod.endswith(".__init__"):
                mod = mod[:-9]
            libs[mod] = rel
        return libs

    def _library_imports(self):
        """``{library module: set of importing files}`` over tracked Python and notebooks."""
        users = collections.defaultdict(set)
        names = set(self.libraries)
        for rel in self.code:
            if not rel.endswith((".py", ".ipynb")) or "tests" in rel.split("/"):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(_source(self.root, rel) or "")
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    mods = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
                for m in mods:
                    while m and m not in names:
                        m = m.rpartition(".")[0]
                    if m and m != "rubinwork" and self.libraries.get(m) != rel:
                        users[m].add(rel)
        return users

    # ---- rendering ----------------------------------------------------------------------

    def page(self, path, title, body, mermaid=False):
        depth_prefix = "../" * path.count("/")
        nav = "".join(f"<a href='{depth_prefix}{u}'>{t}</a>" for t, u in (
            ("Home", "index.html"), ("Products", "index.html#products"),
            ("Studies", "index.html#studies"), ("Libraries", "index.html#libraries"),
            ("Search", "index.html#search")))
        text = (f"<!doctype html><html><head><meta charset='utf-8'><title>{_esc(title)}"
                f"</title><link rel='stylesheet' href='{depth_prefix}style.css'>"
                f"{MERMAID if mermaid else ''}</head><body><nav>{nav}</nav><main>"
                f"{body}</main></body></html>")
        target = self.out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def markdown(self, text, src_rel, here):
        """Markdown to HTML, with repository links pointed at site pages."""
        body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
        body = re.sub(r"<pre><code class=\"language-mermaid\">(.*?)</code></pre>",
                      lambda m: f"<pre class='mermaid'>{html.unescape(m.group(1))}</pre>",
                      body, flags=re.S)
        return re.sub(r"(href|src)=\"([^\"]+)\"",
                      lambda m: f"{m.group(1)}=\"{self._rewrite(m.group(2), src_rel, here)}\"",
                      body)

    def _rewrite(self, url, src_rel, here):
        if re.match(r"^[a-z]+:|^#|^//", url):
            return url
        path, _, frag = url.partition("#")
        frag = f"#{frag}" if frag else ""
        target = posixpath.normpath(posixpath.join(posixpath.dirname(src_rel), path))
        if target.startswith(".."):
            return url
        if target in self.md_set or target in self.code_set:
            return self.link(here, self.repo_page(target)) + frag
        if f"{target}/README.md" in self.md_set:
            return self.link(here, self.repo_page(f"{target}/README.md")) + frag
        if f"{target}/study.md" in self.md_set:
            return self.link(here, self.repo_page(f"{target}/study.md")) + frag
        if target in self.img_set:
            return self.link(here, f"repo/{target}")
        return GITHUB + target + frag

    def product_tags(self, here, products):
        return " ".join(f"<a class='tag' href='{self.link(here, f'products/{p}.html')}'>"
                        f"{p}</a>" for p in sorted(products))


def _docstring(path):
    try:
        doc = ast.get_docstring(ast.parse(path.read_text())) or ""
    except (SyntaxError, OSError):
        return ""
    return re.sub(r"``([^`]+)``", r"`\1`", doc)


def _variant_descriptions(path):
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return {}
    variants = doc.get("variants") if isinstance(doc, dict) else None
    if not isinstance(variants, dict):
        return {}
    return {k: " ".join(str(v.get("description", "")).split())
            for k, v in variants.items() if isinstance(v, dict)}


def _product_pages(s, builds):
    from rubinwork.products import catalog
    by_variant = collections.defaultdict(list)
    for row in builds:
        by_variant[(row["product"], row["variant"])].append(row)
    edges = set()
    inputs = {}
    for (product, variant), rows in by_variant.items():
        cur = [r for r in rows if r["current"]]
        shown = cur[0] if cur else max(rows, key=lambda r: (r.get("created") or "",
                                                            r["build"]))
        try:
            ins = dict(catalog.manifest(product, variant, shown["build"]).get("inputs")
                       or {})
        except Exception:
            ins = {}
        inputs[(product, variant)] = ins
        edges |= {(up, product) for up in ins if up != product}
    for p in s.products:
        here = f"products/{p}.html"
        pkg = s.root / "rubinwork" / "products" / p
        doc = _docstring(pkg / "__init__.py")
        desc = _variant_descriptions(pkg / "variants.yaml")
        ups = sorted(u for u, d in edges if d == p)
        downs = sorted(d for u, d in edges if u == p)
        graph = "\n".join([f"    {u} --> {p}" for u in ups] + [f"    {p} --> {d}"
                                                               for d in downs])
        out = [f"<h1>{p}</h1>",
               f"<p class='muted'>Source: <a href='{s.link(here, s.repo_page(f'rubinwork/products/{p}/__init__.py'))}'>"
               f"rubinwork/products/{p}/</a>. Upstream: {s.product_tags(here, ups) or 'none'}"
               f" · Downstream: {s.product_tags(here, downs) or 'none'}</p>"]
        if graph:
            out.append(f"<pre class='mermaid'>flowchart LR\n{graph}</pre>")
        out.append(s.markdown(doc, f"rubinwork/products/{p}/__init__.py", here))
        out += ["<h2>Variants and builds</h2>", "<table><tr><th>variant</th><th>current</th>"
                "<th>builds</th><th>inputs of the build shown</th><th>description</th></tr>"]
        for (prod, variant), rows in sorted(by_variant.items()):
            if prod != p:
                continue
            cur = [r["build"] for r in rows if r["current"]]
            names = ", ".join(sorted(r["build"] for r in rows))
            if len(rows) > 6:
                names = f"{len(rows)} builds, latest {max(r['build'] for r in rows)}"
            ins = "; ".join(f"{k}: {v}" for k, v in sorted(inputs[(prod, variant)].items()))
            out.append(f"<tr><td><code>{_esc(variant)}</code></td><td>{_esc(cur[0] if cur else '')}"
                       f"</td><td>{_esc(names)}</td><td>{_esc(ins)}</td>"
                       f"<td>{_esc(desc.get(variant, ''))}</td></tr>")
        out.append("</table>")
        declared = [(k, e) for k, st in s.studies.items() for e in st.uses if e.product == p]
        out.append("<h2>Readers</h2>")
        if declared:
            out.append("<p>Declared in <code>uses:</code>: " + ", ".join(
                f"<a href='{s.link(here, s.repo_page(k + '/study.md'))}'>{_esc(k)}</a> "
                f"<code>{_esc(e.variant)}@{_esc(e.build)}</code>" for k, e in sorted(
                    declared, key=lambda x: x[0])) + "</p>")
        rows = s.readers.get(p, [])
        if rows:
            out.append("<table><tr><th>file (scanned)</th><th>belongs to</th>"
                       "<th>literal variants</th></tr>")
            for kind, owner, rel, vs in sorted(rows, key=lambda r: (r[0] != "study", r[2])):
                out.append(f"<tr><td><a href='{s.link(here, s.repo_page(rel))}'>{_esc(rel)}</a>"
                           f"</td><td>{s.owner_link(here, rel)}</td>"
                           f"<td>{_esc(', '.join(vs))}</td></tr>")
            out.append("</table>")
        if not declared and not rows:
            out.append("<p>No reader found.</p>")
        s.page(here, f"product {p}", "\n".join(out), mermaid=True)
    return edges


def _doc_pages(s):
    for rel in s.md:
        here = s.repo_page(rel)
        text = (s.root / rel).read_text(errors="replace")
        head = ""
        if rel.endswith("/study.md") and rel[:-9] in s.studies:
            st = s.studies[rel[:-9]]
            try:
                _, text = _split_front_matter(text)
            except ValueError:
                pass
            scanned = {p for p, rows in s.readers.items()
                       for kind, owner, _, _ in rows if kind == "study" and owner == st.key}
            uses = ", ".join(f"<a href='{s.link(here, f'products/{e.product}.html')}'>"
                             f"{e.product}</a> <code>{_esc(e.variant)}@{_esc(e.build)}</code>"
                             for e in st.uses) or "none"
            files = sorted(f for f in s.code_set | s.md_set
                           if f.startswith(st.key + "/") and f != rel)
            head = (f"<div class='meta'><b>{_esc(st.key)}</b> · status <b>{_esc(st.status)}"
                    f"</b> · started {_esc(st.meta.get('started'))} · last log "
                    f"{_esc(st.last_log_date())}<br><b>Question:</b> {_esc(st.meta.get('question'))}"
                    f"<br><b>Answer:</b> {_esc(st.meta.get('answer')) or '<i>none yet</i>'}"
                    f"<br><b>Products declared:</b> {uses}<br><b>Products its code reaches:</b> "
                    f"{s.product_tags(here, scanned) or 'none'}<br><b>Files:</b> " + ", ".join(
                        f"<a href='{s.link(here, s.repo_page(f))}'>{_esc(f[len(st.key) + 1:])}</a>"
                        for f in files) + "</div>")
        else:
            head = (f"<p class='muted'>{_esc(rel)} · belongs to {s.owner_link(here, rel)} · "
                    f"<a href='{GITHUB}{rel}'>on GitHub</a></p>")
        s.page(here, rel, head + s.markdown(text, rel, here), mermaid="mermaid" in text)


def _code_pages(s):
    fmt = HtmlFormatter(linenos="table", cssclass="hl")
    for rel in s.code:
        here = s.repo_page(rel)
        if rel.endswith(".ipynb"):
            src, lexer = _source(s.root, rel) or "", PythonLexer()
        else:
            src = (s.root / rel).read_text(errors="replace")
            try:
                lexer = (PythonLexer() if rel.endswith("Snakefile")
                         else get_lexer_for_filename(rel))
            except Exception:
                lexer = TextLexer()
        lib = next((m for m, f in s.libraries.items() if f == rel), None)
        users = sorted(s.lib_users.get(lib, ())) if lib else []
        imports = sorted(m for m, fs in s.lib_users.items() if rel in fs)
        meta = [f"<b>{_esc(rel)}</b> · belongs to {s.owner_link(here, rel)} · "
                f"<a href='{GITHUB}{rel}'>on GitHub</a>"]
        if s.reaches.get(rel):
            meta.append("Reaches products: " + s.product_tags(here, s.reaches[rel]))
        if imports:
            meta.append("Imports libraries: " + ", ".join(
                f"<a href='{s.link(here, s.repo_page(s.libraries[m]))}'>{_esc(m)}</a>"
                for m in imports))
        if users:
            meta.append(f"Imported by {len(users)} files: " + ", ".join(
                f"<a href='{s.link(here, s.repo_page(u))}'>{_esc(u)}</a>" for u in users))
        if rel.endswith(".ipynb"):
            meta.append("Code cells only.")
        s.page(here, rel, "<div class='meta'>" + "<br>".join(meta) + "</div>"
               + highlight(src, lexer, fmt))
    return fmt.get_style_defs(".hl")


def _topic_pages(s):
    for t in s.topics:
        here = f"topics/{t}.html"
        studies = sorted(k for k in s.studies if k.startswith(t + "/"))
        docs = [m for m in s.md if m.startswith(t + "/") and not any(
            m.startswith(k + "/") for k in studies)]
        loose = [c for c in s.code if c.startswith(t + "/") and not any(
            c.startswith(k + "/") for k in studies)]
        out = [f"<h1>{t}</h1>"]
        if f"{t}/README.md" in s.md_set:
            out.append(f"<p><a href='{s.link(here, s.repo_page(f'{t}/README.md'))}'>"
                       f"{t}/README.md</a></p>")
        out += ["<h2>Studies</h2>", _study_table(s, here, studies)]
        out += ["<h2>Docs</h2>", "<ul>" + "".join(
            f"<li><a href='{s.link(here, s.repo_page(m))}'>{_esc(m)}</a></li>"
            for m in docs) + "</ul>"]
        out += [f"<h2>Code in no study ({len(loose)} files)</h2>", "<ul>" + "".join(
            f"<li><a href='{s.link(here, s.repo_page(c))}'>{_esc(c)}</a>"
            f" {s.product_tags(here, s.reaches.get(c, ()))}</li>" for c in loose) + "</ul>"]
        s.page(here, f"topic {t}", "\n".join(out))


def _study_table(s, here, keys):
    scanned = collections.defaultdict(set)
    for p, rows in s.readers.items():
        for kind, owner, _, _ in rows:
            if kind == "study":
                scanned[owner].add(p)
    rows = ["<table><tr><th>study</th><th>status</th><th>last log</th><th>question</th>"
            "<th>products</th></tr>"]
    for k in keys:
        st = s.studies[k]
        prods = {e.product for e in st.uses} | scanned[k]
        rows.append(f"<tr><td><a href='{s.link(here, s.repo_page(k + '/study.md'))}'>"
                    f"{_esc(k)}</a></td><td>{_esc(st.status)}</td>"
                    f"<td>{_esc(st.last_log_date())}</td><td>{_esc(st.meta.get('question'))}"
                    f"</td><td>{s.product_tags(here, prods)}</td></tr>")
    return "\n".join(rows) + "</table>"


def _index(s, edges, builds):
    here = "index.html"
    cur = collections.Counter(r["product"] for r in builds if r["current"])
    nvar = collections.Counter((r["product"]) for r in {(r["product"], r["variant"]): r
                                                        for r in builds}.values())
    graph = "\n".join(f"    {u} --> {d}" for u, d in sorted(edges))
    out = ["<h1>rubin-work</h1>",
           "<p>Products, studies, docs and code of this repository, and the links between "
           "them. Generated by <code>python -m rubinwork.studies site</code> (and by "
           "<code>/review</code>). Product reads are scanned from the code, so a reader "
           "here is an inference unless the study declares it in <code>uses:</code>.</p>",
           "<p>Overviews: " + ", ".join(
               f"<a href='{s.link(here, s.repo_page(m))}'>{m}</a>" for m in (
                   "rubinwork/products/README.md", "PRODUCT_MAP.md", "PRODUCTS.md",
                   "STUDIES.md", "CLAUDE.md", "notes/status/organization_plan.md")
               if m in s.md_set) + "</p>",
           "<h2 id='search'>Search</h2><input id='q' placeholder='Search page titles and "
           "study questions'><ul id='hits'></ul>",
           "<h2 id='products'>Products</h2>",
           f"<pre class='mermaid'>flowchart LR\n{graph}</pre>",
           "<table><tr><th>product</th><th>what it is</th><th>variants</th>"
           "<th>current</th><th>readers (files)</th></tr>"]
    for p in s.products:
        first = _docstring(s.root / "rubinwork" / "products" / p / "__init__.py").split(
            "\n\n")[0]
        first = re.sub(r"^The `\w+` product: ", "", " ".join(first.split()))
        out.append(f"<tr><td><a href='products/{p}.html'>{p}</a></td><td>{_esc(first)}</td>"
                   f"<td>{nvar[p]}</td><td>{cur[p]}</td>"
                   f"<td>{len(s.readers.get(p, []))}</td></tr>")
    out.append("</table>")
    out.append("<h2 id='studies'>Studies, by topic</h2>")
    for t in s.topics:
        keys = sorted(k for k in s.studies if k.startswith(t + "/"))
        out.append(f"<h3><a href='topics/{t}.html'>{t}</a> <span class='muted'>"
                   f"({len(keys)} studies)</span></h3>")
        if keys:
            out.append(_study_table(s, here, keys))
    out += ["<h2 id='libraries'>Libraries</h2>",
            "<table><tr><th>module</th><th>summary</th><th>imported by (files)</th></tr>"]
    for m, rel in sorted(s.libraries.items()):
        if m.count(".") > 2:
            continue
        first = " ".join(_docstring(s.root / rel).split("\n\n")[0].split())
        out.append(f"<tr><td><a href='{s.repo_page(rel)}'>{_esc(m)}</a></td>"
                   f"<td>{_esc(first[:160])}</td><td>{len(s.lib_users.get(m, ()))}</td></tr>")
    out.append("</table>")
    entries = ([{"t": f"product {p}", "u": f"products/{p}.html"} for p in s.products]
               + [{"t": f"study {k}: {st.meta.get('question') or ''}",
                   "u": s.repo_page(k + "/study.md")} for k, st in s.studies.items()]
               + [{"t": f"topic {t}", "u": f"topics/{t}.html"} for t in s.topics]
               + [{"t": m, "u": s.repo_page(m)} for m in s.md + s.code])
    out.append("<script>const IDX=" + json.dumps(entries) + ";"
               "const q=document.getElementById('q'),h=document.getElementById('hits');"
               "q.addEventListener('input',()=>{const w=q.value.toLowerCase().split(/\\s+/)"
               ".filter(Boolean);h.innerHTML='';if(!w.length)return;let n=0;"
               "for(const e of IDX){const t=e.t.toLowerCase();if(w.every(x=>t.includes(x)))"
               "{const li=document.createElement('li');li.innerHTML="
               "`<a href='${e.u}'>${e.t.replace(/</g,'&lt;')}</a>`;h.appendChild(li);"
               "if(++n>=60)break;}}});</script>")
    s.page(here, "rubin-work", "\n".join(out), mermaid=True)


def build_site(root=None, out=None, builds=None):
    """Write the whole site.

    Parameters
    ----------
    root : `str` or `pathlib.Path`, optional
        Repository root, `repo_root` by default.
    out : `str` or `pathlib.Path`, optional
        Output directory, `site_root` by default. Replaced as a whole.
    builds : `list` [`dict`], optional
        As from `rubinwork.products.catalog.list_products`; read by default.

    Returns
    -------
    index : `pathlib.Path`
        The site's ``index.html``.
    """
    from rubinwork.products import catalog

    root = pathlib.Path(root) if root else repo_root()
    out = pathlib.Path(out) if out else site_root()
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    s = _Site(root, tmp)
    builds = catalog.list_products() if builds is None else builds
    edges = _product_pages(s, builds)
    _doc_pages(s)
    hl_css = _code_pages(s)
    _topic_pages(s)
    _index(s, edges, builds)
    for rel in s.images:
        dst = tmp / "repo" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, dst)
    (tmp / "style.css").write_text(CSS + hl_css)
    (tmp / MARKER).write_text("generated by rubinwork.studies.site\n")
    if out.exists():
        if not (out / MARKER).is_file():
            raise FileExistsError(f"{out} exists and was not written by this builder; "
                                  "not replacing it")
        shutil.rmtree(out)
    os.replace(tmp, out)
    return out / "index.html"


def write_site(root=None, out=None):
    """`build_site` with the defaults; the command-line entry."""
    return build_site(root, out)
