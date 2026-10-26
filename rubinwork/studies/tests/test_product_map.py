"""Tests for ``PRODUCT_MAP.md``: the reader scan and the generated text."""

import os
import subprocess

from rubinwork.studies import new, product_map


def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def make_repo(tmp_path):
    root = tmp_path / "repo"
    for d in ("rubinwork/products/miw", "rubinwork/products/coadds", "aos/code"):
        (root / d).mkdir(parents=True)
        (root / d / "__init__.py").write_text("")
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.org")
    git(root, "config", "user.name", "t")
    new.create_study("aos", "alpha", "Q?", root=root, started="2026-09-01",
                     check_catalog=False)
    files = {
        # an import, and a catalog call with a literal variant
        "aos/alpha/code/a.py": ("from rubinwork.products import catalog\n"
                                "p = catalog.path('miw', 'd12-x')\n"),
        # a docstring mention is not a read
        "aos/code/doc.py": '"""Reads rubinwork.products.miw."""\n',
        # a product's builder reading another product
        "rubinwork/products/coadds/build.py": "from rubinwork.products import miw\n",
        # a topic Snakefile rule handing a resolved product path to a study script
        "aos/Snakefile": ("def miw_dir(d):\n    return _resolve('miw', d, 'x', [])\n\n"
                          "rule r:\n    input: lambda w: miw_dir(w.d)\n"
                          "    shell: 'python alpha/code/b.py'\n"),
        "aos/alpha/code/b.py": "X = 1\n",
    }
    for rel, text in files.items():
        (root / rel).write_text(text)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init")
    return root


def test_scan_readers(tmp_path):
    root = make_repo(tmp_path)
    got = product_map.scan_readers(root)
    miw = {(k, o, f, tuple(v)) for k, o, f, v in got["miw"]}
    assert ("study", "aos/alpha", "aos/alpha/code/a.py", ("d12-x",)) in miw
    assert ("product", "coadds", "rubinwork/products/coadds/build.py", ()) in miw
    assert ("study", "aos/alpha", "aos/alpha/code/b.py", ()) in miw
    assert not any(f == "aos/code/doc.py" for _, _, f, _ in got["miw"])


def test_markdown(tmp_path, monkeypatch):
    root = make_repo(tmp_path)
    monkeypatch.setenv("RUBINWORK_DATA", str(tmp_path / "data"))
    os.makedirs(tmp_path / "data" / "products")
    text = product_map.product_map_markdown(root, builds=[])
    assert text.startswith("# Product map")
    assert "| [aos/alpha](aos/alpha/study.md) |" in text
    assert "```mermaid" in text
