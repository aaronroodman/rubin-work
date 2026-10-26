"""Tests for the HTML site: pages exist, links resolve, a foreign directory is refused."""

import re
import subprocess

import pytest

from rubinwork.studies import site

from rubinwork.studies.tests.test_product_map import make_repo


def test_build_site(tmp_path):
    root = make_repo(tmp_path)
    (root / "aos" / "alpha" / "notes.md").write_text("# Notes\n\nSee [a](code/a.py).\n")
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "commit", "-q", "-m", "notes"], check=True)
    out = tmp_path / "site"
    index = site.build_site(root, out, builds=[])
    assert index.is_file()
    assert (out / "products" / "miw.html").is_file()
    assert (out / "repo" / "aos" / "alpha" / "study.md.html").is_file()
    page = (out / "repo" / "aos" / "alpha" / "notes.md.html").read_text()
    assert "href=\"code/a.py.html\"" in page
    for f in out.rglob("*.html"):
        for u in re.findall(r"(?:href|src)=['\"]([^'\"#]+)", f.read_text()):
            if not re.match(r"^[a-z]+:|^\$\{", u):
                assert (f.parent / u).exists(), f"{f}: {u}"
    # rebuilding replaces its own output
    site.build_site(root, out, builds=[])


def test_refuses_foreign_directory(tmp_path):
    root = make_repo(tmp_path)
    out = tmp_path / "elsewhere"
    out.mkdir()
    (out / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        site.build_site(root, out, builds=[])
    assert (out / "keep.txt").is_file()
