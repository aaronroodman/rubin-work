"""Tests for ``/review``, on a temporary git repo and data root."""

import datetime
import json
import os
import subprocess

import pytest

from rubinwork.studies import new, review


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def make_build(data, product, variant, build):
    d = data / "products" / product / variant / build
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"status": "complete", "files": {}}))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setenv("RUBINWORK_DATA", str(data))
    make_build(data, "miw", "d12-a", "20261001")
    make_build(data, "miw", "d12-a", "20261010")
    os.symlink("20261010", data / "products/miw/d12-a/current")
    root = tmp_path / "repo"
    for d in ("aos/code/alpha", "aos/code/beta", "optatmo/code", "rubinwork/common"):
        (root / d).mkdir(parents=True)
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.org")
    git(root, "config", "user.name", "t")
    new.create_study("aos", "pinned", "Q?", root=root, started="2026-09-01",
                     check_catalog=False,
                     uses=[__import__("rubinwork.studies", fromlist=["x"])
                           .UsesEntry("miw", "d12-a", "20261001")])
    files = {
        "aos/code/alpha/a.py": "import b\n",
        "aos/code/beta/b.py": '"""See ../aos/output/x in docs."""\nX = 1\n',
        "aos/code/flat.py": "import b\n",
        "optatmo/code/o.py": "P = '../aos/output/wfs/x.parquet'\n",
        "aos/code/alpha/p.py": "P = '/data/products/miw/d12-a/current/x.parquet'\n",
        "rubinwork/common/lib.py": "from rubinwork.products import catalog\n",
        "aos/pinned/code/s.py": "from rubinwork.products import catalog\n",
    }
    for rel, text in files.items():
        (root / rel).write_text(text)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "init", "--date=2026-09-01T12:00:00")
    return root


def test_rule_violations(repo):
    found, unparsed = review.rule_violations(repo)
    got = {(v.path, v.rule, v.allowed) for v in found}
    assert ("aos/code/alpha/a.py", "no-study-import", False) in got
    assert ("aos/code/flat.py", "no-study-import", False) in got
    assert ("optatmo/code/o.py", "no-foreign-output", True) in got
    assert ("aos/code/alpha/p.py", "hand-built-product-path", False) in got
    assert ("rubinwork/common/lib.py", "library-imports-product", False) in got
    # A docstring is not a read; a study may import a product.
    assert not any(v.path in ("aos/code/beta/b.py", "aos/pinned/code/s.py") for v in found)
    assert unparsed == []


def test_review(repo, monkeypatch):
    commit_date = datetime.date.fromisoformat(
        git(repo, "log", "-1", "--format=%ad", "--date=short"))
    result = review.review(root=repo, today=commit_date + datetime.timedelta(days=20))
    assert [s[0] for s in result["stale"]] == ["aos/pinned"]
    assert result["stale"][0][2] == 20
    assert result["superseded"] == [("aos/pinned", "miw: d12-a@20261001", "20261010")]
    assert result["uncovered"] == {"aos": 4, "optatmo": 1}
    assert (repo / "STUDIES.md").is_file() and (repo / "PRODUCTS.md").is_file()
    text = review.report_markdown(result)
    assert "## Rule violations, new" in text and "aos/code/alpha/a.py" in text
    fresh = review.review(root=repo, today=commit_date)
    assert fresh["stale"] == []
    assert all(v is False for v in fresh["regenerated"].values())


def test_answered_without_answer_flagged(repo):
    path = repo / "aos/pinned/study.md"
    path.write_text(path.read_text().replace("status: active", "status: answered"))
    result = review.review(root=repo, regenerate=False)
    assert result["no_answer"] == ["aos/pinned"]


def test_notebook_percent_continuation(tmp_path):
    src = ["print('a %d'\n", "      % 1)\n"]
    nb = {"cells": [{"cell_type": "code", "source": src},
                    {"cell_type": "code", "source": ["%matplotlib inline\n", "x = 1\n"]}]}
    (tmp_path / "n.ipynb").write_text(json.dumps(nb))
    source = review._source(tmp_path, "n.ipynb")
    assert "% 1)" in source and "%matplotlib" not in source
