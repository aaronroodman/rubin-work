"""Tests for ``/wrap``: gathering and applying a Log entry, in a temporary git repo."""

import json
import subprocess

import pytest

from rubinwork import studies
from rubinwork.studies import new, wrap


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A temporary git repo holding one committed study."""
    monkeypatch.setenv("RUBINWORK_DATA", str(tmp_path / "data"))
    root = tmp_path / "repo"
    (root / "aos").mkdir(parents=True)
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@example.org")
    git(root, "config", "user.name", "t")
    new.create_study("aos", "demo", "How big?", root=root, started="2026-10-01",
                     check_catalog=False)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "start demo")
    return root


def test_gather_nothing_new(repo):
    got = wrap.gather("aos/demo", root=repo)
    assert got.has_study_md and got.last_log_date == "2026-10-01"
    assert got.commits == [] and got.runs == [] and not got.anything_new


def test_gather_commits_runs_and_pngs(repo):
    (repo / "aos/demo/code/a.py").write_text("x = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "demo: analysis 1")
    run_dir = repo / "aos/demo/output/20261002_first"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps({
        "created": "2026-10-02T10:00:00", "git_commit": "abc1234",
        "command": "python a.py", "uses": {"miw": ["d12-a@20261007"]}}))
    (run_dir / "plot.png").write_bytes(b"png")
    got = wrap.gather("aos/demo", root=repo)
    assert [c.subject for c in got.commits] == ["demo: analysis 1"]
    assert [r.name for r in got.runs] == ["20261002_first"]
    assert got.uses_from_runs == {"miw": "d12-a@20261007"}
    assert got.pngs == ["output/20261002_first/plot.png"]
    assert got.anything_new


def test_apply_writes_entry_and_commits(repo):
    run_dir = repo / "aos/demo/output/20261002_first"
    run_dir.mkdir(parents=True)
    (run_dir / "plot.png").write_bytes(b"png")
    sha = wrap.apply("aos/demo", "first look", "Width 12 pixels.", date="2026-10-02",
                     status="answered", answer="About 12 pixels: one ring.",
                     uses={"miw": "d12-a@20261007"},
                     figures=["output/20261002_first/plot.png"], root=repo)
    assert sha == git(repo, "rev-parse", "--short", "HEAD")
    study = studies.load_study(repo / "aos/demo")
    assert study.notes == []
    assert study.status == "answered"
    assert study.meta["answer"] == "About 12 pixels: one ring."
    assert study.meta["uses"] == {"miw": "d12-a@20261007"}
    assert study.log_entries() == [("2026-10-02", "first look"),
                                   ("2026-10-01", "started")]
    assert (repo / "aos/demo/figures/plot.png").is_file()
    assert git(repo, "status", "--porcelain") == ""
    assert "Generated with AI" in git(repo, "log", "-1", "--format=%B")
    # After the commit, the work is cited and nothing is new.
    assert not wrap.gather("aos/demo", root=repo).commits


def test_apply_bad_status(repo):
    with pytest.raises(ValueError):
        wrap.apply("aos/demo", "x", "y", status="finished", root=repo)


def test_insert_log_entry_order():
    text = "# t\n\n## Log\n\nIntro.\n\n### 2026-10-01 — a\n\nA.\n\n## Next\n\nN.\n"
    out = wrap.insert_log_entry(text, "2026-10-05", "b", "B.")
    assert out.index("### 2026-10-05 — b") < out.index("### 2026-10-01 — a")
    assert out.index("Intro.") < out.index("### 2026-10-05")
    assert out.endswith("## Next\n\nN.\n")


def test_insert_log_entry_needs_log():
    with pytest.raises(ValueError):
        wrap.insert_log_entry("# t\n", "2026-10-05", "b", "B.")


def test_set_front_matter_keeps_other_lines():
    text = ("---\nstudy: s\nanswer: long\n  continued\nuses: {}  # none\n"
            "started: 2026-10-01\n---\nbody\n")
    out = wrap.set_front_matter(text, answer="a: b", status="done")
    assert "answer: 'a: b'\n" in out and "continued" not in out
    assert "uses: {}  # none\n" in out
    assert out.endswith("status: done\n---\nbody\n")


def test_normalize_requotes_invalid_yaml():
    text = "---\nstudy: s\nanswer: No. Reason: none\n---\nbody\n"
    out = wrap.normalize(text)
    assert out != text
    meta, notes = studies.parse_front_matter(out.split("---\n")[1])
    assert notes == [] and meta["answer"] == "No. Reason: none"
    assert wrap.normalize(out) == out


def test_apply_with_nothing_to_do_makes_no_commit(repo):
    head = git(repo, "rev-parse", "HEAD")
    assert wrap.apply("aos/demo", root=repo) is None
    assert git(repo, "rev-parse", "HEAD") == head


def test_gather_backfill(repo):
    (repo / "aos/docs/studies").mkdir(parents=True)
    (repo / "aos/docs/studies/oldie.md").write_text("# oldie\n")
    (repo / "aos/oldie").mkdir()
    (repo / "aos/oldie/x.py").write_text("x = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "oldie work")
    got = wrap.gather("aos/oldie", root=repo)
    assert not got.has_study_md
    assert [c.subject for c in got.commits] == ["oldie work"]
    assert got.docs[0] == "aos/docs/studies/oldie.md"
