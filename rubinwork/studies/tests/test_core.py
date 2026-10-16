"""Tests for the ``study.md`` parser and validator."""

import pathlib

import pytest

from rubinwork import studies
from rubinwork.products import catalog

REPO = studies.repo_root()
REAL_STUDIES = ("aos/miw_hull", "aos/unpaired_intra_vs_extra", "camera/itl_dip")

GOOD = """\
---
study: demo
topic: aos
status: active
question: Does it work?
answer: Yes, about 3% of sky.
uses:
  fam_tables: danish_1_2@20261005
  miw: d12-a@20261007, d12-b
started: 2026-08-02
---

# demo

## Question

Q.

## Current answer

A.

## Log

### 2026-10-02 — second

### 2026-09-01 — first
"""


def write_study(root, topic, name, text=GOOD):
    """Write a study.md at ``root/topic/name`` and return its path."""
    directory = root / topic / name
    directory.mkdir(parents=True)
    path = directory / "study.md"
    path.write_text(text)
    return path


@pytest.mark.parametrize("key", REAL_STUDIES)
def test_real_studies_parse_as_they_are(key):
    study = studies.load_study(REPO / key / "study.md")
    problems = studies.validate(study, check_catalog=False)
    assert [p for p in problems if p.level == "error"] == []
    assert study.key == key
    assert study.status in studies.STATUSES
    assert study.meta["answer"]
    assert study.last_log_date() is not None


def test_real_studies_found():
    keys = {f"{p.parent.parent.name}/{p.parent.name}"
            for p in studies.find_studies(REPO)}
    assert set(REAL_STUDIES) <= keys


def test_real_uses_resolve_in_catalog():
    if not catalog.products_root().is_dir():
        pytest.skip("no data root here")
    for key in REAL_STUDIES:
        study = studies.load_study(REPO / key)
        errors = [p for p in studies.validate(study) if p.level == "error"]
        assert errors == [], key


def test_parse_good(tmp_path):
    study = studies.load_study(write_study(tmp_path, "aos", "demo"))
    assert study.meta["started"] == "2026-08-02"
    assert study.uses == [
        studies.UsesEntry("fam_tables", "danish_1_2", "20261005"),
        studies.UsesEntry("miw", "d12-a", "20261007"),
        studies.UsesEntry("miw", "d12-b", "current"),
    ]
    assert study.sections() == ["Question", "Current answer", "Log"]
    assert study.log_entries() == [("2026-10-02", "second"), ("2026-09-01", "first")]
    assert study.last_log_date() == "2026-10-02"
    assert studies.validate(study, check_catalog=False) == []


def test_invalid_yaml_value_kept_as_text(tmp_path):
    text = GOOD.replace("answer: Yes, about 3% of sky.",
                        "answer: No. Reason: it was\n  never tried.")
    study = studies.load_study(write_study(tmp_path, "aos", "demo", text))
    assert study.meta["answer"] == "No. Reason: it was never tried."
    problems = studies.validate(study, check_catalog=False)
    assert [p.level for p in problems] == ["warning"]


def test_status_answered_allowed_and_bad_status_rejected(tmp_path):
    ok = studies.load_study(write_study(
        tmp_path, "aos", "a", GOOD.replace("study: demo", "study: a")
        .replace("status: active", "status: answered")))
    assert studies.validate(ok, check_catalog=False) == []
    bad = studies.load_study(write_study(
        tmp_path, "aos", "b", GOOD.replace("study: demo", "study: b")
        .replace("status: active", "status: finished")))
    messages = [p.message for p in studies.validate(bad, check_catalog=False)]
    assert any("status" in m for m in messages)


def test_missing_keys_and_mismatched_names(tmp_path):
    text = GOOD.replace("question: Does it work?\n", "").replace("topic: aos", "topic: wfs")
    study = studies.load_study(write_study(tmp_path, "aos", "demo", text))
    messages = [p.message for p in studies.validate(study, check_catalog=False)]
    assert any("'question'" in m for m in messages)
    assert any("topic" in m and "wfs" in m for m in messages)


def test_no_front_matter(tmp_path):
    path = write_study(tmp_path, "aos", "demo", "# demo\n")
    with pytest.raises(ValueError):
        studies.load_study(path)


def test_uses_resolve_through_catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("RUBINWORK_DATA", str(tmp_path / "data"))
    build = tmp_path / "data/products/fam_tables/danish_1_2/20261005"
    build.mkdir(parents=True)
    (build / "manifest.json").write_text('{"status": "complete", "files": {}}')
    text = GOOD.replace("  miw: d12-a@20261007, d12-b\n", "")
    study = studies.load_study(write_study(tmp_path / "repo", "aos", "demo", text))
    assert studies.validate(study) == []
    study.meta["uses"] = {"fam_tables": "danish_1_2@20990101"}
    problems = studies.validate(study)
    assert len(problems) == 1 and "does not resolve" in problems[0].message


def test_format_uses_round_trip():
    entries = studies.parse_uses({"miw": "b@2, a@1", "fam_tables": "x@1"})
    assert studies.format_uses(entries + entries[:1]) == {
        "fam_tables": "x@1", "miw": "a@1, b@2"}
    assert studies.parse_uses({}) == []
    with pytest.raises(ValueError):
        studies.parse_uses(["miw"])


def test_find_studies_skips_hidden_and_filters_topic(tmp_path):
    write_study(tmp_path, "aos", "demo")
    write_study(tmp_path, "camera", "dip", GOOD.replace("demo", "dip"))
    write_study(tmp_path, "aos", ".ipynb_checkpoints")
    found = studies.find_studies(tmp_path)
    assert [p.parent.name for p in found] == ["demo", "dip"]
    assert [p.parent.name for p in studies.find_studies(tmp_path, "camera")] == ["dip"]


def test_studies_index(tmp_path):
    from rubinwork.studies import index
    write_study(tmp_path, "aos", "demo")
    write_study(tmp_path, "camera", "dip", GOOD.replace("demo", "dip")
                .replace("answer: Yes, about 3% of sky.", "answer: a | b"))
    text = index.studies_markdown(tmp_path)
    rows = [line for line in text.splitlines() if line.startswith("| [")]
    assert [r.split("]")[0] for r in rows] == ["| [aos/demo", "| [camera/dip"]
    assert "a \\| b" in rows[1]
    assert "fam_tables: danish_1_2@20261005; miw: d12-a@20261007, d12-b" in rows[0]
    assert "Last updated:** 2026-10-02" in text


def test_studies_index_is_deterministic(tmp_path):
    from rubinwork.studies import index
    write_study(tmp_path, "aos", "demo")
    assert index.write_index(tmp_path) is True
    first = (tmp_path / "STUDIES.md").read_text()
    assert index.write_index(tmp_path) is False
    assert (tmp_path / "STUDIES.md").read_text() == first
