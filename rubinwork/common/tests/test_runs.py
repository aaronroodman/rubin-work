"""Tests for `rubinwork.common.study_run` and the catalog's read recording.

``RUBINWORK_DATA`` points at a pytest ``tmp_path`` throughout.
"""

import json

import pytest

from rubinwork.common import runs, study_run
from rubinwork.products import catalog, manifest


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    monkeypatch.setenv("RUBINWORK_DATA", str(tmp_path))
    monkeypatch.setattr(runs, "_reads", {})
    monkeypatch.setattr(runs, "_active", [])
    monkeypatch.setattr(manifest, "stack_state", lambda *a, **k: {})


def make_build(tmp_path, product, variant, build):
    build_dir = tmp_path / "products" / product / variant / build
    build_dir.mkdir(parents=True)
    (build_dir / "a.csv").write_text("x\n1\n")
    manifest.write(build_dir, product=product, variant=variant, build=build)
    return build_dir


def read_run(run):
    return json.loads((run.path / "run.json").read_text())


def test_study_run_makes_dated_directory(tmp_path):
    run = study_run("aos/demo", "first_look", date="20261016")
    assert run.path == tmp_path / "studies/aos/demo/20261016_first_look"
    rec = read_run(run)
    assert rec["study"] == "aos/demo"
    assert rec["label"] == "first_look"
    assert rec["uses"] == {}
    assert rec["git_commit"]
    assert "pytest" in rec["command"] or "python" in rec["command"]


def test_run_json_records_builds_read_after_start(tmp_path):
    make_build(tmp_path, "fam_tables", "danish_1_2", "20261005")
    make_build(tmp_path, "fam_tables", "danish_1_2", "20261006")
    catalog.set_current("fam_tables", "danish_1_2", "20261006")
    make_build(tmp_path, "miw", "d12-a", "20261007")
    run = study_run("aos/demo", "x", date="20261016")
    catalog.load("fam_tables", "danish_1_2")
    catalog.path("miw", "d12-a", "20261007")
    catalog.path("fam_tables", "danish_1_2", "20261005")
    assert read_run(run)["uses"] == {
        "fam_tables": ["danish_1_2@20261005", "danish_1_2@20261006"],
        "miw": ["d12-a@20261007"],
    }


def test_reads_before_start_are_included(tmp_path):
    make_build(tmp_path, "miw", "d12-a", "20261007")
    catalog.path("miw", "d12-a", "20261007")
    run = study_run("aos/demo", "x", date="20261016")
    assert read_run(run)["uses"] == {"miw": ["d12-a@20261007"]}


def test_failed_lookup_is_not_recorded(tmp_path):
    run = study_run("aos/demo", "x", date="20261016")
    with pytest.raises(catalog.ProductNotFound):
        catalog.path("miw", "nope")
    assert read_run(run)["uses"] == {}


@pytest.mark.parametrize("study,label", [
    ("demo", "x"), ("aos/demo/sub", "x"), ("aos/demo", "Bad Label"), ("aos/demo", ""),
])
def test_bad_arguments(study, label):
    with pytest.raises(ValueError):
        study_run(study, label)


def test_rerun_same_label_reuses_directory():
    first = study_run("aos/demo", "x", date="20261016")
    second = study_run("aos/demo", "x", date="20261016")
    assert first.path == second.path
    assert len(runs._active) == 1


def test_catalog_shares_data_root(tmp_path):
    assert catalog.data_root() == runs.data_root() == tmp_path
