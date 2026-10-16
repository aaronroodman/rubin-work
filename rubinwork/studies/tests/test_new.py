"""Tests for study creation (``/start-study``).

Every test works in a temporary checkout and a temporary ``RUBINWORK_DATA``, never in
the real tree.
"""

import pytest

from rubinwork import studies
from rubinwork.studies import new
from rubinwork.studies.__main__ import main


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A temporary repo with two topics and a data root with one build."""
    monkeypatch.setenv("RUBINWORK_DATA", str(tmp_path / "data"))
    build = tmp_path / "data/products/fam_tables/danish_1_2/20261014"
    build.mkdir(parents=True)
    (build / "manifest.json").write_text('{"status": "complete", "files": {}}')
    root = tmp_path / "repo"
    for topic in ("aos", "camera", "rubinwork", "notes"):
        (root / topic).mkdir(parents=True)
    return root


def test_topics_exclude_support_dirs(checkout):
    assert new.topics(checkout) == ["aos", "camera"]


def test_create_study(checkout, tmp_path):
    uses = [studies.UsesEntry("fam_tables", "danish_1_2", "20261014")]
    directory = new.create_study("aos", "coadd_vs_miw",
                                 "Why do coadds disagree: a test?", uses,
                                 root=checkout, started="2026-10-16")
    assert directory == checkout / "aos/coadd_vs_miw"
    for sub in new.SUBDIRS:
        assert (directory / sub).is_dir()
    data_dir = tmp_path / "data/studies/aos/coadd_vs_miw"
    assert data_dir.is_dir()
    assert (directory / "output").is_symlink()
    assert (directory / "output").resolve() == data_dir.resolve()
    study = studies.load_study(directory)
    assert study.notes == []
    assert study.meta["question"] == "Why do coadds disagree: a test?"
    assert study.uses == uses
    assert study.log_entries() == [("2026-10-16", "started")]
    assert studies.validate(study) == []
    assert "aos/coadd_vs_miw" in (checkout / "STUDIES.md").read_text()


def test_create_study_without_uses(checkout):
    directory = new.create_study("camera", "dip", "How deep?", root=checkout)
    study = studies.load_study(directory)
    assert study.meta["uses"] == {}
    assert studies.validate(study) == []


@pytest.mark.parametrize("topic,name", [("aos", "Bad"), ("aos", "x"), ("nope", "fine")])
def test_bad_names_refused(checkout, topic, name):
    with pytest.raises(ValueError):
        new.create_study(topic, name, "Q?", root=checkout)
    assert not (checkout / topic / name).exists()


def test_new_topic_allowed_when_asked(checkout):
    new.create_study("survey", "first", "Q?", root=checkout, new_topic=True)
    assert (checkout / "survey/first/study.md").is_file()


def test_duplicate_name_in_any_topic_refused(checkout):
    new.create_study("aos", "dup", "Q?", root=checkout)
    with pytest.raises(ValueError, match="already exists"):
        new.create_study("camera", "dup", "Q?", root=checkout)


def test_unresolved_uses_refused_before_anything_is_made(checkout):
    from rubinwork.products import catalog
    uses = [studies.UsesEntry("fam_tables", "danish_9", "current")]
    with pytest.raises(catalog.ProductNotFound):
        new.create_study("aos", "bad_uses", "Q?", uses, root=checkout)
    assert not (checkout / "aos/bad_uses").exists()


def test_parse_uses_arg():
    assert new.parse_uses_arg("miw=d12-a@20261014") == studies.UsesEntry(
        "miw", "d12-a", "20261014")
    assert new.parse_uses_arg("miw=d12-a").build == "current"
    with pytest.raises(ValueError):
        new.parse_uses_arg("miw")


def test_cli_new(checkout, monkeypatch, capsys):
    monkeypatch.setattr(studies.core, "repo_root", lambda: checkout)
    monkeypatch.setattr(new, "repo_root", lambda: checkout)
    monkeypatch.setattr(studies.index, "repo_root", lambda: checkout)
    assert main(["new", "aos", "cli_study", "--question", "Q?",
                 "--uses", "fam_tables=danish_1_2@20261014"]) == 0
    assert (checkout / "aos/cli_study/study.md").is_file()
    assert main(["new", "aos", "cli_study", "--question", "Q?"]) == 1
    assert "not created" in capsys.readouterr().err


def test_product_variants_real_repo():
    variants = new.product_variants()
    assert "fam_tables" in variants
    path, names = variants["fam_tables"]
    assert path.name == "variants.yaml" and "danish_1_2" in names


def test_current_is_pinned(checkout):
    uses = [studies.UsesEntry("fam_tables", "danish_1_2", "current")]
    import os
    os.symlink("20261014", checkout.parent / "data/products/fam_tables/danish_1_2/current")
    directory = new.create_study("aos", "pinned", "Q?", uses, root=checkout)
    assert studies.load_study(directory).meta["uses"] == {
        "fam_tables": "danish_1_2@20261014"}
