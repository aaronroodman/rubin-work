"""Tests for the product catalog and the manifest writer.

Every test points ``RUBINWORK_DATA`` at a pytest ``tmp_path``, so nothing reads
or writes the real data root on S3DF.
"""

import json
import pathlib
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from rubinwork.products import catalog, manifest  # noqa: E402


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A temporary data root, with ``products/`` and ``studies/`` in place.

    Returns
    -------
    root : `pathlib.Path`
        The temporary data root.
    """
    monkeypatch.setenv("RUBINWORK_DATA", str(tmp_path))
    (tmp_path / "products").mkdir()
    (tmp_path / "studies").mkdir()
    return tmp_path


def make_build(data_root, product, variant, build, n_rows=3, current=True,
               **kwargs):
    """Write one parquet file and its manifest as a build.

    Parameters
    ----------
    current : `bool`, optional
        Point the variant's ``current`` symlink at this build afterwards,
        which a real build never does for itself. On by default so that the
        tests about something else can just read ``"current"``; the tests about
        the rule pass ``current=False``.

    Returns
    -------
    build_dir : `pathlib.Path`
        The build directory.
    """
    build_dir = data_root / "products" / product / variant / build
    build_dir.mkdir(parents=True)
    pd.DataFrame({"visit": range(n_rows), "z4_um": [0.1] * n_rows}).to_parquet(
        build_dir / "table.parquet")
    manifest.write(build_dir, product=product, variant=variant, build=build,
                   **kwargs)
    if current:
        catalog.set_current(product, variant, build)
    return build_dir


def test_data_root_from_env(data_root):
    assert catalog.data_root() == data_root
    assert catalog.products_root() == data_root / "products"
    assert catalog.studies_root() == data_root / "studies"


def test_data_root_default(monkeypatch):
    monkeypatch.delenv("RUBINWORK_DATA", raising=False)
    assert str(catalog.data_root()) == catalog.DATA_ROOT_DEFAULT


def test_path_resolves_current(data_root):
    make_build(data_root, "fam_tables", "danish_1_2", "20261005")
    assert catalog.path("fam_tables", "danish_1_2").name == "20261005"
    assert catalog.path("fam_tables", "danish_1_2", "20261005").name == "20261005"


def test_a_build_does_not_move_current(data_root):
    make_build(data_root, "fam_tables", "danish_1_2", "20261005")
    make_build(data_root, "fam_tables", "danish_1_2", "20261006", current=False)
    assert catalog.path("fam_tables", "danish_1_2").name == "20261005"
    catalog.set_current("fam_tables", "danish_1_2", "20261006")
    assert catalog.path("fam_tables", "danish_1_2").name == "20261006"


def test_no_current_at_all_says_how_to_set_one(data_root):
    make_build(data_root, "fam_tables", "danish_1_2", "20261005", current=False)
    with pytest.raises(catalog.ProductNotFound, match="set-current"):
        catalog.path("fam_tables", "danish_1_2")


def test_manifest_contents(data_root):
    make_build(data_root, "miw", "d12-A_50_34_i_5rot", "20261007",
               n_rows=7, config={"n_rot_bins": 5},
               inputs={"fam_tables": "danish_1_2@20261005"})
    man = catalog.manifest("miw", "d12-A_50_34_i_5rot")
    assert man["product"] == "miw"
    assert man["variant"] == "d12-A_50_34_i_5rot"
    assert man["build"] == "20261007"
    assert man["status"] == "complete"
    assert man["config"] == {"n_rot_bins": 5}
    assert man["inputs"] == {"fam_tables": "danish_1_2@20261005"}
    assert man["files"]["table.parquet"]["rows"] == 7
    assert man["files"]["table.parquet"]["bytes"] > 0
    assert "created" in man
    # Run inside the rubin-work checkout, so git state is available.
    assert man["git_commit"] is not None
    assert isinstance(man["git_dirty"], bool)


def test_manifest_is_valid_json_on_disk(data_root):
    build_dir = make_build(data_root, "miw", "v1", "20261007")
    with open(build_dir / "manifest.json") as f:
        assert json.load(f)["build"] == "20261007"


def test_load_single_file(data_root):
    make_build(data_root, "fam_tables", "danish_1_2", "20261005", n_rows=4)
    df = catalog.load("fam_tables", "danish_1_2")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert list(df.columns) == ["visit", "z4_um"]


def test_load_named_file(data_root):
    build_dir = make_build(data_root, "fam_tables", "danish_1_2", "20261005")
    pd.DataFrame({"visit": [1, 2]}).to_parquet(build_dir / "visits.parquet")
    manifest.write(build_dir, product="fam_tables", variant="danish_1_2",
                   build="20261005")
    assert len(catalog.load("fam_tables", "danish_1_2", file="visits.parquet")) == 2


def test_load_requires_file_when_several(data_root):
    build_dir = make_build(data_root, "fam_tables", "danish_1_2", "20261005")
    pd.DataFrame({"visit": [1, 2]}).to_parquet(build_dir / "visits.parquet")
    manifest.write(build_dir, product="fam_tables", variant="danish_1_2",
                   build="20261005")
    with pytest.raises(ValueError, match="2 files"):
        catalog.load("fam_tables", "danish_1_2")


def test_load_non_tabular_returns_path(data_root):
    build_dir = make_build(data_root, "value_added", "v1", "20261007")
    (build_dir / "efd.db").write_bytes(b"not really duckdb")
    manifest.write(build_dir, product="value_added", variant="v1",
                   build="20261007")
    out = catalog.load("value_added", "v1", file="efd.db")
    assert isinstance(out, pathlib.Path)
    assert out.name == "efd.db"


def test_load_unknown_file_raises(data_root):
    make_build(data_root, "fam_tables", "danish_1_2", "20261005")
    with pytest.raises(catalog.ProductNotFound, match="nope.parquet"):
        catalog.load("fam_tables", "danish_1_2", file="nope.parquet")


def test_unknown_product_variant_and_build(data_root):
    make_build(data_root, "fam_tables", "danish_1_2", "20261005")
    with pytest.raises(catalog.ProductNotFound, match="no product"):
        catalog.path("nope", "danish_1_2")
    with pytest.raises(catalog.ProductNotFound, match="no variant"):
        catalog.path("fam_tables", "nope")
    with pytest.raises(catalog.ProductNotFound, match="no build"):
        catalog.path("fam_tables", "danish_1_2", "19991231")


def test_error_lists_what_is_there(data_root):
    make_build(data_root, "fam_tables", "danish_1_2", "20261005")
    with pytest.raises(catalog.ProductNotFound, match="danish_1_2"):
        catalog.path("fam_tables", "nope")


def test_build_without_manifest_is_not_in_the_catalog(data_root):
    (data_root / "products" / "miw" / "v1" / "20261007").mkdir(parents=True)
    with pytest.raises(catalog.ProductNotFound, match="manifest.json"):
        catalog.manifest("miw", "v1", "20261007")


def test_list_products(data_root):
    make_build(data_root, "fam_tables", "danish_1_2", "20261005")
    make_build(data_root, "fam_tables", "danish_1_2", "20261006")
    make_build(data_root, "miw", "d12-A", "20261007")
    rows = catalog.list_products()
    assert [(r["product"], r["variant"], r["build"]) for r in rows] == [
        ("fam_tables", "danish_1_2", "20261005"),
        ("fam_tables", "danish_1_2", "20261006"),
        ("miw", "d12-A", "20261007"),
    ]
    assert [r["current"] for r in rows] == [False, True, True]  # make_build set each
    assert all(r["status"] == "complete" for r in rows)
    assert all(r["n_files"] == 1 for r in rows)


def test_list_products_one_product(data_root):
    make_build(data_root, "fam_tables", "danish_1_2", "20261005")
    make_build(data_root, "miw", "d12-A", "20261007")
    assert [r["product"] for r in catalog.list_products("miw")] == ["miw"]


def test_list_products_skips_a_build_with_no_manifest(data_root):
    make_build(data_root, "miw", "v1", "20261007")
    (data_root / "products" / "miw" / "v1" / "20261008").mkdir()
    statuses = {r["build"]: r["status"] for r in catalog.list_products("miw")}
    assert statuses == {"20261007": "complete"}


def test_list_products_skips_an_incomplete_build(data_root):
    make_build(data_root, "miw", "v1", "20261007")
    make_build(data_root, "miw", "v1", "20261008", current=False,
               status="failed")
    assert [r["build"] for r in catalog.list_products("miw")] == ["20261007"]


def test_list_products_skips_a_corrupt_manifest(data_root):
    make_build(data_root, "miw", "v1", "20261007")
    broken = make_build(data_root, "miw", "v1", "20261008", current=False)
    (broken / "manifest.json").write_text("{not json")
    assert [r["build"] for r in catalog.list_products("miw")] == ["20261007"]


def test_list_products_empty_root(data_root):
    assert catalog.list_products() == []


def test_list_is_an_alias(data_root):
    assert catalog.list is catalog.list_products


def test_register_external_variant(data_root):
    build_dir = catalog.register(
        "miw", "official-w_2026_40", "20261001",
        location="/sdf/group/rubin/shared/official/miw/w_2026_40",
        config={"release": "w_2026_40"})
    catalog.set_current("miw", "official-w_2026_40", "20261001")
    man = catalog.manifest("miw", "official-w_2026_40")
    assert man["location"].endswith("w_2026_40")
    assert man["files"] == {}
    assert man["status"] == "complete"
    assert catalog.path("miw", "official-w_2026_40") == build_dir.resolve()


def test_register_an_existing_directory_records_its_files(data_root):
    build_dir = data_root / "products" / "fam_tables" / "danish_1_2" / "20250901"
    build_dir.mkdir(parents=True)
    pd.DataFrame({"visit": [1, 2, 3, 4, 5]}).to_parquet(build_dir / "donuts.parquet")
    catalog.register("fam_tables", "danish_1_2", "20250901")
    man = catalog.manifest("fam_tables", "danish_1_2", "20250901")
    assert man["files"]["donuts.parquet"]["rows"] == 5


def test_register_does_not_move_current(data_root):
    make_build(data_root, "fam_tables", "danish_1_2", "20261005")
    catalog.register("fam_tables", "danish_1_2", "20261006")
    assert catalog.path("fam_tables", "danish_1_2").name == "20261005"


def test_file_stats_skips_work_dirs_and_the_manifest(data_root):
    build_dir = make_build(data_root, "miw", "v1", "20261007")
    (build_dir / "_work").mkdir()
    (build_dir / "_work" / "scratch.parquet").write_bytes(b"x")
    (build_dir / ".hidden").write_bytes(b"x")
    stats = manifest.file_stats(build_dir)
    assert set(stats) == {"table.parquet"}


def test_current_symlink_is_relative(data_root):
    make_build(data_root, "miw", "v1", "20261007")
    link = data_root / "products" / "miw" / "v1" / "current"
    assert link.is_symlink()
    assert str(pathlib.Path(link).readlink()) == "20261007"


def test_set_current_refuses_an_incomplete_build(data_root):
    make_build(data_root, "miw", "v1", "20261007")
    make_build(data_root, "miw", "v1", "20261008", current=False,
               status="failed")
    with pytest.raises(catalog.ProductNotFound, match="'failed'"):
        catalog.set_current("miw", "v1", "20261008")
    assert catalog.path("miw", "v1").name == "20261007"


def test_set_current_refuses_a_build_with_no_manifest(data_root):
    make_build(data_root, "miw", "v1", "20261007")
    (data_root / "products" / "miw" / "v1" / "20261008").mkdir()
    with pytest.raises(catalog.ProductNotFound, match="manifest.json"):
        catalog.set_current("miw", "v1", "20261008")
    assert catalog.path("miw", "v1").name == "20261007"


def test_asking_for_an_incomplete_build_says_why(data_root):
    make_build(data_root, "miw", "v1", "20261008", current=False,
               status="interrupted")
    with pytest.raises(catalog.ProductNotFound, match="'interrupted'"):
        catalog.path("miw", "v1", "20261008")
    with pytest.raises(catalog.ProductNotFound, match="'interrupted'"):
        catalog.load("miw", "v1", "20261008")


def test_a_current_symlink_at_an_incomplete_build_is_refused(data_root):
    # A build completed, became current, and was later reran into failure in
    # place.  Nothing should read it just because the symlink survived.
    make_build(data_root, "miw", "v1", "20261007")
    build_dir = data_root / "products" / "miw" / "v1" / "20261007"
    manifest.write(build_dir, product="miw", variant="v1", build="20261007",
                   status="failed")
    with pytest.raises(catalog.ProductNotFound, match="'failed'"):
        catalog.path("miw", "v1")


def test_is_complete(data_root):
    good = make_build(data_root, "miw", "v1", "20261007")
    bad = make_build(data_root, "miw", "v1", "20261008", current=False,
                     status="failed")
    nothing = data_root / "products" / "miw" / "v1" / "20261009"
    nothing.mkdir()
    assert catalog.is_complete(good)
    assert not catalog.is_complete(bad)
    assert not catalog.is_complete(nothing)


def test_cli_set_current(data_root, capsys):
    make_build(data_root, "fam_tables", "danish_1_2", "20261005")
    make_build(data_root, "fam_tables", "danish_1_2", "20261006", current=False)
    assert catalog._main(
        ["set-current", "fam_tables", "danish_1_2", "20261006"]) == 0
    assert catalog.path("fam_tables", "danish_1_2").name == "20261006"


def test_cli_set_current_refuses_and_exits_nonzero(data_root, capsys):
    make_build(data_root, "fam_tables", "danish_1_2", "20261005")
    make_build(data_root, "fam_tables", "danish_1_2", "20261006", current=False,
               status="failed")
    assert catalog._main(
        ["set-current", "fam_tables", "danish_1_2", "20261006"]) == 1
    assert "refusing to move current" in capsys.readouterr().err
    assert catalog.path("fam_tables", "danish_1_2").name == "20261005"


def test_cli_list(data_root, capsys):
    make_build(data_root, "fam_tables", "danish_1_2", "20261005")
    assert catalog._main(["list"]) == 0
    assert "fam_tables/danish_1_2/20261005" in capsys.readouterr().out


def test_git_state_outside_a_repo(tmp_path):
    commit, dirty = manifest.git_state(tmp_path)
    assert commit is None and dirty is None
