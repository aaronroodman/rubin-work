"""Tests for the ``value_added`` product: the reader's path resolution, the shims, the
catalog route, the scratch-versus-catalog manifest rule, and the build comparison.

The path tests are what matter most here. Before phase 3a the reader sat in
``value_added/code/`` and resolved its default database from its own ``parents[1]``; from
``rubinwork/products/value_added/`` that names the products package, so the anchor had to
change to the repository root. A silent regression there would point every reader at a
non-existent database, or worse at the wrong one.

The shim tests assert **module identity**, not just importability: five scripts and three
notebooks do a bare-name ``import efd_db`` after a ``sys.path`` insert, and two of them
hand each other the connection it opens, which separate copies would break.

Nothing here opens the live database.
"""

import pathlib
import subprocess
import sys

import pytest


from rubinwork.products import catalog, value_added  # noqa: E402
from rubinwork.products.value_added import compare_builds  # noqa: E402
from rubinwork.products.value_added import reader as va_reader  # noqa: E402
from rubinwork.products.value_added import write_manifest  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

EFD_DB_IMPORTERS = (
    "aos/cwfs_lut/code/run_cwfs_lut.py",
    "aos/giant_donut_pupil/code/find_giant_nights.py",
)
"""Code files that used the ``efd_db`` shim and now import the package."""

EFD_DB_NOTEBOOKS = (
    "aos/bending_mode_test/notebooks/bending_mode_test_lut_trim.ipynb",
    "aos/notebooks/bounce/_exectest.ipynb",
    "aos/intra_extra/notebooks/blur_intra_extra_offset.ipynb",
)
"""Notebooks that import ``efd_db`` the same way."""

SHIMMED_BUILDERS = (
    "build_efd_db", "build_optical_state", "build_fam_dz", "build_m1m3_thermal_r2",
    "m1m3_thermal_r2", "merge_db_shards", "backfill_commanded_vmodes",
    "backfill_optical_state_pointing",
)
"""Modules that moved into ``builders/`` and left a shim at the old path."""


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A temporary data root, with ``products/`` and ``studies/`` in place."""
    monkeypatch.setenv("RUBINWORK_DATA", str(tmp_path))
    (tmp_path / "products").mkdir()
    (tmp_path / "studies").mkdir()
    return tmp_path


def _write_db(path, n_visits=3):
    """Write a tiny database with the real schema and a few rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = va_reader.open_db(path, readonly=False, create=True)
    try:
        va_reader.create_schema(con)
        con.execute(
            "INSERT INTO visit_telemetry (visit_id, day_obs, seq_num) "
            "VALUES " + ", ".join(f"({1000 + i}, 20251116, {i})"
                                  for i in range(n_visits)))
        con.commit()
    finally:
        con.close()
    return path


# ---- the default database path, the one thing the move breaks -------------------

def test_old_tree_path_is_anchored_at_the_repo_root():
    """Not at the reader's own parents[N], which the move changed."""
    assert va_reader.old_tree_db_path() == (
        REPO_ROOT / "value_added" / "output" / "aos_efd.duckdb")


def test_old_tree_path_is_not_inside_the_products_package():
    """The regression the move could have introduced: parents[1] of the moved reader is
    ``rubinwork/products``, so a naive hop-count fix would have resolved there."""
    path = va_reader.old_tree_db_path()
    assert "rubinwork/products" not in str(path)
    assert path.parent.parent.name == "value_added"


def test_aos_efd_db_overrides_everything(monkeypatch, tmp_path):
    scratch = tmp_path / "scratch" / "aos_efd.duckdb"
    monkeypatch.setenv("AOS_EFD_DB", str(scratch))
    assert va_reader.default_db_path() == scratch


def test_default_path_prefers_the_catalog_over_the_old_tree(monkeypatch, data_root):
    """Catalog first: an existing old-tree database does not shadow a current build."""
    monkeypatch.delenv("AOS_EFD_DB", raising=False)
    old = data_root / "old_tree" / "aos_efd.duckdb"
    _write_db(old)
    monkeypatch.setattr(va_reader, "old_tree_db_path", lambda: old)
    build = data_root / "products" / "value_added" / "v1" / "20261013"
    _write_db(build / va_reader.DB_FILE)
    write_manifest.write("v1", build="20261013", build_dir=build)
    catalog.set_current("value_added", "v1", "20261013")
    assert va_reader.default_db_path() == build / va_reader.DB_FILE


def test_default_path_falls_back_to_the_old_tree_without_a_build(monkeypatch, data_root):
    """With no catalog build, an existing old-tree database is used."""
    monkeypatch.delenv("AOS_EFD_DB", raising=False)
    old = data_root / "old_tree" / "aos_efd.duckdb"
    _write_db(old)
    monkeypatch.setattr(va_reader, "old_tree_db_path", lambda: old)
    assert va_reader.default_db_path() == old


def test_default_path_falls_back_to_the_old_tree_path_when_nothing_exists(
        monkeypatch, data_root):
    """So a caller opening a missing database gets the familiar error and location."""
    monkeypatch.delenv("AOS_EFD_DB", raising=False)
    missing = data_root / "no_such_tree" / "aos_efd.duckdb"
    monkeypatch.setattr(va_reader, "old_tree_db_path", lambda: missing)
    assert va_reader.default_db_path() == missing


def test_catalog_db_path_is_none_without_a_build(data_root):
    assert va_reader.catalog_db_path() is None


# ---- the catalog route ---------------------------------------------------------

def test_connect_opens_a_catalog_build(data_root):
    build = data_root / "products" / "value_added" / "v1" / "20261013"
    _write_db(build / va_reader.DB_FILE, n_visits=5)
    write_manifest.write("v1", build="20261013", build_dir=build)
    catalog.set_current("value_added", "v1", "20261013")
    con = value_added.connect()
    try:
        assert con.execute("SELECT COUNT(*) FROM visit_telemetry").fetchone()[0] == 5
    finally:
        con.close()


def test_connect_raises_without_a_build(data_root):
    with pytest.raises(catalog.ProductNotFound, match="no complete value_added build"):
        value_added.connect()


def test_load_is_an_alias_of_connect(data_root):
    build = data_root / "products" / "value_added" / "v1" / "20261013"
    _write_db(build / va_reader.DB_FILE)
    write_manifest.write("v1", build="20261013", build_dir=build)
    catalog.set_current("value_added", "v1", "20261013")
    con = value_added.load()
    try:
        assert con.execute("SELECT COUNT(*) FROM visit_telemetry").fetchone()[0] == 3
    finally:
        con.close()


def test_a_build_without_a_manifest_is_invisible(data_root):
    build = data_root / "products" / "value_added" / "v1" / "20261013"
    _write_db(build / va_reader.DB_FILE)
    assert va_reader.catalog_db_path() is None


def test_build_dir_is_under_the_products_root(data_root):
    assert value_added.build_dir("v1", "20261013") == (
        data_root / "products" / "value_added" / "v1" / "20261013")
    assert value_added.build_dir("v1") == (
        data_root / "products" / "value_added" / "v1")


# ---- scratch builds write no manifest ------------------------------------------

def test_a_catalog_path_is_a_catalog_build(data_root):
    db = value_added.build_dir("v1", "20261013") / va_reader.DB_FILE
    assert write_manifest.is_catalog_build(db)


def test_a_scratch_path_is_not_a_catalog_build(data_root, tmp_path):
    assert not write_manifest.is_catalog_build(tmp_path / "ref" / "aos_efd.duckdb")


def test_the_old_tree_database_is_not_a_catalog_build(data_root):
    """So the live database is never registered by accident."""
    assert not write_manifest.is_catalog_build(va_reader.old_tree_db_path())


def test_finish_build_writes_nothing_for_a_scratch_database(data_root, tmp_path):
    db = _write_db(tmp_path / "ref" / va_reader.DB_FILE)
    assert write_manifest.finish_build(db) is None
    assert not (db.parent / "manifest.json").exists()


def test_a_shard_under_the_variant_work_dir_is_not_a_catalog_build(data_root):
    """The defect of 2026-10-15: 11 shards raced on one ``manifest.json``."""
    shard = (value_added.build_dir("v1") / va_reader.WORK_DIR / "shards"
             / "state_20261015_221551_03.duckdb")
    assert not write_manifest.is_catalog_build(shard)


def test_concurrent_finish_build_calls_on_one_directory_all_succeed(data_root,
                                                                    monkeypatch):
    """Two writers in one directory must not delete each other's temp file.

    A shard directory no longer reaches `write`, but `merge_db_shards` and a per-table
    builder can finish together in one build directory, and the fixed temp name made
    that a `FileNotFoundError` for the loser.

    The interleaving is forced rather than raced: each writer waits on a barrier between
    creating its temp file and renaming it, which is exactly the window a shared temp
    name loses. Left to chance the two threads almost never overlap there, so the test
    would pass against the defect.
    """
    import concurrent.futures
    import threading

    from rubinwork.products import manifest as manifest_mod

    build = value_added.build_dir("v1", "20261013")
    db = _write_db(build / va_reader.DB_FILE)

    barrier = threading.Barrier(2, timeout=30)
    real_replace = manifest_mod.os.replace

    def replace_after_both_temps_exist(src, dst):
        barrier.wait()
        return real_replace(src, dst)

    monkeypatch.setattr(manifest_mod.os, "replace", replace_after_both_temps_exist)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(write_manifest.finish_build, db,
                               build_options={"writer": str(i)})
                   for i in range(2)]
        paths = [f.result() for f in futures]
    assert all(p is not None and p.exists() for p in paths)
    assert not list(build.glob(".manifest.json.*"))


def test_finish_build_writes_a_manifest_for_a_catalog_build(data_root):
    build = value_added.build_dir("v1", "20261013")
    db = _write_db(build / va_reader.DB_FILE)
    path = write_manifest.finish_build(db, build_options={"what": "telemetry"})
    assert path is not None and path.exists()
    import json
    doc = json.loads(path.read_text())
    assert doc["product"] == "value_added"
    assert doc["variant"] == "v1"
    assert doc["config"]["build_options"] == {"what": "telemetry"}


def test_manifest_records_the_registries_and_the_span(data_root):
    build = value_added.build_dir("v1", "20261013")
    _write_db(build / va_reader.DB_FILE, n_visits=4)
    path = write_manifest.write("v1", build="20261013", build_dir=build)
    import json
    cfg = json.loads(path.read_text())["config"]
    assert cfg["tables"] == list(value_added.TABLES)
    assert cfg["registries"] == list(value_added.REGISTRIES)
    assert cfg["day_obs_min"] == 20251116 and cfg["day_obs_max"] == 20251116
    assert cfg["n_visit_telemetry_rows"] == 4


def test_manifest_records_no_upstream_build(data_root):
    """The telemetry comes from the EFD and ConsDB, which are external services, and the
    per-variant repository upstreams are per row rather than per build."""
    build = value_added.build_dir("v1", "20261013")
    _write_db(build / va_reader.DB_FILE)
    path = write_manifest.write("v1", build="20261013", build_dir=build)
    import json
    assert not json.loads(path.read_text()).get("inputs")


def test_manifest_needs_a_build_or_a_build_dir():
    with pytest.raises(ValueError, match="needs `build` or `build_dir`"):
        write_manifest.write("v1")


# ---- the shims -----------------------------------------------------------------

@pytest.mark.skipif(not ((REPO_ROOT / "value_added/code/efd_db.py").exists()), reason='shims removed')
def test_the_efd_db_shim_gives_the_same_module_object():
    """Not a copy: two importers hand each other the connection it opens."""
    sys.path.insert(0, str(REPO_ROOT / "value_added" / "code"))
    import efd_db

    assert efd_db is va_reader


@pytest.mark.skipif(not ((REPO_ROOT / "value_added/code/build_efd_db.py").exists()), reason='shims removed')
def test_every_builder_shim_gives_the_same_module_object():
    """Checked in a **fresh interpreter** per shim.

    In this one, several builders have already been imported as top-level modules by
    their own siblings' ``sys.path`` bootstrap, so ``sys.modules`` legitimately holds two
    entries for one file and an in-process check would compare the wrong pair. A
    subprocess sees only the import under test.
    """
    for name in SHIMMED_BUILDERS:
        script = (
            "import sys\n"
            "sys.path.insert(0, 'value_added/code')\n"
            f"import {name} as shim\n"
            f"import rubinwork.products.value_added.builders.{name} as pkg\n"
            "assert shim is pkg, (shim, pkg)\n"
            f"print('{name} ok')\n")
        out = subprocess.run([sys.executable, "-c", script],
                             cwd=REPO_ROOT, capture_output=True, text=True)
        assert out.returncode == 0, f"{name}: {out.stderr[-2000:]}"
        assert f"{name} ok" in out.stdout


@pytest.mark.skipif(not ((REPO_ROOT / "value_added/code/build_efd_db.py").exists()), reason='shims removed')
def test_a_shim_exists_for_every_moved_file():
    code = REPO_ROOT / "value_added" / "code"
    for name in SHIMMED_BUILDERS:
        assert (code / f"{name}.py").exists(), name
    assert (code / "run_build.sh").exists()


@pytest.mark.skipif(not ((REPO_ROOT / "value_added/code/build_efd_db.py").exists()), reason='shims removed')
def test_the_shims_are_shims_and_not_copies():
    """A shim is short and re-exports; a leftover copy would be hundreds of lines."""
    code = REPO_ROOT / "value_added" / "code"
    for name in SHIMMED_BUILDERS + (("efd_db",) if (code / "efd_db.py").exists() else ()):
        text = (code / f"{name}.py").read_text()
        n_lines = len(text.splitlines())
        assert n_lines < 60, f"{name}.py is {n_lines} lines; a shim should be short"
        assert "sys.modules[__name__] = _real" in text, name


def test_former_shim_importers_import_the_package():
    for rel in EFD_DB_IMPORTERS:
        text = (REPO_ROOT / rel).read_text()
        assert "from rubinwork.products import value_added as efd_db" in text, rel
        assert "'value_added' / 'code'" not in text, rel


def test_the_notebooks_still_import_efd_db_by_bare_name():
    """Read as JSON rather than executed: the point is only that the import they contain
    is one the shim satisfies."""
    import json

    for rel in EFD_DB_NOTEBOOKS:
        path = REPO_ROOT / rel
        if not path.exists():
            pytest.skip(f"{rel} is not in this checkout")
        doc = json.loads(path.read_text())
        src = "\n".join("".join(c.get("source", [])) for c in doc["cells"]
                        if c.get("cell_type") == "code")
        assert "efd_db" in src, rel


# ---- the moved builders --------------------------------------------------------

def test_no_builder_imports_efd_db_by_bare_name():
    """A moved builder's own directory no longer holds efd_db.py, so a bare-name import
    would resolve to the shim only by accident of sys.path order."""
    builders = REPO_ROOT / "rubinwork" / "products" / "value_added" / "builders"
    for path in sorted(builders.glob("*.py")):
        for line in path.read_text().splitlines():
            stripped = line.strip()
            assert stripped != "import efd_db", f"{path.name}: {line}"
            assert not stripped.startswith("from common."), f"{path.name}: {line}"
            assert not stripped.startswith("import common."), f"{path.name}: {line}"


def test_every_builder_resolves_the_repo_root_at_parents_4():
    """Was parents[2] from value_added/code/; the builders are two levels deeper now."""
    builders = REPO_ROOT / "rubinwork" / "products" / "value_added" / "builders"
    for path in sorted(builders.glob("*.py")):
        text = path.read_text()
        assert "parents[2]" not in text, f"{path.name} still resolves parents[2]"
        if "parents[4]" in text:
            assert path.resolve().parents[4] == REPO_ROOT


def test_the_builders_help_runs():
    """A smoke test that each builder imports under ``python -m`` and parses arguments."""
    for name in ("build_efd_db", "build_optical_state", "build_fam_dz",
                 "build_m1m3_thermal_r2", "merge_db_shards"):
        out = subprocess.run(
            [sys.executable, "-m",
             f"rubinwork.products.value_added.builders.{name}", "--help"],
            cwd=REPO_ROOT, capture_output=True, text=True)
        assert out.returncode == 0, f"{name}: {out.stderr[-2000:]}"


# ---- the comparison ------------------------------------------------------------

def test_compare_keys_cover_every_table_and_registry():
    assert set(compare_builds.TABLE_KEYS) == set(
        value_added.TABLES) | set(value_added.REGISTRIES)


def test_compare_keys_come_from_the_reader():
    assert compare_builds.TABLE_KEYS == dict(value_added.TABLE_KEYS)


def test_visit_telemetry_has_no_expected_difference():
    """It carries no build-time timestamp, so it must match exactly."""
    assert "visit_telemetry" not in compare_builds.EXPECTED_DIFF


def test_every_expected_difference_is_a_timestamp():
    """The list cannot be widened to hide a value difference."""
    for table, cols in compare_builds.EXPECTED_DIFF.items():
        assert table in compare_builds.TABLE_KEYS, table
        for col in cols:
            assert col in ("attempted_at", "updated_at", "created_at", "computed_at"), \
                f"{table}.{col} is not a build-time timestamp"


def test_compare_finds_identical_databases_identical(tmp_path, capsys):
    a = _write_db(tmp_path / "a" / va_reader.DB_FILE)
    b = _write_db(tmp_path / "b" / va_reader.DB_FILE)
    assert compare_builds.main([str(a), str(b)]) == 0
    assert "builds match" in capsys.readouterr().out


def test_compare_catches_a_row_count_difference(tmp_path):
    a = _write_db(tmp_path / "a" / va_reader.DB_FILE, n_visits=3)
    b = _write_db(tmp_path / "b" / va_reader.DB_FILE, n_visits=4)
    assert compare_builds.main([str(a), str(b)]) == 1


def test_compare_catches_a_value_difference(tmp_path):
    a = _write_db(tmp_path / "a" / va_reader.DB_FILE)
    b = _write_db(tmp_path / "b" / va_reader.DB_FILE)
    con = va_reader.open_db(b, readonly=False)
    try:
        con.execute("UPDATE visit_telemetry SET seq_num = 99 WHERE visit_id = 1000")
        con.commit()
    finally:
        con.close()
    assert compare_builds.main([str(a), str(b)]) == 1


def test_compare_handles_list_columns(tmp_path):
    """optical_state stores the 50 DOF and 34 v-modes as DuckDB LIST, which pandas holds
    as object arrays that ``!=`` cannot compare elementwise."""
    assert compare_builds._values_equal([1.0, 2.0], [1.0, 2.0])
    assert not compare_builds._values_equal([1.0, 2.0], [1.0, 3.0])
    assert not compare_builds._values_equal([1.0, 2.0], [1.0, 2.0, 3.0])
    assert not compare_builds._values_equal([1.0], 1.0)


def test_compare_treats_nan_as_equal_to_nan():
    """An unrecovered visit stores NaN rather than NULL, so NaN must not read as a
    difference between two identical builds."""
    assert compare_builds._values_equal(float("nan"), float("nan"))
    assert compare_builds._values_equal([float("nan"), 1.0], [float("nan"), 1.0])


# ---------------------------------------------------------------------------
# Resolving the FAM and MIW inputs: old tree versus product catalog
# ---------------------------------------------------------------------------
# Both builders resolved their input old-tree-first, so a rebuild meant to run against the
# catalog silently read whatever was left in aos/output. For the MIW the two are not the
# same build -- they differ on the outermost radial ring of the field grid, the convex-hull
# and NaN-rim fix -- so the override has to be explicit and has to be tested.


def test_resolve_miw_decomp_returns_none_without_an_override():
    """No override means the historical MiwCornerLookup behaviour, unchanged."""
    import argparse

    from rubinwork.products.value_added.builders import build_optical_state as bos

    a = argparse.Namespace(miw_decomp_path=None, miw_build=None)
    assert bos.resolve_miw_decomp(a) is None


def test_resolve_miw_decomp_prefers_an_explicit_path(tmp_path):
    """--miw-decomp-path wins over --miw-build, so a hand-staged build can be used."""
    import argparse

    from rubinwork.products.value_added.builders import build_optical_state as bos

    decomp = tmp_path / "intrinsic_split_decomp.parquet"
    decomp.write_bytes(b"")
    a = argparse.Namespace(miw_decomp_path=str(decomp), miw_build="d12-whatever@20260101")
    assert bos.resolve_miw_decomp(a) == decomp


def test_resolve_miw_decomp_rejects_a_missing_path(tmp_path):
    """Fail naming the path, rather than later inside pandas."""
    import argparse

    from rubinwork.products.value_added.builders import build_optical_state as bos

    a = argparse.Namespace(miw_decomp_path=str(tmp_path / "nope.parquet"), miw_build=None)
    with pytest.raises(FileNotFoundError):
        bos.resolve_miw_decomp(a)


def test_resolve_miw_decomp_follows_a_catalog_intrinsic_ref(tmp_path, monkeypatch):
    """Once the registry row holds ``miw:<variant>@<build>``, no flag is needed; a
    joined old-tree name resolves to its variant's current build; an unknown name to
    `None`."""
    import argparse

    from rubinwork.products.value_added.builders import build_optical_state as bos

    seen = []

    def fake_path(product, variant, build):
        seen.append((product, variant, build))
        return tmp_path

    monkeypatch.setattr(catalog, "path", fake_path)
    (tmp_path / "intrinsic_split_decomp.parquet").write_bytes(b"")
    a = argparse.Namespace(miw_decomp_path=None, miw_build=None)
    got = bos.resolve_miw_decomp(a, "miw:d12-A_50_34_i_5rot@20261014")
    assert got == tmp_path / "intrinsic_split_decomp.parquet"
    assert seen == [("miw", "d12-A_50_34_i_5rot", "20261014")]
    assert bos.resolve_miw_decomp(a, "danish_1_2_A_50_34_i_5rot") == got
    assert seen[-1] == ("miw", "d12-A_50_34_i_5rot", catalog.CURRENT)
    assert bos.resolve_miw_decomp(a, "not_a_miw_build") is None


def test_resolve_paths_prefers_the_catalog_when_asked(tmp_path, monkeypatch):
    """With both trees populated, prefer='catalog' must not return the old tree.

    This is the regression that matters: the old tree is still on disk, so the default
    'old' order wins silently and a build meant to read the catalog does not.
    """
    from rubinwork.products.value_added.builders import build_fam_dz

    old = tmp_path / "aos_output" / "danish_x"
    old.mkdir(parents=True)
    cat = tmp_path / "catalog" / "danish_x"
    cat.mkdir(parents=True)
    for base in (old, cat):
        (base / "fits.parquet").write_bytes(b"")
        (base / "visits.parquet").write_bytes(b"")

    monkeypatch.setattr(catalog, "path", lambda product, variant, *a, **k: cat)

    fits_old, _ = build_fam_dz.resolve_paths(old.parent, "danish_x", None, prefer="old")
    fits_cat, _ = build_fam_dz.resolve_paths(old.parent, "danish_x", None,
                                             prefer="catalog")
    assert fits_old == old / "fits.parquet"
    assert fits_cat == cat / "fits.parquet"


def test_resolve_paths_rejects_an_unknown_prefer():
    """A typo must not quietly fall back to the old tree."""
    from rubinwork.products.value_added.builders import build_fam_dz

    with pytest.raises(ValueError):
        build_fam_dz.resolve_paths(pathlib.Path("/nonexistent"), "x", None, prefer="catalogue")


# ---------------------------------------------------------------------------
# Registry rows move to catalog names (plan section 6, item c3)
# ---------------------------------------------------------------------------


def test_registry_migration_leaves_param_set_alone(tmp_path):
    """``param_set`` derives the primary key, so migrating it would break the join.

    ``fam_variant_id`` is ``fam__<param_set>__<route>__<prefix>__<scheme>`` and
    ``register_fam_variant`` writes ``param_set`` back from its CLI argument on conflict,
    so a catalog reference stored there would be inconsistent with the id and then
    silently reverted.
    """
    import duckdb

    from rubinwork.products.value_added.builders import migrate_registry_to_catalog as mig

    db = tmp_path / "reg.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE fam_variant (fam_variant_id VARCHAR, param_set VARCHAR, "
                "fits_path VARCHAR)")
    con.execute("INSERT INTO fam_variant VALUES ('fam__ps__batoid__z1toz6__50_34', "
                "'ps', '/old/tree/ps/fits.parquet')")
    con.execute("CREATE TABLE state_variant (variant_id VARCHAR, intrinsic_route VARCHAR, "
                "intrinsic_ref VARCHAR)")
    con.execute("INSERT INTO state_variant VALUES ('v__miw', 'miw', 'joined_build_name')")
    con.close()

    con = duckdb.connect(str(db))
    try:
        updates = mig.plan_updates(con, fam_ref="fam_tables:danish_1_2@20261014",
                                   miw_ref="miw:d12-A@20261014")
        cols = {(t, c) for t, _k, c, _o, _n in updates}
        assert ("fam_variant", "fits_path") in cols
        assert ("state_variant", "intrinsic_ref") in cols
        assert ("fam_variant", "param_set") not in cols

        mig.apply_updates(con, updates)
        assert con.execute("SELECT param_set FROM fam_variant").fetchone()[0] == "ps"
        assert (con.execute("SELECT fits_path FROM fam_variant").fetchone()[0]
                == "fam_tables:danish_1_2@20261014")
    finally:
        con.close()


def test_registry_migration_is_idempotent(tmp_path):
    """Re-running after a successful pass must be a no-op, not a double rewrite."""
    import duckdb

    from rubinwork.products.value_added.builders import migrate_registry_to_catalog as mig

    db = tmp_path / "reg.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE fam_variant (fam_variant_id VARCHAR, param_set VARCHAR, "
                "fits_path VARCHAR)")
    con.execute("INSERT INTO fam_variant VALUES ('fam__ps', 'ps', "
                "'fam_tables:danish_1_2@20261014')")
    con.execute("CREATE TABLE state_variant (variant_id VARCHAR, intrinsic_route VARCHAR, "
                "intrinsic_ref VARCHAR)")
    con.close()

    con = duckdb.connect(str(db))
    try:
        assert mig.plan_updates(con, fam_ref="fam_tables:danish_1_2@20261014") == []
    finally:
        con.close()


# ---- fetch_log provenance ------------------------------------------------------

def test_fetch_log_gains_git_commit_on_an_old_schema(tmp_path):
    """An existing database without the column is migrated, old rows kept as NULL."""
    import duckdb

    db = tmp_path / va_reader.DB_FILE
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE fetch_log (day_obs INTEGER, group_name VARCHAR, "
                "status VARCHAR, n_rows INTEGER, error VARCHAR, attempted_at TIMESTAMP, "
                "PRIMARY KEY (day_obs, group_name))")
    con.execute("INSERT INTO fetch_log VALUES (20251116, 'm1m3', 'ok', 3, NULL, NULL)")
    va_reader.create_schema(con)
    va_reader.log_fetch(con, 20251117, "m1m3", "ok", n_rows=4)
    rows = dict(con.execute("SELECT day_obs, git_commit FROM fetch_log").fetchall())
    con.close()
    assert rows[20251116] is None
    assert rows[20251117]


def test_compare_table_handles_pandas_na_and_reports_max_abs():
    """A nullable column's pd.NA must not raise; differences report their size."""
    import pandas as pd

    a = pd.DataFrame({"k": [1, 2, 3], "x": pd.array([1, None, 3], dtype="Int64"),
                      "v": [[1.0, 2.0], [0.0, 0.0], [5.0, 5.0]]})
    b = pd.DataFrame({"k": [1, 2, 3], "x": pd.array([1, None, 4], dtype="Int64"),
                      "v": [[1.0, 2.5], [0.0, 0.0], [5.0, 5.0]]})
    rep = compare_builds.compare_table(a, b, ("k",))
    assert dict(rep["differing"]) == {"x": 1, "v": 1}
    assert rep["max_abs"] == {"x": 1.0, "v": 0.5}


def test_set_current_on_a_nested_snapshot_links_at_the_variant(data_root):
    """``current`` for a ``snapshots/<date>`` build is ``v1/current``, not
    ``v1/snapshots/current``, and resolves to the snapshot's database."""
    snap = value_added.build_dir("v1", "snapshots/20261016")
    db = _write_db(snap / va_reader.DB_FILE)
    write_manifest.write("v1", build_dir=snap, build="20261016")
    link = catalog.set_current("value_added", "v1", "snapshots/20261016")
    variant_dir = value_added.build_dir("v1")
    assert link == variant_dir / "current"
    assert not (variant_dir / "snapshots" / "current").exists()
    assert va_reader.catalog_db_path("v1").resolve() == db.resolve()
