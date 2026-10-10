"""Tests for the ``fam_tables`` product: its variant definitions and `load`.

The `load` tests point ``RUBINWORK_DATA`` at a pytest ``tmp_path``, so nothing reads
the real data root on S3DF. The ``variants.yaml`` tests read the repository's own config
files, to check that the variant definitions still agree with ``aos/param_sets.yaml`` and
``aos/snake_config.yaml`` — the sources they were transcribed from in phase 2.
"""

import json
import pathlib
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from rubinwork.products import catalog, manifest  # noqa: E402
from rubinwork.products import fam_tables  # noqa: E402
from rubinwork.products.fam_tables import reader as ft_load  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]


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


def make_fam_build(data_root, variant="danish_1_2", build="20261009", n_rows=4,
                   tables=fam_tables.TABLES, current=True):
    """Write a fake ``fam_tables`` build: one parquet per table, plus its manifest.

    Parameters
    ----------
    current : `bool`, optional
        Point ``current`` at this build afterwards, which a real build never
        does for itself.

    Returns
    -------
    build_dir : `pathlib.Path`
        The build directory.
    """
    build_dir = data_root / "products" / "fam_tables" / variant / build
    build_dir.mkdir(parents=True)
    for i, table in enumerate(tables):
        # A distinct column per table, so a mixed-up return order is detectable.
        pd.DataFrame({"visit": range(n_rows), f"{table}_col": [i] * n_rows}).to_parquet(
            build_dir / f"{table}.parquet", index=False)
    manifest.write(build_dir, "fam_tables", variant, build)
    if current:
        catalog.set_current("fam_tables", variant, build)
    return build_dir


# ---- variants.yaml -------------------------------------------------------------

def test_variants_are_the_in_use_three():
    """The registered variants are the three in-use ones, and not the obsolete names."""
    assert fam_tables.variants() == ["danish_1_2", "danish_1_3_test", "danish_1_3_v1000"]


def test_obsolete_variants_absent():
    """The obsolete FAM names are deliberately not registered (phase 2 decision)."""
    for name in ("danish_1_0", "danish_1_1_1", "danish_1_2_0_wep17_7_0_2025"):
        assert name not in fam_tables.variants()


def test_variant_names_follow_the_rules():
    """Section 5: lowercase, at most 24 characters."""
    for name in fam_tables.variants():
        assert name == name.lower()
        assert len(name) <= 24, f"{name} is {len(name)} characters (max 24)"


def test_every_variant_has_the_required_fields():
    """Each entry carries the long param_set key, a builder and its Butler inputs."""
    for name in fam_tables.variants():
        cfg = fam_tables.variant_config(name)
        for field in ("param_set", "builder", "butler_repo", "fam_collections",
                      "collection_phrase", "coord_sys"):
            assert cfg.get(field), f"{name} is missing {field}"
        assert cfg["builder"] in ("snakemake", "blitz")


def test_unknown_variant_raises():
    with pytest.raises(KeyError, match="not a fam_tables variant"):
        fam_tables.variant_config("no_such_variant")


def test_variants_match_param_sets_yaml():
    """Each variant still agrees with its ``aos/param_sets.yaml`` entry.

    This is the transcription phase 2 made; if ``param_sets.yaml`` changes, the variant
    definition has to change with it, or a build would record the wrong configuration.
    """
    param_sets = yaml.safe_load((REPO_ROOT / "aos" / "param_sets.yaml").read_text())
    for name in fam_tables.variants():
        cfg = fam_tables.variant_config(name)
        src = param_sets[cfg["param_set"]]
        assert src.get("dir_name", cfg["param_set"]) == name
        for field in ("butler_repo", "fam_programs", "fam_collections",
                      "collection_phrase", "day_obs_min", "day_obs_max"):
            assert cfg.get(field) == src.get(field), f"{name}.{field}"


def test_chunks_match_snake_config():
    """A ``snakemake`` variant's chunks still match ``aos/snake_config.yaml``."""
    snake = yaml.safe_load((REPO_ROOT / "aos" / "snake_config.yaml").read_text())
    configured = snake["param_sets"]

    # One deliberate difference, A2 (2026-10-10).  variants.yaml narrows the third
    # danish_1_2 chunk to [20260418, 20260513] for the new product tree; the old
    # aos/ tree keeps [20260418, 20260531], the name of the directory it already
    # built, so a routine `snakemake` run there does not rebuild 219 visits
    # through mktable.  Both ranges hold the same 219 visits (max day_obs 20260513).
    known_diffs = [(((20260418, 20260513), None, None, None, None),
                    ((20260418, 20260531), None, None, None, None))]

    def norm(chunk):
        if not isinstance(chunk, dict):
            return (tuple(chunk), None, None, None, None)
        return (tuple(chunk["day_obs"]), chunk.get("collection"), chunk.get("phrase"),
                chunk.get("butler_repo"), chunk.get("programs"))

    def reconcile(mine, theirs):
        """``mine`` with each known allowed difference rewritten to ``theirs``."""
        out = []
        for m, t in zip(mine, theirs):
            out.append(t if (m, t) in known_diffs else m)
        return out

    for name in fam_tables.variants():
        cfg = fam_tables.variant_config(name)
        entry = configured.get(cfg["param_set"])
        if entry is None:
            # Built outside the Snakemake pipeline: the blitz recast writes the
            # combined tables directly, so it declares no chunks.
            assert cfg["builder"] == "blitz"
            assert not cfg.get("chunks")
            continue
        mine = [norm(c) for c in cfg.get("chunks", [])]
        theirs = [norm(c) for c in entry.get("chunks", [])]
        assert len(mine) == len(theirs), f"{name} chunk count differs from snake_config.yaml"
        assert reconcile(mine, theirs) == theirs, \
            f"{name} chunks differ from snake_config.yaml"
        assert cfg["coord_sys"] == entry.get("coord_sys", "OCS")


# ---- load ----------------------------------------------------------------------

def test_load_returns_three_tables_in_order(data_root):
    make_fam_build(data_root)
    donuts, visits, fits = fam_tables.load("danish_1_2", build="20261009")
    assert list(donuts.columns) == ["visit", "donuts_col"]
    assert list(visits.columns) == ["visit", "visits_col"]
    assert list(fits.columns) == ["visit", "fits_col"]
    assert len(donuts) == 4


def test_load_resolves_current(data_root):
    """Once ``current`` is set, the default build needs no name."""
    make_fam_build(data_root, build="20261009")
    donuts, visits, fits = fam_tables.load("danish_1_2")
    assert len(donuts) == 4


def test_load_current_stays_put_until_set_current(data_root):
    make_fam_build(data_root, build="20261008", n_rows=2)
    make_fam_build(data_root, build="20261009", n_rows=7, current=False)
    donuts, _, _ = fam_tables.load("danish_1_2")
    assert len(donuts) == 2, "a new build must not move current"
    catalog.set_current("fam_tables", "danish_1_2", "20261009")
    donuts, _, _ = fam_tables.load("danish_1_2")
    assert len(donuts) == 7


def test_load_subset_of_tables(data_root):
    """The donut table is 12.4 GB for danish_1_2, so skipping it has to work."""
    make_fam_build(data_root)
    visits, fits = fam_tables.load("danish_1_2", build="20261009",
                                  tables=("visits", "fits"))
    assert list(visits.columns) == ["visit", "visits_col"]
    assert list(fits.columns) == ["visit", "fits_col"]


def test_load_rejects_unknown_table(data_root):
    make_fam_build(data_root)
    with pytest.raises(ValueError, match="not fam_tables tables"):
        fam_tables.load("danish_1_2", build="20261009", tables=("donuts", "zernikes"))


def test_load_table_one_table(data_root):
    make_fam_build(data_root)
    fits = ft_load.load_table("fits", "danish_1_2", build="20261009")
    assert list(fits.columns) == ["visit", "fits_col"]


def test_load_table_rejects_unknown_name(data_root):
    make_fam_build(data_root)
    with pytest.raises(ValueError, match="not a fam_tables table"):
        ft_load.load_table("zernikes", "danish_1_2", build="20261009")


def test_load_missing_build_raises(data_root):
    make_fam_build(data_root, build="20261009")
    with pytest.raises(catalog.ProductNotFound):
        fam_tables.load("danish_1_2", build="19990101")


def test_load_missing_variant_raises(data_root):
    with pytest.raises(catalog.ProductNotFound):
        fam_tables.load("danish_1_3_test")


def test_load_missing_table_raises(data_root):
    """A build without all three tables fails on the one that is absent."""
    make_fam_build(data_root, tables=("visits", "fits"))
    with pytest.raises(catalog.ProductNotFound):
        fam_tables.load("danish_1_2", build="20261009")
    visits, fits = fam_tables.load("danish_1_2", build="20261009",
                                   tables=("visits", "fits"))
    assert len(visits) == 4


def test_product_name_is_registered_under(data_root):
    """The build lands under the product name the catalog lists."""
    make_fam_build(data_root)
    assert ft_load.PRODUCT == "fam_tables"
    listed = catalog.list_products("fam_tables")
    assert [b["variant"] for b in listed] == ["danish_1_2"]


# ---- generated aos/param_sets.yaml ---------------------------------------------

def test_param_sets_yaml_is_up_to_date():
    """aos/param_sets.yaml matches what variants.yaml generates.

    It is a generated view of variants.yaml, so a hand edit to it, or a change to
    variants.yaml without regenerating, would silently diverge.
    """
    from rubinwork.products.fam_tables import gen_param_sets
    path = gen_param_sets.default_path()
    if not path.exists():
        pytest.skip(f"{path} not present")
    assert path.read_text() == gen_param_sets.render(), (
        "aos/param_sets.yaml is stale; regenerate with "
        "python -m rubinwork.products.fam_tables.gen_param_sets")


def test_unregistered_variant_is_hidden_but_resolvable():
    """danish_1_0 is emitted for frozen provenance but is not a product build."""
    assert "danish_1_0" not in fam_tables.variants()
    assert "danish_1_0" in fam_tables.reader.variants(registered_only=False)
    cfg = fam_tables.variant_config("danish_1_0")
    assert cfg["param_set"] == "fam_danish_1_0_wep17_3_0_bin2x"


def test_generated_param_sets_carry_the_wfs_half():
    """The FAM/CWFS triplet link survives generation, with its per-entry fields."""
    import yaml
    from rubinwork.products.fam_tables import gen_param_sets
    doc = yaml.safe_load(gen_param_sets.render())
    wfs = doc["fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x"]["wfs_collections"]
    assert set(wfs) == {"refitWcs", "refitWcs_2025", "paired_3mm", "ai_donut", "tarts"}
    assert wfs["paired_3mm"]["seq_offset"] == 0
    assert wfs["ai_donut"]["dataset_type"] == "aggregateZernikesRaw"
    assert wfs["tarts"]["reader"] == "unpaired"


# ---- write_manifest ------------------------------------------------------------

def test_manifest_records_the_per_chunk_collections(data_root):
    """danish_1_2 is composite: one `collections` field would be wrong for most visits."""
    from rubinwork.products.fam_tables import write_manifest as wm

    build_dir = make_fam_build(data_root, build="20261010", current=False)
    wm.write("danish_1_2", build="20261010")
    man = catalog.manifest("fam_tables", "danish_1_2", "20261010")
    assert man["status"] == "complete"
    assert man["variant"] == "danish_1_2"
    chunks = man["config"]["chunks"]
    assert len(chunks) == 10
    assert {c["day_obs"][0] for c in chunks} >= {20260315, 20260713, 20251116}
    embargo = [c for c in chunks if c.get("butler_repo") == "/repo/embargo"]
    assert [c["day_obs"] for c in embargo] == [[20260713, 20260713]]
    assert man["config"]["param_set"] == "fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x"
    # Row counts read back off disk, not from a builder's counter.
    assert man["files"]["donuts.parquet"]["rows"] == 4
    assert set(man["files"]) == {"donuts.parquet", "visits.parquet", "fits.parquet"}
    assert (build_dir / "manifest.json").is_file()


def test_manifest_does_not_move_current(data_root):
    from rubinwork.products.fam_tables import write_manifest as wm

    make_fam_build(data_root, build="20261009")
    make_fam_build(data_root, build="20261010", current=False)
    wm.write("danish_1_2", build="20261010")
    assert catalog.path("fam_tables", "danish_1_2").name == "20261009"


def test_manifest_takes_the_long_param_set_key(data_root):
    from rubinwork.products.fam_tables import write_manifest as wm

    make_fam_build(data_root, build="20261010", current=False)
    wm.write("fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x", build="20261010")
    assert catalog.manifest("fam_tables", "danish_1_2", "20261010")["variant"] \
        == "danish_1_2"


def test_manifest_records_build_options(data_root):
    from rubinwork.products.fam_tables import write_manifest as wm

    make_fam_build(data_root, build="20261010", current=False)
    wm.write("danish_1_2", build="20261010",
             build_options={"builder": "snakemake", "mktable_no_thermal": "1"})
    config = catalog.manifest("fam_tables", "danish_1_2", "20261010")["config"]
    assert config["build_options"]["mktable_no_thermal"] == "1"


def test_manifest_into_an_out_of_tree_build_dir(tmp_path, data_root):
    """The reference builds live outside the data tree, under --out-dir."""
    from rubinwork.products.fam_tables import write_manifest as wm

    out = tmp_path / "refbuild" / "danish_1_3_v1000" / "moved"
    out.mkdir(parents=True)
    pd.DataFrame({"visit": [1, 2]}).to_parquet(out / "donuts.parquet", index=False)
    path = wm.write("danish_1_3_v1000", build_dir=out)
    man = json.loads(path.read_text())
    assert man["build"] == "moved"
    assert man["variant"] == "danish_1_3_v1000"
    assert man["files"]["donuts.parquet"]["rows"] == 2


def test_manifest_cli(data_root, capsys):
    from rubinwork.products.fam_tables import write_manifest as wm

    make_fam_build(data_root, build="20261010", current=False)
    assert wm.main(["--variant", "danish_1_2", "--build", "20261010",
                    "--option", "builder=snakemake"]) == 0
    man = catalog.manifest("fam_tables", "danish_1_2", "20261010")
    assert man["config"]["build_options"] == {"builder": "snakemake"}
    assert "wrote" in capsys.readouterr().out


def test_manifest_cli_status_keeps_a_build_out_of_the_catalog(data_root):
    from rubinwork.products.fam_tables import write_manifest as wm

    make_fam_build(data_root, build="20261010", current=False)
    wm.main(["--variant", "danish_1_2", "--build", "20261010",
             "--status", "tables-only"])
    assert catalog.list_products("fam_tables") == []
    with pytest.raises(catalog.ProductNotFound, match="tables-only"):
        catalog.path("fam_tables", "danish_1_2", "20261010")
