"""Tests for the ``cwfs_tables`` product: its variant definitions, `load`, and the
FAM/CWFS triplet link it owns.

The `load` tests point ``RUBINWORK_DATA`` at a pytest ``tmp_path``, so nothing reads the
real data root on S3DF. The ``variants.yaml`` tests read the repository's own config
files, to check that the definitions still agree with the generated
``aos/param_sets.yaml`` -- which the external package reads the collections through, and
which phase 3a requires to be byte-identical.
"""

import json
import pathlib
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from rubinwork.products import catalog, cwfs_tables, fam_tables  # noqa: E402
from rubinwork.products.cwfs_tables import reader as cw_reader  # noqa: E402
from rubinwork.products.cwfs_tables import write_manifest  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

EXPECTED_VARIANTS = {"d12-refitWcs", "d12-refitWcs_2025", "d12-paired_3mm",
                     "d12-ai_donut", "d12-tarts"}
"""The registered variants: the five that pair with the ``danish_1_2`` FAM build."""


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


def _write_build(root, variant="d12-refitWcs", build="20261010", rows=4):
    """Write a fake build: both tables plus a manifest, so the catalog sees it."""
    d = root / "products" / cwfs_tables.reader.PRODUCT / variant / build
    d.mkdir(parents=True)
    for i, table in enumerate(cwfs_tables.TABLES):
        pd.DataFrame({"day_obs": [20251126] * rows,
                      "seq_num": list(range(rows)),
                      "table_tag": [table] * rows,
                      "value": [float(i)] * rows}).to_parquet(d / f"{table}.parquet")
    write_manifest.write(variant, build=build, build_dir=d)
    return d


# ---- variants ------------------------------------------------------------------

def test_variants_are_the_registered_five():
    assert set(cwfs_tables.variants()) == EXPECTED_VARIANTS


def test_unregistered_variant_is_hidden_but_resolvable():
    """``d10-wep17_3_0`` is carried for byte-identity of the generated file only."""
    assert "d10-wep17_3_0" not in cwfs_tables.variants()
    assert "d10-wep17_3_0" in cw_reader.variants(registered_only=False)
    assert cwfs_tables.variant_config("d10-wep17_3_0")["fam_variant"] == "danish_1_0"


def test_variant_names_follow_the_rules():
    """Lowercase-ish, at most 24 characters, and prefixed with the upstream short code."""
    for name in cw_reader.variants(registered_only=False):
        assert len(name) <= 24, f"{name} is {len(name)} characters"
        assert name.startswith(("d12-", "d10-")), name
        assert " " not in name


def test_every_variant_has_the_required_fields():
    for name in cw_reader.variants(registered_only=False):
        cfg = cwfs_tables.variant_config(name)
        for field in ("fam_variant", "upstream_name", "collection"):
            assert cfg.get(field), f"{name} is missing {field}"


def test_every_fam_variant_is_a_real_fam_variant():
    """A dangling ``fam_variant`` would make the build unresolvable."""
    known = set(fam_tables.reader.variants(registered_only=False))
    for name in cw_reader.variants(registered_only=False):
        assert cwfs_tables.fam_variant(name) in known, name


def test_unknown_variant_raises():
    with pytest.raises(KeyError, match="not a cwfs_tables variant"):
        cwfs_tables.variant_config("no_such_variant")


def test_reader_fields_survive():
    """The per-entry builder fields the variants differ by."""
    assert cwfs_tables.variant_config("d12-paired_3mm")["seq_offset"] == 0
    assert (cwfs_tables.variant_config("d12-ai_donut")["dataset_type"]
            == "aggregateZernikesRaw")
    tarts = cwfs_tables.variant_config("d12-tarts")
    assert tarts["reader"] == "unpaired"
    assert tarts["dataset_type"] == "aggregateAOSVisitTableAvg"


# ---- the FAM/CWFS triplet link, which this product owns ------------------------

def test_fam_tables_no_longer_carries_wfs_collections():
    """Phase 3a moved the link here; two owners would drift apart."""
    for name in fam_tables.reader.variants(registered_only=False):
        cfg = fam_tables.variant_config(name)
        assert "wfs_collections" not in cfg, f"{name} still carries wfs_collections"


def test_wfs_collections_for_rebuilds_the_map():
    wfs = cwfs_tables.wfs_collections_for("danish_1_2")
    assert set(wfs) == {"refitWcs", "refitWcs_2025", "paired_3mm", "ai_donut", "tarts"}
    # An entry with only a collection is a bare string, which is the shape
    # aos/param_sets.yaml has today.
    assert isinstance(wfs["refitWcs"], str)
    assert wfs["paired_3mm"]["seq_offset"] == 0
    assert wfs["tarts"]["reader"] == "unpaired"


def test_wfs_collections_for_unknown_fam_variant_is_empty():
    assert cwfs_tables.wfs_collections_for("danish_1_3_test") == {}


def test_generated_param_sets_match_the_variants():
    """Every variant's collection is what the external package will actually read.

    ``run_wfs_mktable`` resolves its collection through
    ``intrinsics_lib.load_param_sets()``, i.e. out of ``aos/param_sets.yaml``, so a
    variant whose collection disagrees with that file would build the wrong data.
    """
    param_sets = yaml.safe_load((REPO_ROOT / "aos" / "param_sets.yaml").read_text())
    for name in cw_reader.variants(registered_only=False):
        cfg = cwfs_tables.variant_config(name)
        fam_cfg = fam_tables.variant_config(cfg["fam_variant"])
        entry = param_sets[fam_cfg["param_set"]]["wfs_collections"][cfg["upstream_name"]]
        collection = entry["collection"] if isinstance(entry, dict) else entry
        assert collection == cfg["collection"], name


def test_param_sets_yaml_is_up_to_date():
    """The generated file on disk matches what the two variants.yaml files produce."""
    from rubinwork.products.fam_tables import gen_param_sets
    on_disk = (REPO_ROOT / "aos" / "param_sets.yaml").read_text()
    assert on_disk == gen_param_sets.render(), (
        "aos/param_sets.yaml is stale — regenerate with "
        "python -m rubinwork.products.fam_tables.gen_param_sets")


# ---- the comparison key --------------------------------------------------------

def test_donut_key_excludes_the_absent_donut_id():
    """The CWFS tables carry no donut-id column, so the fam_tables key cannot apply."""
    assert "extra_donut_id" not in cwfs_tables.DONUT_KEY
    assert cwfs_tables.DONUT_KEY == ("day_obs", "seq_num", "detector",
                                     "thx_OCS", "thy_OCS")


def test_compare_builds_uses_the_product_key():
    from rubinwork.products.cwfs_tables import compare_builds
    assert compare_builds.TABLE_KEYS["donuts"] == list(cwfs_tables.DONUT_KEY)
    assert compare_builds.TABLE_KEYS["visits"] == list(cwfs_tables.VISIT_KEY)


def test_key_is_unique_detects_a_duplicate(tmp_path):
    from rubinwork.products.cwfs_tables import compare_builds
    p = tmp_path / "donuts.parquet"
    pd.DataFrame({"day_obs": [20251126, 20251126],
                  "seq_num": [31, 31],
                  "detector": ["R00_SW0", "R00_SW0"],
                  "thx_OCS": [0.1, 0.1],
                  "thy_OCS": [0.2, 0.2]}).to_parquet(p)
    unique, n_unique, n_rows = compare_builds.key_is_unique(
        p, list(cwfs_tables.DONUT_KEY))
    assert not unique
    assert (n_unique, n_rows) == (1, 2)


# ---- load ----------------------------------------------------------------------

def test_load_returns_two_tables_in_order(data_root):
    _write_build(data_root)
    donuts, visits = cwfs_tables.load("d12-refitWcs", build="20261010")
    assert list(donuts["table_tag"])[0] == "donuts"
    assert list(visits["table_tag"])[0] == "visits"


def test_load_resolves_current(data_root):
    d = _write_build(data_root)
    catalog.set_current(cwfs_tables.reader.PRODUCT, "d12-refitWcs", d.name)
    donuts, visits = cwfs_tables.load("d12-refitWcs")
    assert len(donuts) == 4


def test_load_subset_of_tables(data_root):
    _write_build(data_root)
    (visits,) = cwfs_tables.load("d12-refitWcs", build="20261010", tables=("visits",))
    assert list(visits["table_tag"])[0] == "visits"


def test_load_rejects_unknown_table(data_root):
    _write_build(data_root)
    with pytest.raises(ValueError, match="not cwfs_tables tables"):
        cwfs_tables.load("d12-refitWcs", build="20261010", tables=("fits",))


def test_load_table_rejects_unknown_name(data_root):
    with pytest.raises(ValueError, match="not a cwfs_tables table"):
        cwfs_tables.load_table("fits", "d12-refitWcs", build="20261010")


def test_load_missing_build_raises(data_root):
    _write_build(data_root)
    with pytest.raises(catalog.ProductNotFound):
        cwfs_tables.load("d12-refitWcs", build="29991231")


def test_build_dir_does_not_need_to_exist(data_root):
    d = cwfs_tables.build_dir("d12-tarts", "20261010")
    assert d.name == "20261010"
    assert d.parent.name == "d12-tarts"
    assert not d.exists()


# ---- write_manifest ------------------------------------------------------------

def test_manifest_records_the_variant_config(data_root):
    d = _write_build(data_root, variant="d12-tarts")
    doc = json.loads((d / "manifest.json").read_text())
    assert doc["product"] == "cwfs_tables"
    assert doc["variant"] == "d12-tarts"
    assert doc["config"]["collection"] == (
        cwfs_tables.variant_config("d12-tarts")["collection"])
    assert doc["config"]["reader"] == "unpaired"


def test_manifest_records_the_fam_build_as_an_input(data_root):
    d = cwfs_tables.build_dir("d12-refitWcs", "20261010")
    d.mkdir(parents=True)
    for table in cwfs_tables.TABLES:
        pd.DataFrame({"day_obs": [20251126]}).to_parquet(d / f"{table}.parquet")
    write_manifest.write("d12-refitWcs", build="20261010", fam_build="20260920")
    doc = json.loads((d / "manifest.json").read_text())
    assert doc["inputs"] == {"fam_tables": "danish_1_2@20260920"}


def test_manifest_counts_rows_off_disk(data_root):
    d = _write_build(data_root, rows=7)
    doc = json.loads((d / "manifest.json").read_text())
    assert doc["files"]["donuts.parquet"]["rows"] == 7


def test_manifest_cli(data_root, capsys):
    d = cwfs_tables.build_dir("d12-refitWcs", "20261010")
    d.mkdir(parents=True)
    for table in cwfs_tables.TABLES:
        pd.DataFrame({"day_obs": [20251126]}).to_parquet(d / f"{table}.parquet")
    assert write_manifest.main(["--variant", "d12-refitWcs", "--build", "20261010",
                                "--fam-build", "20260920",
                                "--option", "coord_sys=OCS"]) == 0
    doc = json.loads((d / "manifest.json").read_text())
    assert doc["config"]["build_options"] == {"coord_sys": "OCS"}
    assert "wrote" in capsys.readouterr().out
