"""Tests for the ``miw`` product: its variant definitions, `load`, the generated
``aos/mi_config.yaml``, and the external-variant registration.

The `load` tests point ``RUBINWORK_DATA`` at a pytest ``tmp_path``, so nothing reads the
real data root on S3DF. The ``variants.yaml`` tests read the repository's own config
files, to check that the definitions still agree with the generated ``aos/mi_config.yaml``
-- which the external package ``lsst.ts.intrinsic.wavefront`` reads its per-entry
configuration through, and whose body phase 3a requires to be byte-identical.
"""

import hashlib
import json
import pathlib
import sys

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from rubinwork.products import catalog, fam_tables, miw  # noqa: E402
from rubinwork.products.miw import gen_mi_config  # noqa: E402
from rubinwork.products.miw import reader as mi_reader  # noqa: E402
from rubinwork.products.miw import write_manifest  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

PREMOVE_BODY = pathlib.Path(__file__).with_name("mi_config_premove_body.yaml")
"""``aos/mi_config.yaml`` from ``defaults:`` on, as it stood before the phase 3a move."""

EXPECTED_VARIANTS = [
    "d12-A_50_34_i", "d12-A_50_34_i_5rot",
    "d13t-A_50_34_i", "d13t-A_50_34_i_5rot",
    "d13v1000-A_50_34_i", "d13v1000-A_50_34_i_rbr", "d13v1000-A_50_50_i_rbr",
    "d13v1000-A_50_34_i_5rot", "d13v1000-A_50_50_i_rbr_5rot",
    "d13v1000-A_22_12_i", "d13v1000-A_22_12_i_5rot", "d13v1000-A_50_34_i_rbr_5rot",
]
"""Every variant, in ``variants.yaml`` order, which is generated-file order."""

EXPECTED_EXTERNAL = ["official-20260528a", "official-gmegias-v3", "staged-v1"]
"""The external variants: two official calibration collections and the staged copy."""


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


def _write_build(root, variant="d12-A_50_34_i", build="20261010", rows=4,
                 tables=miw.TABLES, cwfs=()):
    """Write a fake build: the named tables, the split plot, and a manifest."""
    d = root / "products" / miw.PRODUCT / variant / build
    d.mkdir(parents=True)
    for i, table in enumerate(tables):
        pd.DataFrame({"day_obs": [20260315] * rows,
                      "seq_num": list(range(rows)),
                      "table_tag": [table] * rows,
                      "value": [float(i)] * rows}).to_parquet(d / f"{table}.parquet")
    (d / "intrinsic_split.pdf").write_bytes(b"%PDF-1.4 fake")
    for name in cwfs:
        wd = d / "wfs" / name
        wd.mkdir(parents=True)
        pd.DataFrame({"day_obs": [20260315] * rows,
                      "seq_num": list(range(rows))}).to_parquet(
                          wd / "zk_intrinsic.parquet")
    write_manifest.write(variant, build=build, build_dir=d)
    return d


# ---- variants ------------------------------------------------------------------

def test_variants_are_the_twelve_in_file_order():
    assert miw.variants() == EXPECTED_VARIANTS


def test_external_variants_are_separate_from_built_ones():
    assert miw.external_variants() == EXPECTED_EXTERNAL
    assert not set(miw.variants()) & set(miw.external_variants())


def test_variant_names_are_short_code_plus_dir_name():
    for variant in miw.variants():
        cfg = miw.variant_config(variant)
        assert variant == f"{cfg['short_code']}-{cfg['dir_name']}"


def test_every_variant_has_the_required_fields():
    for variant in miw.variants():
        cfg = miw.variant_config(variant)
        for field in ("fam_variant", "param_set", "mi_name", "dir_name", "text"):
            assert field in cfg, f"{variant} is missing {field}"


def test_every_fam_variant_is_a_real_fam_variant():
    known = set(fam_tables.variants(registered_only=False))
    for variant in miw.variants():
        assert miw.fam_variant(variant) in known


def test_param_set_matches_the_fam_variant():
    for variant in miw.variants():
        cfg = miw.variant_config(variant)
        fam = fam_tables.variant_config(cfg["fam_variant"])
        assert cfg["param_set"] == fam["param_set"]


def test_mi_keys_returns_the_long_param_set_and_the_mi_name():
    assert miw.mi_keys("d12-A_50_34_i_5rot") == (
        "fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x", "pathA_50_34_i_5rot")


def test_unknown_variant_raises():
    with pytest.raises(KeyError, match="not a miw variant"):
        miw.variant_config("d12-nope")


def test_external_variant_has_no_fam_variant():
    with pytest.raises(KeyError, match="external"):
        miw.fam_variant("official-20260528a")


def test_every_external_variant_has_a_location():
    for variant in miw.external_variants():
        cfg = miw.variant_config(variant)
        assert cfg.get("location"), f"{variant} has no location"
        assert cfg.get("kind"), f"{variant} has no kind"


# ---- build_from and the rotator bins -------------------------------------------

def test_build_source_resolves_build_from_to_a_product_variant():
    assert miw.build_source("d12-A_50_34_i_5rot") == "d12-A_50_34_i"
    assert miw.build_source("d13v1000-A_50_50_i_rbr_5rot") == "d13v1000-A_50_50_i_rbr"


def test_build_source_of_a_self_building_variant_is_itself():
    assert miw.build_source("d12-A_50_34_i") == "d12-A_50_34_i"


def test_build_from_stays_within_one_fam_variant():
    for variant in miw.variants():
        source = miw.build_source(variant)
        assert miw.fam_variant(source) == miw.fam_variant(variant)


def test_default_rotator_bins_are_the_nine_from_defaults():
    assert len(miw.rotator_bins("d12-A_50_34_i")) == 9


def test_split_rotator_select_narrows_to_five_bins():
    bins = miw.split_rotator_bins("d12-A_50_34_i_5rot")
    assert bins == [(-65.0, -55.0), (-20.0, -10.0), (-3.0, 3.0), (10.0, 20.0),
                    (55.0, 65.0)]


def test_split_without_rotator_select_uses_every_bin_of_the_source():
    assert len(miw.split_rotator_bins("d12-A_50_34_i")) == 9


# ---- the generated aos/mi_config.yaml ------------------------------------------

def test_mi_config_yaml_is_up_to_date():
    """The generated file on disk is what the generator produces."""
    assert gen_mi_config.main(["--check"]) == 0


def test_mi_config_body_is_byte_identical_to_the_premove_file():
    """The acceptance test for the phase 3a move.

    The body -- everything from ``defaults:`` on -- is what the external package parses.
    The header above it is replaced by the DO-NOT-EDIT notice, whose content moved into
    ``variants.yaml``, so only the body is compared.
    """
    text = gen_mi_config.render()
    lines = text.split("\n")
    body = "\n".join(lines[lines.index("defaults:"):])
    assert body == PREMOVE_BODY.read_text()


def test_mi_config_md5_is_recorded():
    text = gen_mi_config.render()
    assert hashlib.md5(text.encode()).hexdigest() == gen_mi_config.CONFIG_MD5


def test_generated_file_has_no_trailing_newline():
    """The file on disk ends without one; the external package is indifferent, but
    byte-identity is not."""
    assert not gen_mi_config.render().endswith("\n")


def test_generated_file_carries_every_variant_as_an_entry():
    doc = yaml.safe_load(gen_mi_config.render())
    entries = {(ps, e["name"])
               for ps, lst in doc["measured_intrinsics"].items() for e in lst}
    assert entries == {miw.mi_keys(v) for v in miw.variants()}


def test_generated_file_keeps_the_defaults_block():
    doc = yaml.safe_load(gen_mi_config.render())
    assert doc["defaults"]["path"] == "A"
    assert doc["defaults"]["filter"] == ["i"]
    assert len(doc["defaults"]["rotator_bins"]) == 9


def test_generated_build_from_names_an_mi_name_not_a_variant():
    """The runners take ``--mi-name``, so the generated file must carry the parent's
    ``mi_name``, not the product variant name that ``variants.yaml`` uses."""
    doc = yaml.safe_load(gen_mi_config.render())
    for lst in doc["measured_intrinsics"].values():
        names = {e["name"] for e in lst}
        for entry in lst:
            if "build_from" in entry:
                assert entry["build_from"] in names


def test_external_variants_are_not_in_the_generated_file():
    text = gen_mi_config.render()
    for variant in miw.external_variants():
        assert variant not in text


def test_entry_index_fields_must_match_the_text_block():
    """A text block edited without its index fields fails loudly."""
    doc = mi_reader.groups()
    group = dict(doc["groups"][0])
    entry = dict(group["entries"][0], mi_name="pathA_not_this")
    with pytest.raises(ValueError, match="is not the `name:`"):
        gen_mi_config._check_entry(group, entry)


def test_variant_name_mismatch_is_caught():
    doc = mi_reader.groups()
    group = dict(doc["groups"][0])
    entry = dict(group["entries"][0], variant="wrong-name")
    with pytest.raises(ValueError, match="should be"):
        gen_mi_config._check_entry(group, entry)


# ---- comparison keys ------------------------------------------------------------

def test_sidecar_key_needs_the_centroid():
    """``day_obs, seq_num, detector`` is not unique on either sidecar: many donuts share
    a detector in one exposure."""
    from rubinwork.products.miw import compare_builds

    assert compare_builds.DONUT_SIDECAR_KEY == [
        "day_obs", "seq_num", "detector", "centroid_x_extra", "centroid_y_extra"]


def test_every_compared_table_has_a_key():
    from rubinwork.products.miw import compare_builds

    for name in ("intrinsic_split_maps", "intrinsic_split_decomp",
                 "intrinsic_split_rms", "zk_intrinsic", "fits"):
        assert compare_builds.TABLE_KEYS[name]
    for name in ("intrinsic_grid", "dz_fits", "intrinsic_cov_edge"):
        assert compare_builds.BIN_TABLE_KEYS[name]


def test_key_is_unique_detects_a_duplicate(tmp_path):
    from rubinwork.products.miw import compare_builds

    path = tmp_path / "t.parquet"
    pd.DataFrame({"j": [4, 4, 5], "part": [0, 0, 1]}).to_parquet(path)
    unique, n_unique, n_rows = compare_builds.key_is_unique(path, ["j", "part"])
    assert not unique
    assert (n_unique, n_rows) == (2, 3)


# ---- load ----------------------------------------------------------------------

def test_load_returns_the_tables_in_order(data_root):
    _write_build(data_root)
    tables = miw.load("d12-A_50_34_i", build="20261010")
    assert len(tables) == len(miw.TABLES)
    for table, df in zip(miw.TABLES, tables):
        assert df["table_tag"].iloc[0] == table


def test_load_resolves_current(data_root):
    _write_build(data_root)
    catalog.set_current("miw", "d12-A_50_34_i", "20261010")
    maps, = miw.load("d12-A_50_34_i", tables=("intrinsic_split_maps",))
    assert maps["table_tag"].iloc[0] == "intrinsic_split_maps"


def test_load_subset_of_tables(data_root):
    _write_build(data_root)
    fits, = miw.load("d12-A_50_34_i", build="20261010", tables=("fits",))
    assert fits["table_tag"].iloc[0] == "fits"


def test_load_rejects_unknown_table(data_root):
    _write_build(data_root)
    with pytest.raises(ValueError, match="not miw tables"):
        miw.load("d12-A_50_34_i", build="20261010", tables=("nope",))


def test_load_table_rejects_unknown_name(data_root):
    with pytest.raises(ValueError, match="not a miw table"):
        miw.load_table("nope", "d12-A_50_34_i")


def test_load_missing_build_raises(data_root):
    with pytest.raises(catalog.ProductNotFound):
        miw.load("d12-A_50_34_i", build="19990101")


def test_build_dir_does_not_need_to_exist(data_root):
    d = miw.build_dir("d12-A_50_34_i", "20991231")
    assert d.name == "20991231"
    assert not d.exists()


def test_load_maps_is_the_old_load_miw():
    assert miw.load_miw is miw.load_maps


def test_js_default_is_the_twenty_one_term_noll_set():
    assert miw.JS_DEFAULT == tuple(j for j in range(4, 27) if j not in (20, 21))
    assert len(miw.JS_DEFAULT) == 21


def test_load_maps_reads_a_grid(tmp_path):
    path = tmp_path / "maps.parquet"
    cols = {"thx_deg": [0.0, 1.0], "thy_deg": [0.0, 1.0]}
    for j in miw.JS_DEFAULT:
        cols[f"Z{j}_OCS"] = [0.1 * j, 0.2 * j]
    pd.DataFrame(cols).to_parquet(path)
    pts, zk, df = miw.load_maps(path)
    assert pts.shape == (2, 2)
    assert zk.shape == (2, max(miw.JS_DEFAULT) + 1)
    assert zk[0, 4] == pytest.approx(0.4)
    assert zk[0, 20] == 0.0       # not in the Noll set, so left zero
    assert len(df) == 2


def test_load_maps_rejects_a_missing_column(tmp_path):
    path = tmp_path / "maps.parquet"
    pd.DataFrame({"thx_deg": [0.0], "thy_deg": [0.0], "Z4_OCS": [1.0]}).to_parquet(path)
    with pytest.raises(KeyError, match="missing"):
        miw.load_maps(path)


# ---- the manifest --------------------------------------------------------------

def test_manifest_records_the_variant_config(data_root):
    d = _write_build(data_root)
    man = json.loads((d / "manifest.json").read_text())
    assert man["product"] == "miw"
    assert man["variant"] == "d12-A_50_34_i"
    assert man["config"]["mi_name"] == "pathA_50_34_i"
    assert "text" in man["config"]      # the literal configuration YAML travels with it


def test_manifest_records_the_fam_build_as_an_input(data_root):
    d = miw.build_dir("d12-A_50_34_i", "20261011")
    d.mkdir(parents=True)
    write_manifest.write("d12-A_50_34_i", build_dir=d, fam_build="20260920")
    man = json.loads((d / "manifest.json").read_text())
    assert man["inputs"]["fam_tables"] == "danish_1_2@20260920"


def test_manifest_of_a_build_from_variant_records_its_parent(data_root):
    d = miw.build_dir("d12-A_50_34_i_5rot", "20261011")
    d.mkdir(parents=True)
    write_manifest.write("d12-A_50_34_i_5rot", build_dir=d, fam_build="20260920")
    man = json.loads((d / "manifest.json").read_text())
    assert man["inputs"]["miw"] == "d12-A_50_34_i"


def test_manifest_of_a_self_building_variant_names_no_parent(data_root):
    d = _write_build(data_root)
    man = json.loads((d / "manifest.json").read_text())
    assert "miw" not in (man.get("inputs") or {})


def test_manifest_counts_rows_off_disk(data_root):
    d = _write_build(data_root, rows=7)
    man = json.loads((d / "manifest.json").read_text())
    assert man["files"]["fits.parquet"]["rows"] == 7


def test_manifest_write_needs_a_build_or_a_dir(data_root):
    with pytest.raises(ValueError, match="needs"):
        write_manifest.write("d12-A_50_34_i")


def test_manifest_cli(data_root, capsys):
    d = miw.build_dir("d12-A_50_34_i", "20261012")
    d.mkdir(parents=True)
    rc = write_manifest.main(["--variant", "d12-A_50_34_i", "--build-dir", str(d),
                              "--option", "coord_sys=OCS"])
    assert rc == 0
    assert "wrote" in capsys.readouterr().out
    man = json.loads((d / "manifest.json").read_text())
    assert man["config"]["build_options"]["coord_sys"] == "OCS"


# ---- external registration -----------------------------------------------------

def test_register_external_writes_a_location_and_no_files(data_root):
    d = write_manifest.register_external("official-20260528a")
    man = json.loads((d / "manifest.json").read_text())
    assert man["location"] == (
        "LSSTCam/calib/DM-55048/intrinsicZernikes.v1.0/intrinsicsGen.20260528a")
    assert man["files"] == {}
    assert d.name == write_manifest.EXTERNAL_BUILD


def test_register_external_staged_copy_points_into_the_repo(data_root):
    d = write_manifest.register_external("staged-v1")
    man = json.loads((d / "manifest.json").read_text())
    assert man["location"] == "aos/calibration/miw/intrinsic_split_maps_v1.parquet"
    assert man["config"]["source_mi_name"] == "pathA_50_34_i_5rot"


def test_register_external_rejects_a_built_variant(data_root):
    with pytest.raises(KeyError, match="not an external miw variant"):
        write_manifest.register_external("d12-A_50_34_i")


def test_register_external_all(data_root, capsys):
    rc = write_manifest.main(["--register-external", "all"])
    assert rc == 0
    out = capsys.readouterr().out
    for variant in miw.external_variants():
        assert variant in out
    listed = {b["variant"] for b in catalog.list_products("miw")}
    assert listed == set(miw.external_variants())


def test_external_variants_are_visible_to_the_catalog(data_root):
    write_manifest.register_external("official-gmegias-v3")
    path = catalog.path("miw", "official-gmegias-v3",
                        build=write_manifest.EXTERNAL_BUILD)
    assert path.exists()


# ---- the staged copy the external variant points at ----------------------------

def test_staged_parquet_and_its_provenance_are_in_the_repo():
    cfg = miw.variant_config("staged-v1")
    assert (REPO_ROOT / cfg["location"]).exists()
    assert (REPO_ROOT / cfg["provenance"]).exists()


def test_staged_provenance_agrees_with_the_variant_entry():
    cfg = miw.variant_config("staged-v1")
    prov = yaml.safe_load((REPO_ROOT / cfg["provenance"]).read_text())
    assert prov["param_set"] == cfg["source_param_set"]
    assert prov["mi_name"] == cfg["source_mi_name"]
    assert prov["git_sha"] == cfg["source_git_sha"]


# ---- the shims at the old paths -------------------------------------------------

def test_miw_io_shim_gives_the_same_module_object():
    sys.path.insert(0, str(REPO_ROOT / "aos" / "code"))
    try:
        import miw_io
    finally:
        sys.path.pop(0)
    assert miw_io is mi_reader
    assert miw_io.load_miw is mi_reader.load_maps
    assert miw_io.JS_DEFAULT is mi_reader.JS_DEFAULT


def test_miw_corner_shim_gives_the_same_module_object():
    from rubinwork import miw_corner

    sys.path.insert(0, str(REPO_ROOT / "aos" / "code"))
    try:
        import miw_corner_intrinsic
    finally:
        sys.path.pop(0)
    assert miw_corner_intrinsic is miw_corner
    for name in ("MiwCornerLookup", "decomp_path", "load_decomposition",
                 "corner_field_points", "corner_z4_height_um", "DEFAULT_PARAM_SET",
                 "DEFAULT_MI_NAME"):
        assert getattr(miw_corner_intrinsic, name) is getattr(miw_corner, name)
