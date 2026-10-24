"""Tests for the ``dof_lut`` product: its variant definitions, the reader, the manifest
with one upstream, and the schema-metadata comparison.

The `load` tests point ``RUBINWORK_DATA`` at a pytest ``tmp_path``, so nothing reads the
real data root on S3DF. The ``variants.yaml`` tests read the repository's own config, to
check that each variant's MIW settings entry still exists and that the derived
``(param_set, mi_name)`` pair still resolves through the generated ``mi_config.yaml``.

Two things here that no earlier product's tests cover: the **parquet schema metadata**
(where a LUT's mode counts and visit count are recorded, and nowhere else), and the
conditional ``lut_by_rotbin`` table, which a build legitimately lacks when its fit table
has no rotator-angle column.
"""

import pathlib

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


from rubinwork.products import catalog, dof_lut, fam_tables, miw  # noqa: E402
from rubinwork.products.dof_lut import compare_builds  # noqa: E402
from rubinwork.products.dof_lut import reader as dl_reader  # noqa: E402
from rubinwork.products.dof_lut import write_manifest  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

EXPECTED_VARIANTS = {"d12-A_50_34_i_5rot", "d12-A_50_34_i", "d13t-A_50_34_i_5rot"}
"""The three registered variants: the three jobs aos/Snakefile planned before the move."""

SAMPLE_METADATA = {
    "param_set": "fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x",
    "mi_name": "pathA_50_34_i_5rot",
    "reduce": "median", "n_dof": "50", "n_keep": "34",
    "n_visits": "863", "k_min": "1", "k_max": "6",
}
"""The eight schema-metadata fields, with the production reference build's values."""


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


def _write_build(root, variant="d12-A_50_34_i_5rot", build="20261011", n_dof=4,
                 manifest=True, scale=1.0, by_rotbin=True, metadata=None):
    """Write a fake build: the tables with their schema metadata, provenance, plot."""
    d = root / "products" / dl_reader.PRODUCT / variant / build
    d.mkdir(parents=True)
    meta = {k.encode(): v.encode()
            for k, v in (metadata or SAMPLE_METADATA).items()}

    lut = pa.table({
        "dof_index": pa.array(list(range(n_dof))),
        "dof_label": pa.array([f"dof{i}" for i in range(n_dof)]),
        "dof_unit": pa.array(["um"] * n_dof),
        "value": pa.array([float(i) * scale for i in range(n_dof)]),
        "mean": pa.array([float(i) * scale for i in range(n_dof)]),
        "scatter_mad": pa.array([0.1 * scale] * n_dof),
        "n_visits": pa.array([863] * n_dof),
    })
    pq.write_table(lut.replace_schema_metadata(meta), str(d / "lut.parquet"))

    kj = [(k, j) for k in (1, 2) for j in (4, 5, 6)]
    lut_dz = pa.table({
        "k": pa.array([k for k, _ in kj]),
        "j": pa.array([j for _, j in kj]),
        "dz_raw": pa.array([0.01 * (k + j) * scale for k, j in kj]),
        "dz_fit": pa.array([0.009 * (k + j) * scale for k, j in kj]),
        "dz_resid": pa.array([0.001 * (k + j) * scale for k, j in kj]),
        "n_visits": pa.array([863] * len(kj)),
    })
    pq.write_table(lut_dz.replace_schema_metadata(meta), str(d / "lut_dz.parquet"))

    if by_rotbin:
        centres = [-30.0, 0.0, 30.0]
        rows = [(c, i) for c in centres for i in range(n_dof)]
        pd.DataFrame({
            "rotbin_center": [c for c, _ in rows],
            "rotbin_lo": [c - 15.0 for c, _ in rows],
            "rotbin_hi": [c + 15.0 for c, _ in rows],
            "dof_index": [i for _, i in rows],
            "dof_label": [f"dof{i}" for _, i in rows],
            "value": [float(i) * scale for _, i in rows],
            "n_visits": [100] * len(rows),
        }).to_parquet(d / "lut_by_rotbin.parquet")

    with open(d / dl_reader.PROVENANCE, "w") as f:
        yaml.safe_dump({"param_set": SAMPLE_METADATA["param_set"],
                        "mi_name": SAMPLE_METADATA["mi_name"],
                        "n_dof": 50, "n_keep": 34, "reduce": "median",
                        "n_visits": 863, "k_min": 1, "k_max": 6,
                        "bands": ["i"], "programs": ["BLOCK-T614_triplets"],
                        "use_alt_window": False}, f, sort_keys=False)
    for plot in dof_lut.PLOTS:
        (d / plot).write_bytes(b"%PDF-1.4\n")
    if manifest:
        write_manifest.write(variant, build=build, build_dir=d,
                             fam_build="20260920", miw_build="20261007")
        # So `load(variant)` with no explicit build resolves, which is the normal route.
        catalog.set_current(dl_reader.PRODUCT, variant, build)
    return d


# ---- variants ------------------------------------------------------------------

def test_variants_are_the_registered_three():
    assert set(dof_lut.variants()) == EXPECTED_VARIANTS


def test_variant_names_follow_the_rules():
    """At most 32 characters, prefixed with the upstream MIW short code, no spaces."""
    for name in dl_reader.variants(registered_only=False):
        assert len(name) <= 32, f"{name} is {len(name)} characters"
        assert name.split("-")[0] in {"d12", "d13t"}, name
        assert " " not in name


def test_every_variant_names_a_real_miw_variant():
    for name in dl_reader.variants(registered_only=False):
        assert dof_lut.miw_variant(name) in miw.variants(registered_only=False)


def test_variant_name_carries_its_miw_variant():
    """The name is the MIW variant, since a LUT is built per (param_set, mi_name)."""
    for name in dl_reader.variants(registered_only=False):
        assert name == dof_lut.miw_variant(name)


def test_fam_variant_follows_from_the_miw_variant():
    """The real upstream is a fam_tables variant, derived rather than restated."""
    for name in dl_reader.variants(registered_only=False):
        fam = dof_lut.fam_variant(name)
        assert fam == miw.fam_variant(dof_lut.miw_variant(name))
        assert fam in fam_tables.variants(registered_only=False)


def test_mi_keys_resolve_through_the_generated_mi_config():
    """The (param_set, mi_name) the builder takes must exist in aos/mi_config.yaml."""
    with open(REPO_ROOT / "aos" / "mi_config.yaml") as f:
        entries = (yaml.safe_load(f) or {}).get("measured_intrinsics") or {}
    for name in dl_reader.variants(registered_only=False):
        param_set, mi_name = dof_lut.mi_keys(name)
        assert param_set in entries, f"{name}: {param_set} not in mi_config.yaml"
        assert mi_name in [e.get("name") for e in entries[param_set]], \
            f"{name}: {mi_name} not an entry of {param_set}"


def test_dir_name_is_a_string_not_a_yaml_integer():
    """Quoted in variants.yaml, so a later all-digit name cannot become an int."""
    for name in dl_reader.variants(registered_only=False):
        assert isinstance(dof_lut.dir_name(name), str)


def test_dir_names_are_unique():
    names = [dof_lut.dir_name(v) for v in dl_reader.variants(registered_only=False)]
    assert len(names) == len(set(names))


def test_dir_names_match_the_old_tree_layout():
    """``<fam dir_name>_<miw dir_name>``, which is what aos/output/lut/ keys on."""
    for name in dl_reader.variants(registered_only=False):
        fam = dof_lut.fam_variant(name)
        assert dof_lut.dir_name(name) == f"{fam}_{miw.dir_name(name)}"


def test_variants_yaml_has_no_unknown_keys():
    allowed = {"miw_variant", "dir_name", "registered"}
    doc = yaml.safe_load(open(dl_reader.VARIANTS_FILE))
    for name, entry in (doc.get("variants") or {}).items():
        assert set(entry) <= allowed, f"{name} has unexpected keys: {set(entry) - allowed}"


def test_unknown_variant_raises_and_lists_the_defined_ones():
    with pytest.raises(KeyError) as excinfo:
        dof_lut.variant_config("not-a-variant")
    assert "d12-A_50_34_i_5rot" in str(excinfo.value)


def test_no_build_settings_are_restated_here():
    """The builder resolves its own knobs from analysis_config.yaml's `lut:` section.

    Copying them into variants.yaml would be a second place to edit and a silent drift
    when the two disagree, so an entry must carry no `reduce`/`prefix`/`n_keep` key.
    """
    doc = yaml.safe_load(open(dl_reader.VARIANTS_FILE))
    knobs = {"n_dof", "n_keep", "reduce", "use_alt_window", "drop_bad_fit", "prefix"}
    for name, entry in (doc.get("variants") or {}).items():
        assert not (set(entry) & knobs), f"{name} restates builder knobs"


# ---- the reader ----------------------------------------------------------------

def test_load_returns_the_three_tables_in_order(data_root):
    _write_build(data_root)
    lut, lut_dz, by_rotbin = dof_lut.load("d12-A_50_34_i_5rot")
    assert list(lut.columns)[:2] == ["dof_index", "dof_label"]
    assert list(lut_dz.columns)[:2] == ["k", "j"]
    assert "rotbin_center" in by_rotbin.columns


def test_load_skips_a_missing_conditional_table(data_root):
    """lut_by_rotbin is written only when the fit table has a rotator-angle column."""
    _write_build(data_root, by_rotbin=False)
    tables = dof_lut.load("d12-A_50_34_i_5rot")
    assert len(tables) == 2


def test_load_does_not_skip_an_explicitly_requested_rotbin(data_root):
    """Asking for it by name and silently getting nothing back would hide its absence."""
    _write_build(data_root, by_rotbin=False)
    with pytest.raises(catalog.ProductNotFound):
        dof_lut.load("d12-A_50_34_i_5rot", tables=("lut_by_rotbin",))


def test_load_one_table(data_root):
    _write_build(data_root)
    lut = dof_lut.load_table("lut", "d12-A_50_34_i_5rot")
    assert len(lut) == 4


def test_load_rejects_a_table_that_is_not_one(data_root):
    _write_build(data_root)
    with pytest.raises(ValueError):
        dof_lut.load("d12-A_50_34_i_5rot", tables=("lut", "nope"))


def test_load_table_rejects_a_table_that_is_not_one(data_root):
    _write_build(data_root)
    with pytest.raises(ValueError):
        dof_lut.load_table("nope", "d12-A_50_34_i_5rot")


def test_build_metadata_reads_the_schema_fields(data_root):
    """The mode counts live only in the parquet schema metadata."""
    _write_build(data_root)
    meta = dof_lut.build_metadata("d12-A_50_34_i_5rot")
    assert meta["n_keep"] == "34"
    assert meta["n_dof"] == "50"
    assert set(dof_lut.SCHEMA_METADATA_FIELDS) <= set(meta)


def test_build_metadata_reads_lut_dz_too(data_root):
    _write_build(data_root)
    assert (dof_lut.build_metadata("d12-A_50_34_i_5rot", table="lut_dz")
            == dof_lut.build_metadata("d12-A_50_34_i_5rot", table="lut"))


def test_build_metadata_rejects_a_table_without_metadata(data_root):
    _write_build(data_root)
    with pytest.raises(ValueError):
        dof_lut.build_metadata("d12-A_50_34_i_5rot", table="lut_by_rotbin")


def test_build_provenance_reads_the_frozen_yaml(data_root):
    _write_build(data_root)
    prov = dof_lut.build_provenance("d12-A_50_34_i_5rot")
    assert prov["n_keep"] == 34
    assert prov["bands"] == ["i"]
    assert prov["use_alt_window"] is False


def test_a_build_without_a_manifest_is_invisible(data_root):
    _write_build(data_root, manifest=False)
    with pytest.raises(catalog.ProductNotFound):
        dof_lut.load("d12-A_50_34_i_5rot", build="20261011")


def test_build_dir_is_under_the_products_root(data_root):
    d = dof_lut.build_dir("d12-A_50_34_i_5rot", "20261011")
    assert d == catalog.products_root() / "dof_lut" / "d12-A_50_34_i_5rot" / "20261011"


# ---- the manifest --------------------------------------------------------------

def test_manifest_records_both_upstreams(data_root):
    """The miw build supplies the refit_mi fits, the fam_tables build the visits."""
    d = _write_build(data_root)
    import json
    with open(d / "manifest.json") as f:
        got = json.load(f)
    assert got["inputs"] == {"fam_tables": "danish_1_2@20260920",
                             "miw": "d12-A_50_34_i_5rot@20261007"}


def test_manifest_config_records_the_derived_keys(data_root):
    """param_set, mi_name and fam_variant are derived, so the manifest records them."""
    d = _write_build(data_root)
    import json
    with open(d / "manifest.json") as f:
        config = json.load(f)["config"]
    assert config["fam_variant"] == "danish_1_2"
    assert config["mi_name"] == "pathA_50_34_i_5rot"
    assert config["param_set"] == SAMPLE_METADATA["param_set"]
    assert config["miw_variant"] == "d12-A_50_34_i_5rot"


def test_manifest_claims_miw_as_an_input(data_root):
    """Since 2026-10-13 the LUT fits the post-MIW DZ, so miw IS a real upstream.

    Was ``test_manifest_does_not_claim_miw_as_an_input``: the plain-FAM LUT opened no
    intrinsic map, so recording miw would have claimed a dependency the build did not
    have. The post-MIW LUT reads that build's ``refit_mi`` fits.
    """
    d = _write_build(data_root)
    import json
    with open(d / "manifest.json") as f:
        got = json.load(f)
    assert got["inputs"]["miw"] == "d12-A_50_34_i_5rot@20261007"


def test_manifest_refuses_half_the_provenance(data_root):
    """Both upstreams or neither, as the coadds manifest does."""
    with pytest.raises(ValueError, match="together"):
        write_manifest.write("d12-A_50_34_i_5rot", build="20261011",
                             build_dir=data_root, fam_build="20260920")
    with pytest.raises(ValueError, match="together"):
        write_manifest.write("d12-A_50_34_i_5rot", build="20261011",
                             build_dir=data_root, miw_build="20261007")


def test_manifest_needs_a_build_or_a_build_dir():
    with pytest.raises(ValueError):
        write_manifest.write("d12-A_50_34_i_5rot")


def test_manifest_counts_every_file(data_root):
    d = _write_build(data_root)
    import json
    with open(d / "manifest.json") as f:
        files = json.load(f)["files"]
    for name in ("lut.parquet", "lut_dz.parquet", "lut_by_rotbin.parquet",
                 dl_reader.PROVENANCE, "lut.pdf"):
        assert name in files, name
    assert files["lut.parquet"]["rows"] == 4


# ---- the comparison ------------------------------------------------------------

def test_compare_finds_identical_builds_identical(data_root, capsys):
    a = _write_build(data_root, build="20261011")
    b = _write_build(data_root, build="20261012")
    assert compare_builds.main([str(a), str(b)]) == 0
    assert "builds match" in capsys.readouterr().out


def test_compare_catches_a_value_difference(data_root):
    a = _write_build(data_root, build="20261011")
    b = _write_build(data_root, build="20261012", scale=1.5)
    assert compare_builds.main([str(a), str(b)]) == 1


def test_compare_catches_a_schema_metadata_difference(data_root, capsys):
    """A build with a different n_keep compares clean on every row -- but is not the same
    measurement. This is what no earlier product's comparison would have caught."""
    a = _write_build(data_root, build="20261011")
    b = _write_build(data_root, build="20261012",
                     metadata={**SAMPLE_METADATA, "n_keep": "22"})
    assert compare_builds.main([str(a), str(b)]) == 1
    assert "n_keep" in capsys.readouterr().out


def test_compare_treats_a_rotbin_absent_from_both_as_agreement(data_root, capsys):
    a = _write_build(data_root, build="20261011", by_rotbin=False)
    b = _write_build(data_root, build="20261012", by_rotbin=False)
    assert compare_builds.main([str(a), str(b)]) == 0
    assert "absent from both" in capsys.readouterr().out


def test_compare_catches_a_rotbin_on_only_one_side(data_root):
    a = _write_build(data_root, build="20261011", by_rotbin=True)
    b = _write_build(data_root, build="20261012", by_rotbin=False)
    assert compare_builds.main([str(a), str(b)]) == 1


def test_compare_keys_come_from_the_reader(data_root):
    """So the comparison keys and the documented keys cannot drift apart."""
    assert compare_builds.TABLE_KEYS == dict(dl_reader.TABLE_KEYS)


def test_compare_key_uniqueness_check(data_root):
    d = _write_build(data_root)
    unique, n_unique, n_rows = compare_builds.key_is_unique(
        d / "lut.parquet", ["dof_index"])
    assert unique and n_unique == n_rows == 4
