"""Tests for the ``coadds`` product: its variant definitions, both readers, the manifest
with two upstreams, and the npz comparison.

The `load` tests point ``RUBINWORK_DATA`` at a pytest ``tmp_path``, so nothing reads the
real data root on S3DF. The ``variants.yaml`` tests read the repository's own config, to
check that each variant's MIW upstream still exists and that the block-selection settings
are the ones the builder understands.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest
import yaml


from rubinwork.products import catalog, coadds, miw  # noqa: E402
from rubinwork.products.coadds import compare_builds, reader as co_reader  # noqa: E402
from rubinwork.products.coadds import write_manifest  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

EXPECTED_VARIANTS = {"d12-A_50_34_i_5rot", "d12-A_50_34_i_5rot_v2"}
"""The two registered variants: the production run and its all-extra-Zernikes sibling."""


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


def _write_build(root, variant="d12-A_50_34_i_5rot", build="20261010", n_blocks=3,
                 manifest=True, scale=1.0, block_order=None):
    """Write a fake build: both tables, both npz files, the plots, and a manifest."""
    d = root / "products" / co_reader.PRODUCT / variant / build
    d.mkdir(parents=True)
    blocks = list(block_order if block_order is not None else range(n_blocks))
    for table in coadds.TABLES:
        pd.DataFrame({"block": blocks,
                      "day_obs": [20260316] * len(blocks),
                      "table_tag": [table] * len(blocks),
                      "value": [float(b) * scale for b in blocks]}).to_parquet(
                          d / f"{table}.parquet")
    # Each block's map is a function of its BLOCK VALUE, not of its row position, so a
    # build that holds the same blocks in a different order holds the same data -- which
    # is what the alignment test needs to distinguish from a real difference.
    def _block_map(block, n_terms):
        rng = np.random.default_rng(1000 + int(block))
        return rng.normal(size=(n_terms, 5, 5)) * scale

    for name, terms in (("block_grids", [5, 6, 7, 8]),
                        ("block_grids_ext", [12, 13])):
        grids = np.stack([_block_map(b, len(terms)) for b in blocks])
        np.savez(d / f"{name}.npz",
                 block=np.array(blocks),
                 terms=np.array(terms),
                 grids=grids,
                 miw=grids * 0.5,
                 xbins=np.linspace(-1.8, 1.8, 6))
    for plot in coadds.PLOTS:
        (d / plot).write_bytes(b"%PDF-1.4\n")
    if manifest:
        write_manifest.write(variant, build=build, build_dir=d,
                             fam_build="20260920", miw_build="20261010")
    return d


# ---- variants ------------------------------------------------------------------

def test_variants_are_the_registered_two():
    assert set(coadds.variants()) == EXPECTED_VARIANTS


def test_variant_names_follow_the_rules():
    """Lowercase-ish, at most 32 characters, prefixed with the upstream short code."""
    for name in co_reader.variants(registered_only=False):
        assert len(name) <= 32, f"{name} is {len(name)} characters"
        assert name.startswith("d12-"), name
        assert " " not in name


def test_every_variant_names_a_real_miw_variant():
    for name in co_reader.variants(registered_only=False):
        assert coadds.miw_variant(name) in miw.variants(registered_only=False)


def test_variant_name_carries_its_miw_variant():
    """The name is the MIW variant, optionally suffixed -- that is the naming rule."""
    for name in co_reader.variants(registered_only=False):
        assert name.startswith(coadds.miw_variant(name))


def test_fam_variant_follows_from_the_miw_variant():
    """It is not restated in variants.yaml, so the two cannot drift apart."""
    for name in co_reader.variants(registered_only=False):
        assert coadds.fam_variant(name) == miw.fam_variant(coadds.miw_variant(name))
        assert "fam_variant" not in coadds.variant_config(name)


def test_dir_name_is_a_string_not_a_yaml_integer():
    """An unquoted 50_34 parses as the integer 5034: YAML treats _ as a digit separator."""
    for name in co_reader.variants(registered_only=False):
        assert isinstance(coadds.dir_name(name), str), name


def test_dir_names_match_the_old_tree():
    assert coadds.dir_name("d12-A_50_34_i_5rot") == "50_34"
    assert coadds.dir_name("d12-A_50_34_i_5rot_v2") == "50_34_v2"


def test_dir_names_are_unique():
    """A duplicate would make two variants share an old-tree directory."""
    names = [coadds.dir_name(v) for v in co_reader.variants(registered_only=False)]
    assert len(set(names)) == len(names)


def test_unknown_variant_raises_and_lists_the_defined_ones():
    with pytest.raises(KeyError, match="not a coadds variant"):
        coadds.variant_config("nope")


def test_the_two_variants_differ_only_in_ext_terms():
    """The archive README records block_grids.npz as identical between the two runs."""
    a = coadds.variant_config("d12-A_50_34_i_5rot")
    b = coadds.variant_config("d12-A_50_34_i_5rot_v2")
    differing = {k for k in set(a) | set(b)
                 if a.get(k) != b.get(k)} - {"dir_name"}
    assert differing == {"ext_terms"}


# ---- the block-selection settings ----------------------------------------------

def test_build_settings_are_fields_the_builder_knows():
    """Every setting must be a real flag, or the Snakefile builds a broken command."""
    for name in co_reader.variants(registered_only=False):
        assert set(coadds.build_settings(name)) <= set(coadds.BUILD_FIELDS)


def test_build_settings_drop_nulls():
    """A null means `use the builder's own default`, so it is not passed on."""
    settings = coadds.build_settings("d12-A_50_34_i_5rot")
    assert "bands" not in settings          # null in variants.yaml
    assert "cwfs_inner" not in settings     # null in variants.yaml
    assert settings["max_blur"] == 1.4
    assert settings["bounce_programs"] == ["BLOCK-T720", "BLOCK-T724"]


def test_every_build_field_is_a_builder_argument():
    """BUILD_FIELDS is checked against the builder's own parser, not against a list."""
    from rubinwork.products.coadds.builders import run_coadd_blocks_miw as builder
    source = pathlib.Path(builder.__file__).read_text()
    for field in coadds.BUILD_FIELDS:
        flag = '--' + field.replace('_', '-')
        assert f'"{flag}"' in source, f"{flag} is not an argument of the builder"


def test_variants_yaml_has_no_unknown_keys():
    doc = yaml.safe_load(co_reader.VARIANTS_FILE.read_text())
    allowed = set(coadds.BUILD_FIELDS) | {"miw_variant", "dir_name", "registered"}
    for name, entry in doc["variants"].items():
        assert set(entry) <= allowed, f"{name}: unexpected {set(entry) - allowed}"


# ---- load ----------------------------------------------------------------------

def test_load_returns_both_tables_in_order(data_root):
    _write_build(data_root)
    summary, metrics = coadds.load("d12-A_50_34_i_5rot", build="20261010")
    assert list(summary["table_tag"])[0] == "blocks_summary"
    assert list(metrics["table_tag"])[0] == "coadd_metrics"
    assert len(summary) == 3


def test_load_one_table(data_root):
    _write_build(data_root)
    metrics, = coadds.load("d12-A_50_34_i_5rot", build="20261010",
                           tables=("coadd_metrics",))
    assert list(metrics["table_tag"])[0] == "coadd_metrics"


def test_load_rejects_a_table_that_is_not_one(data_root):
    _write_build(data_root)
    with pytest.raises(ValueError, match="not coadds tables"):
        coadds.load("d12-A_50_34_i_5rot", build="20261010", tables=("donuts",))


def test_load_table_rejects_a_table_that_is_not_one(data_root):
    with pytest.raises(ValueError, match="not a coadds table"):
        coadds.load_table("donuts", "d12-A_50_34_i_5rot")


def test_load_grids_returns_the_arrays(data_root):
    _write_build(data_root)
    grids = coadds.load_grids(variant="d12-A_50_34_i_5rot", build="20261010")
    assert set(grids.files) >= {"block", "terms", "grids", "miw"}
    assert grids["grids"].shape == (3, 4, 5, 5)


def test_load_grids_reads_the_ext_file(data_root):
    _write_build(data_root)
    ext = coadds.load_grids("block_grids_ext", variant="d12-A_50_34_i_5rot",
                            build="20261010")
    assert list(ext["terms"]) == [12, 13]


def test_load_grids_rejects_an_unknown_file(data_root):
    with pytest.raises(ValueError, match="not a coadds grid file"):
        coadds.load_grids("block_grids_extra", variant="d12-A_50_34_i_5rot")


def test_load_grids_needs_a_variant():
    with pytest.raises(ValueError, match="needs `variant`"):
        coadds.load_grids("block_grids")


def test_a_build_without_a_manifest_is_invisible(data_root):
    _write_build(data_root, manifest=False)
    with pytest.raises(catalog.ProductNotFound):
        coadds.load("d12-A_50_34_i_5rot", build="20261010")


def test_build_dir_is_under_the_products_root(data_root):
    d = coadds.build_dir("d12-A_50_34_i_5rot", "20261010")
    assert d == data_root / "products" / "coadds" / "d12-A_50_34_i_5rot" / "20261010"
    assert not d.exists()        # a path to write into, not a resolved build


# ---- the manifest, with TWO upstreams ------------------------------------------

def test_manifest_records_both_upstreams(data_root):
    d = _write_build(data_root)
    got = catalog.manifest("coadds", "d12-A_50_34_i_5rot", build="20261010")
    assert got["inputs"] == {"fam_tables": "danish_1_2@20260920",
                             "miw": "d12-A_50_34_i_5rot@20261010"}
    assert got["status"] == "complete"
    assert got["product"] == "coadds"


def test_manifest_refuses_to_record_only_one_upstream(data_root):
    d = data_root / "products" / "coadds" / "d12-A_50_34_i_5rot" / "20261011"
    d.mkdir(parents=True)
    with pytest.raises(ValueError, match="both a fam_tables build and a miw build"):
        write_manifest.write("d12-A_50_34_i_5rot", build="20261011", build_dir=d,
                             fam_build="20260920")
    with pytest.raises(ValueError, match="both a fam_tables build and a miw build"):
        write_manifest.write("d12-A_50_34_i_5rot", build="20261011", build_dir=d,
                             miw_build="20261010")


def test_manifest_config_resolves_the_fam_variant(data_root):
    _write_build(data_root)
    config = catalog.manifest("coadds", "d12-A_50_34_i_5rot",
                              build="20261010")["config"]
    assert config["variant"] == "d12-A_50_34_i_5rot"
    assert config["miw_variant"] == "d12-A_50_34_i_5rot"
    assert config["fam_variant"] == "danish_1_2"
    assert config["ext_terms"] == "secondary"


def test_manifest_counts_every_file(data_root):
    _write_build(data_root)
    files = catalog.manifest("coadds", "d12-A_50_34_i_5rot",
                             build="20261010")["files"]
    for name in ("blocks_summary.parquet", "coadd_metrics.parquet",
                 "block_grids.npz", "block_grids_ext.npz", *coadds.PLOTS):
        assert name in files, name
    assert files["blocks_summary.parquet"]["rows"] == 3


def test_manifest_needs_a_build_or_a_build_dir():
    with pytest.raises(ValueError, match="needs `build` or `build_dir`"):
        write_manifest.write("d12-A_50_34_i_5rot")


def test_write_manifest_cli(data_root, capsys):
    d = data_root / "products" / "coadds" / "d12-A_50_34_i_5rot" / "20261012"
    d.mkdir(parents=True)
    pd.DataFrame({"block": [0]}).to_parquet(d / "blocks_summary.parquet")
    rc = write_manifest.main(["--variant", "d12-A_50_34_i_5rot", "--build", "20261012",
                              "--fam-build", "20260920", "--miw-build", "20261010",
                              "--option", "ext_terms=secondary"])
    assert rc == 0
    got = catalog.manifest("coadds", "d12-A_50_34_i_5rot", build="20261012")
    assert got["config"]["build_options"] == {"ext_terms": "secondary"}


# ---- the npz comparison --------------------------------------------------------

def test_compare_npz_identical(data_root, tmp_path):
    a = _write_build(data_root, build="20261010")
    b = _write_build(data_root, build="20261011")
    rep = compare_builds.compare_npz(a / "block_grids.npz", b / "block_grids.npz")
    assert rep["differing"] == []
    assert rep["n_arrays"] == 5
    assert rep["only_a"] == [] and rep["only_b"] == []
    assert rep["aligned"] is True


def test_compare_npz_catches_a_value_difference(data_root):
    a = _write_build(data_root, build="20261010")
    b = _write_build(data_root, build="20261011", scale=1.0001)
    rep = compare_builds.compare_npz(a / "block_grids.npz", b / "block_grids.npz")
    names = {n for n, _ in rep["differing"]}
    assert "grids" in names and "miw" in names
    assert all(mx is not None and mx > 0 for n, mx in rep["differing"])


def test_compare_npz_tolerance_is_zero_by_default(data_root):
    """A difference of one part in 1e12 is a difference, not a rounding artefact."""
    a = _write_build(data_root, build="20261010")
    b = _write_build(data_root, build="20261011", scale=1 + 1e-12)
    assert compare_builds.compare_npz(a / "block_grids.npz",
                                      b / "block_grids.npz")["differing"]


def test_compare_npz_aligns_on_block(data_root):
    """A different block ORDER is not a data difference once the rows are aligned."""
    a = _write_build(data_root, build="20261010", block_order=[0, 1, 2])
    b = _write_build(data_root, build="20261011", block_order=[2, 0, 1])
    rep = compare_builds.compare_npz(a / "block_grids.npz", b / "block_grids.npz")
    assert rep["aligned"] is False          # the stored order differs
    # `block` itself differs, but every per-block array realigns to match.
    assert {n for n, _ in rep["differing"]} == {"block"}


def test_compare_npz_reports_different_block_sets(data_root):
    a = _write_build(data_root, build="20261010", n_blocks=3)
    b = _write_build(data_root, build="20261011", n_blocks=2)
    rep = compare_builds.compare_npz(a / "block_grids.npz", b / "block_grids.npz")
    assert rep["aligned"] is False
    assert {n for n, _ in rep["differing"]} >= {"block", "grids", "miw"}


def test_compare_npz_reports_an_array_only_one_side_has(data_root, tmp_path):
    a = _write_build(data_root, build="20261010")
    extra = tmp_path / "extra.npz"
    with np.load(a / "block_grids.npz", allow_pickle=True) as d:
        arrays = {k: d[k] for k in d.files}
    arrays["surprise"] = np.arange(3)
    np.savez(extra, **arrays)
    rep = compare_builds.compare_npz(a / "block_grids.npz", extra)
    assert rep["only_b"] == ["surprise"]


def test_compare_builds_cli_on_identical_builds(data_root, capsys):
    a = _write_build(data_root, build="20261010")
    b = _write_build(data_root, build="20261011")
    rc = compare_builds.main([str(a), str(b)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "builds match" in out
    assert "values: identical" in out


def test_compare_builds_cli_fails_on_a_difference(data_root, capsys):
    a = _write_build(data_root, build="20261010")
    b = _write_build(data_root, build="20261011", scale=2.0)
    rc = compare_builds.main([str(a), str(b)])
    assert rc == 1
    assert "DIFFERENCES FOUND" in capsys.readouterr().out


def test_block_key_is_the_comparison_key():
    assert coadds.BLOCK_KEY == ("block",)
    assert compare_builds.TABLE_KEYS["blocks_summary"] == ["block"]
    assert compare_builds.TABLE_KEYS["coadd_metrics"] == ["block"]


# ---- the shim at the old path ---------------------------------------------------

@pytest.mark.skipif(not ((REPO_ROOT / "aos/code/coadd/run_coadd_blocks_miw.py").exists()), reason='shim removed')
def test_the_old_path_shim_gives_the_same_module():
    """aos/moresnippets.ipynb and two study scripts import the builder by bare name."""
    sys.path.insert(0, str(REPO_ROOT / "aos" / "code" / "coadd"))
    import run_coadd_blocks_miw as shim
    from rubinwork.products.coadds.builders import run_coadd_blocks_miw as real
    assert shim is real
    for name in ("assign_blocks", "block_summary", "main", "THERMAL_VARS"):
        assert getattr(shim, name) is getattr(real, name), name
