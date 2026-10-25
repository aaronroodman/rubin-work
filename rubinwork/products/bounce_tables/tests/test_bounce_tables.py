"""Tests for the ``bounce_tables`` product: its variant definitions, the reader, the
one-upstream manifest, the build comparison, and the library split the move created.

The `load` tests point ``RUBINWORK_DATA`` at a pytest ``tmp_path``, so nothing reads the
real data root on S3DF. The ``variants.yaml`` tests read the repository's own config, to
check that each variant's MIW settings entry still exists and that the derived
``(param_set, mi_name)`` pair still resolves through the generated ``mi_config.yaml``.

Three things here that no earlier product's tests cover:

- The **live ``bounce:`` override** in ``aos/analysis_config.yaml`` still resolves to the
  five-point BLOCK-T720 elevation sweep. A build that silently fell back to
  ``DEFAULT_BOUNCES`` would measure one leg instead of five and still look healthy.
- The **library split**: ``aos/bounce/code/bounce_lib.py`` re-exports every name the
  reduction moved out under, and `rubinwork.bounce` imports nothing from `aos/`.
- The ``bounce_kj_stats`` key needs ``night``: without it only 756 of 1,890 rows are
  distinct, so a key-less comparison would align the wrong rows.
"""

import pathlib
import subprocess
import sys

import pandas as pd
import pytest
import yaml


from rubinwork.bounce import legs as bl  # noqa: E402
from rubinwork.products import bounce_tables, catalog, miw  # noqa: E402
from rubinwork.products.bounce_tables import compare_builds  # noqa: E402
from rubinwork.products.bounce_tables import reader as bt_reader  # noqa: E402
from rubinwork.products.bounce_tables import write_manifest  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

EXPECTED_VARIANTS = {"d12-A_50_34_i_5rot_july", "d12-A_50_34_i_5rot",
                     "d12-A_50_34_i", "d13t-A_50_34_i_5rot"}
"""The four registered variants: the three jobs aos/Snakefile plans, plus `_july`.

`_july` is the only one with output in the old tree (13 files) and is its own variant
rather than a build of the plain one, Aaron's call 2026-10-11, because its input is a
different fit table.
"""

ALLOWED_KEYS = {"miw_variant", "fits", "dir_name", "registered"}
"""Every key a variants.yaml entry may carry. Guards against restating builder knobs."""


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


def _write_build(root, variant="d12-A_50_34_i_5rot_july", build="20261012",
                 manifest=True, scale=1.0, skip=()):
    """Write a fake build: the four tables, in the shapes the reference build has."""
    d = root / "products" / bt_reader.PRODUCT / variant / build
    d.mkdir(parents=True)

    # night is a STRING here, with a literal 'all' pooled row, as on the real table.
    kj_rows = [(b, n, c, k, j)
               for b in ("T720_elevation",)
               for n in ("all", "20260420")
               for c in ("Elev=40", "Elev=60")
               for k in (1, 2) for j in (4, 5)]
    if "bounce_kj_stats" not in skip:
        pd.DataFrame({
            "bounce": [r[0] for r in kj_rows],
            "night": [r[1] for r in kj_rows],
            "comparison": [r[2] for r in kj_rows],
            "k": [r[3] for r in kj_rows],
            "j": [r[4] for r in kj_rows],
            "delta": [0.01 * (r[3] + r[4]) * scale for r in kj_rows],
            "err": [0.001 * scale] * len(kj_rows),
        }).to_parquet(d / "bounce_kj_stats.parquet")

    dof_rows = [(b, c, n, kind, i)
                for b in ("T720_elevation",)
                for c in ("Elev=40",)
                for n in ("all", "20260420")
                for kind in ("dof", "vmode")
                for i in range(3)]
    if "bounce_dof_stats" not in skip:
        pd.DataFrame({
            "bounce": [r[0] for r in dof_rows],
            "comparison": [r[1] for r in dof_rows],
            "night": [r[2] for r in dof_rows],
            "kind": [r[3] for r in dof_rows],
            "index": [r[4] for r in dof_rows],
            "unit": ["um" if r[3] == "dof" else "dimensionless" for r in dof_rows],
            "delta": [float(r[4]) * scale for r in dof_rows],
            "delta_err": [0.1 * scale] * len(dof_rows),
            "delta_rbr": [float(r[4]) * 0.9 * scale for r in dof_rows],
            "delta_rbr_err": [0.1 * scale] * len(dof_rows),
        }).to_parquet(d / "bounce_dof_stats.parquet")

    if "bounce_fwhm_metric" not in skip:
        pd.DataFrame({
            "bounce": ["T720_elevation", "T724_rotator"],
            "comparison": ["Elev=40", "Rot=60"],
            "n_pairs": [22, 31],
            "fwhm_before": [0.298630 * scale, 0.208190 * scale],
            "fwhm_after_50_34": [0.038143 * scale, 0.019094 * scale],
        }).to_parquet(d / "bounce_fwhm_metric.parquet")

    # night is an INT64 here: this table has per-night rows only, no pooled row.
    if "bounce_fwhm_vs_bvalue" not in skip:
        pd.DataFrame({
            "bounce": ["T720_elevation"] * 2,
            "comparison": ["Elev=40"] * 2,
            "night": [20260420, 20260513],
            "b_value": [40.0, 40.0],
            "fwhm_before": [0.3 * scale, 0.29 * scale],
            "fwhm_after_default": [0.038 * scale, 0.037 * scale],
        }).to_parquet(d / "bounce_fwhm_vs_bvalue.parquet")

    if manifest:
        write_manifest.write(variant, build=build, build_dir=d, miw_build="20261007")
        # So `load(variant)` with no explicit build resolves, which is the normal route.
        catalog.set_current(bt_reader.PRODUCT, variant, build)
    return d


# ---- variants ------------------------------------------------------------------

def test_variants_are_the_registered_four():
    assert set(bounce_tables.variants()) == EXPECTED_VARIANTS


def test_variant_names_follow_the_rules():
    for name in bounce_tables.variants(registered_only=False):
        assert len(name) <= 32, f"{name} is {len(name)} characters, over the limit of 32"
        assert name == name.lower() or any(c.isupper() for c in name), name


def test_every_variant_names_a_real_miw_variant():
    known = set(miw.variants(registered_only=False))
    for name in bounce_tables.variants(registered_only=False):
        assert bounce_tables.miw_variant(name) in known, name


def test_variant_name_carries_its_miw_variant():
    """The name is <miw variant>[_<fits suffix>], so a reader can trace the upstream."""
    for name in bounce_tables.variants(registered_only=False):
        assert name.startswith(bounce_tables.miw_variant(name)), name


def test_fam_variant_follows_from_the_miw_variant():
    for name in bounce_tables.variants(registered_only=False):
        assert bounce_tables.fam_variant(name) == miw.fam_variant(
            bounce_tables.miw_variant(name))


def test_mi_keys_resolve_through_the_generated_mi_config():
    """Each variant's (param_set, mi_name) is a pair mi_config.yaml actually defines."""
    for name in bounce_tables.variants(registered_only=False):
        param_set, mi_name = bounce_tables.mi_keys(name)
        assert param_set and mi_name, name
        assert bounce_tables.mi_keys(name) == miw.mi_keys(
            bounce_tables.miw_variant(name))


def test_the_july_variant_reads_a_different_fits_table():
    """What makes `_july` its own variant rather than a build of the plain one."""
    assert bounce_tables.fits_name("d12-A_50_34_i_5rot_july") == "fits_july.parquet"
    assert bounce_tables.fits_name("d12-A_50_34_i_5rot") == "fits.parquet"
    assert (bounce_tables.miw_variant("d12-A_50_34_i_5rot_july")
            == bounce_tables.miw_variant("d12-A_50_34_i_5rot"))


def test_every_variant_names_its_fits_table():
    for name in bounce_tables.variants(registered_only=False):
        fits = bounce_tables.fits_name(name)
        assert fits.endswith(".parquet"), name
        assert "/" not in fits, f"{name}: fits must be a file name, not a path"


def test_dir_name_is_a_string_not_a_yaml_integer():
    """YAML reads an unquoted 50_34 as the integer 5034: an underscore is a digit
    separator. Every dir_name must come back as a str."""
    for name in bounce_tables.variants(registered_only=False):
        assert isinstance(bounce_tables.dir_name(name), str), name


def test_dir_names_are_unique():
    names = [bounce_tables.dir_name(v)
             for v in bounce_tables.variants(registered_only=False)]
    assert len(names) == len(set(names))


def test_dir_names_match_the_old_tree_layout():
    """<fam dir_name>_<miw dir_name>, plus the fits suffix where there is one.

    The same construction dof_lut's dir_names follow, since both products were built per
    (param_set, mi_name) pair into the same old tree.
    """
    for name in bounce_tables.variants(registered_only=False):
        fam = bounce_tables.fam_variant(name)
        expected = f"{fam}_{miw.dir_name(bounce_tables.miw_variant(name))}"
        got = bounce_tables.dir_name(name)
        assert got == expected or got == f"{expected}_july", \
            f"{name}: {got} is neither {expected} nor {expected}_july"


def test_variants_yaml_has_no_unknown_keys():
    for name in bounce_tables.variants(registered_only=False):
        extra = set(bounce_tables.variant_config(name)) - ALLOWED_KEYS
        assert not extra, f"{name} carries unexpected keys {sorted(extra)}"


def test_no_build_settings_are_restated_here():
    """The bounce legs and mode counts stay in analysis_config.yaml.

    Restating them would be a second place to edit, and the live override is keyed on
    the long param_set rather than on a variant name.
    """
    knobs = {"bounces", "n_dof", "n_keep", "rbr_kappa", "rbr_power", "oic_rho",
             "n_keep_reduced", "min_detectors", "fit_prefix", "focal_k_range"}
    for name in bounce_tables.variants(registered_only=False):
        overlap = set(bounce_tables.variant_config(name)) & knobs
        assert not overlap, f"{name} restates builder knobs {sorted(overlap)}"


def test_unknown_variant_raises_and_lists_the_defined_ones():
    with pytest.raises(KeyError) as e:
        bounce_tables.variant_config("not-a-variant")
    assert "d12-A_50_34_i_5rot_july" in str(e.value)


# ---- the live analysis_config override -----------------------------------------

def test_the_live_bounce_override_still_resolves_the_five_point_sweep():
    """The BLOCK-T720 legs come from analysis_config.yaml, not from DEFAULT_BOUNCES.

    The override at aos/analysis_config.yaml redefines T720 as a five-point elevation
    sweep. A build that failed to resolve it would measure one leg instead of five and
    produce a short but healthy-looking table, so this is checked against the file rather
    than against the code default.
    """
    param_set, mi_name = bounce_tables.mi_keys("d12-A_50_34_i_5rot_july")
    cfg = bl.resolve_config(
        param_set, mi_name,
        analysis_config=REPO_ROOT / "aos" / "analysis_config.yaml")
    t720 = [b for b in cfg["bounces"] if b["name"] == "T720_elevation"]
    assert t720, "the resolved config defines no T720_elevation bounce"
    n_legs = len(t720[0]["comparisons"])
    assert n_legs == 5, (
        f"the resolved T720_elevation bounce has {n_legs} comparison legs "
        f"(count, dimensionless), expected 5; DEFAULT_BOUNCES has 1, so a build that "
        f"silently fell back to the code default would look like this")
    # The override also replaces pupil_j_range, because the phase-1 fit table carries
    # no nollIndices column.
    assert len(cfg["pupil_j_range"]) == 21


def test_default_bounces_is_only_the_fallback():
    """The library default is one T720 leg, which is why the override matters."""
    t720 = [b for b in bl.DEFAULT_BOUNCES if b["name"] == "T720_elevation"]
    assert len(t720) == 1
    assert len(t720[0]["comparisons"]) == 1


# ---- the library split ---------------------------------------------------------

def test_bounce_lib_reexports_every_moved_name():
    """aos/bounce/code/bounce_lib.py keeps working for its existing callers."""
    sys.path.insert(0, str(REPO_ROOT / "aos" / "code"))
    sys.path.insert(0, str(REPO_ROOT / "aos" / "bounce" / "code"))
    import bounce_lib

    for name in ("run_bounce", "form_pairs", "paired_delta", "paired_deltas_kj",
                 "paired_deltas_matrix", "rbr_module", "rbr_deltas", "solver_deltas",
                 "filter_visits", "bounce_program_mask", "leg_b_value", "leg_pointing",
                 "to_long_df", "stats_per_kj", "resolve_config",
                 "DEFAULT", "DEFAULT_BOUNCES", "DOF22", "CAM_HEX_DOF"):
        assert hasattr(bounce_lib, name), f"bounce_lib no longer exports {name}"
        assert getattr(bl, name, None) is not None


def test_the_library_imports_nothing_from_a_topic():
    """A library never reaches into a study (rule 2 of the plan's section 2).

    Checked on the IMPORT statements, not on the prose: the docstrings name
    ``aos/bounce/code/bounce_lib.py`` deliberately, to say where the plotters went.
    `aos_fwhm` is reached lazily inside `rubinwork.bounce.analysis`, by path insert, and
    is a library itself.
    """
    for name in ("legs.py", "__init__.py", "analysis.py"):
        text = (REPO_ROOT / "rubinwork" / "bounce" / name).read_text()
        imports = [ln.strip() for ln in text.splitlines()
                   if ln.strip().startswith(("import ", "from "))]
        for line in imports:
            for banned in ("bounce_lib", "run_bounce", "cwfs_lut", "cwfs_lut_lib"):
                assert banned not in line, f"{name} imports {banned}: {line}"
        # No module-level sys.path insert of a topic either: analysis.py's reach into
        # aos/code for aos_fwhm is inside a function, deliberately and documented.
        for line in text.splitlines():
            if line.startswith("sys.path"):
                raise AssertionError(f"{name} has a module-level path insert: {line}")


def test_the_four_smatrix_imports_no_longer_reach_into_aos_bounce():
    """The wrong-way imports the plan's section 2 lists as needing removal."""
    for name in ("run_oic_compare.py", "run_regularized_compare.py"):
        text = (REPO_ROOT / "smatrix" / "regularized_inversion" / "code"
                / name).read_text()
        assert "import bounce_lib" not in text, f"{name} still imports bounce_lib"
        assert "import run_bounce" not in text, f"{name} still imports run_bounce"
        assert "rubinwork.bounce" in text, f"{name} does not use the library"


def test_rbr_module_resolves_to_the_installed_package():
    """The sys.path insert into smatrix/code is gone."""
    mod = bl.rbr_module()
    assert mod.__name__ == "rubinwork.smatrix.regularized_inversion"
    for fn in ("invert_truncated", "invert_oic", "invert_range_penalty",
               "oic_authority", "achieved_residual", "dof_range_vector"):
        assert hasattr(mod, fn), fn


def test_resolve_config_layers_the_overrides_over_default():
    """And the three command-line overrides land on the right keys."""
    cfg = bl.resolve_config(
        "fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x", "pathA_50_34_i_5rot",
        analysis_config=REPO_ROOT / "aos" / "analysis_config.yaml",
        rbr_kappa=7.5, rbr_power=2, no_rbr=True)
    assert cfg["rbr_kappa"] == 7.5
    assert cfg["rbr_power"] == 2
    assert cfg["rbr_enable"] is False
    # Untouched DEFAULT keys still come through.
    assert cfg["n_dof"] == 50 and cfg["n_keep"] == 34


def test_the_analysis_returns_every_field_both_halves_need():
    """The contract between the builder and the study half."""
    from rubinwork.bounce.analysis import BounceAnalysis

    fields = set(BounceAnalysis.__dataclass_fields__)
    # What the builder's write_tables reads.
    for name in ("fit_table", "bounce_results", "df_kj", "fwhm_rows",
                 "fwhm_bvalue_rows", "DOF_all", "dof_ranges", "svd_ok"):
        assert name in fields, name
    # The per-visit arrays the three per-ordinal PDF pages need, which is why the study
    # calls the analysis instead of reading the product tables back.
    for name in ("C_all", "DOF5_all", "C5_all", "DOFSUM_all"):
        assert name in fields, name


# ---- the reader ----------------------------------------------------------------

def test_load_returns_the_four_tables_in_order(data_root):
    _write_build(data_root)
    kj, dof, fwhm, bval = bounce_tables.load("d12-A_50_34_i_5rot_july")
    assert list(kj.columns)[:5] == ["bounce", "night", "comparison", "k", "j"]
    assert set(dof["kind"]) == {"dof", "vmode"}
    assert len(fwhm) == 2
    assert len(bval) == 2


def test_load_one_table(data_root):
    _write_build(data_root)
    (fwhm,) = bounce_tables.load("d12-A_50_34_i_5rot_july",
                                 tables=("bounce_fwhm_metric",))
    assert len(fwhm) == 2


def test_load_rejects_a_table_that_is_not_one(data_root):
    _write_build(data_root)
    with pytest.raises(ValueError, match="not bounce_tables tables"):
        bounce_tables.load("d12-A_50_34_i_5rot_july", tables=("lut",))


def test_load_table_rejects_a_table_that_is_not_one(data_root):
    _write_build(data_root)
    with pytest.raises(ValueError, match="is not a bounce_tables table"):
        bounce_tables.load_table("lut", "d12-A_50_34_i_5rot_july")


def test_a_build_without_a_manifest_is_invisible(data_root):
    _write_build(data_root, build="20261012", manifest=False)
    with pytest.raises(catalog.ProductNotFound):
        bounce_tables.load("d12-A_50_34_i_5rot_july", build="20261012")


def test_build_dir_is_under_the_products_root(data_root):
    d = bounce_tables.build_dir("d12-A_50_34_i_5rot_july", "20261012")
    assert d == (data_root / "products" / "bounce_tables"
                 / "d12-A_50_34_i_5rot_july" / "20261012")
    assert not d.exists()


# ---- the manifest --------------------------------------------------------------

def test_manifest_records_the_one_upstream(data_root):
    d = _write_build(data_root)
    doc = yaml.safe_load((d / "manifest.json").read_text())
    assert doc["inputs"] == {"miw": "d12-A_50_34_i_5rot@20261007"}


def test_manifest_does_not_claim_fam_tables_as_an_input(data_root):
    """The build reads the MIW build's fit table, not the FAM tables.

    The opposite assignment from dof_lut, which reads the plain FAM tables. Recording
    fam_tables here would claim a dependency the build does not have.
    """
    d = _write_build(data_root)
    doc = yaml.safe_load((d / "manifest.json").read_text())
    assert "fam_tables" not in doc["inputs"]
    assert doc["config"]["fam_variant"] == "danish_1_2"


def test_manifest_config_records_the_derived_keys(data_root):
    d = _write_build(data_root)
    cfg = yaml.safe_load((d / "manifest.json").read_text())["config"]
    assert cfg["variant"] == "d12-A_50_34_i_5rot_july"
    assert cfg["miw_variant"] == "d12-A_50_34_i_5rot"
    assert cfg["fits"] == "fits_july.parquet"
    assert cfg["param_set"] == "fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x"
    assert cfg["mi_name"] == "pathA_50_34_i_5rot"


def test_manifest_records_the_min_detectors_option(data_root):
    """The quality cut is a build option, not a variant property: the same variant can
    be built with a different one."""
    d = data_root / "products" / "bounce_tables" / "d12-A_50_34_i_5rot_july" / "b"
    _write_build(data_root, build="b0", manifest=False)
    src = data_root / "products" / "bounce_tables" / "d12-A_50_34_i_5rot_july" / "b0"
    src.rename(d)
    write_manifest.write("d12-A_50_34_i_5rot_july", build="b", build_dir=d,
                         miw_build="20261007", build_options={"min_detectors": "160"})
    cfg = yaml.safe_load((d / "manifest.json").read_text())["config"]
    assert cfg["build_options"] == {"min_detectors": "160"}


def test_manifest_needs_a_build_or_a_build_dir():
    with pytest.raises(ValueError, match="needs `build` or `build_dir`"):
        write_manifest.write("d12-A_50_34_i_5rot_july")


def test_manifest_counts_every_file(data_root):
    d = _write_build(data_root)
    files = yaml.safe_load((d / "manifest.json").read_text())["files"]
    for table in bounce_tables.TABLES:
        assert f"{table}.parquet" in files, table
        assert files[f"{table}.parquet"]["rows"] > 0


# ---- the comparison ------------------------------------------------------------

def test_compare_finds_identical_builds_identical(data_root, capsys):
    a = _write_build(data_root, build="20261012", manifest=False)
    b = _write_build(data_root, build="20261013", manifest=False)
    assert compare_builds.main([str(a), str(b)]) == 0
    assert "builds match" in capsys.readouterr().out


def test_compare_catches_a_value_difference(data_root):
    a = _write_build(data_root, build="20261012", manifest=False)
    b = _write_build(data_root, build="20261013", manifest=False, scale=1.01)
    assert compare_builds.main([str(a), str(b)]) == 1


def test_compare_catches_a_missing_table(data_root):
    a = _write_build(data_root, build="20261012", manifest=False)
    b = _write_build(data_root, build="20261013", manifest=False,
                     skip=("bounce_fwhm_metric",))
    assert compare_builds.main([str(a), str(b)]) == 1


def test_compare_keys_come_from_the_reader(data_root):
    assert compare_builds.TABLE_KEYS == dict(bounce_tables.TABLE_KEYS)


def test_the_kj_key_needs_night(data_root):
    """Without `night` only 756 of 1,890 real rows are distinct, so a key-less
    comparison would align the wrong rows against each other."""
    a = _write_build(data_root, build="20261012", manifest=False)
    path = a / "bounce_kj_stats.parquet"
    with_night, n_u, n_r = compare_builds.key_is_unique(
        path, list(bounce_tables.TABLE_KEYS["bounce_kj_stats"]))
    assert with_night, f"{n_u} unique of {n_r} rows (counts, dimensionless)"
    without, n_u2, _ = compare_builds.key_is_unique(
        path, ["bounce", "comparison", "k", "j"])
    assert not without, "dropping night should NOT leave the rows distinct"
    assert n_u2 < n_u


def test_compare_treats_a_pdf_absent_from_both_as_agreement(data_root, capsys):
    """Seven of the nine PDFs are conditional on what the run finds."""
    a = _write_build(data_root, build="20261012", manifest=False)
    b = _write_build(data_root, build="20261013", manifest=False)
    assert compare_builds.main([str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "absent from both" in out
    assert "0 of 9 PDFs present on both sides" in out


def test_compare_catches_a_pdf_on_only_one_side(data_root):
    a = _write_build(data_root, build="20261012", manifest=False)
    b = _write_build(data_root, build="20261013", manifest=False)
    (a / "bounce_summary.pdf").write_bytes(b"%PDF-1.4\n")
    assert compare_builds.main([str(a), str(b)]) == 1


# ---- the study readers ---------------------------------------------------------

def test_the_cross_study_reader_resolves_old_tree_then_catalog(data_root):
    """aos/cwfs_lut/code/bounce_compare.py is the one reader outside the bounce study.

    It hardwired aos/output/bounce/<run>/ and now resolves the same way the aos/Snakefile
    resolver does.
    """
    sys.path.insert(0, str(REPO_ROOT / "aos" / "cwfs_lut" / "code"))
    import bounce_compare

    _write_build(data_root)
    # An empty repo root, so the old tree misses and the catalog is reached.
    path = bounce_compare._bounce_stats_path(
        "danish_1_2_A_50_34_i_5rot_july", root=data_root / "no_such_repo")
    assert path is not None and path.name == "bounce_dof_stats.parquet"
    assert "products/bounce_tables" in str(path)


def test_the_cross_study_reader_maps_a_run_to_its_variant():
    sys.path.insert(0, str(REPO_ROOT / "aos" / "cwfs_lut" / "code"))
    import bounce_compare

    assert (bounce_compare._bounce_variant("danish_1_2_A_50_34_i_5rot_july")
            == "d12-A_50_34_i_5rot_july")
    assert bounce_compare._bounce_variant("not-a-run") is None
    # The study's own default is the one variant with data.
    assert bounce_compare.DEFAULT_BOUNCE_RUN == bounce_tables.dir_name(
        "d12-A_50_34_i_5rot_july")


# ---- the Snakefiles ------------------------------------------------------------

def test_the_product_snakefile_declares_all_four_tables():
    text = (REPO_ROOT / "rubinwork" / "products" / "bounce_tables"
            / "Snakefile").read_text()
    for table in bounce_tables.TABLES:
        assert f"{table}.parquet" in text, table


def test_the_aos_rule_survives_as_the_plotting_half():
    """The first moved product whose aos/ rule is not removed."""
    text = (REPO_ROOT / "aos" / "Snakefile").read_text()
    assert "rule bounce:" in text
    assert "bounce/output/{d}/bounce_summary.pdf" in text
    # No bounce TABLE is an input, output or `rule all` target any more; the comment
    # explaining the move still names bounce_kj_stats.parquet, so check code lines only.
    code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    for line in code:
        for table in bounce_tables.TABLES:
            assert f"{table}.parquet" not in line, (
                f"aos/Snakefile still names the bounce table {table}: {line.strip()}")


def test_the_builder_module_runs_its_help():
    """A smoke test that the builder imports and its parser is well formed."""
    out = subprocess.run(
        [sys.executable, "-m",
         "rubinwork.products.bounce_tables.builders.build_bounce_tables", "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "--min-detectors" in out.stdout
    assert "--variant" in out.stdout
