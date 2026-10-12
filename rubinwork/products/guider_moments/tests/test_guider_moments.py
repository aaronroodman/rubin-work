"""Tests for the ``guider_moments`` product: the night-partition layout, the shims, the
catalog route, the scratch-versus-catalog manifest rule, and the build comparison.

Two things matter most. First, the **variant is the schema version**: ``v6`` has to equal
``guiderMoments.SCHEMA_VERSION`` or a night would be filed under a schema it was not
written with, and nothing else would notice. Second the **shims assert module identity**,
not just importability: the atmosphere study imports ``guiderMoments`` by bare name and two
study runners import ``guiderEdgeRecovery`` the same way, and three scripts in two topics
import the HSM estimator, where separate copies would silently unmatch the
guider-vs-science-CCD moment comparison.

Nothing here reads the real data root or queries the Butler.
"""

import json
import pathlib
import subprocess
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

from rubinwork.products import catalog, guider_moments  # noqa: E402
from rubinwork.products.guider_moments import compare_builds  # noqa: E402
from rubinwork.products.guider_moments import reader as gm_reader  # noqa: E402
from rubinwork.products.guider_moments import write_manifest  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

SHIMMED_BUILDERS = ("run_guider_moments", "combine_moments", "list_guider_exposures")
"""Builders that moved into ``builders/`` and left a shim in ``guider/code/``."""

HSM_IMPORTERS = (
    "rubinwork/products/guider_moments/builders/run_guider_moments.py",
    "guider/code/extract_adjacent_psf_moments.py",
    "optatmo/code/extract_psf_moments.py",
)
"""Every code file that uses the shared galsim-HSM estimator."""

NIGHT = 20260706


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A temporary data root, with ``products/`` and ``studies/`` in place."""
    monkeypatch.setenv("RUBINWORK_DATA", str(tmp_path))
    (tmp_path / "products").mkdir()
    (tmp_path / "studies").mkdir()
    return tmp_path


def _write_night(directory, night=NIGHT, n_rows=4):
    """Write a night's three tables with the real keys and a few rows."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    exp_id = night * 100000 + 44
    detectors = [f"R0{i}_SG0" for i in range(n_rows)]
    pd.DataFrame({"expId": [exp_id] * n_rows, "dayObs": [night] * n_rows,
                  "detector": detectors,
                  "Mxx_coadd": [0.1 * i for i in range(n_rows)]}).to_parquet(
        directory / f"guider_moments_{night}.parquet", index=False)
    pd.DataFrame({"expid": [exp_id] * n_rows, "detector": detectors,
                  "trackid": list(range(n_rows)),
                  "fwhm": [1.0 + 0.1 * i for i in range(n_rows)]}).to_parquet(
        directory / f"guider_stars_{night}.parquet", index=False)
    pd.DataFrame({"expId": [exp_id], "n_guiders": [n_rows]}).to_parquet(
        directory / f"guider_metrics_{night}.parquet", index=False)
    (directory / gm_reader.SKIPPED_FILE).write_text("no skipped visits\n")
    (directory / gm_reader.DETECTOR_COUNTS_FILE.format(night=night)).write_text(
        f"{n_rows}guider=1\n")
    return directory


# ---- the variant IS the schema version ----------------------------------------

def test_variant_matches_the_schema_version():
    """``v6`` and ``guiderMoments.SCHEMA_VERSION`` cannot drift."""
    from rubinwork.products.guider_moments.guiderMoments import SCHEMA_VERSION

    assert gm_reader.PRODUCT_VARIANT == f"v{SCHEMA_VERSION}"
    assert gm_reader.schema_version() == SCHEMA_VERSION


def test_every_defined_variant_declares_its_schema_version():
    for variant in gm_reader.variants():
        assert gm_reader.variant_config(variant)["schema_version"] == int(variant[1:])


def test_unknown_variant_raises():
    with pytest.raises(KeyError):
        gm_reader.variant_config("v999")


def test_variant_config_carries_the_pipeline_settings():
    """The reference build is issued with these, not with argparse defaults."""
    cfg = gm_reader.variant_config()
    assert cfg["butler"]["collections"] == ["LSSTCam/raw/guider", "LSSTCam/raw/all"]
    assert cfg["butler"]["view"] == "dvcs"
    assert cfg["guider_hz"] == 5.0
    assert cfg["tracker"]["min_snr"] == 10.0
    assert cfg["tracker"]["residual_scale"] == 10.0
    assert cfg["moments"]["ap_nsigma"] == 4.0
    assert cfg["moments"]["cen_niter"] == 3


def test_variants_yaml_agrees_with_guider_snake_config():
    """The product's settings mirror the pipeline config the Snakefile reads.

    Not generated from it -- the product owns its own definition -- but a divergence here
    would mean the product and the study pipeline run different tracker settings.
    """
    import yaml

    with open(REPO_ROOT / "guider" / "snake_config.yaml") as handle:
        snake = yaml.safe_load(handle)
    cfg = gm_reader.variant_config()
    assert cfg["butler"]["collections"] == snake["butler"]["collections"]
    assert cfg["butler"]["view"] == snake["butler"]["view"]
    assert cfg["butler"]["image_types"] == snake["image_types"]
    assert cfg["guider_hz"] == snake["guider_hz"]
    for key, value in cfg["tracker"].items():
        assert snake["tracker"][key] == value, key
    for key, value in cfg["moments"].items():
        assert snake["moments"][key] == value, key


# ---- the night-partition layout ------------------------------------------------

def test_night_dir_is_a_partition_not_a_dated_build(data_root):
    directory = gm_reader.night_dir(NIGHT)
    assert directory == (data_root / "products" / "guider_moments" / "v6"
                         / "nights" / str(NIGHT))


def test_nights_is_empty_without_any_build(data_root):
    assert gm_reader.nights() == []


def test_nights_lists_only_complete_nights(data_root):
    complete = _write_night(gm_reader.night_dir(NIGHT))
    write_manifest.write(NIGHT, night_root=complete)
    _write_night(gm_reader.night_dir(20260707), night=20260707)  # no manifest
    assert gm_reader.nights() == [NIGHT]


def test_catalog_path_resolves_a_night(data_root):
    directory = _write_night(gm_reader.night_dir(NIGHT))
    write_manifest.write(NIGHT, night_root=directory)
    assert catalog.path("guider_moments", "v6", f"nights/{NIGHT}") == directory.resolve()


def test_catalog_list_products_sees_the_night(data_root):
    """An incremental product's partitions sit one level deeper than a dated build."""
    directory = _write_night(gm_reader.night_dir(NIGHT))
    write_manifest.write(NIGHT, night_root=directory)
    entries = [e for e in catalog.list_products("guider_moments")]
    assert [(e["variant"], e["build"]) for e in entries] == [("v6", f"nights/{NIGHT}")]
    assert entries[0]["status"] == "complete"


def test_load_table_and_load_night(data_root):
    directory = _write_night(gm_reader.night_dir(NIGHT))
    write_manifest.write(NIGHT, night_root=directory)
    mom = guider_moments.load_table("moments", NIGHT)
    assert len(mom) == 4
    tables = guider_moments.load_night(NIGHT)
    assert sorted(tables) == ["metrics", "moments", "stars"]
    assert len(tables["stars"]) == 4


def test_load_table_reads_a_scratch_night_root(tmp_path):
    """A reference build has no manifest, so it is reachable only by path."""
    directory = _write_night(tmp_path / "ref" / str(NIGHT))
    mom = guider_moments.load_table("moments", NIGHT, night_root=directory)
    assert len(mom) == 4


def test_load_table_rejects_an_unknown_table():
    with pytest.raises(ValueError):
        gm_reader._table_file("psfmoments", NIGHT)


# ---- the manifest, and the scratch-versus-catalog rule -------------------------

def test_manifest_records_the_night_and_summit_utils(data_root):
    directory = _write_night(gm_reader.night_dir(NIGHT))
    path = write_manifest.write(NIGHT, night_root=directory)
    man = json.loads(path.read_text())
    assert man["product"] == "guider_moments"
    assert man["variant"] == "v6"
    assert man["build"] == f"nights/{NIGHT}"
    assert man["config"]["day_obs"] == NIGHT
    assert set(man["config"]["summit_utils"]) == {"commit", "branch", "version"}
    assert man["config"]["schema_version"] == 6


def test_manifest_records_no_upstream_product(data_root):
    """The inputs are Butler raw data and summit_utils, both external."""
    directory = _write_night(gm_reader.night_dir(NIGHT))
    man = json.loads(write_manifest.write(NIGHT, night_root=directory).read_text())
    assert not man.get("inputs")


def test_manifest_counts_the_files_on_disk(data_root):
    directory = _write_night(gm_reader.night_dir(NIGHT))
    man = json.loads(write_manifest.write(NIGHT, night_root=directory).read_text())
    assert man["files"][f"guider_moments_{NIGHT}.parquet"]["rows"] == 4


def test_is_catalog_build_is_decided_by_path(data_root, tmp_path):
    assert write_manifest.is_catalog_build(gm_reader.night_dir(NIGHT))
    assert not write_manifest.is_catalog_build(tmp_path / "ref" / str(NIGHT))


def test_a_scratch_night_cannot_be_registered(data_root, tmp_path):
    """The reference build must not be able to write a manifest, even by accident."""
    scratch = _write_night(tmp_path / "ref" / str(NIGHT))
    with pytest.raises(ValueError):
        write_manifest.write(NIGHT, night_root=scratch)
    assert not (scratch / "manifest.json").exists()


def test_an_incomplete_night_is_not_in_the_catalog(data_root):
    directory = _write_night(gm_reader.night_dir(NIGHT))
    write_manifest.write(NIGHT, night_root=directory, status="running")
    with pytest.raises(catalog.ProductNotFound):
        catalog.path("guider_moments", "v6", f"nights/{NIGHT}")
    assert gm_reader.nights() == []


# ---- the comparison ------------------------------------------------------------

def test_compare_night_reports_identical_builds(tmp_path):
    a = _write_night(tmp_path / "a" / str(NIGHT))
    b = _write_night(tmp_path / "b" / str(NIGHT))
    reports = compare_builds.compare_night(a, b, NIGHT)
    assert sorted(reports) == ["metrics", "moments", "stars"]
    for table, rep in reports.items():
        assert rep["differing"] == [], table
        assert rep["unique_a"] is True, table
        assert rep["compared"] > 0, table
    assert compare_builds.main([str(a), str(b), str(NIGHT)]) == 0


def test_compare_night_catches_a_changed_value(tmp_path):
    a = _write_night(tmp_path / "a" / str(NIGHT))
    b = _write_night(tmp_path / "b" / str(NIGHT))
    path = b / f"guider_moments_{NIGHT}.parquet"
    df = pd.read_parquet(path)
    df.loc[0, "Mxx_coadd"] += 1e-9
    df.to_parquet(path, index=False)
    reports = compare_builds.compare_night(a, b, NIGHT)
    assert [c for c, _ in reports["moments"]["differing"]] == ["Mxx_coadd"]
    assert compare_builds.main([str(a), str(b), str(NIGHT)]) == 1


def test_compare_text_checks_both_sidecars(tmp_path):
    a = _write_night(tmp_path / "a" / str(NIGHT))
    b = _write_night(tmp_path / "b" / str(NIGHT))
    assert all(compare_builds.compare_text(a, b, NIGHT).values())
    (b / gm_reader.SKIPPED_FILE).write_text("2026070600044\tmissing\n")
    results = compare_builds.compare_text(a, b, NIGHT)
    assert results[gm_reader.SKIPPED_FILE] is False
    assert results[gm_reader.DETECTOR_COUNTS_FILE.format(night=NIGHT)] is True


def test_table_keys_are_the_verified_ones():
    """Read off the 3a reference build, not guessed: note the expId/expid case split."""
    assert gm_reader.TABLE_KEYS["moments"] == ("expId", "detector")
    assert gm_reader.TABLE_KEYS["stars"] == ("expid", "detector", "trackid")
    assert gm_reader.TABLE_KEYS["metrics"] == ("expId",)


# ---- the shims: module identity, one fresh interpreter each --------------------

def _identity_check(code):
    """Run `code` in a fresh interpreter, so no sibling's path bootstrap interferes."""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(REPO_ROOT))
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout.strip()


@pytest.mark.parametrize("name", SHIMMED_BUILDERS)
def test_builder_shim_gives_the_same_module_object(name):
    assert _identity_check(
        "import sys; sys.path.insert(0, 'guider/code');"
        f"import {name};"
        f"from rubinwork.products.guider_moments.builders import {name} as real;"
        f"print({name} is real)") == "True"


def test_guider_moments_shim_gives_the_same_module_object():
    """The atmosphere study imports three names from it by bare name."""
    assert _identity_check(
        "import sys; sys.path.insert(0, 'guider/code');"
        "import guiderMoments;"
        "from rubinwork.products.guider_moments import guiderMoments as real;"
        "print(guiderMoments is real and "
        "guiderMoments.measureStampMoments is real.measureStampMoments and "
        "guiderMoments._FWHM_PER_SIGMA == real._FWHM_PER_SIGMA)") == "True"


def test_guider_edge_recovery_shim_gives_the_same_module_object():
    """Two study runners import ``makeTrackerConfig`` from it by bare name."""
    assert _identity_check(
        "import sys; sys.path.insert(0, 'guider/code');"
        "import guiderEdgeRecovery;"
        "from rubinwork import guider_edge_recovery as real;"
        "print(guiderEdgeRecovery is real and "
        "guiderEdgeRecovery.makeTrackerConfig is real.makeTrackerConfig)") == "True"


def test_hsm_moments_shim_gives_the_same_module_object():
    """The matched-estimator property depends on one estimator, not two copies."""
    assert _identity_check(
        "import sys; sys.path.insert(0, 'optatmo/code');"
        "import moments_hsm;"
        "from rubinwork import hsm_moments as real;"
        "print(moments_hsm is real and "
        "moments_hsm.measure_hsm_moments is real.measure_hsm_moments)") == "True"


def test_every_hsm_importer_uses_the_library():
    """Asserted on the import lines, not on whole-file text: the prose legitimately
    mentions the old name."""
    for rel in HSM_IMPORTERS:
        lines = [ln.strip() for ln in (REPO_ROOT / rel).read_text().splitlines()
                 if ln.strip().startswith(("import ", "from "))]
        assert any("rubinwork.hsm_moments" in ln for ln in lines), rel
        assert not any(ln.startswith(("import moments_hsm", "from moments_hsm"))
                       for ln in lines), rel


def test_the_builder_has_no_relative_hop_to_another_topic():
    """The trap the move removed: __file__/../../optatmo/code."""
    text = (REPO_ROOT / "rubinwork" / "products" / "guider_moments" / "builders"
            / "run_guider_moments.py").read_text()
    assert "optatmo" not in text


# ---- the Snakefiles ------------------------------------------------------------

def test_product_snakefile_declares_all_five_combine_outputs():
    """``detector_counts_<night>.txt`` was written but undeclared in guider/.

    The night is fixed by ``--config`` rather than being a wildcard, so the paths are
    f-strings; this asserts on the five output keys and the file stems they name.
    """
    text = (REPO_ROOT / "rubinwork" / "products" / "guider_moments"
            / "Snakefile").read_text()
    combine = text.split("rule combine:")[1].split("rule manifest:")[0]
    for key, stem in (("moments=", "guider_moments_"), ("stars=", "guider_stars_"),
                      ("metrics=", "guider_metrics_"), ("skipped=", "skipped_visits.txt"),
                      ("counts=", "detector_counts_")):
        assert key in combine, key
        assert stem in combine, stem


def test_product_snakefile_has_no_night_wildcard():
    """One night per invocation, so ``seqNum`` is the only wildcard.

    A ``{night}`` wildcard made ``rule combine`` raise ``RuleException``: its
    ``skipped_visits.txt`` output and its log carry no wildcard, and snakemake requires
    every output and log of a rule to carry the same ones.
    """
    text = (REPO_ROOT / "rubinwork" / "products" / "guider_moments"
            / "Snakefile").read_text()
    assert "wildcards.night" not in text
    assert "night = r" not in text


def test_guider_snakefile_no_longer_has_the_product_rules():
    """Asserted on the rule declarations, so a comment explaining the move is fine."""
    lines = (REPO_ROOT / "guider" / "Snakefile").read_text().splitlines()
    declared = {ln.split()[1].rstrip(":") for ln in lines
                if ln.startswith(("rule ", "checkpoint "))}
    assert declared == {"all", "psfmoments", "movie", "movies", "rtvplot", "rtvplots",
                        "plots"}


def test_guider_snakefile_resolves_the_product_old_tree_then_catalog():
    text = (REPO_ROOT / "guider" / "Snakefile").read_text()
    assert "def momentsDir" in text
    assert "_catalog.path" in text
