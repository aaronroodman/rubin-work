import pytest

from rubinwork.products.giant_donuts import reader


def test_variants_resolve():
    for v in reader.variants():
        cfg = reader.variant_config(v)
        assert {"mask", "fit_noll", "fwhm_arcsec", "step_mm", "import"} <= set(cfg)
        assert all({"night", "run", "fit_dir"} <= set(s) for s in cfg["import"])


def test_unknown_variant():
    with pytest.raises(KeyError):
        reader.variant_config("no_such_variant")


def test_mask_name_matches_pupil_study_convention():
    # As pass6_common.mask_name_s at (detector, camera, M2) offsets in mm.
    assert (reader.mask_name("science", "extra")
            == "RubinObsc_v4.02s_r_det+0.0_cam+4.0_m2+4.0.yaml")
    assert (reader.mask_name("SW1", "intra")
            == "RubinObsc_v4.02s_r_det-1.5_cam-4.0_m2-4.0.yaml")
