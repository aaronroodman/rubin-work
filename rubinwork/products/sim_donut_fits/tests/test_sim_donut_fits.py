import numpy as np

from rubinwork.donut_sim import fit
from rubinwork.products.sim_donut_fits import reader


def test_variants_resolve():
    for v in reader.variants():
        cfg = reader.variant_config(v)
        assert {"sims", "noll", "binning", "flux_e", "sky_e_per_px", "lstsq"} <= set(cfg)
        assert set(cfg["noll"]) == {"fam", "cwfs", "giant"}
        for noll in cfg["noll"].values():
            assert min(noll) >= 4
        assert isinstance(cfg["flux_e"], float) and isinstance(cfg["sky_e_per_px"], float)


def test_bin_stamp_matches_ts_wep():
    from lsst.ts.wep.utils import binArray
    rng = np.random.default_rng(1)
    img = rng.normal(size=(181, 181))
    ref = binArray(img, 2)
    ref = ref[:-1, :-1] if ref.shape[0] % 2 == 0 else ref
    np.testing.assert_allclose(fit.bin_stamp(img, 2), ref)
    assert fit.bin_stamp(img, 1).shape == (181, 181)


def test_centre_offset():
    # binning 1: Danish's centre is +0.5 px from the middle pixel's centre
    assert fit.centre_offset_px(181, 1) == -0.5
    # 181 -> 180 -> 90 -> 89: chief ray at binned 44.75, Danish centre 44.5
    assert fit.centre_offset_px(181, 2) == 0.25
    # 801 -> 400 -> 399: kernel side 199 is odd, so Danish centres on the middle pixel
    assert fit.centre_offset_px(801, 2) == 0.75
    assert fit.centre_offset_px(801, 1) == -0.5
