import numpy as np

from rubinwork.donut_sim import raytrace
from rubinwork.products.sim_donuts import reader
from rubinwork.products.sim_donuts.builders.build import _point_kept


def test_variants_resolve():
    for v in reader.variants():
        cfg = reader.variant_config(v)
        assert {"intrinsic", "bands", "fwhm_arcsec", "oversample", "configs"} <= set(cfg)
        for conf in cfg["configs"].values():
            assert conf["npix"] % 2 == 1, "Danish needs odd stamps"


def test_point_subset():
    conf = {"rings": [0, 2], "spokes": [0, 5]}
    assert _point_kept("r02_a05", conf)
    assert not _point_kept("r01_a05", conf)
    assert not _point_kept("r02_a06", conf)
    assert _point_kept("r07_a33", {})


def test_stop_grid_annulus():
    x, y = raytrace.stop_grid(0.1, r_inner=1.0, r_outer=2.0)
    r = np.hypot(x, y)
    assert r.min() >= 1.0 and r.max() <= 2.0
    assert abs(len(x) * 0.01 - np.pi * 3.0) < 0.1     # area of the annulus, m²


def test_render_conserves_and_centres():
    rng = np.random.default_rng(4)
    dx, dy = rng.normal(0, 25e-6, (2, 20000)) + np.array([[30e-6], [-20e-6]])
    sharp, lost = raytrace.render(dx, dy, 2.0, 41, 10e-6, 3, 0.0, 10.312)
    assert lost == 0
    np.testing.assert_allclose(sharp.sum(), 2.0 * len(dx), rtol=1e-12)
    img, _ = raytrace.render(dx, dy, 2.0, 41, 10e-6, 3, 1.0, 10.312)
    yy, xx = np.indices(img.shape) - 20
    cx, cy = (img * xx).sum() / img.sum(), (img * yy).sum() / img.sum()
    np.testing.assert_allclose([cx, cy], [3.0, -2.0], atol=0.1)    # pixels
