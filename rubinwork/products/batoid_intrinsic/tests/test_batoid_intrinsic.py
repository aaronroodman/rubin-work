import numpy as np
import pandas as pd

from rubinwork.donut_sim import optics
from rubinwork.products.batoid_intrinsic import reader
from rubinwork.products.batoid_intrinsic.builders.build import dz_fits


def test_variants_resolve():
    for v in reader.variants():
        cfg = reader.variant_config(v)
        assert {"model", "obscuration", "bands", "jmax", "configs", "grids"} <= set(cfg)
        for conf in cfg["configs"].values():
            assert len(conf["optics"]) == len(conf["offsets_mm"])
            assert conf["grid"] in cfg["grids"]


def test_science_grid():
    pts = optics.science_grid(rings=3, spokes=4, radius_deg=1.75)
    assert len(pts) == 12
    r = np.hypot([p[1] for p in pts], [p[2] for p in pts])
    assert r.max() < 1.75 and r.min() > 0


def test_dz_fit_round_trip():
    rng = np.random.default_rng(2)
    jmax, kmax, radius = 8, 6, 1.75
    pts = optics.science_grid(rings=5, spokes=9, radius_deg=radius)
    thx = np.array([p[1] for p in pts])
    thy = np.array([p[2] for p in pts])
    coef = np.zeros((kmax + 1, jmax + 1))
    coef[1:, 4:] = rng.normal(size=(kmax, jmax - 3))
    z = reader.evaluate_dz(coef, thx, thy, radius)
    df = pd.DataFrame(z[:, 4:], columns=reader.noll_columns(jmax))
    df["thx_deg"], df["thy_deg"], df["grid"] = thx, thy, "science"
    df["band"], df["kind"], df["config"], df["side"] = "r", "ta", "fam", "extra"
    cfg = {"jmax": jmax, "dz_kmax": kmax, "grids": {"science": {"radius_deg": radius}}}
    (h,) = dz_fits(df, cfg)
    np.testing.assert_allclose(h.data, coef, atol=1e-10)
    assert h.header["RESMAX"] < 1e-10
