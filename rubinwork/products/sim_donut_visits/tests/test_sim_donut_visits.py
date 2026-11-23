import numpy as np

from rubinwork.donut_sim import observe
from rubinwork.products import sim_donut_visits


def test_variants_resolve():
    for v in sim_donut_visits.variants():
        cfg = sim_donut_visits.variant_config(v)
        assert cfg["n_visits"] > 0
        assert all(isinstance(x, float) for x in cfg["mag_range"] + cfg["sky_mag_arcsec2"])


def test_magnitude_draw_in_range_and_faint_heavy():
    m = observe.draw_mags(np.random.default_rng(1), 20000, 10.5, 15.42, 0.25)
    assert m.min() >= 10.5 and m.max() <= 15.42
    assert abs(np.mean(m > 14.42) - 0.465) < 0.02   # (1 - 10**-0.25) / (1 - 10**-1.23)


def test_sky_dark_r():
    # SMTN-002 dark r sky 21.2 mag/arcsec², 30 s, 0.2 arcsec pixels: about 880 e-.
    assert abs(observe.sky_e_per_px(21.2, 28.36, 30.0, 0.2) - 877) < 5
