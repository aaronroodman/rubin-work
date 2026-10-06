"""Tests for the CWFS look-up-table dependence study.

Run with ``pytest aos/code/cwfs_lut/test_cwfs_lut_lib.py``.

The one that matters most is `test_only_the_four_tilts_convert_to_arcsec`: the stored tilts are
deg and the bounce test's are arcsec, and getting that wrong corrupts only the tilt entries
while leaving the decentres and bending modes correct, so nothing downstream would complain.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[2]))

import cwfs_lut_lib as C


def test_only_the_four_tilts_convert_to_arcsec():
    """DOF 3, 4, 8, 9 scale by 3600 arcsec/deg; the other 46 are untouched."""
    v = np.arange(50, dtype=float) + 1.0
    out = C.to_bounce_units(v)
    tilts = list(C.HEX_TILT_DOF)
    others = [j for j in range(50) if j not in tilts]
    np.testing.assert_allclose(out[tilts], v[tilts] * 3600.0)
    np.testing.assert_allclose(out[others], v[others])


def test_to_bounce_units_does_not_mutate_the_input():
    v = np.ones(50)
    C.to_bounce_units(v)
    np.testing.assert_allclose(v, np.ones(50))


def test_to_bounce_units_handles_a_stack():
    a = np.ones((7, 50))
    out = C.to_bounce_units(a)
    assert out.shape == (7, 50)
    np.testing.assert_allclose(out[:, 3], 3600.0)
    np.testing.assert_allclose(out[:, 0], 1.0)


def test_to_bounce_units_rejects_a_vmode_vector():
    """A 34-element v-mode vector passed here would be silently mangled, so it must raise."""
    with pytest.raises(ValueError, match='50-element'):
        C.to_bounce_units(np.zeros(34))


def test_dof_labels_cover_the_whole_vector():
    """Every index 0 to 49 has a name, and the bending-mode numbering starts at 1."""
    for j in range(50):
        name, unit = C.dof_label(j)
        assert name and unit
    assert C.dof_label(10)[0] == 'M1M3 bending 1'
    assert C.dof_label(29)[0] == 'M1M3 bending 20'
    assert C.dof_label(30)[0] == 'M2 bending 1'
    assert C.dof_label(49)[0] == 'M2 bending 20'


def test_dof_label_reports_arcsec_only_in_bounce_units():
    assert C.dof_label(3)[1] == 'deg'
    assert C.dof_label(3, bounce_units=True)[1] == 'arcsec'
    # A decentre is µm either way.
    assert C.dof_label(1, bounce_units=True)[1] == 'µm'


def test_ofc_dof_ordering_is_m2_then_camera():
    """DOF 0-4 are the M2 hexapod and 5-9 the camera hexapod, not the reverse."""
    assert C.dof_label(0)[0].startswith('M2 hexapod')
    assert C.dof_label(5)[0].startswith('camera hexapod')


def test_huber_trend_recovers_a_planted_slope():
    rng = np.random.default_rng(0)
    x = rng.uniform(20.0, 85.0, 2000)
    y = 3.5 * x - 40.0 + rng.standard_normal(2000)
    res = C.huber_trend(x, y)
    assert abs(res['slope'] - 3.5) < 0.05
    assert abs(res['intercept'] + 40.0) < 2.0
    assert res['n'] == 2000
    assert res['pearson_r'] > 0.99


def test_huber_trend_resists_a_one_sided_tail():
    """Outliers must not drag the slope, which is why this is Huber and not least squares."""
    rng = np.random.default_rng(1)
    x = rng.uniform(20.0, 85.0, 2000)
    y = 2.0 * x + rng.standard_normal(2000)
    y[:120] += 400.0
    assert abs(C.huber_trend(x, y)['slope'] - 2.0) < 0.1


def test_huber_trend_refuses_a_narrow_span():
    """A slope fitted over a few deg of angle extrapolates badly and is not reportable."""
    rng = np.random.default_rng(2)
    x = rng.uniform(69.0, 71.0, 1000)          # 2 deg span, under min_span
    res = C.huber_trend(x, 2.0 * x + rng.standard_normal(1000))
    assert not np.isfinite(res['slope'])
    assert res['span'] < 5.0


def test_huber_trend_refuses_too_few_points():
    rng = np.random.default_rng(3)
    x = rng.uniform(20.0, 85.0, 50)
    assert not np.isfinite(C.huber_trend(x, 2.0 * x)['slope'])


def test_huber_trend_drops_non_finite_pairs():
    rng = np.random.default_rng(4)
    x = rng.uniform(20.0, 85.0, 1000)
    y = 1.5 * x + rng.standard_normal(1000)
    y[:10] = np.nan
    x[990:] = np.inf
    res = C.huber_trend(x, y)
    assert res['n'] == 980
    assert abs(res['slope'] - 1.5) < 0.05


def _frame(slope_per_deg=4.0, n=1200, seed=0):
    """Synthetic visits with a planted elevation slope on DOF 1 and a tilt on DOF 3."""
    rng = np.random.default_rng(seed)
    elev = rng.uniform(20.0, 85.0, n)
    d = {'elevation_deg': elev,
         'rotator_angle_deg': rng.uniform(-80.0, 80.0, n)}
    for j in range(50):
        if j == 1:
            d[f'dof{j}_olr'] = slope_per_deg * elev + rng.standard_normal(n)
        elif j == 3:
            d[f'dof{j}_olr'] = 1e-4 * elev + 1e-5 * rng.standard_normal(n)
        else:
            d[f'dof{j}_olr'] = rng.standard_normal(n)
    return pd.DataFrame(d)


def test_trend_table_recovers_the_planted_dof_slope():
    tab = C.trend_table(_frame(slope_per_deg=4.0), 'elevation_deg', verbose=False)
    row = tab[tab['dof'] == 1].iloc[0]
    assert abs(row['slope'] - 4.0) < 0.05
    assert row['unit'] == 'µm'


def test_trend_table_reports_tilt_slopes_in_arcsec_when_asked():
    """The tilt slope must scale by 3600 arcsec/deg and the decentre slope must not."""
    df = _frame()
    deg = C.trend_table(df, 'elevation_deg', verbose=False)
    arc = C.trend_table(df, 'elevation_deg', bounce_units=True, verbose=False)
    s_deg = deg[deg['dof'] == 3]['slope'].iloc[0]
    s_arc = arc[arc['dof'] == 3]['slope'].iloc[0]
    assert abs(s_arc / s_deg - 3600.0) < 1e-6
    assert arc[arc['dof'] == 3]['unit'].iloc[0] == 'arcsec'
    # DOF 1 is a decentre and is identical in both.
    np.testing.assert_allclose(deg[deg['dof'] == 1]['slope'].iloc[0],
                               arc[arc['dof'] == 1]['slope'].iloc[0])


def test_trend_table_rejects_a_missing_angle_column():
    with pytest.raises(KeyError, match='rotator_angle_deg'):
        C.trend_table(_frame().drop(columns=['rotator_angle_deg']), 'rotator_angle_deg',
                      verbose=False)


def test_intrinsic_spread_differences_are_miw_minus_batoid():
    a = C.trend_table(_frame(slope_per_deg=4.0, seed=1), 'elevation_deg', verbose=False)
    b = C.trend_table(_frame(slope_per_deg=6.0, seed=1), 'elevation_deg', verbose=False)
    cmp = C.intrinsic_spread(a, b, verbose=False)
    row = cmp[cmp['dof'] == 1].iloc[0]
    assert row['slope_diff'] > 0, 'MIW minus batoid, so a larger MIW slope is positive'
    assert abs(row['slope_diff'] - 2.0) < 0.1


def test_olr_columns_are_in_index_order():
    cols = C.olr_columns()
    assert cols[0] == 'dof0_olr' and cols[-1] == 'dof49_olr' and len(cols) == 50


def test_bounce_comparable_dof_excludes_tilts_and_bending():
    """The bounce comparison covers hexapod pistons and decentres only."""
    assert set(C.BOUNCE_COMPARABLE_DOF).isdisjoint(C.HEX_TILT_DOF)
    assert max(C.BOUNCE_COMPARABLE_DOF) < 10
