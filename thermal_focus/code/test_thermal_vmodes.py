"""Tests for the all-v-mode thermal study.

Run with ``pytest thermal_focus/code/test_thermal_vmodes.py``.

The two that matter are `test_sign_matches_v1_convention`, which pins the response against the
published v-mode-1 convention, and `test_only_the_planted_mode_is_called_thermal`, which checks
the screening rule on a frame where the answer is known by construction.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))

import thermal_focus_lib as L
import thermal_vmodes as TV

FEATURES = ['truss_temp_mean_c', 'm1m3_z_gradient_c_per_m', 'm1m3_y_gradient_c_per_m',
            'm1m3_radial_gradient_c_per_m', 'm1m3_x_gradient_c_per_m']


def _frame(thermal_mode=3, slope=-0.30, n_nights=20, n_visits=40, seed=0):
    """Synthetic per-visit frame with one mode given a truss-temperature slope.

    `slope` is in dimensionless v-mode amplitude per °C, applied to the optical state; the
    stored ``v*_olr`` column is its negative.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for night in range(20260101, 20260101 + n_nights):
        truss = 5.0 + 6.0 * rng.random()
        for s in range(n_visits):
            t = truss + 0.1 * rng.standard_normal()
            r = dict(day_obs=night, seq_num=s, truss_temp_mean_c=t)
            for c in FEATURES[1:]:
                r[c] = 0.1 * rng.standard_normal()
            for k in range(1, TV.N_MODES + 1):
                state = (slope * t if k == thermal_mode else 0.0) + rng.standard_normal()
                r[f'v{k}_olr'] = -state
            rows.append(r)
    return pd.DataFrame(rows)


def test_sign_matches_v1_convention():
    """The response is the optical state, the negative of the stored open-loop column.

    The stub rows satisfy the stored identity ``v_olr = v_meas - v_trim``, which holds on the
    real table to 6.7e-16 (dimensionless v-mode amplitude); a frame violating it could not
    occur and would make the comparison below meaningless.
    """
    v1 = np.array([2.0, 3.0])
    v1_trim = np.array([2.25, 1.5])
    df = pd.DataFrame({'v1': v1, 'v1_trim': v1_trim, 'v1_olr': v1 - v1_trim})
    out = TV.attach_mode_response(df, 1)
    np.testing.assert_allclose(out['y'], [0.25, -1.5])
    # Same quantity thermal_focus_lib forms as v1_trim + MEASURED_SIGN * v1.
    np.testing.assert_allclose(out['y'], df['v1_trim'] + L.MEASURED_SIGN * df['v1'])


def test_attach_rejects_a_mode_the_variant_lacks():
    df = pd.DataFrame({'v1_olr': [0.0]})
    with pytest.raises(KeyError, match='v13_olr'):
        TV.attach_mode_response(df, 13)


def test_only_the_planted_mode_is_called_thermal():
    """A frame with one thermal mode yields exactly that mode above the FDR cut."""
    tab = TV.mode_table(_frame(thermal_mode=3), features=FEATURES, verbose=False)
    called = sorted(tab.loc[tab['thermal'], 'mode'].tolist())
    assert called == [3], f'expected only v3, got {called}'
    assert tab.loc[tab['mode'] == 3, 'skill'].iloc[0] > 0.05


def test_pure_noise_calls_nothing_thermal():
    """With no planted signal the screening rule must not manufacture a detection."""
    df = _frame(thermal_mode=0, seed=7)       # mode 0 never matches, so no mode is thermal
    tab = TV.mode_table(df, features=FEATURES, verbose=False)
    assert int(tab['thermal'].sum()) == 0


def test_null_is_median_based_not_mean_based():
    """The intercept-only null uses the median, so a one-sided tail does not inflate it."""
    rng = np.random.default_rng(3)
    y = rng.standard_normal(600)
    y[:30] += 50.0                             # one-sided outlier population
    df = pd.DataFrame({'y': y, 'day_obs': np.repeat(np.arange(20260101, 20260111), 60)})
    n0 = TV.null_nmad(df)
    assert n0 < 2.0, f'median-based null should resist the tail, got {n0:.3f}'


def test_noise_floor_separates_structured_from_flat_modes():
    """Modes given between-night offsets show a high between/within ratio; others do not."""
    rng = np.random.default_rng(1)
    rows = []
    for night in range(20260101, 20260131):
        off = {k: (3.0 * rng.standard_normal() if k <= 6 else 0.0)
               for k in range(1, TV.N_MODES + 1)}
        for s in range(40):
            r = dict(day_obs=night, seq_num=s)
            for k in range(1, TV.N_MODES + 1):
                r[f'v{k}_olr'] = off[k] + rng.standard_normal()
            rows.append(r)
    tab = TV.noise_floor_table(pd.DataFrame(rows), verbose=False)
    structured = sorted(tab.loc[tab['signal_to_noise'] > 2.0, 'mode'].tolist())
    assert structured == [1, 2, 3, 4, 5, 6], f'got {structured}'


def test_mode_table_rejects_missing_features():
    """A feature the frame does not carry fails loudly rather than being dropped."""
    with pytest.raises(KeyError, match='cam_AverageTemp'):
        TV.mode_table(_frame(), features=['truss_temp_mean_c', 'cam_AverageTemp'],
                      verbose=False)


def test_response_columns_are_in_mode_order():
    cols = TV.response_columns(34)
    assert cols[0] == 'v1_olr' and cols[-1] == 'v34_olr' and len(cols) == 34
