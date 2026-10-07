"""Tests for the per-channel thermal screen.

Run with ``pytest thermal_focus/code/test_thermal_vmodes_channels.py``.

The two that matter are `test_planted_channel_is_the_leading_one`, which checks the screen
recovers which channel a mode was built to respond to, and
`test_near_duplicate_channels_are_skipped`, which pins the collinearity guard -- without it a
mode's "leading 4" were four thermometers on the same structure and the combined fit lost skill
against the single best channel.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))

import thermal_vmodes_channels as TVC                            # noqa: E402


def _frame(n_nights=24, n_visits=30, planted_mode=2, planted_channel='m1m3_z_gradient_c_per_m',
           slope=0.8, n_modes=4, seed=0):
    """Synthetic frame where one mode responds to one named channel.

    The planted channel varies between nights; a duplicate of it is included so the
    collinearity guard has something to catch.
    """
    rng = np.random.default_rng(seed)
    nights = 20250801 + np.arange(n_nights)
    day_obs = np.repeat(nights, n_visits)
    n = len(day_obs)
    per_night = dict(zip(nights, rng.normal(0.0, 1.0, n_nights)))
    drive = np.array([per_night[d] for d in day_obs]) + rng.normal(0, 0.05, n)

    df = pd.DataFrame({'day_obs': day_obs, 'seq_num': np.arange(n)})
    df[planted_channel] = drive
    # A near-copy of the driver: same signal, tiny independent noise. Any sane duplicate rule
    # must treat this as redundant.
    df[planted_channel.replace('z_gradient', 'y_gradient')] = drive + rng.normal(0, 0.01, n)
    df['truss_temp_mean_c'] = rng.normal(10.0, 2.0, n)
    df['cam_AverageTemp'] = rng.normal(11.0, 2.0, n)
    df['cam_AmbAirtemp'] = rng.normal(9.0, 2.0, n)

    for k in range(1, n_modes + 1):
        y = rng.normal(0.0, 0.2, n)
        if k == planted_mode:
            y = y + slope * drive
        df[f'v{k}_olr'] = -y                       # stored open-loop is the negated state
    return df


def test_attach_differences_adds_only_computable_columns():
    df = _frame()
    out = TVC.attach_differences(df, verbose=False)
    assert 'cam_minus_ambient_c' in out.columns
    # Camera-minus-truss needs both, which the frame has.
    assert 'cam_minus_truss_c' in out.columns
    # The lens asymmetries have no inputs in this frame and must be skipped, not filled.
    assert 'l1_minus_l2_x_c' not in out.columns
    expect = df['cam_AverageTemp'] - df['cam_AmbAirtemp']
    assert np.allclose(out['cam_minus_ambient_c'], expect)


def test_channel_table_covers_base_and_differences():
    tab = TVC.channel_table()
    assert set(tab['kind']) == {'base', 'difference'}
    assert tab['channel'].is_unique
    # Every difference names inputs that are themselves base channels, or the grid would try to
    # fit a column no loader produces.
    base = set(c for c, _l, _u in TVC.BASE_CHANNELS)
    for _name, a, b, _lab, _unit in TVC.DIFF_CHANNELS:
        assert a in base and b in base


def test_single_channel_fit_reports_sign():
    df = _frame(slope=0.8)
    row = TVC.single_channel_fit(df, 2, 'm1m3_z_gradient_c_per_m', n_splits=4)
    # The response is the optical state, so a positive planted slope must come back positive.
    assert row['spearman_rho'] > 0.5
    assert row['slope'] > 0
    assert row['n'] > 0


def test_single_channel_fit_sign_follows_the_plant():
    neg = TVC.single_channel_fit(_frame(slope=-0.8), 2, 'm1m3_z_gradient_c_per_m', n_splits=4)
    assert neg['spearman_rho'] < -0.5
    assert neg['slope'] < 0


def test_planted_channel_is_the_leading_one():
    df = TVC.attach_differences(_frame(), verbose=False)
    grid = TVC.channel_grid(df, n_modes=4, n_splits=4, verbose=False)
    best = TVC.leading_channels(grid, rho_strong=0.4)
    assert len(best), 'the planted mode should clear the threshold'
    top = best.iloc[0]
    assert int(top['mode']) == 2
    # The planted channel or its deliberate near-copy; both are the same physical driver.
    assert top['channel'] in ('m1m3_z_gradient_c_per_m', 'm1m3_y_gradient_c_per_m')


def test_unplanted_modes_do_not_clear_the_threshold():
    df = TVC.attach_differences(_frame(), verbose=False)
    grid = TVC.channel_grid(df, n_modes=4, n_splits=4, verbose=False)
    best = TVC.leading_channels(grid, rho_strong=0.4)
    assert set(best['mode'].astype(int)) == {2}


def test_near_duplicate_channels_are_skipped():
    df = _frame()
    ranked = ['m1m3_z_gradient_c_per_m', 'm1m3_y_gradient_c_per_m', 'truss_temp_mean_c',
              'cam_AverageTemp']
    lead, dropped = TVC.select_lead(df, ranked, n_lead=3, dup_rho=0.9)
    assert 'm1m3_z_gradient_c_per_m' in lead
    assert 'm1m3_y_gradient_c_per_m' not in lead, 'the near-copy must be dropped'
    assert any(c == 'm1m3_y_gradient_c_per_m' for c, _k, _r in dropped)
    assert len(lead) <= 3


def test_select_lead_keeps_independent_channels():
    df = _frame()
    ranked = ['m1m3_z_gradient_c_per_m', 'truss_temp_mean_c', 'cam_AverageTemp']
    lead, dropped = TVC.select_lead(df, ranked, n_lead=3, dup_rho=0.9)
    assert lead == ranked, 'independent channels must all survive'
    assert not dropped


def test_combined_fit_beats_the_null_on_the_planted_mode():
    df = TVC.attach_differences(_frame(), verbose=False)
    grid = TVC.channel_grid(df, n_modes=4, n_splits=4, verbose=False)
    res = TVC.combined_fit(df, 2, grid, n_lead=4, n_splits=4, verbose=False)
    assert res['nmad_combined'] < res['nmad_null']
    assert res['skill_combined'] > 0.3
    assert len(res['residual']) == res['n']
    # The prediction is out of fold, so it must be finite everywhere the response is.
    assert np.isfinite(res['prediction']).sum() == res['n']


def test_nmad_summary_matches_the_fit():
    df = TVC.attach_differences(_frame(), verbose=False)
    grid = TVC.channel_grid(df, n_modes=4, n_splits=4, verbose=False)
    tab, results = TVC.combined_table(df, grid, rho_strong=0.4, n_splits=4, verbose=False)
    summary = TVC.nmad_summary(results)
    assert len(summary) == len(tab)
    r = results[0]
    row = summary[summary['mode'] == r['mode']].iloc[0]
    assert np.isclose(row['nmad_combined'], r['nmad_combined'])


def test_pivot_is_modes_by_channels():
    df = TVC.attach_differences(_frame(), verbose=False)
    grid = TVC.channel_grid(df, n_modes=4, n_splits=4, verbose=False)
    wide = TVC.pivot_rho(grid)
    assert list(wide.index) == [1, 2, 3, 4]
    assert len(wide.columns) == len(grid.attrs['channels'])


def test_modes_carrying_reads_the_content_table():
    # Two Noll terms, three modes: mode 2 holds all the Z11 by construction.
    content = pd.DataFrame([[0.5, 0.1, 0.2], [0.0, 0.9, 0.1]],
                           index=pd.Index([4, 11], name='noll'),
                           columns=['v1', 'v2', 'v3'])
    assert TVC.modes_carrying(content, 11, top=1) == [(2, 0.9)]
    assert TVC.modes_carrying(content, 4, top=1) == [(1, 0.5)]
    assert TVC.modes_carrying(content, 99) == []
