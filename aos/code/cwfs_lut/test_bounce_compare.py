"""Tests for the bounce-test comparison.

Run with ``pytest aos/code/cwfs_lut/test_bounce_compare.py``.

Three guard the failure modes that would be silent. `test_the_unit_column_is_not_trusted` pins
that the bounce file's wrong ``arcsec`` label cannot reach a comparison;
`test_vmode_index_is_zero_based_against_one_based_columns` pins the off-by-one that would rotate
the whole v-mode comparison by one mode; and `test_bounce_slope_sign_follows_the_throw_direction`
pins the sign, which matters because the elevation legs include four negative throws and one
positive.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[2]))

import bounce_compare as B                                       # noqa: E402


def _bounce_frame(kind='dof', n_entries=50, bounce='T724_rotator', throw=60.0,
                  deltas=None, with_rbr=True, unit='arcsec'):
    """Synthetic bounce stats table for one leg.

    `unit` defaults to the wrong ``arcsec`` label the real file carries, so the tests see what
    production sees.
    """
    deltas = (np.arange(n_entries, dtype=float) + 1.0) if deltas is None else np.asarray(deltas)
    if bounce == 'T724_rotator':
        geom = dict(elevation_deg=70.0, ref_elevation_deg=70.0,
                    rot_angle_deg=throw, ref_rot_angle_deg=0.0)
    else:
        geom = dict(elevation_deg=70.0 + throw, ref_elevation_deg=70.0,
                    rot_angle_deg=-0.5, ref_rot_angle_deg=-0.5)
    rows = []
    for i in range(n_entries):
        rows.append(dict(bounce=bounce, comparison='leg', reference='ref', block='B',
                         night='all', day_obs='all', n_visits=31, n_pairs=31,
                         kind=kind, index=i, unit=unit,
                         delta=float(deltas[i]), delta_err=1.0,
                         delta_rbr=float(deltas[i]) if with_rbr else np.nan,
                         delta_rbr_err=1.0 if with_rbr else np.nan,
                         significance=float(deltas[i]) / 1.0,
                         dof_range=1.0, ratio_to_range=0.1, ratio_to_range_rbr=0.1,
                         **geom))
    return pd.DataFrame(rows)


def _survey_frame(n=1200, slope_per_deg=4.0, dof_index=1, vmode_index=0, seed=0):
    """Synthetic per-visit frame with a planted slope on one DOF and one v-mode."""
    rng = np.random.default_rng(seed)
    base = {
        'day_obs': np.repeat(20260401 + np.arange(n // 60), 60)[:n],
        'elevation_deg': rng.uniform(20.0, 85.0, n),
        'rotator_angle_deg': rng.uniform(-80.0, 80.0, n),
    }
    rot = base['rotator_angle_deg']
    # Built in one dict and concatenated once: 168 successive inserts would make pandas warn
    # about fragmentation on every call and bury the test output.
    cols = {}
    for j in range(50):
        y = rng.normal(0.0, 1.0, n)
        if j == dof_index:
            y = y + slope_per_deg * rot
        cols[f'dof{j}'] = y
        cols[f'dof{j}_olr'] = y
    for k in range(1, 35):
        y = rng.normal(0.0, 0.01, n)
        if k - 1 == vmode_index:
            y = y + 1e-3 * rot
        cols[f'v{k}'] = y
        cols[f'v{k}_olr'] = y
    return pd.DataFrame({**base, **cols})


def test_missing_bounce_file_raises_with_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match='bounce_dof_stats'):
        B.load_bounce_stats(run='no_such_run', root=tmp_path)


def test_bounce_slope_divides_by_the_throw():
    stats = _bounce_frame(deltas=np.full(50, 120.0), throw=60.0)
    tab = B.bounce_slope(stats, bounce='T724_rotator', kind='dof', arm='svd')
    assert np.allclose(tab['slope'], 2.0)
    assert np.allclose(tab['slope_err'], 1.0 / 60.0)
    assert np.allclose(tab['throw_deg'], 60.0)


def test_bounce_slope_sign_follows_the_throw_direction():
    """A negative throw flips the slope. The elevation legs are mostly negative throws."""
    stats = _bounce_frame(deltas=np.full(50, 80.0), bounce='T720_elevation', throw=-40.0)
    tab = B.bounce_slope(stats, bounce='T720_elevation', kind='dof', arm='svd')
    assert np.allclose(tab['slope'], -2.0)
    # The error is a magnitude and must stay positive regardless of the throw's sign.
    assert (tab['slope_err'] > 0).all()


def test_the_unit_column_is_not_trusted():
    """The file's wrong ``arcsec`` label must not survive into a comparison."""
    stats = _bounce_frame(unit='arcsec')
    assert 'arcsec' in set(stats['unit'])
    tab = B.bounce_slope(stats.drop(columns=['unit']), bounce='T724_rotator', kind='dof',
                         arm='svd').set_index('index')
    for j in B.C.HEX_TILT_DOF:
        assert tab.loc[j, 'unit'] == 'deg'
    assert tab.loc[1, 'unit'] == 'µm'


def test_load_bounce_stats_drops_the_unit_column(tmp_path):
    run = 'fake_run'
    d = tmp_path / 'aos' / 'output' / 'bounce' / run
    d.mkdir(parents=True)
    _bounce_frame().to_parquet(d / 'bounce_dof_stats.parquet', index=False)
    stats = B.load_bounce_stats(run=run, root=tmp_path)
    assert 'unit' not in stats.columns
    assert len(stats) == 50


def test_rbr_arm_rejected_where_it_is_all_nan():
    """RBR is solved in DOF space only, so a v-mode RBR request must raise, not return NaN."""
    stats = _bounce_frame(kind='vmode', n_entries=34, with_rbr=False)
    with pytest.raises(ValueError, match='delta_rbr'):
        B.bounce_slope(stats, bounce='T724_rotator', kind='vmode', arm='rbr')


def test_unknown_bounce_arm_raises():
    with pytest.raises(ValueError, match='unknown bounce arm'):
        B.bounce_slope(_bounce_frame(), bounce='T724_rotator', kind='dof', arm='nope')


def test_vmode_index_is_zero_based_against_one_based_columns():
    """Bounce ``index`` 0 is ``v1``; a one-off here rotates the whole comparison silently."""
    df = _survey_frame(vmode_index=0)
    tab = B.survey_slopes(df, 'rotator_angle_deg', arm='deviation', kind='vmode')
    by = tab.set_index('index')
    assert by.loc[0, 'label'] == 'v1'
    assert by.loc[33, 'label'] == 'v34'
    # The plant is on v1, which is index 0, and must be the strongest entry.
    assert abs(by.loc[0, 'slope']) > 10 * abs(by.loc[1, 'slope'])
    assert tab['unit'].eq('dimensionless').all()


def test_survey_slopes_recovers_the_planted_dof_slope():
    df = _survey_frame(slope_per_deg=4.0, dof_index=1)
    tab = B.survey_slopes(df, 'rotator_angle_deg', arm='deviation', kind='dof')
    row = tab.set_index('index').loc[1]
    assert abs(row['slope'] - 4.0) < 0.05
    assert row['label'] == 'M2 hexapod dx'


def test_survey_arms_reach_different_columns():
    """The deviation and open-loop arms must read `dofN` and `dofN_olr`, not the same column."""
    df = _survey_frame()
    df['dof1_olr'] = df['dof1'] * 2.0
    dev = B.survey_slopes(df, 'rotator_angle_deg', arm='deviation', kind='dof')
    olr = B.survey_slopes(df, 'rotator_angle_deg', arm='open_loop', kind='dof')
    s_dev = dev.set_index('index').loc[1, 'slope']
    s_olr = olr.set_index('index').loc[1, 'slope']
    assert abs(s_olr / s_dev - 2.0) < 0.01


def test_lateral_sum_pairs_m2_with_camera_on_the_same_axis():
    """A transposed pairing would give a plausible-looking wrong number."""
    assert B.LATERAL_SUM_AXES == (('dz', 0, 5), ('dx', 1, 6), ('dy', 2, 7))
    paired = [j for _a, m, c in B.LATERAL_SUM_AXES for j in (m, c)]
    assert not set(paired) & set(B.C.HEX_TILT_DOF)


def test_lateral_sum_is_invariant_to_the_hexapod_split():
    """Moving amplitude from M2 dx to camera dx leaves the summed slope unchanged.

    This is the degeneracy the lateral-sum comparison exists to be robust against.
    """
    df = _survey_frame(slope_per_deg=4.0, dof_index=1)
    moved = df.copy()
    moved['dof6'] = df['dof6'] + df['dof1']
    moved['dof1'] = 0.0
    x = df['rotator_angle_deg'].to_numpy(float)
    a = B.C.huber_trend(x, (df['dof1'] + df['dof6']).to_numpy(float))['slope']
    b = B.C.huber_trend(x, (moved['dof1'] + moved['dof6']).to_numpy(float))['slope']
    assert abs(a - b) < 1e-6


def test_cosine_similarity_is_one_for_identical_vectors():
    v = np.array([1.0, -2.0, 3.0, 0.5])
    assert abs(B.cosine_similarity(v, v) - 1.0) < 1e-12
    # Scale-invariant: the same direction at half the amplitude still scores +1.
    assert abs(B.cosine_similarity(v, 0.5 * v) - 1.0) < 1e-12


def test_cosine_similarity_is_minus_one_for_a_flipped_vector():
    v = np.array([1.0, -2.0, 3.0, 0.5])
    assert abs(B.cosine_similarity(v, -v) + 1.0) < 1e-12


def test_cosine_similarity_is_nan_without_length():
    assert not np.isfinite(B.cosine_similarity(np.zeros(4), np.ones(4)))


def test_compare_per_dof_difference_sigma_uses_both_errors():
    bounce = pd.DataFrame([dict(index=0, label='M2 hexapod dz', unit='µm', slope=10.0,
                                slope_err=3.0, significance=10.0)])
    survey = pd.DataFrame([dict(index=0, label='M2 hexapod dz', unit='µm', slope=20.0,
                                slope_err=4.0)])
    cmp = B.compare_per_dof(bounce, survey).iloc[0]
    assert abs(cmp['difference'] - 10.0) < 1e-12
    # Quadrature sum of 3 and 4 is 5, so 10 / 5 = 2.
    assert abs(cmp['difference_sigma'] - 2.0) < 1e-12
    assert abs(cmp['ratio'] - 2.0) < 1e-12


def test_subspace_screens_on_bounce_significance():
    """An insignificant bounce term must not move the cosine similarity."""
    idx = list(range(4))
    bounce = pd.DataFrame([dict(index=i, label=f'e{i}', unit='µm', slope=1.0, slope_err=1.0,
                                significance=10.0 if i < 3 else 0.5) for i in idx])
    survey = pd.DataFrame([dict(index=i, label=f'e{i}', unit='µm',
                                slope=1.0 if i < 3 else -500.0, slope_err=1.0) for i in idx])
    agr = B.compare_subspaces(bounce, survey, subspaces=(('test', tuple(idx)),),
                              min_significance=3.0).iloc[0]
    assert agr['n_terms'] == 4
    assert agr['n_significant'] == 3
    assert abs(agr['cosine_similarity'] - 1.0) < 1e-12


def test_elevation_legs_return_one_row_per_leg():
    """Per-leg points, deliberately not one pooled slope."""
    throws = [-40.0, -30.0, -20.0, -10.0, 5.0]
    parts = []
    for t in throws:
        f = _bounce_frame(bounce='T720_elevation', throw=t,
                          deltas=np.full(50, 2.0 * t))
        f['comparison'] = f'Elev={70 + t:.0f}'
        parts.append(f)
    stats = pd.concat(parts, ignore_index=True).drop(columns=['unit'])
    legs, fits = B.compare_elevation_legs(stats, _survey_frame(), dof_indices=(1, 2))
    assert len(legs) == len(throws) * 2
    assert set(fits['index']) == {1, 2}
    # Deltas were planted exactly linear in throw, so the fit must recover the slope and the
    # chi2/dof must be near zero -- the guard that the weighting and sign are right.
    row = fits.set_index('index').loc[1]
    assert abs(row['slope_linear'] - 2.0) < 1e-6
    assert row['chi2_linear'] < 1e-6
    assert row['dof_linear'] == len(throws) - 1


def test_elevation_fit_weights_by_the_per_leg_error():
    """A leg with a large error must not drag the slope the way an unweighted fit would."""
    throws = [-40.0, -30.0, -20.0, -10.0, 5.0]
    parts = []
    for t in throws:
        f = _bounce_frame(bounce='T720_elevation', throw=t, deltas=np.full(50, 2.0 * t))
        f['comparison'] = f'Elev={70 + t:.0f}'
        if t == -20.0:
            f['delta'] = 500.0                    # a wild leg
            f['delta_err'] = 1e6                  # that says so
        parts.append(f)
    stats = pd.concat(parts, ignore_index=True).drop(columns=['unit'])
    _legs, fits = B.compare_elevation_legs(stats, _survey_frame(), dof_indices=(1,))
    assert abs(fits.iloc[0]['slope_linear'] - 2.0) < 1e-3


def test_bounce_legs_reports_the_cross_throw():
    """The rotator leg pins elevation, which is what makes it the clean test."""
    legs = B.bounce_legs(_bounce_frame().drop(columns=['unit']), bounce='T724_rotator')
    assert len(legs) == 1
    assert abs(legs.iloc[0]['throw_deg'] - 60.0) < 1e-9
    assert abs(legs.iloc[0]['cross_throw_deg']) < 1e-9


def test_compare_all_rejects_a_mismatched_angle():
    """T724 throws the rotator, so asking for elevation must raise rather than fit noise."""
    stats = _bounce_frame().drop(columns=['unit'])
    with pytest.raises(ValueError, match='throws'):
        B.compare_all(stats, {'v': _survey_frame()}, 'elevation_deg',
                      bounce='T724_rotator', verbose=False)
