"""Tests for the optical-state builder's handling of nights with no wavefront.

Both cases here were real defects found in the first full-span batch run, where the
pre-20251102 era -- which has exposures in ConsDB but no corner-WFS quicklook -- either
crashed the build or filled it with rows carrying no information.

Run directly (``python code/test_build_optical_state.py``) or under pytest.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[0]))

import build_optical_state as bos

ZK_NOLL = [4, 5, 6]


def test_intrinsic_skips_filters_ofc_does_not_know():
    """An engineering filter must yield NaN, not a RuntimeError.

    ``OTHER:PINHOLE`` and the literal ``'none'`` both reach this function from ConsDB.
    ``get_intrinsic_zernikes`` raises on either, which crashed four shards per variant.
    """
    bands = ['r', 'OTHER:PINHOLE', 'none', '', None, 'i']
    rots = np.zeros(len(bands))
    out = bos.intrinsic_batoid(bands, rots, ZK_NOLL)

    assert out.shape == (len(bands), 4 * len(ZK_NOLL))
    assert np.isfinite(out[0]).all(), 'r band should have an intrinsic'
    assert np.isfinite(out[5]).all(), 'i band should have an intrinsic'
    for i, band in enumerate(bands):
        if i not in (0, 5):
            assert np.isnan(out[i]).all(), f'band {band!r} should give NaN'


def test_intrinsic_skips_nonfinite_rotator_angle():
    out = bos.intrinsic_batoid(['r', 'r'], [0.0, np.nan], ZK_NOLL)
    assert np.isfinite(out[0]).all()
    assert np.isnan(out[1]).all()


def _patch(monkey, meta, opd_complete):
    """Stub ConsDB access so recover_night runs without a database.

    ``opd_complete`` chooses whether the corner Zernikes come back usable.
    """
    n = len(meta)
    n_z = len(ZK_NOLL)
    dev = (np.zeros((n, 4 * n_z)) if opd_complete
           else np.full((n, 4 * n_z), np.nan))
    n_opd = n if opd_complete else 0
    monkey(bos, 'visit_metadata', lambda cdb, day_obs: meta)
    monkey(bos, 'measured_deviation',
           lambda *a, **k: (dev, n_opd, dev.copy()))


def test_night_with_no_complete_corner_opd_writes_no_rows(monkeypatch):
    """The pre-20251102 era must record empty, not all-NaN rows.

    Writing rows whose recovered and open-loop columns are entirely NaN puts 18,733
    uninformative visits in the table, where a query that forgets to screen them gets
    NaN and no error.
    """
    meta = pd.DataFrame({'visit_id': [2025051100001, 2025051100002],
                         'band': ['r', 'r'],
                         'rotator_angle_deg': [0.0, 0.0],
                         'altitude_deg': [70.0, 70.0],
                         'img_type': ['science', 'science']})
    _patch(monkeypatch.setattr, meta, opd_complete=False)

    res = bos.recover_night(None, 20250511, None, n_modes=12,
                            intrinsic_route='batoid', zk_noll=ZK_NOLL, verbose=False)

    assert len(res['visit_ids']) == 0, 'no rows for a night with no wavefront'
    assert res['n_opd'] == 0
    assert res['v_modes'].shape == (0, 12)
    assert res['dof'].shape == (0, 50)
    assert res['fwhm_cwfs_arcsec'].size == 0


def test_night_with_complete_corner_opd_still_recovers(monkeypatch):
    """The n_opd == 0 shortcut must not swallow a night that does have data."""
    meta = pd.DataFrame({'visit_id': [2026031800001, 2026031800002],
                         'band': ['r', 'r'],
                         'rotator_angle_deg': [0.0, 59.5],
                         'altitude_deg': [70.0, 40.0],
                         'img_type': ['science', 'science']})
    _patch(monkeypatch.setattr, meta, opd_complete=True)

    n_modes = 12
    recover = lambda z: (np.zeros(50), np.zeros(n_modes), np.zeros_like(z), 0.0)
    res = bos.recover_night(None, 20260318, None, n_modes=n_modes,
                            intrinsic_route='batoid', zk_noll=ZK_NOLL,
                            verbose=False, recover=recover)

    assert len(res['visit_ids']) == 2
    assert res['n_opd'] == 2
    assert res['ok'].all()
    # Pointing travels with the visit, in the row order the recovery used: the look-up-table
    # study reads these against the open-loop DOF, so a transposition would be silent.
    assert res['elevation_deg'].tolist() == [70.0, 40.0]
    assert res['rotator_angle_deg'].tolist() == [0.0, 59.5]


def test_hexapod_tilt_range_is_deg_not_arcsec():
    """The four hexapod tilt DOF are deg, which `ofc_svd.DOF_UNITS_50` calls arcsec.

    Stored `dof` and `dof_olr` inherit whatever unit the shipped normalization weights
    use, and that is deg: the M2 hexapod rx/ry range is 0.12 deg (432 arcsec), not
    0.12 arcsec. Anything compared against the bounce-test tables, which follow the
    arcsec convention, needs these four entries scaled by 3600 arcsec/deg.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'aos' / 'code'))
    sys.path.insert(0, str(root / 'smatrix' / 'code'))
    import aos_state
    import regularized_inversion as RI

    se = aos_state.make_state_estimator(dof_set='all_50', n_modes=34)
    r = np.asarray(RI.dof_range_vector(aos_state.CornerSvdShim(se)), float)

    assert r.shape == (50,)
    # M2 hexapod rx, ry then camera hexapod rx, ry.
    np.testing.assert_allclose(r[[3, 4]], 0.12, rtol=1e-6)
    np.testing.assert_allclose(r[[8, 9]], 0.24, rtol=1e-6)
    # The decentres are µm in both conventions, and are the scale that makes the tilt
    # entries unambiguous: a 0.12 arcsec tilt range alongside a 6700 µm decentre range
    # would be physically absurd.
    np.testing.assert_allclose(r[[1, 2]], 6700.0, rtol=1e-6)


def test_commanded_projector_leaves_lut_tilts_in_deg():
    """The hexapod LUT must reach the v-mode basis in deg, unscaled.

    This function used to multiply the four tilt entries by 3600 arcsec/deg, inflating the
    projected ``v_modes_lut`` norm by about 1082x. v1 hid the bug, moving only 1.9%, since
    v1 is almost pure defocus.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'aos' / 'code'))
    import aos_state

    se = aos_state.make_state_estimator(dof_set='all_50', n_modes=34)
    n_modes = 34
    lut = {f'lut_dof{k}': [0.0] for k in range(10)}
    lut['lut_dof3'] = [0.01]          # deg of M2 hexapod rx, a typical LUT tilt
    row = dict(visit_id=[2026031800001], **lut,
               **{f'dof{k}': [0.0] for k in range(50)})

    class _Con:
        """Stand in for the DuckDB connection the projector queries."""
        def execute(self, sql):
            class _R:
                def df(_self):
                    return pd.DataFrame(row)
            return _R()

    project = bos.make_commanded_projector(se, n_modes)
    v_lut, _, _ = project(_Con(), np.array([2026031800001], 'int64'))

    want = aos_state.vmodes_from_dofs(
        np.array([[0.0, 0.0, 0.0, 0.01] + [0.0] * 46]), se, n_modes=n_modes)
    np.testing.assert_allclose(v_lut, want, rtol=1e-12, atol=0,
                               err_msg='LUT tilt was rescaled on its way to the basis')


def _main():
    class _M:
        """Minimal monkeypatch stand-in so this runs without pytest."""
        def __init__(self):
            self._undo = []

        def setattr(self, obj, name, value):
            self._undo.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def undo(self):
            for obj, name, old in reversed(self._undo):
                setattr(obj, name, old)

    tests = [test_intrinsic_skips_filters_ofc_does_not_know,
             test_intrinsic_skips_nonfinite_rotator_angle,
             test_night_with_no_complete_corner_opd_writes_no_rows,
             test_night_with_complete_corner_opd_still_recovers,
             test_hexapod_tilt_range_is_deg_not_arcsec,
             test_commanded_projector_leaves_lut_tilts_in_deg]
    bad = 0
    for t in tests:
        m = _M()
        try:
            t(m) if 'monkeypatch' in t.__code__.co_varnames else t()
            print(f'  ok   {t.__name__}')
        except Exception as exc:
            bad += 1
            print(f'  FAIL {t.__name__}: {type(exc).__name__}: {exc}')
        finally:
            m.undo()
    print(f'\n{len(tests) - bad} of {len(tests)} passed')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(_main())
