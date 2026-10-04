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
                         'rotator_angle_deg': [0.0, 0.0],
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
             test_night_with_complete_corner_opd_still_recovers]
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
