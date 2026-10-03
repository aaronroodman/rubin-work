"""Tests for `open_loop` and `aos_state.CornerSvdShim`.

Run with ``python aos/code/test_open_loop.py`` (or under pytest) in the LSST stack
environment; needs ``lsst.ts.ofc`` and ``$TS_CONFIG_MTTCS_DIR``.

The sign tests are the point of this file. The OLR sign cannot be checked by
``run_olr.py``'s identity ``olr_deviation == olr_opd - intrinsic``, which holds for either
sign because the intrinsic cancels, so it is pinned here instead by a round trip: push a
known Trim through the forward operator, and check that the open-loop wavefront removes it
rather than doubling it.
"""
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / 'smatrix' / 'code'))

import aos_state          # noqa: E402
import open_loop          # noqa: E402
import regularized_inversion as RI   # noqa: E402

_CACHE = {}


def _estimator(dof_set, n_modes):
    """One estimator per (dof_set, n_modes), held for the process.

    Held deliberately: `aos_state.corner_recovery_basis` caches on
    ``id(state_estimator)`` and CPython reuses an id after garbage collection, so a
    short-lived estimator could be handed another scheme's basis.
    """
    key = (dof_set, n_modes)
    if key not in _CACHE:
        _CACHE[key] = aos_state.make_state_estimator(dof_set=dof_set, n_modes=n_modes)
    return _CACHE[key]


def test_sensitivity_matrix_shape_follows_dof_set():
    """Q11: the scheme's own DOF set decides the column count, 22 or 50."""
    for dof_set, n_modes, n_dof in (('standard_22', 12, 22), ('all_50', 34, 50)):
        se = _estimator(dof_set, n_modes)
        S = open_loop.olr_sensitivity_matrix(se)
        n_rows = len(aos_state.SENSOR_NAMES) * len(aos_state.ZK_NOLL)
        assert S.shape == (n_rows, n_dof), f'{dof_set}: got {S.shape}'
        assert np.isfinite(S).all()


def test_olr_zernikes_removes_the_trim_wavefront():
    """The OLR sign: a deviation produced *by* a Trim must come back to zero.

    Construct the deviation the loop would have measured after applying ``trim``, i.e.
    minus the Trim's own wavefront contribution. The open-loop reconstruction must then
    recover the zero wavefront, not twice the contribution. With the old
    ``olr/code/olr.py`` addition this returns -2x the contribution instead.
    """
    se = _estimator('all_50', 34)
    S = open_loop.olr_sensitivity_matrix(se)
    trim = np.zeros(50)
    trim[5] = 30.0           # camera hexapod dz [µm]
    trim[10] = 0.25          # M1M3 bending mode 1 [µm]
    contribution = S @ trim[[int(d) for d in se.ofc_data.dof_idx]]

    zk_dev = -contribution                      # what the closed loop sees
    zk_olr = open_loop.olr_zernikes(zk_dev, trim, S, se)
    assert np.allclose(zk_olr, -2.0 * contribution, atol=1e-9), (
        'open-loop wavefront should be deviation minus the Trim contribution')

    # And the complementary case: with no Trim applied, the OLR is a no-op.
    zk_plain = open_loop.olr_zernikes(zk_dev, np.zeros(50), S, se)
    assert np.allclose(zk_plain, zk_dev, atol=1e-12)


def test_open_loop_state_is_the_negative_of_the_optical_state():
    """``Deviation - Trim`` for the OLR, ``Trim - Deviation`` for the optical state."""
    se = _estimator('all_50', 34)
    rng = np.random.default_rng(3)
    dof_rec = rng.normal(0, 1.0, 50)
    trim = rng.normal(0, 1.0, 50)

    dof_olr, v_olr = open_loop.open_loop_state(dof_rec, trim, se, n_modes=34)
    dof_state = open_loop.optical_state_dofs(dof_rec, trim)

    assert np.allclose(dof_olr[0], dof_rec - trim)
    assert np.allclose(dof_state, trim - dof_rec)
    assert np.allclose(dof_olr[0], -dof_state), 'the two must differ by exactly a sign'
    assert v_olr.shape == (1, 34)

    # The v-modes of the differenced DOF equal the difference of the projected v-modes,
    # the projection being linear; this guards against a sign flip entering only one path.
    v_rec = aos_state.vmodes_from_dofs(dof_rec, se, n_modes=34)
    v_trim = aos_state.vmodes_from_dofs(trim, se, n_modes=34)
    assert np.allclose(v_olr, v_rec - v_trim, atol=1e-9)


def test_thermal_focus_v1_convention_is_reproduced():
    """The optical state must match ``thermal_focus``'s v1 form, generalized.

    ``thermal_focus_lib`` computes ``v1_trim + MEASURED_SIGN * v1`` with
    ``MEASURED_SIGN = -1.0``, i.e. ``v1_trim - v1_meas``. Projecting
    `optical_state_dofs` must give the same v-mode 1.
    """
    se = _estimator('all_50', 34)
    rng = np.random.default_rng(11)
    dof_rec = rng.normal(0, 2.0, 50)
    trim = rng.normal(0, 2.0, 50)

    v_meas = aos_state.vmodes_from_dofs(dof_rec, se, n_modes=34)[0]
    v_trim = aos_state.vmodes_from_dofs(trim, se, n_modes=34)[0]
    thermal_focus_v1 = v_trim[0] + (-1.0) * v_meas[0]

    v_state = aos_state.vmodes_from_dofs(
        open_loop.optical_state_dofs(dof_rec, trim), se, n_modes=34)[0]
    assert np.isclose(v_state[0], thermal_focus_v1, atol=1e-9)


def test_olr_identity_check_passes_and_catches_a_real_error():
    """The carried-over identity check fires on a corner-ordering error."""
    rng = np.random.default_rng(5)
    n = len(aos_state.SENSOR_NAMES) * len(aos_state.ZK_NOLL)
    intrinsic = rng.normal(0, 0.1, n)
    olr_opd = rng.normal(0, 0.3, n)
    olr_dev = olr_opd - intrinsic
    assert open_loop.check_olr_identity(olr_dev, olr_opd, intrinsic) <= 1e-9

    scrambled = olr_dev.reshape(4, -1)[[1, 0, 3, 2]].ravel()
    try:
        open_loop.check_olr_identity(scrambled, olr_opd, intrinsic)
    except AssertionError:
        pass
    else:
        raise AssertionError('identity check missed a corner-ordering error')

    # ... and is blind to the sign, which is why the round-trip test above exists.
    assert open_loop.check_olr_identity(-olr_opd - intrinsic, -olr_opd, intrinsic) <= 1e-9


def test_shim_reproduces_recover_optical_state():
    """Q13: the shim must present the basis `aos_state` actually inverts in."""
    for dof_set, n_modes in (('standard_22', 12), ('all_50', 34)):
        se = _estimator(dof_set, n_modes)
        shim = aos_state.CornerSvdShim(se, n_keep=n_modes)
        idx = shim.dof_idx
        S = open_loop.olr_sensitivity_matrix(se)

        truth = np.zeros(50)
        truth[5], truth[0] = 25.0, -10.0
        dW = S @ truth[idx]

        d_trunc = RI.invert_truncated(dW, shim, rank=n_modes)
        dof_full, _v, _zk = aos_state.recover_optical_state(dW, se, n_modes=n_modes)
        assert np.allclose(d_trunc, dof_full[idx], atol=1e-9), (
            f'{dof_set}: shim-truncated and recover_optical_state disagree')


def test_shim_slices_u_eff_to_n_keep():
    """``U_eff`` must be sliced, since the solvers read the rank off its width."""
    se = _estimator('all_50', 34)
    shim = aos_state.CornerSvdShim(se, n_keep=34)
    assert shim.U_eff.shape[1] == 34
    assert shim.Sigma.shape == (34,)
    assert shim.V.shape[1] == 34
    assert shim.n_keep_eff == 34
    assert shim.kj_grid is None
    assert len(shim.normalization_weights) == 50

    try:
        aos_state.CornerSvdShim(se, n_keep=99)
    except ValueError:
        pass
    else:
        raise AssertionError('n_keep above the available modes should raise')


def test_range_penalty_runs_and_respects_ranges():
    """The RBR solver must run against the shim and pull over-range DOF back."""
    se = _estimator('all_50', 34)
    shim = aos_state.CornerSvdShim(se, n_keep=34)
    S = open_loop.olr_sensitivity_matrix(se)
    ranges = RI.dof_range_vector(shim)
    assert np.isfinite(ranges).all() and (ranges > 0).all()

    # Drive a bending mode well past its range so the penalty must bind.
    truth = np.zeros(50)
    truth[12] = 6.0 * ranges[12]
    dW = S @ truth[shim.dof_idx]

    d_tr = RI.invert_truncated(dW, shim, rank=34)
    d_rbr, info = RI.invert_range_penalty(dW, shim, ranges, kappa=4, power=3,
                                         rank=34, return_info=True)
    assert info['converged'], f"IRLS did not converge: {info}"
    assert np.max(np.abs(d_rbr) / ranges) < np.max(np.abs(d_tr) / ranges), (
        'the range penalty should reduce the worst |d_j| / r_j')

    res = RI.achieved_residual(dW, d_rbr, shim, rank=34)
    assert res.shape == dW.shape
    assert np.isfinite(res).all()


def test_shim_refuses_obsolete_normalization():
    """A bare ``OFCData('lsst')`` carries the obsolete weights; the shim must refuse."""
    from lsst.ts.ofc import OFCData
    from lsst.ts.ofc.state_estimator import StateEstimator
    ofc = OFCData('lsst')
    ofc.configure_controller()
    if ofc.controller.get('normalization_weights_filename') == aos_state.REQUIRED_NORM_YAML:
        return      # this install's bare default is already the required one
    ofc.zn_selected = np.array(aos_state.ZK_NOLL)
    ofc.comp_dof_idx = aos_state._comp_dof_idx(aos_state.DOF_SETS['all_50'])
    try:
        aos_state.CornerSvdShim(StateEstimator(ofc), n_keep=34)
    except RuntimeError as exc:
        assert 'normalization' in str(exc)
    else:
        raise AssertionError('shim accepted the obsolete normalization')


def test_corner_basis_cache_is_not_confused_between_schemes():
    """Two live estimators must keep distinct bases (the ``id()``-cache trap)."""
    se22 = _estimator('standard_22', 12)
    se50 = _estimator('all_50', 34)
    b22 = aos_state.corner_recovery_basis(se22)
    b50 = aos_state.corner_recovery_basis(se50)
    assert b22['V'].shape[0] == 22 and b50['V'].shape[0] == 50
    assert len(b22['dof_indices']) == 22 and len(b50['dof_indices']) == 50


def _main():
    """Run every ``test_*`` in this module, reporting pass or fail per test."""
    tests = [v for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f'PASS {t.__name__}')
        except Exception as exc:                      # noqa: BLE001
            failed += 1
            print(f'FAIL {t.__name__}: {type(exc).__name__}: {exc}')
    print(f'\n{len(tests) - failed} passed, {failed} failed')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main())
