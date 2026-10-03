"""Open Loop Reproduction (OLR): the forward direction of the corner-wavefront operator.

The Active Optics System (AOS) runs closed loop, so a visit's measured corner wavefront
has already had the accumulated correction (the Trim) applied to it. The OLR undoes that,
reporting the wavefront, degrees of freedom (DOF) and v-modes that **would have been
present had the loop been open**.

Two signed quantities live here and differ only by an overall sign. Conflating them is the
failure mode this module exists to prevent, so both are named explicitly:

===================  ===================  ====================================================
quantity             value                meaning
===================  ===================  ====================================================
optical state        ``Trim - Deviation`` the DOF defining the visit's optical state
**OLR output**       ``Deviation - Trim`` what would have been seen with the loop open
===================  ===================  ====================================================

A measured deviation is equivalent to some DOF vector, and the Trim is applied with the
**opposite** sign in order to drive that deviation toward zero. So ``Trim - Deviation`` is
the optical state, and the open-loop reconstruction is its negative. The optical-state form
is what ``thermal_focus/code/thermal_focus_lib.py`` computes for v-mode 1 alone, as
``v1_trim + MEASURED_SIGN * v1`` with ``MEASURED_SIGN = -1.0``; this module generalizes it
to every v-mode and to the Zernikes.

Superseding ``olr/code/olr.py``
-------------------------------
This module replaces that file's ``build_olr_sensitivity_matrix`` and ``apply_trim``, and
is **not** a port — four things changed:

1. **The sign is fixed.** ``olr/code/olr.py`` computed
   ``olr_opd = zk_opd + sens_mat @ trim``, *adding* the correction. That is the wrong sign
   per the argument above. Note that ``olr/code/run_olr.py``'s identity check
   ``olr_deviation == olr_opd - intrinsic`` does **not** catch this: the intrinsic is
   carried through unchanged and cancels, so the identity holds for either sign. It is
   still worth running (`check_olr_identity`) as a basis and alignment check.
2. **The DOF set comes from the state estimator**, so the column count is 22 for
   ``standard_22`` and 50 for ``all_50`` rather than always 22. ``olr/code/olr.py`` got 22
   by hand-masking ``comp_dof_idx``, and its ``DEFAULT_DOF_INDICES`` was a hand-written
   duplicate of ``DOF_SETS['standard_22']`` that could drift from it silently. Both are
   gone. Keeping 22 everywhere would put the 50/34 open-loop state and its
   deviation-recovered state in different subspaces, defeating the scheme comparison.
3. **The normalization is asserted.** ``olr/code/olr.py`` built a bare
   ``OFCData(name='lsst')``, which resolves `aos_state.OBSOLETE_NORM_YAML` — the
   normalization that *rotates* the v-mode basis rather than rescaling it, and fails
   silently. Everything here comes from `aos_state.make_state_estimator`, which raises
   unless the resolved normalization is `aos_state.REQUIRED_NORM_YAML`.
4. **The Zernike frame is named.** ``olr/code/olr.py`` called its field angles
   ``field_angles_ccs`` while taking them from the same ``sample_points`` this module uses
   at rotator zero, where `aos_state` requires the Optical Coordinate System (OCS). The
   frame is now stated rather than inherited from a variable name: see
   `aos_state.ZK_FRAME` and `aos_state.SMATRIX_ROTATION_ANGLE_DEG`.

There is deliberately **no Double Zernike (DZ) state** anywhere in this module. This work
uses the corner wavefront sensors (CWFS) only, with no Full Array Mode (FAM), and four
corner field points do not uniquely determine a DZ field over the focal plane.

Zernike layout
--------------
Every Zernike vector here is flattened corner-major as
``4 corners x len(aos_state.ZK_NOLL)`` = 84 values, in `aos_state.SENSOR_NAMES` order —
the row order of the sensitivity matrix, and the same layout
`aos_state.recover_optical_state` consumes. `aos_state.ZK_NOLL` already excludes Noll
Z20 and Z21, so the ``np.insert`` zero-padding ``olr/code/olr.py`` needed to reach a dense
23-vector has no counterpart here.

Requires the LSST stack (``lsst.ts.ofc``) and ``$TS_CONFIG_MTTCS_DIR``.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import aos_state  # noqa: E402

__all__ = ['olr_sensitivity_matrix', 'olr_zernikes', 'open_loop_state',
           'optical_state_dofs', 'check_olr_identity', 'cwfs_fwhm',
           'make_fwhm_converter']


def olr_sensitivity_matrix(state_estimator):
    """Forward DOF-to-corner-Zernike sensitivity matrix for one scheme.

    Parameters
    ----------
    state_estimator : `lsst.ts.ofc.state_estimator.StateEstimator`
        From `aos_state.make_state_estimator`, which fixes the DOF set and asserts the
        required normalization.

    Returns
    -------
    sens_mat : `numpy.ndarray`
        Shape ``(84, n_dof)`` — ``4 corners x 21 Zernikes`` rows by the scheme's active
        DOF count (22 for ``standard_22``, 50 for ``all_50``), in µm of wavefront per DOF
        unit (µm for translations and bending modes, arcsec for tilts). Rows are
        corner-major in `aos_state.SENSOR_NAMES` order.

    Notes
    -----
    Unnormalized and untruncated: this is the raw forward operator in physical DOF units,
    so ``sens_mat @ dof_active`` is directly a wavefront. The normalization weights enter
    only the SVD used for *inverting* a wavefront (`aos_state.corner_recovery_basis`).

    Evaluated at camera rotator angle `aos_state.SMATRIX_ROTATION_ANGLE_DEG` = 0.0 deg,
    so the Zernikes it produces are in the telescope frame (`aos_state.ZK_FRAME`, OCS).
    There is deliberately no rotation-angle argument — see that constant.

    Costs about 270 ms, so hold the result rather than rebuilding it per visit.
    """
    field_angles = [state_estimator.ofc_data.sample_points[s]
                    for s in aos_state.SENSOR_NAMES]
    return np.asarray(state_estimator.get_sensitivity_matrix(
        field_angles, aos_state.SMATRIX_ROTATION_ANGLE_DEG,
        normalize=False, truncate=False), dtype=float)


def _active_dofs(dof_state, state_estimator):
    """Slice a 50-element DOF vector down to the scheme's active indices.

    Replaces ``olr/code/olr.py``'s ``dof_state[DEFAULT_DOF_INDICES]``: the indices come
    from the estimator, so they cannot drift from the DOF set the matrix was built for.
    """
    idx = [int(d) for d in state_estimator.ofc_data.dof_idx]
    arr = np.asarray(dof_state, dtype=float)
    if arr.shape[-1] != 50:
        raise ValueError(
            f'dof_state has {arr.shape[-1]} entries; the OFC state is 50 DOF. Pass the '
            f'full 50-vector (as stored in visit_telemetry dof0..49) and let the state '
            f'estimator select its own active subset.')
    return arr[..., idx]


def olr_zernikes(zk_deviation, dof_trim, sens_mat, state_estimator):
    """Open-loop corner Zernikes: what the CWFS would have measured with the loop open.

    Parameters
    ----------
    zk_deviation : `numpy.ndarray`
        Measured deviation Zernikes (OPD minus intrinsic), ``(84,)`` or ``(n, 84)``, µm of
        wavefront, corner-major in `aos_state.SENSOR_NAMES` order, and already derotated
        into the telescope frame (`aos_state.ZK_FRAME`, OCS).
    dof_trim : `numpy.ndarray`
        Commanded Trim, ``(50,)`` or ``(n, 50)``, µm and arcsec — the full OFC state as
        stored in ``visit_telemetry.dof0..49``.
    sens_mat : `numpy.ndarray`
        From `olr_sensitivity_matrix`, built from the same `state_estimator`.
    state_estimator : `lsst.ts.ofc.state_estimator.StateEstimator`
        Supplies the active DOF indices.

    Returns
    -------
    zk_olr : `numpy.ndarray`
        Same shape as `zk_deviation`, µm of wavefront: the open-loop wavefront
        ``zk_deviation - sens_mat @ trim_active``.

    Notes
    -----
    The **subtraction** is the sign correction over ``olr/code/olr.py``, which added. The
    Trim was applied to drive the measured deviation toward zero, so removing its modelled
    wavefront contribution is what reconstructs the open-loop wavefront. See the module
    docstring.
    """
    zk = np.atleast_2d(np.asarray(zk_deviation, dtype=float))
    trim_active = np.atleast_2d(_active_dofs(dof_trim, state_estimator))
    if zk.shape[1] != sens_mat.shape[0]:
        raise ValueError(
            f'zk_deviation has {zk.shape[1]} values per visit but the sensitivity matrix '
            f'has {sens_mat.shape[0]} rows ({len(aos_state.SENSOR_NAMES)} corners x '
            f'{len(aos_state.ZK_NOLL)} Zernikes, corner-major).')
    if trim_active.shape[1] != sens_mat.shape[1]:
        raise ValueError(
            f'trim has {trim_active.shape[1]} active DOF but the sensitivity matrix has '
            f'{sens_mat.shape[1]} columns; the matrix and the DOF selection must come '
            f'from the same state estimator.')
    out = zk - trim_active @ sens_mat.T
    return out.reshape(np.shape(zk_deviation))


def optical_state_dofs(dof_recovered, dof_trim):
    """The visit's optical state in DOF space, ``Trim - Deviation``.

    Parameters
    ----------
    dof_recovered : `numpy.ndarray`
        DOF recovered from the measured deviation alone, ``(50,)`` or ``(n, 50)``, from
        `aos_state.recover_optical_state`. µm and arcsec.
    dof_trim : `numpy.ndarray`
        Commanded Trim, same shape and units.

    Returns
    -------
    dof_state : `numpy.ndarray`
        ``dof_trim - dof_recovered``, µm and arcsec.

    Notes
    -----
    This is the optical state, **not** the OLR output — the two differ by an overall sign,
    and `open_loop_state` returns the other one. Generalizes the v-mode-1 form in
    ``thermal_focus/code/thermal_focus_lib.py`` (``v1_trim + MEASURED_SIGN * v1`` with
    ``MEASURED_SIGN = -1.0``) to all 50 DOF.
    """
    return (np.asarray(dof_trim, dtype=float)
            - np.asarray(dof_recovered, dtype=float))


def open_loop_state(dof_recovered, dof_trim, state_estimator, n_modes):
    """Open-loop DOF and v-modes: ``Deviation - Trim``.

    Parameters
    ----------
    dof_recovered : `numpy.ndarray`
        DOF recovered from the measured deviation alone, ``(50,)`` or ``(n, 50)``, µm and
        arcsec, from `aos_state.recover_optical_state`.
    dof_trim : `numpy.ndarray`
        Commanded Trim, same shape and units.
    state_estimator : `lsst.ts.ofc.state_estimator.StateEstimator`
        The v-mode engine; the same estimator the measured v-modes are reported in, so all
        terms share one basis.
    n_modes : `int`
        v-modes to return — 12 or 34.

    Returns
    -------
    dof_olr : `numpy.ndarray`
        ``(n, 50)`` open-loop DOF, ``dof_recovered - dof_trim``, µm and arcsec.
    v_olr : `numpy.ndarray`
        ``(n, n_modes)`` open-loop v-mode amplitudes, dimensionless, projected through
        `aos_state.vmodes_from_dofs`.

    Notes
    -----
    The sign is ``Deviation - Trim``, the negative of `optical_state_dofs`: this is the
    state that would have been present with the loop open, which is what "Open Loop
    Reproduction" names. Projecting the differenced DOF is equivalent to differencing the
    projected v-modes, the projection being linear, but is done this way so the stored DOF
    and v-modes cannot disagree in sign.
    """
    rec = np.atleast_2d(np.asarray(dof_recovered, dtype=float))
    trim = np.atleast_2d(np.asarray(dof_trim, dtype=float))
    dof_olr = rec - trim
    v_olr = aos_state.vmodes_from_dofs(dof_olr, state_estimator, n_modes=n_modes)
    return dof_olr, v_olr


def make_fwhm_converter():
    """The ts_wep per-Zernike FWHM conversion, or `None` if ts_wep is unavailable.

    Returns
    -------
    conv : `callable` or `None`
        ``lsst.ts.wep.utils.convertZernikesToPsfWidth``, to be passed to `cwfs_fwhm`.
        `None` when ts_wep is not importable, so a build can proceed with the IQ column
        left NaN rather than failing outright.
    """
    try:
        from lsst.ts.wep.utils import convertZernikesToPsfWidth
    except ImportError:
        return None
    return convertZernikesToPsfWidth


def cwfs_fwhm(zk_residual, conv, reduce=np.nanmedian):
    """PSF FWHM contribution of a corner residual wavefront, median over the four CWFS.

    Parameters
    ----------
    zk_residual : `numpy.ndarray`
        Residual wavefront, ``(84,)`` or ``(n, 84)``, µm of wavefront, corner-major in
        `aos_state.SENSOR_NAMES` order — typically the achieved residual left after a
        scheme's correction.
    conv : `callable`
        From `make_fwhm_converter`.
    reduce : `callable`, optional
        Reduction over the four corners. Default `numpy.nanmedian`.

    Returns
    -------
    fwhm : `numpy.ndarray`
        ``(n,)`` arcsec FWHM contribution, one per visit.

    Notes
    -----
    **Evaluated at the four corner sensors only, not over the focal plane**, and this is
    deliberate (decided 2026-10-02). This work uses the CWFS with no Full Array Mode, and
    four corner field points do not uniquely determine a Double Zernike field, so there is
    no DZ state to evaluate on a focal-plane grid. The optical state in the science
    sensors is not well known; extrapolating there would fold that uncertainty into the
    image-quality estimate, so the metric is evaluated only where the wavefront is
    actually measured.

    **Not interchangeable with `aos_fwhm.fp_fwhm`**, which every other wavefront
    image-quality number in this repository uses: that evaluates a DZ field on an
    area-uniform focal-plane grid out to ``FP_RADIUS`` = 1.75 deg and takes the median
    over *that grid*. The ts_wep conversion and the Z4+ quadrature sum are shared; the
    evaluation domain is not. Do not substitute one for the other.

    Zero means no residual wavefront, so this term adds nothing. It is **not** a total
    PSF width and not the seeing floor — it is the AOS wavefront contribution alone, which
    adds in quadrature on top of the delivered image quality.
    """
    import aos_fwhm
    z = np.atleast_2d(np.asarray(zk_residual, dtype=float))
    n_z = len(aos_state.ZK_NOLL)
    n_corners = len(aos_state.SENSOR_NAMES)
    if z.shape[1] != n_corners * n_z:
        raise ValueError(
            f'zk_residual has {z.shape[1]} values per visit; expected '
            f'{n_corners * n_z} ({n_corners} corners x {n_z} Zernikes, corner-major).')
    out = np.full(len(z), np.nan)
    for i, row in enumerate(z):
        if not np.isfinite(row).all():
            continue
        per_corner = aos_fwhm.zj_to_fwhm(row.reshape(n_corners, n_z),
                                         aos_state.ZK_NOLL, conv)
        out[i] = float(reduce(per_corner))
    return out


def check_olr_identity(zk_olr_deviation, zk_olr_opd, zk_intrinsic, atol=1e-9):
    """Assert ``olr_deviation == olr_opd - intrinsic``, carried over from ``run_olr.py``.

    Parameters
    ----------
    zk_olr_deviation, zk_olr_opd, zk_intrinsic : `numpy.ndarray`
        Open-loop deviation, open-loop OPD and intrinsic Zernikes, same shape, µm of
        wavefront.
    atol : `float`, optional
        Absolute tolerance, µm of wavefront.

    Returns
    -------
    max_abs_dev : `float`
        Largest absolute discrepancy, µm of wavefront.

    Raises
    ------
    AssertionError
        If the identity fails anywhere, so a basis or alignment error in the forward
        direction fails loudly at build time rather than silently entering the database.

    Notes
    -----
    **This does not check the sign.** The intrinsic is carried through unchanged and
    cancels from both sides, so the identity holds whether the Trim wavefront is added or
    subtracted. It catches corner-ordering, Zernike-selection and shape errors. The sign
    is fixed by the argument in the module docstring and guarded by the round-trip test in
    ``aos/code/test_open_loop.py``.
    """
    lhs = np.asarray(zk_olr_deviation, dtype=float)
    rhs = np.asarray(zk_olr_opd, dtype=float) - np.asarray(zk_intrinsic, dtype=float)
    finite = np.isfinite(lhs) & np.isfinite(rhs)
    if not finite.any():
        return float('nan')
    dev = np.abs(lhs[finite] - rhs[finite])
    max_abs_dev = float(dev.max())
    if max_abs_dev > atol:
        raise AssertionError(
            f'OLR identity olr_deviation == olr_opd - intrinsic fails: max |discrepancy| '
            f'= {max_abs_dev:.3e} µm of wavefront over {int(finite.sum())} values, '
            f'tolerance {atol:.1e} µm. This is a basis, corner-ordering or '
            f'Zernike-selection error in the forward direction.')
    return max_abs_dev
