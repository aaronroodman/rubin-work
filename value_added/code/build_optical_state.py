#!/usr/bin/env python3
"""Recover the optical state per visit and store it as one variant of the value-added
database.

The optical state is the degree-of-freedom (DOF) vector obtained by running the measured
per-corner Zernike **deviations** (optical path difference, OPD, minus the intrinsic
wavefront) through the sensitivity-matrix singular value decomposition (SVD), plus the
v-mode amplitudes of that state. It is **not** the Trim (accumulated Active Optics System,
AOS, offset) and **not** the Tweak (per-iteration correction).

It is a *family* of variants along three independent axes, so it is stored long, keyed
``(visit_id, variant_id)``, with `state_variant` recording what each variant is:

* **scheme** — ``22_12`` (22 DOF, 12 v-modes; what the AOS runs online) or ``50_34``;
* **intrinsic route** — ``batoid`` (the ts_ofc design intrinsic) or ``miw`` (the Measured
  Intrinsic Wavefront);
* **OPD version** — a tag for the measured-Zernike provenance, so a reprocessing arrives
  as a new variant rather than overwriting the old numbers.

Usage
-----
Register and build a new variant::

    python value_added/code/build_optical_state.py --scheme 50_34 --intrinsic batoid \\
        --opd-version consdb_v1 --day-obs 20251023-20260913

Fill only the gaps in an existing variant::

    python value_added/code/build_optical_state.py --variant v50_34__batoid__consdb_v1 \\
        --day-obs 20251023-20260913 --resume

List what is registered::

    python value_added/code/build_optical_state.py --list

Alongside the measured state, each row stores the **commanded** v-modes — the hexapod
look-up-table (LUT) and the Trim, read from `visit_telemetry` and projected in the variant's
own scheme. Storing them here rather than reprojecting them per analysis is what guarantees
that all three terms of ``v_lut + v_trim - v_meas`` share one basis.

Notes
-----
`aos_state.DOF_SETS['standard_22']` is ``sorted(range(0, 17) + range(30, 35))`` — 10
rigid-body plus M1M3 bending 1–7 plus M2 bending 1–5. It is **not** the first 22 contiguous
indices, so a scalar 22 would silently select DOF 0–21 instead.

The 50/34 scheme passes ``n_modes=34`` explicitly. `aos_state.N_MODES['all_50']` is 20, but
that is only a default: the decomposition for ``all_50`` offers all 50 modes and ``n_modes``
sets `StateEstimator.truncate_index`, so 34 is well-defined.

Every v-mode stored here — measured, LUT and Trim — comes from
`aos_state.make_state_estimator`, the single sanctioned engine. The *inversion* of the
measured corner wavefront uses `aos_state.corner_recovery_basis` instead, because
``StateEstimator.Vh`` does not span the 84-row corner problem; the recovered DOF are then
re-projected onto the estimator basis so the three v-mode terms remain comparable. See
`aos/docs/status/corner_recovery_route_comparison.md`.

The camera rotator angle comes from the ConsDB ``physical_rotator_angle``, **not**
``boresightRotAngle``. It enters only the intrinsic-wavefront lookup, which is evaluated at
the observed angle so that the intrinsic is in the same frame as the ConsDB measured
Zernikes and the deviation is frame-consistent. The deviation is then inverted against a
sensitivity matrix fixed at rotator zero (`aos_state.SMATRIX_ROTATION_ANGLE_DEG`), which
takes the deviation to be in the telescope frame (Optical Coordinate System, OCS) —
`aos_state.ZK_FRAME`.
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'aos' / 'code'))

import efd_db                                                       # noqa: E402
from common.telemetry_clients import make_consdb_client              # noqa: E402

INSTRUMENT = 'lsstcam'

#: scheme -> (ts_ofc DOF-set name, n_dof, n_modes)
#:
#: ``50_34_rbr`` is a **pseudo-scheme** (Q5): it shares the ``all_50`` DOF set and the 34
#: retained modes with ``50_34`` and differs only in the solver, so ``scheme`` no longer
#: determines ``n_dof`` and ``n_modes`` uniquely. The penalty parameters live in
#: `state_variant.notes`, there being no column for them. `RBR_SCHEMES` is what makes the
#: difference operative — without it, ``--scheme 50_34_rbr`` would build the ``all_50``
#: estimator, run the **truncated** solver, and store a silent duplicate of ``50_34``
#: under the RBR variant name.
SCHEMES = {
    '22_12': ('standard_22', 22, 12),
    '50_34': ('all_50', 50, 34),
    '50_34_rbr': ('all_50', 50, 34),
}

#: Schemes solved by Range-Bounded Recovery instead of the truncated SVD, with their
#: penalty parameters [both dimensionless]. kappa = 4 and power = 3 match the bounce test
#: (Q8). The guard in `resolve_solver` refuses any scheme named here whose solver path is
#: not importable, so an RBR variant cannot quietly become a truncated one.
RBR_SCHEMES = {'50_34_rbr': dict(kappa=4, power=3)}

DEFAULT_OFC_VERSION = 'v13'

#: Tilt entries of the 50-element DOF vector: M2 hexapod rx/ry then camera hexapod rx/ry.
#: Stored in deg, which is what the v-mode basis expects. `ofc_svd.DOF_UNITS_50` labels the
#: same four arcsec, so scale by 3600 arcsec/deg before comparing against anything built on
#: that convention -- the bounce-test tables in particular.
HEX_TILT_DOF = [3, 4, 8, 9]


def build_state_estimator(scheme, ofc_version=DEFAULT_OFC_VERSION):
    """The OFC `StateEstimator` for one scheme — the single v-mode engine.

    Parameters
    ----------
    scheme : `str`
        ``'22_12'`` or ``'50_34'``.
    ofc_version : `str`, optional
        ts_ofc configuration version. Must be one whose controller selects
        `aos_state.REQUIRED_NORM_YAML`; `aos_state.make_state_estimator` raises otherwise.

    Returns
    -------
    se : `lsst.ts.ofc.state_estimator.StateEstimator`
        From `aos_state.make_state_estimator`, with `truncate_index` set to `n_modes`.
    n_modes : `int`
        Modes retained — 12 or 34.
    """
    import aos_state
    dof_set, _n_dof, n_modes = SCHEMES[scheme]
    se = aos_state.make_state_estimator(dof_set=dof_set, version=ofc_version,
                                        n_modes=n_modes)
    avail = aos_state.corner_recovery_basis(se)['n_modes']
    if avail < n_modes:
        raise RuntimeError(f'scheme {scheme} wants {n_modes} modes but {dof_set!r} '
                           f'offers only {avail}')
    return se, n_modes


def _same_db(a_path, b_path):
    """Whether two ``--db``-style arguments resolve to the same file.

    `None` means `efd_db.default_db_path`, and either side may be relative or carry
    symlinks, so both are resolved before comparing. DuckDB refuses a second connection to
    one file under a different read-only setting, so this decides whether the telemetry
    reader can simply reuse the write connection.
    """
    def _r(p):
        return (pathlib.Path(p) if p is not None
                else pathlib.Path(efd_db.default_db_path())).expanduser().resolve()
    return _r(a_path) == _r(b_path)


def resolve_solver(scheme, state_estimator, n_modes):
    """The recovery solver for one scheme, with the RBR guard.

    Parameters
    ----------
    scheme : `str`
        Key into `SCHEMES`.
    state_estimator : `lsst.ts.ofc.state_estimator.StateEstimator`
    n_modes : `int`

    Returns
    -------
    recover : `callable`
        ``recover(z_dev)`` -> ``(dof_full, v_modes, zk_model, resid_rms_um)`` for one
        visit's 84-value deviation wavefront. ``zk_model`` is the wavefront the recovered
        DOF reproduce, so ``z_dev - zk_model`` is the residual the image-quality metric
        consumes.
    notes : `str`
        One line describing the solver and its parameters, for `state_variant.notes`.

    Raises
    ------
    RuntimeError
        If `scheme` is in `RBR_SCHEMES` but the shared solver or the corner shim cannot be
        imported. **This guard is the point of the function**: `SCHEMES` maps
        ``50_34_rbr`` to the same DOF set as ``50_34``, so without it the builder would
        happily run the truncated solver and write a silent duplicate of the ``50_34``
        variant under the RBR name.

    Notes
    -----
    The two arms store **different residuals**, and the variant's `notes` records which.
    The truncated arm stores the subspace residual ``z_dev - zk_constrained``. The RBR arm
    stores the **achieved** residual ``dW - S (d / w)``, because the subspace residual
    cannot see a regularizer trading wavefront for amplitude (settled in items 6 and 9) —
    it is independent of the recovered amplitudes, so for a regularized solve it would
    report the truncated arm's number.
    """
    import aos_state

    if scheme not in RBR_SCHEMES:
        def recover(z_dev):
            dof, v, zk_con = aos_state.recover_optical_state(
                z_dev, state_estimator, n_modes=n_modes)
            resid = float(np.sqrt(np.mean((z_dev - zk_con) ** 2)))
            return dof, v, zk_con, resid
        return recover, (f'truncated SVD at {n_modes} modes; resid_rms_um is the '
                         f'subspace residual z_dev - zk_constrained [µm of wavefront]')

    params = RBR_SCHEMES[scheme]
    try:
        sys.path.insert(0, str(_ROOT / 'smatrix' / 'code'))
        import regularized_inversion as RI
        shim = aos_state.CornerSvdShim(state_estimator, n_keep=n_modes)
        ranges = RI.dof_range_vector(shim)
    except Exception as exc:
        raise RuntimeError(
            f'scheme {scheme!r} is a Range-Bounded Recovery variant but its solver path '
            f'is unavailable ({type(exc).__name__}: {exc}). Refusing to fall back to the '
            f'truncated solver: that would write rows indistinguishable from the 50_34 '
            f'variant under the RBR variant name. Fix the import or build 50_34 instead.'
        ) from exc
    if not (np.isfinite(ranges).all() and (ranges > 0).all()):
        raise RuntimeError(
            f'scheme {scheme!r}: dof_range_vector returned non-positive or non-finite '
            f'ranges, which invert_range_penalty rejects')
    idx = shim.dof_idx

    def recover(z_dev):
        d_active = RI.invert_range_penalty(z_dev, shim, ranges, rank=n_modes, **params)
        dof_full = np.zeros(50)
        dof_full[idx] = d_active
        v = np.asarray(state_estimator.get_vmodes_from_dofs(dof_full), dtype=float)
        res = RI.achieved_residual(z_dev, d_active, shim, rank=n_modes)
        return dof_full, v, z_dev - res, float(np.sqrt(np.mean(res ** 2)))

    notes = (f'range-bounded recovery (RBR): invert_range_penalty, kappa='
             f'{params["kappa"]}, power={params["power"]} (both dimensionless), '
             f'{n_modes} modes, via aos_state.CornerSvdShim; resid_rms_um is the '
             f'ACHIEVED residual dW - S (d / w) [µm of wavefront], not the subspace '
             f'residual')
    return recover, notes


def visit_metadata(cdb, day_obs):
    """Band and rotator angle per visit for one night, from the ConsDB exposure table.

    Returns
    -------
    df : `pandas.DataFrame`
        ``visit_id``, ``day_obs``, ``seq_num``, ``band``, ``img_type``,
        ``rotator_angle_deg`` [deg] and ``altitude_deg`` [deg], ordered by ``seq_num``.

    Notes
    -----
    ``rotator_angle_deg`` is the ConsDB ``physical_rotator_angle``, which is the angle the
    intrinsic-wavefront evaluation needs; ``sky_rotation`` and ``boresightRotAngle`` are
    different quantities and must not be substituted.

    That column lives on ``visit1_quicklook``, not on ``exposure``, so this is a join
    within ``cdb_lsstcam`` — a same-database join, which the ConsDB server handles (unlike
    a cross-database join to ``efd_lsstcam``, which returns HTTP 500).
    """
    q = (f'SELECT e.exposure_id AS visit_id, e.day_obs, e.seq_num, e.band, e.img_type, '
         f'e.altitude, q.physical_rotator_angle '
         f'FROM cdb_{INSTRUMENT}.exposure e '
         f'LEFT JOIN cdb_{INSTRUMENT}.visit1_quicklook q '
         f'  ON q.visit_id = e.exposure_id '
         f'WHERE e.day_obs = {int(day_obs)} ORDER BY e.seq_num')
    df = cdb.query(q).to_pandas()
    if df.empty:
        return df
    df = df.rename(columns={'physical_rotator_angle': 'rotator_angle_deg',
                            'altitude': 'altitude_deg'})
    df['visit_id'] = df['visit_id'].astype('int64')
    for c in ('rotator_angle_deg', 'altitude_deg'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def intrinsic_batoid(bands, rot_angles, zk_noll, ofc_version=DEFAULT_OFC_VERSION,
                     cache=None):
    """Design (batoid) intrinsic wavefront at the four corner sensors.

    Parameters
    ----------
    bands : `iterable` [`str`]
        Per-visit band, lower case.
    rot_angles : `array_like` [`float`]
        Per-visit camera rotator angle [deg].
    zk_noll : `list` [`int`]
        Noll indices to return, in order.
    ofc_version : `str`, optional
    cache : `dict`, optional
        Reused ``(band, rounded rotator angle)`` -> intrinsic lookup, since the evaluation
        is much slower than the visit loop.

    Returns
    -------
    z_int : `numpy.ndarray`
        Shape ``(n_visits, 4 * len(zk_noll))``, µm of wavefront, flattened corner-major to
        match the SVD row order. NaN for a visit with no band or no rotator angle.

    Notes
    -----
    Cached on the rotator angle rounded to 0.5 deg. The intrinsic varies smoothly and
    slowly with rotator angle, so that quantization is far below the measurement scatter,
    and it turns tens of thousands of evaluations into a few hundred.
    """
    import aos_state
    from lsst.ts.ofc import OFCData
    from lsst.ts.ofc.utils.ofc_data_helpers import get_intrinsic_zernikes

    cache = {} if cache is None else cache
    ofcd = OFCData('lsst')
    n_z = len(zk_noll)
    out = np.full((len(rot_angles), 4 * n_z), np.nan)
    cols = [z - ofcd.znmin for z in zk_noll]
    # ConsDB reports a filter string for every exposure, including ones ts_ofc has no
    # intrinsic wavefront for: the literal 'none' for flats, darks, biases and CBP, and
    # names like 'OTHER:PINHOLE' for engineering masks. get_intrinsic_zernikes raises on
    # any of them, so screen against the set it does know rather than enumerating the
    # ones it does not. Science exposures always carry a real band.
    # intrinsic_zk also carries an '' key, a degenerate no-filter entry rather than a
    # real band, so drop it: an exposure with an empty band gets no intrinsic.
    known = {str(k).lower() for k in ofcd.intrinsic_zk} - {''}
    for i, (band, rot) in enumerate(zip(bands, rot_angles)):
        if (not isinstance(band, str) or band.lower() not in known
                or not np.isfinite(rot)):
            continue
        key = (band.lower(), round(float(rot) * 2.0) / 2.0)
        if key not in cache:
            z = get_intrinsic_zernikes(ofcd, key[0].upper(), aos_state.SENSOR_NAMES,
                                       key[1])
            cache[key] = np.asarray(z, float)[:, cols].ravel()
        out[i] = cache[key]
    return out


def measured_deviation(cdb, visit_ids, bands, rot_angles, intrinsic_route,
                       zk_noll, ofc_version=DEFAULT_OFC_VERSION, cache=None,
                       miw_lookup=None):
    """Per-corner Zernike deviation (OPD minus intrinsic) for a set of visits.

    Parameters
    ----------
    cdb : `lsst.summit.utils.ConsDbClient`
    visit_ids : `array_like` [`int`]
    bands : `iterable` [`str`]
    rot_angles : `array_like` [`float`]
        Camera rotator angle [deg].
    intrinsic_route : `str`
        ``'batoid'`` or ``'miw'``.
    zk_noll : `list` [`int`]
        Noll indices, in the SVD's row order.
    ofc_version : `str`, optional
    cache : `dict`, optional
        Passed to `intrinsic_batoid`.
    miw_lookup : `callable`, optional
        Required for ``intrinsic_route='miw'``. Called as
        ``miw_lookup(visit_ids, rot_angles, zk_noll)`` and must return an array shaped
        ``(n_visits, 4 * len(zk_noll))`` of µm of wavefront, corner-major.

    Returns
    -------
    z_dev : `numpy.ndarray`
        Shape ``(n_visits, 4 * len(zk_noll))``, µm of wavefront. NaN where either the OPD
        or the intrinsic is unavailable.
    n_opd : `int`
        Visits with a complete measured OPD.
    opd : `numpy.ndarray`
        The raw measured OPD, same shape and units — returned so the build-time OLR
        identity check has the intrinsic available as ``opd - z_dev``.

    Notes
    -----
    ConsDB ``ccdvisit1_quicklook`` stores the **total** OPD, while
    `aos_state.recover_optical_state` consumes the deviation, so the intrinsic is what
    defines the optical state — which is why the intrinsic route is a variant axis rather
    than an implementation detail.
    """
    import aos_state
    z_opd = aos_state.fetch_corner_zernikes_consdb(cdb, list(visit_ids),
                                                   instrument=INSTRUMENT,
                                                   zk_noll=zk_noll)
    n_v, n_z = len(visit_ids), len(zk_noll)
    opd = np.full((n_v, 4 * n_z), np.nan)
    if len(z_opd):
        want = [f'z{z}_{c}' for c in aos_state.SENSOR_NAMES for z in zk_noll]
        have = [c for c in want if c in z_opd.columns]
        z_opd = z_opd.reindex(index=list(visit_ids))
        if len(have) == len(want):
            opd = z_opd[want].to_numpy(float)
        else:
            missing = set(want) - set(have)
            print(f'    warning: {len(missing)} of {len(want)} corner Zernike columns '
                  f'absent from ConsDB; those entries stay NaN')
            for j, c in enumerate(want):
                if c in z_opd.columns:
                    opd[:, j] = pd.to_numeric(z_opd[c], errors='coerce').to_numpy(float)
    n_opd = int(np.isfinite(opd).all(axis=1).sum())

    if intrinsic_route == 'batoid':
        intr = intrinsic_batoid(bands, rot_angles, zk_noll, ofc_version, cache)
    elif intrinsic_route == 'miw':
        if miw_lookup is None:
            raise RuntimeError(
                "intrinsic_route='miw' needs a miw_lookup; construct one with "
                "aos/code/miw_corner_intrinsic.py's MiwCornerLookup, which evaluates an "
                "existing Measured Intrinsic Wavefront decomposition at the four corner "
                "field points. Running this module as a script does that for you: pass "
                "--intrinsic miw --intrinsic-ref <MIW build name>")
        intr = np.asarray(miw_lookup(visit_ids, rot_angles, zk_noll), float)
        if intr.shape != opd.shape:
            raise ValueError(f'miw_lookup returned {intr.shape}, expected {opd.shape}')
    else:
        raise ValueError(f'unknown intrinsic_route {intrinsic_route!r}')
    return opd - intr, n_opd, opd


def make_commanded_projector(state_estimator, n_modes):
    """A callable projecting the commanded hexapod LUT and Trim onto the v-modes.

    Parameters
    ----------
    state_estimator : `lsst.ts.ofc.state_estimator.StateEstimator`
        From `build_state_estimator` — **the same estimator the measured state's v-modes
        are reported in**. Sharing it is what puts all three v-mode terms in one basis, so
        ``v_lut + v_trim - v_meas`` mixes none.
    n_modes : `int`
        Modes to retain — 12 or 34.

    Returns
    -------
    project : `callable`
        ``project(con, visit_ids)`` -> ``(v_lut, v_trim, trim_dof)``. The two v-mode
        blocks are shaped ``(len(visit_ids), n_modes)`` of dimensionless amplitudes, NaN
        for any visit with no `visit_telemetry` row or a non-finite active DOF;
        ``trim_dof`` is ``(len(visit_ids), 50)`` of raw Trim DOF [µm, deg], which the
        open-loop reconstruction needs in DOF space rather than as v-modes.

    Notes
    -----
    The hexapod LUT is read from `visit_telemetry` as ``lut_dof0..9`` and the Trim as
    ``dof0..49``, both [µm, deg]. The LUT's 40 mirror-bending entries are set to zero rather
    than NaN, since the hexapods command no bending; leaving them NaN would poison every
    projection.

    **No deg -> arcsec conversion is applied to either vector**, because the v-mode basis
    expects the four hexapod tilt entries in deg: one unit of DOF 3 moves the v-modes by
    22.74 (dimensionless v-mode norm per unit DOF 3) against an allowed range of 0.12, which
    is one degree of tilt rather than one arcsec. This function used to scale the LUT tilts
    by 3600 arcsec/deg, which inflated the projected `v_modes_lut` norm by about 1082x
    (dimensionless, as-built over correct); v1 was nearly unaffected, at 1.9%, since v1 is
    almost pure defocus and barely responds to tilt.

    The projection is `aos_state.vmodes_from_dofs`, i.e. ``StateEstimator``'s own
    ``get_vmodes_from_dofs`` — the basis the Main Telescope AOS reports on the summit. The
    12-mode cap that once argued against it was `truncate_index`, a controller-yaml default
    rather than a limit; `aos_state.make_state_estimator` sets it from ``n_modes``, so 34
    modes are returned for the 50/34 scheme.
    """
    import aos_state
    lut_cols = [f'lut_dof{k}' for k in range(10)]
    trim_cols = [f'dof{k}' for k in range(50)]

    def project(con, visit_ids):
        vids = np.asarray(visit_ids, 'int64')
        out_lut = np.full((len(vids), n_modes), np.nan)
        out_trim = np.full((len(vids), n_modes), np.nan)
        sel = ', '.join(['visit_id'] + lut_cols + trim_cols)
        tel = con.execute(
            f'SELECT {sel} FROM visit_telemetry WHERE visit_id IN '
            f'({",".join(str(int(v)) for v in vids)})').df() if len(vids) else None
        if tel is None or not len(tel):
            return out_lut, out_trim, np.full((len(vids), 50), np.nan)
        tel = tel.set_index('visit_id').reindex(index=vids)
        lut_dof = np.zeros((len(vids), 50))
        lut_dof[:, :10] = tel[lut_cols].to_numpy(float)
        trim_dof = tel[trim_cols].to_numpy(float)
        out_lut = aos_state.vmodes_from_dofs(lut_dof, state_estimator, n_modes=n_modes)
        out_trim = aos_state.vmodes_from_dofs(trim_dof, state_estimator, n_modes=n_modes)
        return out_lut, out_trim, trim_dof

    return project


def _empty_night(n_modes):
    """The `recover_night` result for a night with nothing to recover."""
    return dict(visit_ids=np.array([], 'int64'),
                v_modes=np.zeros((0, n_modes)), dof=np.zeros((0, 50)),
                resid_rms_um=np.array([]), ok=np.array([], bool),
                v_modes_olr=np.zeros((0, n_modes)), dof_olr=np.zeros((0, 50)),
                fwhm_cwfs_arcsec=np.array([]), elevation_deg=np.array([]),
                rotator_angle_deg=np.array([]), n_opd=0)


def recover_night(cdb, day_obs, state_estimator, n_modes, intrinsic_route, zk_noll,
                  ofc_version=DEFAULT_OFC_VERSION, cache=None, miw_lookup=None,
                  img_type=None, verbose=True, recover=None, trim_dof=None,
                  fwhm_conv=None, sens_mat=None):
    """Recover the optical state, and the open-loop state, for every visit of one night.

    Parameters
    ----------
    cdb : `lsst.summit.utils.ConsDbClient`
    day_obs : `int`
    state_estimator : `lsst.ts.ofc.state_estimator.StateEstimator`
        From `build_state_estimator`.
    n_modes : `int`
        Modes to retain — 12 or 34.
    intrinsic_route : `str`
    zk_noll : `list` [`int`]
    ofc_version : `str`, optional
    cache : `dict`, optional
    miw_lookup : `callable`, optional
    img_type : `str` or `iterable` [`str`], optional
        Restrict to these ConsDB ``img_type`` values. Default is every exposure that has
        corner Zernikes.
    verbose : `bool`, optional
    recover : `callable`, optional
        From `resolve_solver`. Default is the plain truncated recovery, so the
        signature stays usable from a notebook.
    trim_dof : `numpy.ndarray` or `callable`, optional
        Commanded Trim per visit, ``(n_visits_of_the_night, 50)`` [µm, arcsec], aligned to
        this night's visits in `visit_metadata` order — or a callable taking this night's
        ``visit_ids`` and returning that array, since the visit list is determined here.
        Required for the open-loop columns; omitted, they come back NaN.
    fwhm_conv : `callable`, optional
        From `open_loop.make_fwhm_converter`. Omitted, the image-quality column is NaN.
    sens_mat : `numpy.ndarray`, optional
        From `open_loop.olr_sensitivity_matrix`, used only for the per-visit forward
        identity check.

    Returns
    -------
    out : `dict`
        ``visit_ids`` ``(n,)``; ``v_modes`` and ``v_modes_olr`` ``(n, n_modes)``
        [dimensionless]; ``dof`` and ``dof_olr`` ``(n, 50)`` [µm, arcsec];
        ``resid_rms_um`` ``(n,)`` [µm of wavefront]; ``fwhm_cwfs_arcsec`` ``(n,)``
        [arcsec]; ``ok`` ``(n,)``; ``n_opd`` `int`.

    Notes
    -----
    ``dof`` and ``v_modes`` are the **deviation-recovered** state — from this visit's
    measured deviation alone. ``dof_olr`` and ``v_modes_olr`` are the **open-loop**
    state, ``Deviation - Trim``, which is what would have been present with the loop open;
    the optical state ``Trim - Deviation`` is its negative. See `open_loop`.

    ``fwhm_cwfs_arcsec`` is the median over the **four corner sensors**, not over the
    focal plane, and describes the deviation-recovered arm. See `open_loop.cwfs_fwhm` for
    why, and note it is not interchangeable with `aos_fwhm.fp_fwhm`.
    """
    import aos_state
    import open_loop
    meta = visit_metadata(cdb, day_obs)
    if meta.empty:
        return _empty_night(n_modes)
    if img_type is not None:
        want = [img_type] if isinstance(img_type, str) else list(img_type)
        meta = meta[meta['img_type'].isin(want)]
        if meta.empty:
            if verbose:
                print(f'{day_obs}: no {",".join(want)} exposures')
            return _empty_night(n_modes)
    vids = meta['visit_id'].to_numpy('int64')
    z_dev, n_opd, zk_opd = measured_deviation(
        cdb, vids, meta['band'].tolist(), meta['rotator_angle_deg'].to_numpy(float),
        intrinsic_route, zk_noll, ofc_version, cache, miw_lookup)

    # No exposure has a complete set of corner Zernikes, so there is no optical state to
    # recover for any visit of this night. Record it empty rather than writing rows whose
    # recovered and open-loop columns are all NaN: those carry no information, and a
    # query that forgets to screen them gets NaN with no error. The whole pre-20251102
    # era is like this -- ConsDB has the exposures but no corner-WFS quicklook.
    if n_opd == 0:
        if verbose:
            print(f'{day_obs}: {len(vids)} exposures, none with complete corner OPD; '
                  f'no optical state for this night')
        return _empty_night(n_modes)

    if recover is None:
        def recover(z):
            d, v, zk_con = aos_state.recover_optical_state(
                z, state_estimator, n_modes=n_modes)
            return d, v, zk_con, float(np.sqrt(np.mean((z - zk_con) ** 2)))

    n = len(vids)
    v_modes = np.full((n, n_modes), np.nan)
    dof = np.full((n, 50), np.nan)
    resid = np.full(n, np.nan)
    zk_resid = np.full_like(z_dev, np.nan)
    ok = np.zeros(n, bool)
    for i in range(n):
        row = z_dev[i]
        if not np.isfinite(row).all():
            continue
        d, v, zk_model, r = recover(row)
        dof[i] = d
        v_modes[i] = v
        resid[i] = r
        zk_resid[i] = row - zk_model
        ok[i] = True

    # The open-loop state: Deviation - Trim, in DOF and v-mode space. NaN where the Trim
    # is unavailable, which leaves the deviation-recovered state intact.
    v_olr = np.full((n, n_modes), np.nan)
    dof_olr = np.full((n, 50), np.nan)
    if trim_dof is not None:
        trim = np.asarray(trim_dof(vids) if callable(trim_dof) else trim_dof, float)
        if trim.shape != (n, 50):
            raise ValueError(f'trim_dof has shape {trim.shape}, expected {(n, 50)}; it '
                             f'must be aligned to this night\'s visits')
        good = ok & np.isfinite(trim).all(axis=1)
        if good.any():
            d_o, v_o = open_loop.open_loop_state(dof[good], trim[good],
                                                 state_estimator, n_modes)
            dof_olr[good] = d_o
            v_olr[good] = v_o
            # The forward identity, carried over from run_olr.py: the open-loop deviation
            # and the open-loop OPD differ by exactly the intrinsic, because the same Trim
            # wavefront is removed from each. Run against the real intrinsic (z_opd -
            # z_dev), not a zero placeholder, or the check is vacuous. It is insensitive
            # to the OLR sign (the intrinsic cancels), so it catches basis, corner-order
            # and Zernike-alignment errors; the sign is pinned by the open_loop tests.
            if sens_mat is not None and zk_opd is not None:
                intr = zk_opd[good] - z_dev[good]
                zk_olr_dev = open_loop.olr_zernikes(z_dev[good], trim[good],
                                                    sens_mat, state_estimator)
                zk_olr_opd = open_loop.olr_zernikes(zk_opd[good], trim[good],
                                                    sens_mat, state_estimator)
                open_loop.check_olr_identity(zk_olr_dev, zk_olr_opd, intr, atol=1e-6)

    fwhm = np.full(n, np.nan)
    if fwhm_conv is not None and ok.any():
        fwhm[ok] = open_loop.cwfs_fwhm(zk_resid[ok], fwhm_conv)

    if verbose:
        n_iq = int(np.isfinite(fwhm).sum())
        msg = (f'{day_obs}: {n} exposures, {n_opd} with complete corner OPD, '
               f'{int(ok.sum())} recovered, {int(np.isfinite(dof_olr[:, 0]).sum())} '
               f'open-loop')
        if n_iq:
            msg += (f'; median CWFS FWHM contribution '
                    f'{np.nanmedian(fwhm):.4f} arcsec over {n_iq} visits')
        print(msg)
    return dict(visit_ids=vids, v_modes=v_modes, dof=dof, resid_rms_um=resid, ok=ok,
                v_modes_olr=v_olr, dof_olr=dof_olr, fwhm_cwfs_arcsec=fwhm,
                elevation_deg=meta['altitude_deg'].to_numpy(float),
                rotator_angle_deg=meta['rotator_angle_deg'].to_numpy(float),
                n_opd=n_opd)


def main(argv=None):
    """Command-line entry point."""
    p = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variant', default=None,
                   help='existing registered variant id to fill')
    p.add_argument('--scheme', choices=sorted(SCHEMES), default=None)
    p.add_argument('--intrinsic', choices=('batoid', 'miw'), default=None)
    p.add_argument('--opd-version', default='consdb_v1',
                   help='provenance tag for the measured OPD; a reprocessing gets a new '
                        'value so old and new coexist')
    p.add_argument('--intrinsic-ref', default=None,
                   help="MIW build name for --intrinsic miw, e.g. pathA_50_34_i_5rot; "
                        "defaults to the ts_ofc config version for the batoid route")
    p.add_argument('--ofc-version', default=DEFAULT_OFC_VERSION)
    p.add_argument('--miw-param-set', default=None,
                   help='param_set holding the MIW build named by --intrinsic-ref; '
                        'defaults to the one in aos/code/miw_corner_intrinsic.py')
    p.add_argument('--miw-ccd-height', action='store_true',
                   help='add a standalone per-sensor height-equivalent defocus to the MIW '
                        'Zernike 4. Normally leave this off: the detector heights are '
                        'camera-fixed and the MIW already carries them in its CCS '
                        'component, so this double-counts them')
    p.add_argument('--day-obs', default=None,
                   help='single night, inclusive range, or a comma list')
    p.add_argument('--img-type', default=None,
                   help="restrict to these ConsDB img_types, comma separated, e.g. "
                        "'science,acq'")
    p.add_argument('--resume', action='store_true',
                   help='skip nights already fully populated for this variant')
    p.add_argument('--list', action='store_true',
                   help='list registered variants with their row counts and exit')
    p.add_argument('--db', default=None)
    p.add_argument('--telemetry-db', default=None,
                   help='database to read visit_telemetry (the Trim and hexapod LUT) from, '
                        'when it is not the --db being written. Required for a sharded '
                        'build: a shard writes its own empty database, so without this the '
                        'commanded and open-loop columns are silently all NaN')
    p.add_argument('--consdb-url', default='auto')
    p.add_argument('--quiet', action='store_true')
    a = p.parse_args(argv)

    if a.list:
        con = efd_db.open_db(a.db, readonly=True)
        v = con.execute(
            'SELECT s.variant_id, s.scheme, s.intrinsic_route, s.opd_version, '
            's.n_modes, s.intrinsic_ref, COUNT(o.visit_id) AS n_rows, '
            'SUM(CASE WHEN o.ok THEN 1 ELSE 0 END) AS n_ok '
            'FROM state_variant s LEFT JOIN optical_state o USING (variant_id) '
            'GROUP BY 1, 2, 3, 4, 5, 6 ORDER BY 1').df()
        con.close()
        print(v.to_string(index=False) if len(v) else 'no variants registered')
        return 0

    if not a.day_obs:
        p.error('--day-obs is required unless --list is given')

    con = efd_db.open_db(a.db, create=True)
    if a.variant:
        reg = con.execute('SELECT * FROM state_variant WHERE variant_id = ?',
                          [a.variant]).df()
        if not len(reg):
            con.close()
            p.error(f'variant {a.variant!r} is not registered; define it with '
                    f'--scheme/--intrinsic/--opd-version')
        r = reg.iloc[0]
        scheme, route, opd_version = r['scheme'], r['intrinsic_route'], r['opd_version']
        ofc_version = r['ofc_config_version'] or a.ofc_version
        intrinsic_ref = r['intrinsic_ref']
        vid = a.variant
    else:
        if not (a.scheme and a.intrinsic):
            con.close()
            p.error('give either --variant, or both --scheme and --intrinsic')
        scheme, route, opd_version = a.scheme, a.intrinsic, a.opd_version
        ofc_version = a.ofc_version
        intrinsic_ref = a.intrinsic_ref or (
            f'ofc_{ofc_version}' if route == 'batoid' else None)
        if route == 'miw' and not intrinsic_ref:
            con.close()
            p.error('--intrinsic miw needs --intrinsic-ref naming the MIW build')
        vid = None      # registered below, once the solver notes are known

    import aos_state
    import open_loop
    zk_noll = aos_state.ZK_NOLL
    se, n_modes = build_state_estimator(scheme, ofc_version)

    # Resolve the solver BEFORE registering, so the variant's notes record which solver
    # and which residual its rows hold -- no column describes either. The RBR guard lives
    # here, so a 50_34_rbr run with a broken solver path fails before it writes anything.
    recover, solver_notes = resolve_solver(scheme, se, n_modes)

    if vid is None:
        _dof_set, n_dof, _nm = SCHEMES[scheme]
        vid = efd_db.register_variant(
            con, scheme, route, opd_version, n_dof, n_modes,
            intrinsic_ref=intrinsic_ref, ofc_config_version=ofc_version,
            notes=solver_notes)
        print(f'registered variant {vid}')
    else:
        con.execute('UPDATE state_variant SET notes = ? WHERE variant_id = ?',
                    [solver_notes, vid])

    print(f'variant {vid}: scheme {scheme} ({n_modes} v-modes), intrinsic {route}'
          + (f' [{intrinsic_ref}]' if intrinsic_ref else '')
          + f', OPD {opd_version}, {len(zk_noll)} Zernike terms')
    print(f'  solver: {solver_notes}')

    sens_mat = open_loop.olr_sensitivity_matrix(se)
    print(f'  OLR sensitivity matrix {sens_mat.shape} '
          f'(rows = 4 corners x {len(zk_noll)} Zernikes, cols = active DOF)')
    fwhm_conv = open_loop.make_fwhm_converter()
    if fwhm_conv is None:
        print('  warning: ts_wep unavailable, so fwhm_cwfs_arcsec stays NaN')

    # The MIW route needs its intrinsic evaluated at the corner field points. The build named
    # in the variant's intrinsic_ref is the intrinsic, so a different build is a different
    # variant rather than a switch here.
    miw_lookup = None
    if route == 'miw':
        from miw_corner_intrinsic import DEFAULT_PARAM_SET, MiwCornerLookup
        miw_lookup = MiwCornerLookup(mi_name=intrinsic_ref,
                                     param_set=a.miw_param_set or DEFAULT_PARAM_SET,
                                     add_ccd_height=a.miw_ccd_height)
        print(f'  MIW intrinsic from {miw_lookup.path}')
        print(f'  corner field points [deg, OCS]: '
              + ', '.join(f'{s} ({p[0]:+.4f}, {p[1]:+.4f})'
                          for s, p in zip(aos_state.SENSOR_NAMES, miw_lookup.points)))
        if miw_lookup.z4_height_um is not None:
            print(f'  corner CCD-height Zernike 4 added [µm of wavefront]: '
                  + ', '.join(f'{v:+.4f}' for v in miw_lookup.z4_height_um))

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from build_efd_db import parse_day_obs
    cdb = make_consdb_client(a.consdb_url)
    days = parse_day_obs(a.day_obs, cdb=cdb)
    if a.resume:
        # day_obs comes from visit_id rather than from a join on visit_telemetry: a shard's
        # own visit_telemetry is empty, so the join would find no populated nights and
        # --resume would silently rebuild everything. Verified over all 213,704
        # visit_telemetry rows that visit_id // 100000 == day_obs.
        have = {int(d) for d in con.execute(
            'SELECT DISTINCT visit_id // 100000 AS day_obs FROM optical_state '
            'WHERE variant_id = ?', [vid]).df()['day_obs']}
        skip = [d for d in days if d in have]
        days = [d for d in days if d not in have]
        if skip:
            print(f'--resume: skipping {len(skip)} night(s) already populated')
    if not days:
        con.close()
        print('nothing to do')
        return 0

    # The commanded hexapod LUT and Trim are projected here, in the variant's own scheme, so
    # that all three v-mode terms stored for a visit share one basis and v1_lut + v1_trim -
    # v1_meas mixes none. A visit with no visit_telemetry row gets NaN commanded terms and
    # keeps its measured state.
    project_commanded = make_commanded_projector(se, n_modes)

    # A shard writes its own database, whose visit_telemetry is empty, so the Trim must be
    # read from elsewhere or every commanded and open-loop column comes back NaN without
    # any error. Fail loudly on an empty telemetry source rather than building NaN columns.
    # Reuse the write connection when --telemetry-db names the database already open:
    # DuckDB refuses a second connection to one file under a different read-only setting.
    tel_con = con
    if a.telemetry_db and not _same_db(a.telemetry_db, a.db):
        tel_con = efd_db.open_db(a.telemetry_db, readonly=True)
    n_tel = tel_con.execute('SELECT COUNT(*) FROM visit_telemetry').fetchone()[0]
    if not n_tel:
        con.close()
        if tel_con is not con:
            tel_con.close()
        p.error(
            'visit_telemetry is empty in the database the Trim is read from, so the '
            'commanded v-modes and the entire open-loop arm would be silently NaN. Pass '
            '--telemetry-db pointing at the main database (this is required for every '
            'sharded build, since a shard writes its own empty database).')
    print(f'  Trim and hexapod LUT read from {a.telemetry_db or "the output database"} '
          f'({n_tel} visit_telemetry rows)')

    img_types = ([t.strip() for t in a.img_type.split(',') if t.strip()]
                 if a.img_type else None)
    cache, total, total_ok, total_cmd, total_olr = {}, 0, 0, 0, 0
    for day in days:
        # The Trim is needed inside recover_night for the open-loop state, but the visit
        # list comes from recover_night itself, so the commanded terms are fetched through
        # a callback on that list rather than afterwards. One query per night either way.
        fetched = {}

        def _trim_for(visit_ids):
            v_lut, v_trim, trim_dof = project_commanded(tel_con, visit_ids)
            fetched['lut'], fetched['trim'] = v_lut, v_trim
            return trim_dof

        res = recover_night(
            cdb, day, se, n_modes, route, zk_noll, ofc_version, cache,
            miw_lookup=miw_lookup, img_type=img_types, verbose=not a.quiet,
            recover=recover, trim_dof=_trim_for, fwhm_conv=fwhm_conv,
            sens_mat=sens_mat)
        vids = res['visit_ids']
        if not len(vids):
            continue
        v_lut, v_trim = fetched['lut'], fetched['trim']
        n_cmd = int(np.isfinite(v_lut[:, 0]).sum())
        n_olr = int(np.isfinite(res['dof_olr'][:, 0]).sum())
        if not a.quiet:
            print(f'{day}: {n_cmd} of {len(vids)} with commanded v-modes '
                  f'(hexapod LUT and Trim, dimensionless)')
        efd_db.upsert_optical_state(
            con, vid, vids, res['v_modes'], res['dof'], res['resid_rms_um'], res['ok'],
            v_modes_lut=v_lut, v_modes_trim=v_trim,
            v_modes_olr=res['v_modes_olr'], dof_olr=res['dof_olr'],
            fwhm_cwfs_arcsec=res['fwhm_cwfs_arcsec'],
            elevation_deg=res['elevation_deg'],
            rotator_angle_deg=res['rotator_angle_deg'])
        total += len(vids)
        total_ok += int(res['ok'].sum())
        total_cmd += n_cmd
        total_olr += n_olr
    n_rows = con.execute('SELECT COUNT(*) FROM optical_state WHERE variant_id = ?',
                         [vid]).fetchone()[0]
    con.close()
    if tel_con is not con:
        tel_con.close()
    print(f'\n{total} visits this run, {total_ok} recovered, {total_cmd} with commanded '
          f'v-modes, {total_olr} with open-loop state; {n_rows} rows for variant {vid}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
