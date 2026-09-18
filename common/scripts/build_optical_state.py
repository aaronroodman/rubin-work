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

    python common/scripts/build_optical_state.py --scheme 50_34 --intrinsic batoid \\
        --opd-version consdb_v1 --day-obs 20251023-20260913

Fill only the gaps in an existing variant::

    python common/scripts/build_optical_state.py --variant v50_34__batoid__consdb_v1 \\
        --day-obs 20251023-20260913 --resume

List what is registered::

    python common/scripts/build_optical_state.py --list

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

from common import efd_db                                            # noqa: E402
from common.telemetry_clients import make_consdb_client              # noqa: E402

INSTRUMENT = 'lsstcam'

#: scheme -> (ts_ofc DOF-set name, n_dof, n_modes)
SCHEMES = {
    '22_12': ('standard_22', 22, 12),
    '50_34': ('all_50', 50, 34),
}

DEFAULT_OFC_VERSION = 'v13'

#: hexapod-LUT entries carrying a tilt, in the 50-element DOF vector: camera and M2 rx/ry.
HEX_TILT_LUT = [3, 4, 8, 9]
DEG_TO_ARCSEC = 3600.0


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
    for i, (band, rot) in enumerate(zip(bands, rot_angles)):
        # ConsDB reports the literal string 'none' for exposures taken with no filter --
        # flats, darks, biases and CBP -- rather than a null, and 'none'.upper() is not a
        # filter ts_ofc knows, so it must be screened here with the empty and null cases.
        # Science exposures always carry a real band, so this only skips calibration rows.
        if (not isinstance(band, str) or band.lower() in ('', 'none')
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
                "intrinsic_route='miw' needs a miw_lookup; build the Measured Intrinsic "
                "Wavefront at the corner field points first with "
                "run_make_intrinsic_sidecar.py (see the science_lut study), then pass a "
                "lookup over its zk_intrinsic_MI column")
        intr = np.asarray(miw_lookup(visit_ids, rot_angles, zk_noll), float)
        if intr.shape != opd.shape:
            raise ValueError(f'miw_lookup returned {intr.shape}, expected {opd.shape}')
    else:
        raise ValueError(f'unknown intrinsic_route {intrinsic_route!r}')
    return opd - intr, n_opd


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
        ``project(con, visit_ids)`` -> ``(v_lut, v_trim)``, each shaped
        ``(len(visit_ids), n_modes)`` of dimensionless v-mode amplitudes, NaN for any visit
        with no `visit_telemetry` row or a non-finite active DOF.

    Notes
    -----
    The hexapod LUT is read from `visit_telemetry` as ``lut_dof0..9`` [µm, deg] and the Trim
    as ``dof0..49`` [µm, arcsec]. The LUT's four tilt entries are converted deg -> arcsec and
    its 40 mirror-bending entries are set to zero rather than NaN, since the hexapods command
    no bending; leaving them NaN would poison every projection.

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
            return out_lut, out_trim
        tel = tel.set_index('visit_id').reindex(index=vids)
        lut_dof = np.zeros((len(vids), 50))
        lut_dof[:, :10] = tel[lut_cols].to_numpy(float)
        lut_dof[:, HEX_TILT_LUT] *= DEG_TO_ARCSEC
        trim_dof = tel[trim_cols].to_numpy(float)
        out_lut = aos_state.vmodes_from_dofs(lut_dof, state_estimator, n_modes=n_modes)
        out_trim = aos_state.vmodes_from_dofs(trim_dof, state_estimator, n_modes=n_modes)
        return out_lut, out_trim

    return project


def recover_night(cdb, day_obs, state_estimator, n_modes, intrinsic_route, zk_noll,
                  ofc_version=DEFAULT_OFC_VERSION, cache=None, miw_lookup=None,
                  img_type=None, verbose=True):
    """Recover the optical state for every visit of one night.

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

    Returns
    -------
    visit_ids : `numpy.ndarray`
    v_modes : `numpy.ndarray`
        Shape ``(n, n_modes)``, µm of wavefront.
    dof : `numpy.ndarray`
        Shape ``(n, 50)``, µm and deg.
    resid_rms_um : `numpy.ndarray`
        Per-visit root-mean-square of ``z_dev - zk_constrained`` [µm of wavefront] — the
        part of the measured deviation the retained subspace cannot reproduce.
    ok : `numpy.ndarray` [`bool`]
    """
    import aos_state
    meta = visit_metadata(cdb, day_obs)
    if meta.empty:
        return (np.array([], 'int64'), np.zeros((0, n_modes)), np.zeros((0, 50)),
                np.array([]), np.array([], bool))
    if img_type is not None:
        want = [img_type] if isinstance(img_type, str) else list(img_type)
        meta = meta[meta['img_type'].isin(want)]
        if meta.empty:
            if verbose:
                print(f'{day_obs}: no {",".join(want)} exposures')
            return (np.array([], 'int64'), np.zeros((0, n_modes)), np.zeros((0, 50)),
                    np.array([]), np.array([], bool))
    vids = meta['visit_id'].to_numpy('int64')
    z_dev, n_opd = measured_deviation(
        cdb, vids, meta['band'].tolist(), meta['rotator_angle_deg'].to_numpy(float),
        intrinsic_route, zk_noll, ofc_version, cache, miw_lookup)

    n = len(vids)
    v_modes = np.full((n, n_modes), np.nan)
    dof = np.full((n, 50), np.nan)
    resid = np.full(n, np.nan)
    ok = np.zeros(n, bool)
    for i in range(n):
        row = z_dev[i]
        if not np.isfinite(row).all():
            continue
        d, v, zk_con = aos_state.recover_optical_state(
            row, state_estimator, n_modes=n_modes)
        dof[i] = d
        v_modes[i] = v
        resid[i] = float(np.sqrt(np.mean((row - zk_con) ** 2)))
        ok[i] = True
    if verbose:
        print(f'{day_obs}: {n} exposures, {n_opd} with complete corner OPD, '
              f'{int(ok.sum())} recovered')
    return vids, v_modes, dof, resid, ok


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
                        'defaults to the one in common/miw_corner_intrinsic.py')
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
        _dof_set, n_dof, n_modes = SCHEMES[scheme]
        vid = efd_db.register_variant(
            con, scheme, route, opd_version, n_dof, n_modes,
            intrinsic_ref=intrinsic_ref, ofc_config_version=ofc_version)
        print(f'registered variant {vid}')

    import aos_state
    zk_noll = aos_state.ZK_NOLL
    se, n_modes = build_state_estimator(scheme, ofc_version)
    print(f'variant {vid}: scheme {scheme} ({n_modes} v-modes), intrinsic {route}'
          + (f' [{intrinsic_ref}]' if intrinsic_ref else '')
          + f', OPD {opd_version}, {len(zk_noll)} Zernike terms')

    # The MIW route needs its intrinsic evaluated at the corner field points. The build named
    # in the variant's intrinsic_ref is the intrinsic, so a different build is a different
    # variant rather than a switch here.
    miw_lookup = None
    if route == 'miw':
        from common.miw_corner_intrinsic import DEFAULT_PARAM_SET, MiwCornerLookup
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
        have = {int(d) for d in con.execute(
            'SELECT DISTINCT v.day_obs FROM optical_state o '
            'JOIN visit_telemetry v USING (visit_id) WHERE o.variant_id = ?',
            [vid]).df()['day_obs']}
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

    img_types = ([t.strip() for t in a.img_type.split(',') if t.strip()]
                 if a.img_type else None)
    cache, total, total_ok, total_cmd = {}, 0, 0, 0
    for day in days:
        vids, v_modes, dof, resid, ok = recover_night(
            cdb, day, se, n_modes, route, zk_noll, ofc_version, cache,
            miw_lookup=miw_lookup, img_type=img_types, verbose=not a.quiet)
        if not len(vids):
            continue
        v_lut, v_trim = project_commanded(con, vids)
        n_cmd = int(np.isfinite(v_lut[:, 0]).sum())
        if not a.quiet:
            print(f'{day}: {n_cmd} of {len(vids)} with commanded v-modes '
                  f'(hexapod LUT and Trim, dimensionless)')
        efd_db.upsert_optical_state(con, vid, vids, v_modes, dof, resid, ok,
                                    v_modes_lut=v_lut, v_modes_trim=v_trim)
        total += len(vids)
        total_ok += int(ok.sum())
        total_cmd += n_cmd
    n_rows = con.execute('SELECT COUNT(*) FROM optical_state WHERE variant_id = ?',
                         [vid]).fetchone()[0]
    con.close()
    print(f'\n{total} visits this run, {total_ok} recovered, {total_cmd} with commanded '
          f'v-modes; {n_rows} rows for variant {vid}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
