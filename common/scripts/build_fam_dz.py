#!/usr/bin/env python3
"""Store the Full Array Mode (FAM) Double Zernike (DZ) fits and their v-modes as one
variant of the value-added database.

A FAM visit pair is one extra-focal plus one intra-focal exposure of the science CCDs. The
Active Optics System (AOS) pipeline fits a DZ expansion to the donut wavefronts of that
pair, giving coefficients ``DZ(k, j)`` in µm of wavefront for focal (field) Zernike order
``k`` and pupil Noll index ``j``. This script reads those coefficients from a
``param_set``'s ``fits.parquet``, projects them onto the Optical Feedback Control (OFC)
sensitivity-matrix singular value decomposition (SVD) to get v-mode amplitudes and degrees
of freedom (DOF), and writes both to the `fam_dz` table.

Like `optical_state`, it is a *family* of variants, so the table is long and keyed
``(visit_id, fam_variant_id)`` with `fam_variant` recording what each variant is:

* **param_set** — the Butler collection paired with a processing variant;
* **intrinsic route** — ``batoid`` (the design intrinsic) or ``miw`` (a Measured Intrinsic
  Wavefront build, read from ``<param_set>/<mi_name>/fits.parquet``);
* **prefix** — ``z1toz6`` (focal orders k=1..6) or ``z1toz3`` (k=1..3);
* **scheme** — ``50_34`` (50 DOF, 34 v-modes) or ``22_12``.

The FAM triplet is ordered **intra-focal, extra-focal, in-focus acq** in ascending
``seq_num``. ``fits.parquet`` is keyed on the extra-focal member, so each `fam_dz` row also
stores ``intra_seq_num = seq_num - 1``, ``acq_seq_num = seq_num + 1`` and the matching
``acq_visit_id``, which is what makes the in-focus Corner Wavefront Sensor (CWFS) optical
state in `optical_state` a key lookup rather than a search.

Usage
-----
Register and build the Batoid-intrinsic k=1..6 variant at 50 DOF / 34 v-modes::

    python common/scripts/build_fam_dz.py \\
        --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x

The same for a Measured Intrinsic Wavefront build::

    python common/scripts/build_fam_dz.py \\
        --param-set fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x --mi pathA_50_34_i_5rot

List what is registered::

    python common/scripts/build_fam_dz.py --list

Notes
-----
The pupil Noll indices are read from ``visits.parquet``'s ``nollIndices`` column, never
hardcoded. The canonical set is the 21 indices 4–19 and 22–26: Noll 20 and 21 are absent by
design. A hardcoded contiguous ``range(4, 23)`` would silently carry j=20,21 as all-NaN
columns and drop j=23–26, so the sidecar is the only source used — the same rule
``aos/code/lut/run_build_lut.py`` follows.

The projection is `ofc_svd.project_dz_table` through `ofc_svd.build_ofc_svd`, the
sensitivity-matrix SVD built inside OFC code. The v-modes it returns are in the same basis
as `optical_state`'s, so a FAM v-mode and a CWFS v-mode are directly comparable.

Requires `lsst.ts.ofc` and ``$TS_CONFIG_MTTCS_DIR`` (the AOS environment). It needs no
Butler and no Consolidated Database: ``fits.parquet`` and ``visits.parquet`` are local.
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root
from common import efd_db  # noqa: E402

#: Focal (field) Zernike order range implied by each DZ-fit column prefix.
PREFIX_K_RANGE = {'z1toz3': (1, 3), 'z1toz6': (1, 6)}

#: Degree-of-freedom and v-mode counts for each scheme name.
SCHEMES = {'50_34': (50, 34), '22_12': (22, 12)}

#: Default output root for the AOS param_set tables, relative to the repo root.
DEFAULT_AOS_OUTPUT = 'aos/output'


def resolve_paths(aos_output, param_set, mi=None):
    """Locate the ``fits.parquet`` and ``visits.parquet`` for a param_set.

    Parameters
    ----------
    aos_output : `pathlib.Path`
        The ``aos/output`` directory.
    param_set : `str`
        Butler collection paired with a processing variant.
    mi : `str`, optional
        Measured Intrinsic Wavefront build name. When given, the DZ fits come from
        ``<param_set>/<mi>/fits.parquet`` — the refit against the MIW — while the visits
        sidecar stays at the param_set level, since the pupil Noll set is a property of the
        donut extraction rather than of the intrinsic.

    Returns
    -------
    fits_path, visits_path : `pathlib.Path`

    Raises
    ------
    FileNotFoundError
        Naming the missing file and the param_set, rather than failing later inside pandas.
    """
    base = pathlib.Path(aos_output) / param_set
    visits_path = base / 'visits.parquet'
    fits_path = (base / mi / 'fits.parquet') if mi else (base / 'fits.parquet')
    for p, what in ((fits_path, 'DZ fits'), (visits_path, 'visits sidecar')):
        if not p.exists():
            raise FileNotFoundError(
                f'{what} not found for param_set {param_set!r}'
                + (f', mi {mi!r}' if mi else '') + f': {p}')
    return fits_path, visits_path


def read_pupil_j(visits_path):
    """Canonical pupil Noll indices from the visits sidecar.

    Parameters
    ----------
    visits_path : `pathlib.Path`
        A ``visits.parquet`` carrying ``nollIndices``.

    Returns
    -------
    pupil_j : `list` [`int`]
        The pupil Noll indices the donut fit actually used.

    Raises
    ------
    RuntimeError
        If the column is absent, since guessing the set would silently corrupt the
        coefficient ordering.
    """
    v = pd.read_parquet(visits_path, columns=['nollIndices'])
    if 'nollIndices' not in v.columns or v.empty:
        raise RuntimeError(f'{visits_path} has no usable nollIndices column; the pupil '
                           f'Noll set must not be guessed')
    return [int(j) for j in np.asarray(v['nollIndices'].iloc[0]).tolist()]


def load_fits(fits_path, prefix, pupil_j, k_min, k_max, programs=None,
              day_obs_range=None):
    """Read the per-pair DZ fit table and the coefficient block.

    Parameters
    ----------
    fits_path : `pathlib.Path`
    prefix : `str`
        DZ-fit column prefix, e.g. ``'z1toz6'``.
    pupil_j : `list` [`int`]
        Canonical pupil Noll indices.
    k_min, k_max : `int`
        Inclusive focal (field) Zernike order range.
    programs : `tuple` [`str`], optional
        Keep only these ``science_program`` values.
    day_obs_range : `tuple`, optional
        ``(first, last)`` inclusive.

    Returns
    -------
    meta : `pandas.DataFrame`
        Identity and quality columns, one row per FAM extra/intra-focal pair.
    coeff : `numpy.ndarray`
        Shape (n, n_kj) in ``kj_grid`` order [µm of wavefront].
    err : `numpy.ndarray`
        Matching formal errors [µm of wavefront].

    Notes
    -----
    The ``kj_grid`` order is ``(k, j)`` with pupil ``j`` fastest within each focal order
    ``k``, matching `ofc_svd.build_ofc_svd`; the columns are read in that order here so the
    stored array and the SVD agree without a second lookup.
    """
    kj = [(k, j) for k in range(k_min, k_max + 1) for j in pupil_j]
    coeff_cols = [f'{prefix}_z{j}_c{k}' for k, j in kj]
    err_cols = [f'{c}_err' for c in coeff_cols]
    meta_cols = ['day_obs', 'seq_num', 'visit', 'science_program', 'band', 'alt', 'az',
                 'rotator_angle', 'bad_fit', 'visit_quality_pass', 'n_donuts_1', 'mjd']
    have = set(pq.ParquetFile(fits_path).schema_arrow.names)
    missing = [c for c in coeff_cols if c not in have]
    if missing:
        raise RuntimeError(
            f'{len(missing)} of {len(coeff_cols)} DZ coefficient columns are absent from '
            f'{fits_path} for prefix {prefix!r}, first missing {missing[0]!r}. The pupil '
            f'Noll set from the visits sidecar does not match this fit table.')
    meta_cols = [c for c in meta_cols if c in have]
    err_present = [c for c in err_cols if c in have]
    df = pd.read_parquet(fits_path,
                         columns=meta_cols + coeff_cols + err_present)
    if programs:
        df = df[df['science_program'].isin(programs)]
    if day_obs_range:
        lo, hi = day_obs_range
        if lo is not None:
            df = df[df['day_obs'] >= int(lo)]
        if hi is not None:
            df = df[df['day_obs'] <= int(hi)]
    df = df.sort_values(['day_obs', 'seq_num']).reset_index(drop=True)
    coeff = df[coeff_cols].to_numpy(float)
    if len(err_present) == len(err_cols):
        err = df[err_cols].to_numpy(float)
    else:
        err = None
    meta = pd.DataFrame({
        'visit_id': df['visit'].astype('int64') if 'visit' in df
        else df['day_obs'].astype('int64') * 100000 + df['seq_num'].astype('int64'),
        'day_obs': df['day_obs'].astype('int64'),
        'seq_num': df['seq_num'].astype('int64'),
    })
    if 'n_donuts_1' in df:
        meta['n_donuts'] = df['n_donuts_1'].astype('int64')
    if 'bad_fit' in df:
        meta['bad_fit'] = df['bad_fit'].astype(bool)
    if 'visit_quality_pass' in df:
        meta['quality_pass'] = df['visit_quality_pass'].astype(bool)
    return meta, coeff, err


def project(coeff, pupil_j, k_min, k_max, n_dof, n_modes):
    """Project DZ coefficients onto the OFC v-modes and DOF.

    Parameters
    ----------
    coeff : `numpy.ndarray`
        Shape (n, n_kj) in ``kj_grid`` order [µm of wavefront].
    pupil_j : `list` [`int`]
    k_min, k_max : `int`
    n_dof, n_modes : `int`
        Optical Feedback Control degrees of freedom and retained v-modes.

    Returns
    -------
    v_modes : `numpy.ndarray`
        Shape (n, n_modes) [dimensionless amplitudes].
    dof : `numpy.ndarray`
        Shape (n, n_dof) [µm and deg].

    Notes
    -----
    The SVD is built inside OFC code via `ofc_svd.build_ofc_svd`, never with a bare
    ``numpy.linalg.svd`` on a sensitivity matrix, and is evaluated at camera rotator angle
    0.0 deg by AOS group decision. The 22-DOF scheme passes the explicit index list
    `aos_state.DOF22`, since a scalar 22 would select DOF 0–21 instead of the intended
    10 rigid-body plus 7 M1M3 plus 5 M2 bending modes.
    """
    from lsst.ts.intrinsic.wavefront import ofc_svd as osv
    dof_spec = n_dof
    if n_dof == 22:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'aos' / 'code'))
        import aos_state
        dof_spec = aos_state.DOF22
    svd = osv.build_ofc_svd(pupil_j, k_min, k_max, n_modes, n_dof=dof_spec)
    print(f'  SVD: U_eff={svd.U_eff.shape}, n_dof={svd.n_dof}, '
          f'n_keep_eff={svd.n_keep_eff}, n_kj={len(svd.kj_grid)}')
    if len(svd.kj_grid) != coeff.shape[1]:
        raise RuntimeError(f'kj_grid has {len(svd.kj_grid)} entries but the coefficient '
                           f'block has {coeff.shape[1]} columns')
    A = svd.project_amplitudes(coeff)
    return svd.vmodes(A), svd.dof(A)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--param-set', help='Butler collection + processing variant')
    ap.add_argument('--mi', default=None,
                    help='Measured Intrinsic Wavefront build name; omit for the batoid '
                         'design intrinsic')
    ap.add_argument('--prefix', default='z1toz6', choices=sorted(PREFIX_K_RANGE),
                    help='DZ-fit column prefix (default z1toz6, focal orders k=1..6)')
    ap.add_argument('--scheme', default='50_34', choices=sorted(SCHEMES),
                    help='DOF / v-mode counts (default 50_34)')
    ap.add_argument('--programs', default=None,
                    help='comma-separated science_program filter; omit for all')
    ap.add_argument('--day-obs', default=None, help='YYYYMMDD or YYYYMMDD-YYYYMMDD')
    ap.add_argument('--aos-output', default=None,
                    help=f'override the {DEFAULT_AOS_OUTPUT} directory')
    ap.add_argument('--db-path', default=None, help='override the database path')
    ap.add_argument('--list', action='store_true',
                    help='list registered FAM variants and exit')
    ap.add_argument('--dry-run', action='store_true',
                    help='project and report, but write nothing')
    args = ap.parse_args()

    if args.list:
        print(efd_db.fam_variants(db_path=args.db_path).to_string(index=False))
        return
    if not args.param_set:
        ap.error('--param-set is required unless --list is given')

    repo = pathlib.Path(__file__).resolve().parents[2]
    aos_output = pathlib.Path(args.aos_output) if args.aos_output \
        else repo / DEFAULT_AOS_OUTPUT
    k_min, k_max = PREFIX_K_RANGE[args.prefix]
    n_dof, n_modes = SCHEMES[args.scheme]
    route = 'miw' if args.mi else 'batoid'
    programs = tuple(s.strip() for s in args.programs.split(',')) \
        if args.programs else None
    day_obs_range = None
    if args.day_obs:
        parts = args.day_obs.split('-')
        day_obs_range = (int(parts[0]), int(parts[-1]))

    fits_path, visits_path = resolve_paths(aos_output, args.param_set, args.mi)
    pupil_j = read_pupil_j(visits_path)
    print(f'param_set {args.param_set}  intrinsic {route}'
          + (f' ({args.mi})' if args.mi else '')
          + f'  prefix {args.prefix} (k={k_min}..{k_max})  scheme {args.scheme}')
    print(f'  fits   {fits_path}')
    print(f'  pupil Noll j from the visits sidecar: {pupil_j} (n={len(pupil_j)})')

    meta, coeff, err = load_fits(fits_path, args.prefix, pupil_j, k_min, k_max,
                                 programs=programs, day_obs_range=day_obs_range)
    print(f'  FAM extra/intra-focal pairs read: {len(meta)}')
    if len(meta) == 0:
        raise RuntimeError('no rows survive the program / day_obs filters')
    n_nan = int(np.isnan(coeff).any(axis=1).sum())
    print(f'  rows with any NaN DZ coefficient: {n_nan} of {len(meta)}')

    v_modes, dof = project(coeff, pupil_j, k_min, k_max, n_dof, n_modes)
    print(f'  v-modes {v_modes.shape}, dof {dof.shape}')

    if args.dry_run:
        print('  --dry-run: nothing written')
        return

    con = efd_db.open_db(args.db_path, readonly=False, create=True)
    try:
        efd_db.create_schema(con)
        fvid = efd_db.register_fam_variant(
            con, args.param_set, route, args.prefix, args.scheme, k_min, k_max, pupil_j,
            n_dof, n_modes, intrinsic_ref=(args.mi or 'ofc_v13'),
            fits_path=str(fits_path))
        n = efd_db.upsert_fam_dz(con, fvid, meta, coeff, v_modes, dof,
                                 dz_coeff_err=err)
        con.commit()
    finally:
        con.close()
    print(f'  wrote {n} rows to fam_dz as variant {fvid}')


if __name__ == '__main__':
    main()
