"""Double Zernike focal-plane fitting for Rubin AOS wavefront data.

Fits focal-plane Noll Zernike polynomials (Z1-Z3 or Z1-Z6) to per-image
donut wavefront residuals using robust regression (Huber M-estimator).

Can be used as a library or via the companion CLI script run_dz_fit.py.
"""

import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm
from astropy.table import QTable, join
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ============================================================
# Noll index utilities
# ============================================================

def derive_noll_indices(nZk, noll_indices_arr=None):
    """Derive Noll Zernike indices and index mapping.

    Parameters
    ----------
    nZk : int
        Number of Zernike terms in the data array.
    noll_indices_arr : array-like, optional
        Explicit Noll indices (e.g. from visit_info nollIndices column).
        If None, inferred from nZk assuming contiguous from Z4.

    Returns
    -------
    iZs : list of int
        Noll indices (e.g. [4, 5, 6, ..., 22, 23, 24, 25, 26]).
    iZidx : dict
        Mapping from Noll index to column position in zk arrays.
    """
    if noll_indices_arr is not None:
        iZs = [int(n) for n in noll_indices_arr]
        if len(iZs) != nZk:
            print(f"WARNING: nollIndices length ({len(iZs)}) != zk array width ({nZk})")
            print(f"  nollIndices: {iZs}")
            print(f"  Falling back to contiguous range")
            iZs = list(range(4, 4 + nZk))
    else:
        if nZk == 19:
            iZs = list(range(4, 23))
        else:
            iZs = list(range(4, 4 + nZk))

    iZidx = {iZ: i for i, iZ in enumerate(iZs)}
    return iZs, iZidx


# ============================================================
# Focal-plane Zernike basis
# ============================================================

def focal_plane_zernike_basis(thx_deg, thy_deg, max_noll, fp_radius=1.75):
    """Build focal-plane Noll Zernike basis matrix.

    Coordinates are normalized to the focal-plane radius so that the
    Zernike polynomials are evaluated on a unit disk.  All basis functions
    are dimensionless, so fit coefficients have the same units as the data (μm).

    Parameters
    ----------
    thx_deg, thy_deg : ndarray
        Field angles in degrees.
    max_noll : int
        Maximum Noll index (1-6 supported).
    fp_radius : float
        Focal plane radius in degrees for normalization (default 1.75).

    Returns
    -------
    A : ndarray, shape (n_points, max_noll)
        Design matrix with one column per focal Zernike term.
    labels : list of str
        Labels for each column (e.g. 'Z1_piston', 'Z2_tilt', ...).
    """
    x = thx_deg / fp_radius
    y = thy_deg / fp_radius
    r2 = x**2 + y**2

    cols = []
    labels = []

    if max_noll >= 1:
        cols.append(np.ones_like(x))
        labels.append('Z1_piston')
    if max_noll >= 2:
        cols.append(2.0 * x)
        labels.append('Z2_tilt')
    if max_noll >= 3:
        cols.append(2.0 * y)
        labels.append('Z3_tip')
    if max_noll >= 4:
        cols.append(np.sqrt(3) * (2.0 * r2 - 1.0))
        labels.append('Z4_defocus')
    if max_noll >= 5:
        cols.append(2.0 * np.sqrt(6) * x * y)
        labels.append('Z5_astig45')
    if max_noll >= 6:
        cols.append(np.sqrt(6) * (x**2 - y**2))
        labels.append('Z6_astig0')

    return np.column_stack(cols), labels


# ============================================================
# Core fitting function
# ============================================================

def fit_focal_zernikes(day_obs_arr, seq_num_arr, thx_deg, thy_deg,
                       zk_data, zk_intrinsic, iZs,
                       max_focal_noll=3, include_intrinsic=True,
                       fp_radius=1.75, prefix='z1toz3'):
    """Fit focal-plane Noll Zernikes to per-image wavefront residuals.

    For each image (unique day_obs, seq_num) and each pupil Zernike iZ, fits:
        residual = k1*Zfocal_1 + k2*Zfocal_2 + ... + kN*Zfocal_N
    where residual = zk_data - zk_intrinsic (if include_intrinsic) or zk_data.

    Uses robust regression (Huber M-estimator) with fallback to least squares.

    Parameters
    ----------
    day_obs_arr : ndarray of int
        Day observation IDs per donut.
    seq_num_arr : ndarray of int
        Sequence numbers per donut.
    thx_deg, thy_deg : ndarray
        Field angles in degrees per donut.
    zk_data : ndarray, shape (n_donuts, n_zernikes)
        Measured Zernike values in μm.
    zk_intrinsic : ndarray, shape (n_donuts, n_zernikes)
        Intrinsic model Zernike values in μm.
    iZs : list of int
        Noll indices corresponding to columns of zk_data/zk_intrinsic.
    max_focal_noll : int
        Maximum focal Noll index for fit (default 3).
    include_intrinsic : bool
        If True, subtract intrinsic before fitting (default True).
    fp_radius : float
        Focal plane radius in degrees (default 1.75).
    prefix : str
        Column name prefix for output (e.g. 'z1toz3').

    Returns
    -------
    fit_rows : list of dict
        One dict per image with fit parameters.
    zk_fit_vals : ndarray, shape (n_donuts, n_zernikes)
        Per-donut fitted values.
    zk_rlm_weights : ndarray, shape (n_donuts, n_zernikes)
        Per-donut RLM weights.
    """
    images = sorted(set(zip(day_obs_arr.tolist(), seq_num_arr.tolist())))
    n_donuts = len(day_obs_arr)
    n_zernikes = len(iZs)

    zk_fit_vals = np.zeros((n_donuts, n_zernikes))
    zk_rlm_weights = np.ones((n_donuts, n_zernikes))
    fit_rows = []

    for img_idx, (dobs, snum) in enumerate(images):
        mask = (day_obs_arr == dobs) & (seq_num_arr == snum)
        img_params, fit_vals_i, weights_i = _fit_one_image(
            thx_deg[mask], thy_deg[mask],
            zk_data[mask], zk_intrinsic[mask],
            iZs, max_focal_noll, include_intrinsic, fp_radius, prefix,
            dobs, snum, img_idx)
        zk_fit_vals[mask] = fit_vals_i
        zk_rlm_weights[mask] = weights_i
        fit_rows.append(img_params)

    print(f"Fit '{prefix}' (focal Noll 1-{max_focal_noll}): "
          f"{len(images)} images, {n_donuts} donuts, "
          f"include_intrinsic={include_intrinsic}")

    return fit_rows, zk_fit_vals, zk_rlm_weights


def _fit_one_image(thx_deg, thy_deg, zk_data, zk_intrinsic, iZs,
                   max_focal_noll, include_intrinsic, fp_radius, prefix,
                   dobs, snum, img_idx):
    """Fit one image's donuts. Returns (img_params, fit_vals, rlm_weights)."""
    A, _ = focal_plane_zernike_basis(thx_deg, thy_deg, max_focal_noll, fp_radius)
    n_pts = len(thx_deg)
    n_zernikes = len(iZs)
    n_coeffs = max_focal_noll

    fit_vals = np.zeros((n_pts, n_zernikes))
    rlm_weights = np.ones((n_pts, n_zernikes))
    img_params = {'day_obs': int(dobs), 'seq_num': int(snum),
                  'image_idx': int(img_idx), 'n_donuts': int(n_pts)}

    for j_idx, iZ in enumerate(iZs):
        if include_intrinsic:
            resid = zk_data[:, j_idx] - zk_intrinsic[:, j_idx]
        else:
            resid = zk_data[:, j_idx].copy()

        try:
            rlm_model = sm.RLM(resid, A, M=sm.robust.norms.HuberT())
            rlm_results = rlm_model.fit()
            coeffs = rlm_results.params
            bse = rlm_results.bse
            scale = float(rlm_results.scale)
            weights = rlm_results.weights
        except Exception:
            coeffs, _, _, _ = np.linalg.lstsq(A, resid, rcond=None)
            bse = np.full(n_coeffs, np.nan)
            scale = float(np.std(resid - A @ coeffs))
            weights = np.ones(n_pts)

        for ci in range(n_coeffs):
            img_params[f'{prefix}_z{iZ}_c{ci+1}'] = float(coeffs[ci])
            img_params[f'{prefix}_z{iZ}_c{ci+1}_err'] = float(bse[ci])
        img_params[f'{prefix}_z{iZ}_scale'] = scale

        fit_vals[:, j_idx] = A @ coeffs
        rlm_weights[:, j_idx] = weights

    return img_params, fit_vals, rlm_weights


def fit_focal_zernikes_streaming(input_file, visit_info, coord_sys, iZs,
                                 max_focal_noll=3, include_intrinsic=True,
                                 fp_radius=1.75, prefix='z1toz3'):
    """Streaming variant: read donuts one row group (= one visit) at a time.

    Reads the donuts parquet file written by stream_zernikes_to_parquet,
    where each visit is stored as a single row group. Per-visit reads
    use row-group stats to avoid scanning the whole file.

    Returns fit_rows (list of dicts) — zk_fit_vals and zk_rlm_weights
    are not accumulated (they weren't used downstream anyway).
    """
    pf = pq.ParquetFile(str(input_file))
    fit_rows = []
    total_donuts = 0

    thx_col = f'thx_{coord_sys}'
    thy_col = f'thy_{coord_sys}'
    zk_col = f'zk_{coord_sys}'
    zk_intr_col = f'zk_intrinsic_{coord_sys}'

    # Build a (day_obs, seq_num) -> row_group_idx lookup from row-group stats
    rg_index = {}
    for i in range(pf.num_row_groups):
        meta = pf.metadata.row_group(i)
        d = s = None
        for ci in range(meta.num_columns):
            cmeta = meta.column(ci)
            name = cmeta.path_in_schema
            if name == 'day_obs' and cmeta.statistics is not None:
                d = cmeta.statistics.min
            elif name == 'seq_num' and cmeta.statistics is not None:
                s = cmeta.statistics.min
        if d is not None and s is not None:
            rg_index[(int(d), int(s))] = i

    for img_idx, v in enumerate(visit_info):
        dobs = int(v['day_obs'])
        snum = int(v['seq_num'])
        rg_i = rg_index.get((dobs, snum))
        if rg_i is None:
            continue

        df = pf.read_row_group(rg_i).to_pandas()
        if len(df) == 0:
            continue

        thx_deg = np.rad2deg(df[thx_col].to_numpy(dtype=float))
        thy_deg = np.rad2deg(df[thy_col].to_numpy(dtype=float))
        zk_data = np.stack(df[zk_col].values)
        zk_intrinsic = np.stack(df[zk_intr_col].values)

        img_params, _, _ = _fit_one_image(
            thx_deg, thy_deg, zk_data, zk_intrinsic, iZs,
            max_focal_noll, include_intrinsic, fp_radius, prefix,
            dobs, snum, img_idx)
        fit_rows.append(img_params)
        total_donuts += len(df)

    print(f"Fit '{prefix}' (focal Noll 1-{max_focal_noll}): "
          f"{len(fit_rows)} images, {total_donuts} donuts (streamed), "
          f"include_intrinsic={include_intrinsic}")

    return fit_rows


# ============================================================
# Bad-fit flagging
# ============================================================

def flag_bad_fits(fit_table, prefix, threshold=2.0, min_donuts=200):
    """Flag visits with bad fits based on coefficient magnitude and donut count.

    Parameters
    ----------
    fit_table : QTable
        Fit parameter table (one row per image).
    prefix : str
        Fit prefix (e.g. 'z1toz3').
    threshold : float
        Maximum allowed |coefficient| in μm (default 2.0).
    min_donuts : int
        Minimum donuts required for a valid fit (default 200).

    Returns
    -------
    bad_mask : ndarray of bool
        True for bad-fit rows.
    """
    coeff_cols = [c for c in fit_table.colnames if c.startswith(f'{prefix}_z')
                  and '_c' in c and not c.endswith('_err') and not c.endswith('_scale')]
    coeff_arr = np.column_stack([np.array(fit_table[c]) for c in coeff_cols])
    bad_coeff = np.any(np.abs(coeff_arr) > threshold, axis=1)
    bad_ndonuts = np.array(fit_table['n_donuts']) < min_donuts
    bad_mask = bad_coeff | bad_ndonuts

    n_bad = np.sum(bad_mask)
    n_bad_coeff = np.sum(bad_coeff & ~bad_ndonuts)
    n_bad_ndonuts = np.sum(bad_ndonuts)
    print(f"{prefix}: {n_bad}/{len(fit_table)} visits flagged as bad_fit")
    print(f"  {n_bad_coeff} with |coeff| > {threshold} μm, "
          f"{n_bad_ndonuts} with n_donuts < {min_donuts}")

    if n_bad > 0:
        for i in range(len(fit_table)):
            if not bad_mask[i]:
                continue
            row = fit_table[i]
            reasons = []
            if row['n_donuts'] < min_donuts:
                reasons.append(f"n_donuts={row['n_donuts']}")
            for c in coeff_cols:
                if abs(row[c]) > threshold:
                    reasons.append(f"{c}={row[c]:.3f}")
            print(f"  day_obs={row['day_obs']} seq_num={row['seq_num']}  "
                  + ', '.join(reasons))

    return bad_mask


# ============================================================
# High-level pipeline
# ============================================================

def run_double_zernike_fits(input_file, coord_sys='OCS',
                            output_file=None, bad_fit_threshold=2.0,
                            min_donuts=200):
    """Run the full Double Zernike fitting pipeline.

    Loads input HDF5 (donuts + visits tables), derives Noll indices,
    validates data, runs z1toz3 and z1toz6 fits, flags bad fits, merges
    with visit_info, and saves output.

    Parameters
    ----------
    input_file : str or Path
        Path to HDF5 file containing 'donuts' and 'visits' tables
        (from intrinsics_mktable).
    coord_sys : str
        Coordinate system: 'OCS' or 'CCS'.
    output_file : str or Path, optional
        Output parquet path. If None, derived as {stem}_fits.parquet.
    bad_fit_threshold : float
        Flag fits with |coefficient| > this (μm). Default 2.0.
    min_donuts : int
        Flag fits with fewer donuts than this. Default 200.

    Returns
    -------
    fit_merged : QTable
        Combined fit table with both z1toz3 and z1toz6 results.
    """
    input_file = Path(input_file)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Detect format by suffix: .parquet (new, streaming) or .hdf5 (legacy)
    is_parquet = input_file.suffix == '.parquet'

    if is_parquet:
        visits_file = input_file.parent / f'{input_file.stem}_visits.parquet'
        if output_file is None:
            output_file = input_file.parent / f'{input_file.stem}_fits.parquet'
        print(f"Loading: {input_file} (parquet, one row group per visit)")
        visit_info = QTable.read(str(visits_file))
        print(f"  {len(visit_info)} visits")
    else:
        # Legacy HDF5: donuts and visits in same file
        if output_file is None:
            output_file = input_file.parent / f'{input_file.stem}_fits.parquet'
        print(f"Loading: {input_file} (legacy HDF5)")
        visit_info = QTable.read(str(input_file), path='visits')
        print(f"  {len(visit_info)} visits")

    # Derive Noll indices
    noll_arr = None
    if 'nollIndices' in visit_info.colnames:
        noll_arr = np.array(visit_info['nollIndices'][0])

    if is_parquet:
        # Probe one row group to determine nZk
        pf = pq.ParquetFile(str(input_file))
        df0 = pf.read_row_group(0).to_pandas()
        zk_sample = np.stack(df0[f'zk_{coord_sys}'].values)
        nZk = zk_sample.shape[1]
        iZs, iZidx = derive_noll_indices(nZk, noll_arr)
        print(f"  Noll indices ({len(iZs)} terms): {iZs}")
        del df0, zk_sample, pf

        # Fit via per-visit row-group reads
        rows_z3 = fit_focal_zernikes_streaming(
            input_file, visit_info, coord_sys, iZs,
            max_focal_noll=3, prefix='z1toz3')
        rows_z6 = fit_focal_zernikes_streaming(
            input_file, visit_info, coord_sys, iZs,
            max_focal_noll=6, prefix='z1toz6')
    else:
        # Legacy path: load the whole donuts table via astropy
        aosTable = QTable.read(str(input_file), path='donuts')
        print(f"  {len(aosTable)} donuts, {len(aosTable.columns)} columns")

        zk_data = np.stack(aosTable[f'zk_{coord_sys}'])
        zk_intrinsic = np.stack(aosTable[f'zk_intrinsic_{coord_sys}'])
        nZk = zk_data.shape[1]
        iZs, iZidx = derive_noll_indices(nZk, noll_arr)
        print(f"  Noll indices ({len(iZs)} terms): {iZs}")

        # Validate zk = residual + intrinsic (legacy-only sanity check)
        resid_col = f'zk_residual_{coord_sys}'
        if resid_col in aosTable.colnames:
            zk_resid = np.stack(aosTable[resid_col])
            diff = zk_data - (zk_resid + zk_intrinsic)
            max_abs_diff = np.max(np.abs(diff))
            print(f"  Validation: max |zk - (residual + intrinsic)| = "
                  f"{max_abs_diff:.2e} μm",
                  "PASSED" if max_abs_diff <= 0.01 else "WARNING")

        day_obs_arr = np.array(aosTable['day_obs'])
        seq_num_arr = np.array(aosTable['seq_num'])
        thx_deg = np.rad2deg(np.array(aosTable[f'thx_{coord_sys}']))
        thy_deg = np.rad2deg(np.array(aosTable[f'thy_{coord_sys}']))

        rows_z3, _, _ = fit_focal_zernikes(
            day_obs_arr, seq_num_arr, thx_deg, thy_deg,
            zk_data, zk_intrinsic, iZs,
            max_focal_noll=3, prefix='z1toz3')
        rows_z6, _, _ = fit_focal_zernikes(
            day_obs_arr, seq_num_arr, thx_deg, thy_deg,
            zk_data, zk_intrinsic, iZs,
            max_focal_noll=6, prefix='z1toz6')

    fit_table_z3 = QTable(rows_z3)
    fit_table_z6 = QTable(rows_z6)

    # Flag bad fits
    bad_z3 = flag_bad_fits(fit_table_z3, 'z1toz3', bad_fit_threshold, min_donuts)
    fit_table_z3['z1toz3_bad_fit'] = bad_z3
    bad_z6 = flag_bad_fits(fit_table_z6, 'z1toz6', bad_fit_threshold, min_donuts)
    fit_table_z6['z1toz6_bad_fit'] = bad_z6

    # Combine into single table
    fit_combined = fit_table_z3.copy()
    for col in fit_table_z6.colnames:
        if col.startswith('z1toz6_'):
            fit_combined[col] = fit_table_z6[col]
    fit_combined['bad_fit'] = bad_z3 | bad_z6
    n_bad = np.sum(fit_combined['bad_fit'])
    print(f"\nCombined: {n_bad}/{len(fit_combined)} visits flagged as bad_fit")

    # Merge with visit_info
    fit_merged = join(fit_combined, visit_info,
                      keys=['day_obs', 'seq_num'], join_type='left')
    print(f"Merged with visit_info: {len(fit_merged)} rows, "
          f"{len(fit_merged.columns)} columns")

    # Save
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fit_merged.write(str(output_file), format='parquet', overwrite=True)
    print(f"\nSaved: {output_file}")
    print(f"  {len(fit_merged)} rows x {len(fit_merged.columns)} columns")

    return fit_merged
