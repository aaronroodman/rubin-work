#!/usr/bin/env python3
"""Build the per-donut measured-intrinsic sidecar — Phase 2 step 3 (new).

For every donut in output/<param_set>/donuts.parquet, evaluate the measured
intrinsic Zernike vector from the OCS/CCS spin decomposition (step 2) plus the
per-donut CCD-height Z4, and write a **row-aligned** sidecar

    output/<param_set>/<mi_name>/zk_intrinsic.parquet

with one row per donut (same order as donuts.parquet), a list column
``zk_intrinsic_MI`` (ordered by ``nollIndices``), and key columns
(day_obs, seq_num, detector, centroid_x_intra, centroid_y_intra) so the refit
(step 4) can join robustly.

Reconstruction (per donut, at its visit's rotator angle theta) uses
intrinsic_split.reconstruct_at — the exact inverse of the decomposition:
``A(theta) = O_pol + exp(i*n_spin*s*theta)*roll(C_pol, shift)`` — so the
camera-fixed (CCS) component, including the azimuthal spin phase for
astig/coma/trefoil, is rotated back into OCS before interpolating at the
donut's OCS field position.  Z4 uses the height-removed optical intrinsic
(from step 2's z4_optical) plus the per-donut camera-height Z4.  RSP-only.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from scipy.spatial import Delaunay

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import intrinsic_split as isp
import mi_config as mc


def barycentric_weights(tri, pts):
    """Per-point simplex index + 3 barycentric weights on triangulation ``tri``.
    Points outside the hull get simplex -1 (-> NaN interpolation)."""
    simp = tri.find_simplex(pts)
    T = tri.transform[simp]
    b01 = np.einsum('nij,nj->ni', T[:, :2, :], pts - T[:, 2, :])
    bary = np.column_stack([b01, 1.0 - b01.sum(axis=1)])
    verts = tri.simplices[simp]
    return simp, verts, bary


def interp_field(field_ravel, simp, verts, bary):
    """Linear-interpolate a raveled field at the precomputed barycentric pts."""
    out = np.where(simp >= 0,
                   (field_ravel[verts] * bary).sum(axis=1), np.nan)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--coord-sys', default='OCS')
    args = ap.parse_args()

    cfg = mc.load_mi_config(args.param_set, args.mi_name,
                            config_path=(Path(args.config) if args.config else None))
    b = cfg.get('build', {})
    coord = cfg.get('coord_sys', args.coord_sys)
    base = Path(args.output_root) / args.param_set
    mi_dir = base / args.mi_name

    # ---- decomposition (step 2) ----
    dz = np.load(mi_dir / 'intrinsic_split_decomp.npz')
    A = dz['A']; X = dz['X']; Y = dz['Y']
    jvals = dz['jvals'].tolist()
    O_pol, C_pol = dz['O_pol'], dz['C_pol']
    n_spin, s_arr, part = dz['n_spin'], dz['s'], dz['part']
    nZk = len(jvals)
    j4_col = jvals.index(4) if 4 in jvals else None
    print(f'[sidecar] {args.param_set}/{args.mi_name}: {nZk} Zernikes {jvals}')

    # ---- donut light columns (file order) ----
    cols = ['day_obs', 'seq_num', 'detector',
            f'thx_{coord}', f'thy_{coord}',
            'centroid_x_intra', 'centroid_y_intra',
            'centroid_x_extra', 'centroid_y_extra']
    dd = pq.read_table(str(base / 'donuts.parquet'), columns=cols).to_pandas()
    N = len(dd)
    print(f'  donuts: {N}')

    # ---- rotator angle per visit (deg) from visits.parquet ----
    vt = pq.read_table(str(base / 'visits.parquet'),
                       columns=['day_obs', 'seq_num', 'rotator_angle']).to_pandas()
    rot_lut = {(int(r.day_obs), int(r.seq_num)): float(r.rotator_angle)
               for r in vt.itertuples()}

    # ---- per-donut camera-height Z4 (rotation-independent), computed once ----
    z4_height = np.full(N, np.nan)
    if j4_col is not None:
        try:
            from lsst.obs.lsst import LsstCam
            from ccd_height import compute_ccd_heights, HEIGHT_TO_Z4_UM_PER_MM
            fac = b.get('height_to_z4_factor') or HEIGHT_TO_Z4_UM_PER_MM
            hc = compute_ccd_heights(
                dd, LsstCam.getCamera(),
                source=b.get('height_source', 'batoid_rubin'),
                height_map_dir=b.get('batoid_rubin_height_map_dir'),
                metrology_fits=b.get('height_map_fits'), factor=fac)
            z4_height = np.asarray(hc['Z4_height'], dtype=float)
            print(f'  Z4_height: mean={np.nanmean(z4_height):+.4f} μm')
        except Exception as e:
            print(f'  Z4 height skipped: {type(e).__name__}: {e}')

    # ---- precompute interpolation weights (donut OCS field deg) once ----
    thx_deg = np.rad2deg(np.asarray(dd[f'thx_{coord}'], dtype=float))
    thy_deg = np.rad2deg(np.asarray(dd[f'thy_{coord}'], dtype=float))
    tri = Delaunay(np.column_stack([X.ravel(), Y.ravel()]))
    simp, verts, bary = barycentric_weights(tri, np.column_stack([thx_deg, thy_deg]))

    # ---- reconstruct per visit (fixed theta), fill row-aligned zk array ----
    zk_int = np.full((N, nZk), np.nan)
    dobs = np.asarray(dd['day_obs']).astype(int)
    snum = np.asarray(dd['seq_num']).astype(int)
    order = np.lexsort((snum, dobs))
    # group contiguous (dobs, snum)
    keys = list(zip(dobs[order], snum[order]))
    n_missing_rot = 0
    start = 0
    for i in range(1, len(keys) + 1):
        if i == len(keys) or keys[i] != keys[start]:
            idx = order[start:i]
            d, snv = keys[start]
            theta = rot_lut.get((d, snv))
            if theta is not None and np.isfinite(theta):
                th = np.deg2rad(theta)
                for ij in range(nZk):
                    dec = {'O_pol': O_pol[ij], 'C_pol': C_pol[ij],
                           'n_spin': int(n_spin[ij]), 's': int(s_arr[ij])}
                    recon = isp.reconstruct_at(dec, th, A)
                    fld = recon.real if part[ij] == 0 else recon.imag
                    zk_int[idx, ij] = interp_field(
                        np.ascontiguousarray(fld).ravel(),
                        simp[idx], verts[idx], bary[idx])
            else:
                n_missing_rot += len(idx)
            start = i
    if j4_col is not None:
        # add CCD-height Z4 back, but don't let a missing (out-of-domain) height
        # NaN-poison an otherwise-finite optical Z4
        finite_h = np.isfinite(z4_height)
        zk_int[:, j4_col] = np.where(finite_h, zk_int[:, j4_col] + z4_height,
                                     zk_int[:, j4_col])
        n_missing_h = int((~finite_h).sum())
        if n_missing_h:
            print(f'  CCD-height Z4 missing for {n_missing_h} donuts '
                  f'(Z4 left optical-only there)')
    n_ok = int(np.isfinite(zk_int).all(axis=1).sum())
    print(f'  reconstructed {n_ok}/{N} donuts with a finite intrinsic '
          f'({n_missing_rot} donuts missing rotator)')

    # ---- write row-aligned sidecar ----
    out_tbl = pa.table({
        'day_obs': pa.array(dobs), 'seq_num': pa.array(snum),
        'detector': pa.array(dd['detector'].astype(str)),
        'centroid_x_intra': pa.array(np.asarray(dd['centroid_x_intra'], dtype=float)),
        'centroid_y_intra': pa.array(np.asarray(dd['centroid_y_intra'], dtype=float)),
        'zk_intrinsic_MI': pa.array(list(zk_int)),
    })
    meta = {b'nollIndices': np.array(jvals, dtype=int).tobytes(),
            b'mi_name': args.mi_name.encode(), b'coord_sys': coord.encode(),
            b'n_rows': str(N).encode()}
    out_tbl = out_tbl.replace_schema_metadata(meta)
    mi_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_tbl, str(mi_dir / 'zk_intrinsic.parquet'), compression='snappy')
    print(f'  wrote zk_intrinsic.parquet ({N} rows, '
          f'zk_intrinsic_MI[{nZk}] ordered by {jvals})')


if __name__ == '__main__':
    main()
