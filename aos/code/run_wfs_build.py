#!/usr/bin/env python3
"""wfs_build — original and FAM-corrected corner-WFS wavefront maps.

The in-focus corner-WFS donuts (built by ``run_wfs_mktable.py`` into
``output/<ps>/wfs/donuts.parquet``) sample the wavefront in a thin annulus at
the edge of the science field, *outside* the FAM donut radius.  This step
projects them onto the same focal-plane grid as the FAM measured intrinsic so
the two can be checked for continuity across the r ~ 1.5° boundary.

Each WFS estimate is entered at BOTH of its half-chip field positions
(``thx/thy_<coord>_intra`` and ``..._extra``) with the same ``zk_<coord>``
vector — the cwfs reduction yields one wavefront per corner raft split across
the SW0/SW1 sensors.

Two maps are built (median / RMS / N per grid cell):

* **original**  — the raw WFS ``zk`` binned on the FAM grid.
* **corrected** — ``zk`` minus the per-image FAM double-Zernike field
  reconstruction, evaluated at the WFS field position.  The FAM field model is
  the k = 1..6 fit stored per visit in ``fits.parquet``
  (``z1toz6_z{j}_c{k}``), evaluated with the same focal-plane Zernike basis
  (``fp_radius = 1.75``) used to fit it.  Subtracting it removes the per-visit
  aberration drift, leaving the static wavefront the FAM measured-intrinsic map
  also isolates — so the corrected WFS annulus should join continuously onto
  the FAM grid.

Outputs (per param_set / mi_name):
    output/<ps>/<mi>/wfs_build/wfs_grid.parquet      binned WFS annulus
    output/<ps>/<mi>/wfs_build/wfs_continuity.pdf    FAM + WFS maps per Zernike
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mi_config as mc
from measured_intrinsic import bin_median_focal, bin_count_rms_focal
from dz_fitting import focal_plane_zernike_basis

FP_RADIUS_GRID = 1.8       # focal grid extent (matches FAM build)
FP_RADIUS_BASIS = 1.75     # field normalization of the DZ fit
N_BINS = 73
MAX_K = 6                  # k = 1..6 field modes stored in fits.parquet


def _coeff_lookup(fits_path, noll, prefix='z1toz6', max_k=MAX_K):
    """Per-visit DZ coefficients: {(day_obs, seq_num): (n_j, max_k) array}."""
    df = pd.read_parquet(fits_path)
    lut = {}
    for r in df.itertuples():
        C = np.full((len(noll), max_k), np.nan)
        for ji, j in enumerate(noll):
            for k in range(1, max_k + 1):
                C[ji, k - 1] = getattr(r, f'{prefix}_z{j}_c{k}', np.nan)
        lut[(int(r.day_obs), int(r.seq_num))] = C
    return lut


def _fam_dz_at(thx_deg, thy_deg, coeffs, max_k=MAX_K):
    """FAM DZ field at field points: A(n_pts, max_k) @ coeffs(max_k, n_j)."""
    A, _ = focal_plane_zernike_basis(thx_deg, thy_deg, max_k, FP_RADIUS_BASIS)
    return A @ coeffs.T          # (n_pts, n_j)


def _grid_to_long(grid, rms, count, xcent, ycent, noll):
    """Flatten {j -> 2D} median/rms grids + count to a long cell table."""
    XX, YY = np.meshgrid(xcent, ycent, indexing='ij')
    n = count.astype(int)
    keep = n > 0
    rows = dict(thx_deg=XX[keep], thy_deg=YY[keep], n_donuts=n[keep])
    zk = np.stack([grid[j][keep] for j in noll], axis=1)        # (n_cell, n_j)
    rk = np.stack([rms[j][keep] for j in noll], axis=1)
    return rows, zk, rk


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--coord-sys', default='OCS', choices=['OCS', 'CCS'])
    ap.add_argument('--config', default=None)
    ap.add_argument('--output-root', default='output')
    args = ap.parse_args()
    coord = args.coord_sys

    cfg = mc.load_mi_config(args.param_set, args.mi_name,
                            config_path=(Path(args.config) if args.config else None))
    rot_bins = mc.selected_rotator_bins(cfg)   # rotator_bins filtered by rotator_select
    base_ps = Path(args.output_root) / args.param_set
    base_mi = base_ps / args.mi_name
    # FAM intrinsic grids come from the build SOURCE entry (build_from reuse)
    grid_base = base_ps / mc.build_source(cfg, args.mi_name)
    out = base_mi / 'wfs_build'
    out.mkdir(parents=True, exist_ok=True)

    # ---- WFS donuts ----
    from astropy.table import QTable
    donuts = QTable.read(str(base_ps / 'wfs' / 'donuts.parquet'))
    noll = [int(x) for x in donuts.meta.get('nollIndices')]
    iZidx = {j: i for i, j in enumerate(noll)}
    zk = np.array(donuts[f'zk_{coord}'], dtype=float)            # (n, n_j)
    dobs = np.asarray(donuts['day_obs']).astype(int)
    fam_s = np.asarray(donuts['fam_seq_num']).astype(int)
    seq = np.asarray(donuts['seq_num']).astype(int)             # in-focus seq
    print(f'[wfs_build] {args.param_set}/{args.mi_name}: {len(zk)} WFS donuts, '
          f'{len(noll)} Zernikes')

    # Both half-chip field positions, same zk entered at each.
    tx_i = np.rad2deg(np.asarray(donuts[f'thx_{coord}_intra'], float))
    ty_i = np.rad2deg(np.asarray(donuts[f'thy_{coord}_intra'], float))
    tx_e = np.rad2deg(np.asarray(donuts[f'thx_{coord}_extra'], float))
    ty_e = np.rad2deg(np.asarray(donuts[f'thy_{coord}_extra'], float))
    thx = np.concatenate([tx_i, tx_e])
    thy = np.concatenate([ty_i, ty_e])
    zk2 = np.concatenate([zk, zk], axis=0)
    seq2 = np.concatenate([seq, seq])                          # for FAM key

    # ---- FAM per-image DZ field, evaluated at each WFS position ----
    # fits.parquet is keyed by the in-focus seq's *FAM* partner; the FAM key is
    # the extra-focal seq == fam_seq_num.  We join on (day_obs, fam_seq_num).
    lut = _coeff_lookup(base_ps / 'fits.parquet', noll)
    fam_key2 = np.concatenate([fam_s, fam_s])
    dobs2 = np.concatenate([dobs, dobs])
    dz = np.full_like(zk2, np.nan)
    n_fit = 0
    n_partial = 0          # A2: visits whose FAM coeffs had some (zero-filled) NaN
    for (d, s), C in lut.items():
        m = (dobs2 == d) & (fam_key2 == s)
        if not m.any() or not np.isfinite(C).any():
            continue
        if not np.isfinite(C).all():
            n_partial += 1
        dz[m] = _fam_dz_at(thx[m], thy[m], np.nan_to_num(C))   # NaN coeff -> 0
        n_fit += int(m.sum())
    zk_corr = zk2 - dz
    print(f'  FAM DZ field applied to {n_fit}/{len(zk2)} WFS positions '
          f'({len(lut)} fitted visits; {n_partial} with some zero-filled NaN coeff)')

    # ---- bin both maps on the FAM grid ----
    g_orig, xb, yb, xc, yc = bin_median_focal(thx, thy, zk2, iZidx,
                                              n_bins=N_BINS, fp_radius=FP_RADIUS_GRID)
    cnt, rms_orig, *_ = bin_count_rms_focal(thx, thy, zk2, iZidx,
                                            n_bins=N_BINS, fp_radius=FP_RADIUS_GRID)
    g_corr, *_ = bin_median_focal(thx, thy, zk_corr, iZidx,
                                  n_bins=N_BINS, fp_radius=FP_RADIUS_GRID)
    _, rms_corr, *_ = bin_count_rms_focal(thx, thy, zk_corr, iZidx,
                                          n_bins=N_BINS, fp_radius=FP_RADIUS_GRID)

    rows, zk_o, rk_o = _grid_to_long(g_orig, rms_orig, cnt, xc, yc, noll)
    _, zk_c, rk_c = _grid_to_long(g_corr, rms_corr, cnt, xc, yc, noll)
    df = pd.DataFrame(dict(thx_deg=rows['thx_deg'], thy_deg=rows['thy_deg'],
                           n_donuts=rows['n_donuts']))
    df['zk_orig'] = list(zk_o); df['zk_corr'] = list(zk_c)
    df['zk_rms_orig'] = list(rk_o); df['zk_rms_corr'] = list(rk_c)
    df['nollIndices'] = [list(noll)] * len(df)
    df.to_parquet(out / 'wfs_grid.parquet')
    print(f'  wrote wfs_build/wfs_grid.parquet ({len(df)} WFS cells)')

    # ---- FAM intrinsic reference cloud (all rotator bins pooled) ----
    fam_pts = _load_fam_grid(grid_base, rot_bins, noll)
    _continuity_pdf(out / 'wfs_continuity.pdf', noll, fam_pts,
                    rows, zk_o, zk_c)
    print('  wrote wfs_build/wfs_continuity.pdf')


def _load_fam_grid(base_mi, rot_bins, noll):
    """Pool the per-rotbin FAM measured-intrinsic grids into one point cloud."""
    txs, tys, zks = [], [], []
    for lo, hi in rot_bins:
        gp = base_mi / 'build' / f'rot_{lo:g}_{hi:g}' / 'intrinsic_grid.parquet'
        if not gp.exists():
            continue
        d = pd.read_parquet(gp)
        gnoll = [int(j) for j in np.asarray(d['nollIndices'].iloc[0]).tolist()]
        z = np.vstack(d['zk'].values)
        idx = [gnoll.index(j) if j in gnoll else -1 for j in noll]
        zsel = np.full((len(d), len(noll)), np.nan)
        for c, ii in enumerate(idx):
            if ii >= 0:
                zsel[:, c] = z[:, ii]
        txs.append(np.asarray(d['thx_deg'], float))
        tys.append(np.asarray(d['thy_deg'], float))
        zks.append(zsel)
    if not txs:
        return None
    return dict(thx=np.concatenate(txs), thy=np.concatenate(tys),
                zk=np.vstack(zks))


def _continuity_pdf(path, noll, fam, rows, zk_orig, zk_corr):
    """One page per Zernike: FAM grid + WFS annulus (original | corrected),
    shared color scale, so a continuous wavefront shows no seam at r~1.5°."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        from common.zernike_names import NOLL_NAMES
    except Exception:
        NOLL_NAMES = {}
    wx, wy = rows['thx_deg'], rows['thy_deg']
    with PdfPages(str(path)) as pdf:
        for p, j in enumerate(noll):
            fig, axes = plt.subplots(1, 2, figsize=(11, 5.2),
                                     layout='constrained')
            wfs_cols = {'original': zk_orig[:, p], 'corrected': zk_corr[:, p]}
            fz = fam['zk'][:, p] if fam is not None else None
            # shared scale from the union of BOTH WFS panels + FAM (robust pctl),
            # so the 'original' panel isn't saturated by corrected-only limits
            allv = np.concatenate([wfs_cols['original'], wfs_cols['corrected']]
                                  + ([fz] if fz is not None else []))
            finite = allv[np.isfinite(allv)]
            vlo, vhi = (np.nanpercentile(finite, [2, 98])
                        if finite.size else (-1, 1))
            for ax, (lab, wv) in zip(axes, wfs_cols.items()):
                if fz is not None:
                    ax.scatter(fam['thx'], fam['thy'], c=fz, s=14, marker='s',
                               vmin=vlo, vmax=vhi, cmap='RdBu_r')
                sc = ax.scatter(wx, wy, c=wv, s=26, marker='o', vmin=vlo,
                                vmax=vhi, cmap='RdBu_r', edgecolors='k',
                                linewidths=0.3)
                ax.set_aspect('equal'); ax.set_title(f'WFS {lab}', fontsize=10)
                ax.set_xlabel('thx [deg]'); ax.set_ylabel('thy [deg]')
                fig.colorbar(sc, ax=ax, shrink=0.8, label='Zernike [μm]')
            fig.suptitle(f'Z{j} {NOLL_NAMES.get(j, "")}  —  FAM grid (squares) + '
                         f'WFS annulus (circles)', fontsize=13)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    main()
