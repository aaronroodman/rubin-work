#!/usr/bin/env python3
"""study_wfs_radial — FAM vs corner-WFS agreement at the WFS radius.

The companion to ``study_radialbins``.  Where that study shows the FAM measured
intrinsic vs azimuth for each rotator bin separately, this one collapses the
rotator dimension: in each radial shell it overlays the **rotator-weighted
average** of the FAM measured intrinsic against the rotator-weighted average of
the corner-WFS map (from ``wfs_build``), so the two independent wavefront
estimates can be checked for agreement in the overlap annulus.

Both averages weight each grid cell by its donut count ``n_donuts``; the
plotted error bar is the pooled SE of the combined estimate,
``1.2533·√(Σ nₖ·rmsₖ²) / Σ nₖ`` (median SE with pooled per-cell variance — the
same convention as ``study_radialbins``).

The WFS series uses the **FAM-corrected** map by default (per-image FAM DZ
field removed, matching what the FAM measured intrinsic isolates); the raw WFS
map is drawn faintly for reference.

Writes  output/<param_set>/<mi_name>/wfs_build/study_wfs_radial.pdf .
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mi_config as mc
from run_study_radialbins import wfs_inner_radius_deg

DEFAULT = dict(inner_deg=None, outer_deg=1.725, n_radial_bins=4,
               azimuth_bin_deg=30.0)


def _load_fam(base_mi, rot_bins, noll):
    """Pool per-rotbin FAM grids into (r, az, zk, rms, nd, tab) point clouds."""
    R, AZ, ZK, RMS, ND, TAB = [], [], [], [], [], []
    for lo, hi in rot_bins:
        gp = base_mi / 'build' / f'rot_{lo:g}_{hi:g}' / 'intrinsic_grid.parquet'
        if not gp.exists():
            continue
        d = pd.read_parquet(gp)
        gnoll = [int(j) for j in np.asarray(d['nollIndices'].iloc[0]).tolist()]
        idx = [gnoll.index(j) for j in noll]
        thx = np.asarray(d['thx_deg'], float); thy = np.asarray(d['thy_deg'], float)
        R.append(np.hypot(thx, thy))
        AZ.append(np.degrees(np.arctan2(thy, thx)) % 360.0)
        ZK.append(np.vstack(d['zk'].values)[:, idx])
        RMS.append(np.vstack(d['zk_rms'].values)[:, idx]
                   if 'zk_rms' in d.columns else np.full((len(d), len(noll)), np.nan))
        ND.append(np.asarray(d['n_donuts'], float)
                  if 'n_donuts' in d.columns else np.ones(len(d)))
        TAB.append(np.vstack(d['zk_tabulated'].values)[:, idx]
                   if 'zk_tabulated' in d.columns else np.full((len(d), len(noll)), np.nan))
    if not R:
        return None
    return dict(r=np.concatenate(R), az=np.concatenate(AZ), zk=np.vstack(ZK),
                rms=np.vstack(RMS), nd=np.concatenate(ND), tab=np.vstack(TAB))


def _load_wfs(grid_path, noll, key):
    """WFS grid as (r, az, zk, rms, nd) using zk_<key> / zk_rms_<key>."""
    d = pd.read_parquet(grid_path)
    gnoll = [int(j) for j in d['nollIndices'].iloc[0]]
    idx = [gnoll.index(j) for j in noll]
    thx = np.asarray(d['thx_deg'], float); thy = np.asarray(d['thy_deg'], float)
    return dict(
        r=np.hypot(thx, thy),
        az=np.degrees(np.arctan2(thy, thx)) % 360.0,
        zk=np.vstack(d[f'zk_{key}'].values)[:, idx],
        rms=np.vstack(d[f'zk_rms_{key}'].values)[:, idx],
        nd=np.asarray(d['n_donuts'], float))


def _wavg(cloud, jj, rlo, rhi, azedges, binned_statistic):
    """Donut-weighted average + pooled-median SE per azimuth bin in a shell."""
    m = ((cloud['r'] >= rlo) & (cloud['r'] < rhi)
         & np.isfinite(cloud['zk'][:, jj]))
    if int(m.sum()) < 3:
        return None, None
    nd = np.where(np.isfinite(cloud['nd'][m]), cloud['nd'][m], 0.0)
    val = cloud['zk'][m, jj]
    num, _, _ = binned_statistic(cloud['az'][m], nd * val, 'sum', bins=azedges)
    den, _, _ = binned_statistic(cloud['az'][m], nd, 'sum', bins=azedges)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean = num / den
    rc = np.where(np.isfinite(cloud['rms'][m, jj]), cloud['rms'][m, jj], 0.0)
    enum, _, _ = binned_statistic(cloud['az'][m], nd * rc ** 2, 'sum', bins=azedges)
    with np.errstate(divide='ignore', invalid='ignore'):
        err = 1.2533 * np.sqrt(enum) / den
    mean[~np.isfinite(mean)] = np.nan
    err[~np.isfinite(err)] = np.nan
    return mean, err


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--analysis-config', default=None)
    ap.add_argument('--output-root', default='output')
    args = ap.parse_args()

    cfg = mc.load_mi_config(args.param_set, args.mi_name,
                            config_path=(Path(args.config) if args.config else None))
    sec = {**DEFAULT, **mc.analysis_section(
        'study_wfs_radial', args.param_set, args.mi_name,
        config_path=(Path(args.analysis_config) if args.analysis_config else None))}
    rot_bins = mc.selected_rotator_bins(cfg)   # rotator_bins filtered by rotator_select
    base_mi = Path(args.output_root) / args.param_set / args.mi_name
    # FAM intrinsic grids come from the build SOURCE entry (build_from reuse)
    grid_base = Path(args.output_root) / args.param_set / mc.build_source(cfg, args.mi_name)
    out = base_mi / 'wfs_build' / 'study_wfs_radial.pdf'

    inner = (float(sec['inner_deg']) if sec.get('inner_deg') is not None
             else wfs_inner_radius_deg())
    outer = float(sec['outer_deg']); nrb = int(sec['n_radial_bins'])
    redges = np.linspace(inner, outer, nrb + 1)
    azbin = float(sec['azimuth_bin_deg']); n_az = int(round(360.0 / azbin))
    azedges = np.linspace(0.0, 360.0, n_az + 1)
    azc = 0.5 * (azedges[:-1] + azedges[1:])
    print(f'[study_wfs_radial] {args.param_set}/{args.mi_name}: '
          f'shells {inner:.4f}-{outer}° x{nrb}')

    # noll comes from a FAM grid; load once to get it, then the clouds
    first = None
    for lo, hi in rot_bins:
        gp = grid_base / 'build' / f'rot_{lo:g}_{hi:g}' / 'intrinsic_grid.parquet'
        if gp.exists():
            first = gp; break
    if first is None:
        raise RuntimeError('No FAM intrinsic grids found.')
    noll = [int(j) for j in pd.read_parquet(first)['nollIndices'].iloc[0]]
    fam = _load_fam(grid_base, rot_bins, noll)
    wfs_c = _load_wfs(base_mi / 'wfs_build' / 'wfs_grid.parquet', noll, 'corr')
    wfs_o = _load_wfs(base_mi / 'wfs_build' / 'wfs_grid.parquet', noll, 'orig')
    have_tab = np.isfinite(fam['tab']).any()
    print(f'  {len(noll)} Zernikes; FAM cells {len(fam["r"])}, '
          f'WFS cells {len(wfs_c["r"])} (tabulated: {have_tab})')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import binned_statistic
    try:
        from common.zernike_names import NOLL_NAMES
    except Exception:
        NOLL_NAMES = {}

    n_pages = 0
    with PdfPages(str(out)) as pdf:
        for jj, j in enumerate(noll):
            fig, axes = plt.subplots(nrb, 1, figsize=(15, 3.0 * nrb),
                                     layout='constrained', sharex=True,
                                     sharey=True, squeeze=False)
            axes = axes.ravel()
            for b in range(nrb):
                ax = axes[b]
                fm, fe = _wavg(fam, jj, redges[b], redges[b + 1], azedges, binned_statistic)
                cm, ce = _wavg(wfs_c, jj, redges[b], redges[b + 1], azedges, binned_statistic)
                om, _ = _wavg(wfs_o, jj, redges[b], redges[b + 1], azedges, binned_statistic)
                if fm is not None:
                    ax.errorbar(azc, fm, yerr=fe, marker='o', ms=5, lw=1.2,
                                color='steelblue', capsize=2, elinewidth=0.8,
                                label='FAM (rot-weighted)' if b == 0 else None)
                if om is not None:
                    ax.plot(azc, om, marker='.', ms=4, lw=0.7, color='grey',
                            alpha=0.6, label='WFS original' if b == 0 else None)
                if cm is not None:
                    ax.errorbar(azc, cm, yerr=ce, marker='s', ms=5, lw=1.2,
                                color='crimson', capsize=2, elinewidth=0.8,
                                label='WFS corrected' if b == 0 else None)
                if have_tab:
                    tmask = ((fam['r'] >= redges[b]) & (fam['r'] < redges[b + 1])
                             & np.isfinite(fam['tab'][:, jj]))
                    if int(tmask.sum()) >= 3:
                        tmed, _, _ = binned_statistic(fam['az'][tmask],
                                                      fam['tab'][tmask, jj],
                                                      'median', bins=azedges)
                        ax.plot(azc, tmed, 'k--', lw=1.6, alpha=0.8,
                                label='tabulated' if b == 0 else None)
                ax.axhline(0, color='k', lw=0.4, alpha=0.5); ax.grid(alpha=0.3)
                ax.set_title(f'r ∈ [{redges[b]:.4f}, {redges[b+1]:.4f}]°', fontsize=9)
                ax.set_ylabel(f'Z{j} [μm]', fontsize=8)
                ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 30))
                ax.tick_params(labelsize=7)
            axes[-1].set_xlabel('focal-plane azimuth [deg] (OCS)', fontsize=9)
            axes[0].legend(fontsize=8, ncol=4, loc='upper center')
            fig.suptitle(f'Z{j} {NOLL_NAMES.get(j, "")} — FAM vs corner-WFS, '
                         f'rotator-weighted average vs azimuth   '
                         f'(WFS shells {redges[0]:.3f}–{redges[-1]:.3f}°)',
                         fontsize=12)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig); n_pages += 1
    print(f'  wrote wfs_build/study_wfs_radial.pdf ({n_pages} pages)')


if __name__ == '__main__':
    main()
