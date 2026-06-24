#!/usr/bin/env python3
"""study_radialbins — OCS measured intrinsic at the WFS radius, per rotator bin.

Runs after the per-rotator-bin measured-intrinsic grids are built (alongside
intrinsic_split), reading each build/rot_<lo>_<hi>/intrinsic_grid.parquet
*before* the OCS/CCS split.  For each pupil Zernike j it makes one page of four
panels — the four radial shells spanning the corner-WFS region — and within each
panel overlays the N rotator-angle samples (median measured intrinsic vs
focal-plane azimuth, one colour+marker per rotator bin).  The point is to read
off how consistent the measured intrinsic is across rotator angle *at the radius
the wavefront sensors actually see*.

Radial-bin inner edge — the extra-focal corner-WFS inner field radius
---------------------------------------------------------------------
The corner wavefront CCDs are split into an extra-focal half (``SW0``, the inner
half) and an intra-focal half (``SW1``).  For each extra-focal sensor
(``R00_SW0``, ``R04_SW0``, ``R40_SW0``, ``R44_SW0``) the camera's
``PIXELS -> FIELD_ANGLE`` transform maps the four detector bbox corners to field
angles off the boresight; the radius is hypot(field_x, field_y).  Field angle
(not a sky WCS) is used because it *is* the radial field coordinate — a SkyWcs
would give RA/Dec that you'd then have to reference back to the boresight to get
the same number.  By the 4-fold symmetry of the corner rafts every SW0 sensor
gives the same four corner radii (verified on LsstCam, lsst-scipipe-13.0.0):

    corner radii = [1.5178, 1.5979, 1.6855, 1.7580] deg
                    ^inner                    ^outer

so the INNER corner, **1.5178 deg**, is the inner point of the extra-focal WFS.
This is computed at runtime via ``ccd_height.wfs_field_radius_range(LsstCam)``
(cameraGeom, RSP-only); off-RSP it falls back to the cached 1.5178 deg, and it
is overridable via analysis_config ``study_radialbins.inner_deg``.

The outer edge defaults to **1.725 deg** (the limit used in the AOS online
system).  Four equal-width radial bins between them:

    [1.5178, 1.5696]  [1.5696, 1.6214]  [1.6214, 1.6732]  [1.6732, 1.7250] deg

Writes  output/<param_set>/<mi_name>/study_radialbins.pdf .
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mi_config as mc

DEFAULT = dict(inner_deg=None, outer_deg=1.725, n_radial_bins=4,
               azimuth_bin_deg=15.0)
# extra-focal SW0 inner-corner field radius (see module docstring); used when
# cameraGeom is unavailable (off-RSP) or to document the expected value.
WFS_INNER_FALLBACK_DEG = 1.5178


def wfs_inner_radius_deg():
    """Inner field radius (deg) of the extra-focal corner WFS — the minimum
    bbox-corner field angle over the SW0 sensors (see module docstring for the
    derivation).  cameraGeom is RSP-only; falls back to WFS_INNER_FALLBACK_DEG."""
    try:
        from lsst.obs.lsst import LsstCam
        from ccd_height import wfs_field_radius_range
        r_min, r_max = wfs_field_radius_range(LsstCam.getCamera())
        print(f'  WFS field coverage (cameraGeom): inner {r_min:.4f}°, '
              f'outer {r_max:.4f}°')
        return r_min
    except Exception as e:
        print(f'  (cameraGeom unavailable [{type(e).__name__}]; '
              f'inner = {WFS_INNER_FALLBACK_DEG}°)')
        return WFS_INNER_FALLBACK_DEG


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--config', default=None, help='mi_config.yaml path')
    ap.add_argument('--analysis-config', default=None)
    ap.add_argument('--output-root', default='output')
    args = ap.parse_args()

    cfg = mc.load_mi_config(args.param_set, args.mi_name,
                            config_path=(Path(args.config) if args.config else None))
    sec = {**DEFAULT, **mc.analysis_section(
        'study_radialbins', args.param_set, args.mi_name,
        config_path=(Path(args.analysis_config) if args.analysis_config else None))}
    rot_bins = mc.rotator_bins(cfg)
    sel = mc.rotator_select(cfg)
    if sel is not None:
        sel_set = {(round(lo, 3), round(hi, 3)) for lo, hi in sel}
        rot_bins = [(lo, hi) for lo, hi in rot_bins
                    if (round(lo, 3), round(hi, 3)) in sel_set]
    base = Path(args.output_root) / args.param_set / args.mi_name
    # grids come from the build SOURCE entry (build_from -> reuse parent's grids)
    src_mi = mc.build_source(cfg, args.mi_name)
    grid_base = Path(args.output_root) / args.param_set / src_mi
    out = base / 'study_radialbins.pdf'
    print(f'[study_radialbins] {args.param_set}/{args.mi_name}: '
          f'{len(rot_bins)} rotator bins'
          + (f' (grids from {src_mi})' if src_mi != args.mi_name else ''))

    inner = (float(sec['inner_deg']) if sec.get('inner_deg') is not None
             else wfs_inner_radius_deg())
    outer = float(sec['outer_deg']); nrb = int(sec['n_radial_bins'])
    redges = np.linspace(inner, outer, nrb + 1)
    azbin = float(sec['azimuth_bin_deg']); n_az = int(round(360.0 / azbin))
    azedges = np.linspace(0.0, 360.0, n_az + 1)
    azc = 0.5 * (azedges[:-1] + azedges[1:])
    print('  radial bins (deg): '
          + '  '.join(f'[{redges[i]:.4f},{redges[i+1]:.4f}]' for i in range(nrb)))

    # ---- load each rotator-bin OCS intrinsic grid ----
    samples = []
    for lo, hi in rot_bins:
        gp = grid_base / 'build' / f'rot_{lo:g}_{hi:g}' / 'intrinsic_grid.parquet'
        if not gp.exists():
            print(f'  missing {gp} — skipped'); continue
        df = pd.read_parquet(gp)
        thx = np.asarray(df['thx_deg'], float); thy = np.asarray(df['thy_deg'], float)
        zk = np.vstack(df['zk'].values)
        noll = [int(j) for j in np.asarray(df['nollIndices'].iloc[0]).tolist()]
        # optional per-cell error inputs (present once the build records them)
        rms = np.vstack(df['zk_rms'].values) if 'zk_rms' in df.columns else None
        nd = (np.asarray(df['n_donuts'], float) if 'n_donuts' in df.columns else None)
        tab = (np.vstack(df['zk_tabulated'].values)
               if 'zk_tabulated' in df.columns else None)
        samples.append(dict(
            center=0.5 * (lo + hi), label=f'rot{0.5 * (lo + hi):+.0f}',
            r=np.hypot(thx, thy), az=np.degrees(np.arctan2(thy, thx)) % 360.0,
            zk=zk, rms=rms, nd=nd, tab=tab, noll=noll))
    if not samples:
        raise RuntimeError('No rotator-bin intrinsic grids found.')
    noll = samples[0]['noll']
    have_err = all(s['rms'] is not None and s['nd'] is not None for s in samples)
    have_tab = all(s['tab'] is not None for s in samples)
    print(f'  loaded {len(samples)} rotator samples, {len(noll)} Zernikes  '
          f'(error bars: {have_err}, tabulated curve: {have_tab})')
    if not have_err:
        print('  NOTE: zk_rms/n_donuts absent — rebuild build_intrinsic for '
              'error-on-median bars (lines plotted without errors)')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.stats import binned_statistic
    try:
        from common.zernike_names import NOLL_NAMES
    except Exception:
        NOLL_NAMES = {}

    # colour by rotator angle (diverging), distinct marker per sample
    norm = colors.Normalize(vmin=-65.0, vmax=65.0)
    cmap = plt.get_cmap('turbo')
    markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', 'h', '<', '>', 'p']
    n_pages = 0
    with PdfPages(str(out)) as pdf:
        for jj, j in enumerate(noll):
            # 4x1 column: each radial-shell panel spans the full page width
            fig, axes = plt.subplots(nrb, 1, figsize=(15, 3.0 * nrb),
                                     layout='constrained', sharex=True,
                                     sharey=True, squeeze=False)
            axes = axes.ravel()
            for b in range(nrb):
                ax = axes[b]
                for si, s in enumerate(samples):
                    m = ((s['r'] >= redges[b]) & (s['r'] < redges[b + 1])
                         & np.isfinite(s['zk'][:, jj]))
                    if int(m.sum()) < 5:
                        continue
                    med, _, _ = binned_statistic(s['az'][m], s['zk'][m, jj],
                                                 statistic='median', bins=azedges)
                    kw = dict(marker=markers[si % len(markers)], ms=4, lw=0.9,
                              alpha=0.9, color=cmap(norm(s['center'])),
                              label=s['label'] if b == 0 else None)
                    if have_err:
                        # SE of the median ≈ 1.2533·σ_pooled/√N over the donuts in
                        # each azimuth bin: σ_pooled² = Σ n_c·rms_c² / Σ n_c, so
                        # SE_median = 1.2533·√(Σ n_c·rms_c²) / Σ n_c.
                        rc = np.where(np.isfinite(s['rms'][m, jj]), s['rms'][m, jj], 0.0)
                        nc = np.where(np.isfinite(s['nd'][m]), s['nd'][m], 0.0)
                        num, _, _ = binned_statistic(s['az'][m], nc * rc ** 2,
                                                     statistic='sum', bins=azedges)
                        den, _, _ = binned_statistic(s['az'][m], nc,
                                                     statistic='sum', bins=azedges)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            err = 1.2533 * np.sqrt(num) / den
                        err[~np.isfinite(err)] = np.nan
                        ax.errorbar(azc, med, yerr=err, elinewidth=0.7,
                                    capsize=1.5, **kw)
                    else:
                        ax.plot(azc, med, **kw)
                if have_tab:                              # tabulated (batoid) ref
                    at = np.concatenate([s['az'][(s['r'] >= redges[b])
                                         & (s['r'] < redges[b + 1])
                                         & np.isfinite(s['tab'][:, jj])] for s in samples])
                    vt = np.concatenate([s['tab'][(s['r'] >= redges[b])
                                         & (s['r'] < redges[b + 1])
                                         & np.isfinite(s['tab'][:, jj]), jj] for s in samples])
                    if at.size >= 5:
                        tmed, _, _ = binned_statistic(at, vt, statistic='median',
                                                      bins=azedges)
                        ax.plot(azc, tmed, 'k--', lw=1.8, alpha=0.85,
                                label='tabulated' if b == 0 else None)
                ax.axhline(0, color='k', lw=0.4, alpha=0.5); ax.grid(alpha=0.3)
                ax.set_title(f'r ∈ [{redges[b]:.4f}, {redges[b+1]:.4f}]°', fontsize=9)
                ax.set_ylabel(f'Z{j} [μm]', fontsize=8)
                ax.set_xlim(0, 360); ax.set_xticks(range(0, 361, 30))
                ax.tick_params(labelsize=7)
            axes[-1].set_xlabel('focal-plane azimuth [deg] (OCS)', fontsize=9)
            axes[0].legend(fontsize=7, ncol=len(samples) + (1 if have_tab else 0),
                           loc='upper center', title='rotator bin', title_fontsize=7)
            fig.suptitle(f'Z{j} {NOLL_NAMES.get(j, "")} — OCS measured intrinsic '
                         f'vs azimuth, by rotator bin   (WFS radial shells '
                         f'{redges[0]:.3f}–{redges[-1]:.3f}°)', fontsize=12)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig); n_pages += 1
    print(f'  wrote study_radialbins.pdf ({n_pages} pages, '
          f'{len(samples)} rotator samples)')


if __name__ == '__main__':
    main()
