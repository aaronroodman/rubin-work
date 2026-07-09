"""Reproduce the RubinTV psf_shape_azel panels from our measured PSF-star
moments, to validate our moment calculation against RubinTV.

Layout mirrors rubin-work/aos/code/run_psf_fp_maps.psf_page (psfPlotting style):
  row0: ellipticity whisker | FWHM map | FWHM histogram
  row1: e1 map | e2 map | e histogram
  row2: coma whisker | trefoil markers | kurtosis histogram
Field positions rotated CCS->AzEl by the rotator so the frame matches RubinTV.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import frames

VISITS_PARQUET = '../aos/output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/visits.parquet'
DAY = 20260513
EXTENT, FP_RADIUS = 2.0, 1.75
SCALE_ELLIP, KEY_ELLIP = 0.5, 0.2
SCALE_COMA, KEY_COMA = 0.125, 0.05
TREFOIL_AREA, KEY_TREFOIL = 200.0, 0.1
NSUB = 700   # subsample for whisker/marker panels (declutter dense fields)


def rot_for(seq):
    v = pd.read_parquet(VISITS_PARQUET)
    r = v[(v.day_obs == DAY) & (v.seq_num == seq - 1)]
    return float(r.rotator_angle.iloc[0]) if len(r) else 0.0


def page(df, x, y, title, out):
    fig = plt.figure(figsize=(13, 12))
    gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.9], hspace=0.22, wspace=0.25)

    def setup_map(ax, ttl):
        ax.set_xlim(-EXTENT, EXTENT); ax.set_ylim(-EXTENT, EXTENT)
        ax.set_aspect('equal')
        ax.add_patch(Circle((0, 0), FP_RADIUS, fill=False, ls='--', color='r', lw=0.8))
        ax.set_title(ttl, fontsize=9, pad=3); ax.tick_params(labelsize=7)
        ax.set_xlabel('Δ Az [deg]', fontsize=7); ax.set_ylabel('Δ El [deg]', fontsize=7)

    sub = (np.linspace(0, len(x) - 1, NSUB).astype(int) if len(x) > NSUB
           else np.arange(len(x)))

    def whisker(ax, ttl, ang, amp, scale, key):
        ang, amp = np.asarray(ang)[sub], np.asarray(amp)[sub]
        Q = ax.quiver(x[sub], y[sub], amp * np.cos(ang), amp * np.sin(ang),
                      angles='xy', scale_units='xy', scale=scale, headlength=0,
                      headaxislength=0, width=0.004, pivot='mid', color='k')
        setup_map(ax, ttl)
        ax.quiverkey(Q, 0.14, 0.95, key, f'{key:g}', labelpos='E',
                     coordinates='axes', fontproperties={'size': 7})

    whisker(fig.add_subplot(gs[0, 0]), 'ellipticity',
            0.5 * np.arctan2(df.e2, df.e1), df.e.values, SCALE_ELLIP, KEY_ELLIP)
    ax = fig.add_subplot(gs[0, 1]); setup_map(ax, 'FWHM [arcsec]')
    vlo, vhi = np.nanpercentile(df.fwhm, [2, 98])
    fig.colorbar(ax.scatter(x, y, c=df.fwhm, s=8, cmap='viridis', vmin=vlo, vmax=vhi),
                 ax=ax, shrink=0.85)
    for col, key in [(0, 'e1'), (1, 'e2')]:
        ax = fig.add_subplot(gs[1, col]); setup_map(ax, key)
        v = np.nanpercentile(np.abs(df[key]), 98) or 0.01
        fig.colorbar(ax.scatter(x, y, c=df[key], s=8, cmap='RdBu_r', vmin=-v, vmax=v),
                     ax=ax, shrink=0.85)
    whisker(fig.add_subplot(gs[2, 0]), 'coma',
            np.arctan2(df.coma2, df.coma1), np.hypot(df.coma1, df.coma2),
            SCALE_COMA, KEY_COMA)
    ax = fig.add_subplot(gs[2, 1]); setup_map(ax, 'trefoil')
    tamp = np.hypot(df.trefoil1, df.trefoil2).values[sub]
    tang = (np.degrees(np.arctan2(df.trefoil2, df.trefoil1)) / 3).values[sub]
    for xi, yi, ai, si in zip(x[sub], y[sub], tang, tamp * TREFOIL_AREA):
        ax.scatter(xi, yi, marker=(3, 0, 30 + ai), s=si, color='k', lw=0.1)

    for r, (key, lab, col) in enumerate([('fwhm', 'FWHM [arcsec]', 'steelblue'),
                                         ('e', 'e', 'darkorange'),
                                         ('kurtosis', 'kurtosis', 'firebrick')]):
        ax = fig.add_subplot(gs[r, 2]); v = df[key].values; v = v[np.isfinite(v)]
        ax.hist(v, bins=60, color=col)
        q = np.nanpercentile(v, [25, 50, 75])
        for qq, lw in zip(q, (1, 2, 1)):
            ax.axvline(qq, color='k', lw=lw)
        ax.text(0.97, 0.96, f'{lab}\n25%: {q[0]:.3f}\n50%: {q[1]:.3f}\n75%: {q[2]:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out, dpi=110, bbox_inches='tight'); plt.close(fig)
    print(f'saved {out}')


def main():
    import os
    for seq in [31, 34]:
        pq = f'data/rubintv_{DAY}000{seq}.parquet'
        if not os.path.exists(pq):
            print(f'skip seq {seq} (no {pq})'); continue
        df = pd.read_parquet(pq)
        rot = np.deg2rad(rot_for(seq))
        # rotate CCS field angle -> Az/El frame (sign matched to RubinTV later)
        x, y = frames.rotate_field(df.thx_ccs_deg.values, df.thy_ccs_deg.values, rot)
        print(f'seq {seq}: {len(df)} stars, rotator {np.degrees(rot):.1f} deg')
        for k in ['fwhm', 'e', 'kurtosis']:
            q = np.nanpercentile(df[k].values, [25, 50, 75])
            print(f'  {k:8s} 25/50/75 = {q.round(3)}')
        page(df, x, y, f'20260513 seq={seq} — our PSF-star moments', f'output/rubintv_check_{seq}.png')


if __name__ == '__main__':
    main()
