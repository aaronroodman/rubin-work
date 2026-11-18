"""One page per edge rule (R0-R3) showing its rim points on an extra and an intra donut.

For two sides (by default both sides of star S4848323907389759872 on 20261107), from an
`edge_rules` run: per rule a page with, per side, the data stamp at full resolution and the
inner and outer bands unwrapped about the fit-a center (data over the bin's flat top, 50 %
contour in orange). Data rim points white, fit c model rim points cyan, bins cut by the
per-edge spider cut shaded. A last page per side marks every rule's data radius on the
radial profiles of six bins.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/edge_rules_compare.py \\
        --rules 20261118_edge_rules_20261118_night2_20261107_v402s
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import edge_review as erv  # noqa: E402
import edge_rules as er  # noqa: E402
import psi_gallery as pg  # noqa: E402
import psi_labels as pl  # noqa: E402
import robust_edges as rob  # noqa: E402

KEYS = ('388_S4848323907389759872:extra', '366_S4848323907389759872:intra')
RULE_NAMES = {'R0': 'R0: 50 % of the 0.70-0.90 r_out flat top',
              'R1': 'R1: 50 % of the local plateau 12-25 pixels inside the edge',
              'R2': 'R2: steepest slope (profile smoothed over 2 pixels)',
              'R3': 'R3: blurred step fitted to the dark flank (plateau fixed to R1)'}
COLOURS = {'R0': 'tab:orange', 'R1': 'tab:green', 'R2': 'tab:purple', 'R3': 'tab:blue'}
PROFILE_BINS_DEG = (1.0, 61.0, 91.0, 181.0, 251.0, 331.0)


def load_side(run, fit_dir, key, bins):
    c, _, mask_run = pl._load(run, fit_dir)
    pl._init(*map(str, mask_run))
    row = c[c.key == key].iloc[0]
    img, msk = pg.stamp(pl.STUDY_OUT / run, key)
    model = pl.fit_model(row, pl.mask_for(row), er.FWHM_ARCSEC)
    g = bins[bins.key == key].sort_values('bin').reset_index(drop=True)
    return row, img, msk, model, g


def _stamp(ax, row, img, msk, g, rule):
    cx, cy = row.cx_a, row.cy_a
    vmax = np.nanpercentile(img[~msk], 99.0)
    ax.imshow(np.where(msk, np.nan, img), origin='lower', cmap='gray', vmin=0, vmax=vmax,
              interpolation='none')
    a = np.radians(g.az_ccs_deg.to_numpy())
    for e in ('inner', 'outer'):
        for src, col, ms in (('model', 'cyan', 1.6), ('data', 'w', 1.2)):
            r = g[f'r_{e}_{rule}_{src}'].to_numpy()
            ax.plot(cx + r * np.cos(a), cy + r * np.sin(a), '.', color=col, ms=ms)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{row.key}\nfield {row.field_r_deg:.3f} deg, OCS az {row.field_az_ocs_deg:.0f} '
                 f'deg, rtp {row.rtp_deg:.2f} deg', fontsize=8)


def _band(ax, row, img, msk, g, rule, edge):
    cx, cy, r_out = row.cx_a, row.cy_a, row.r_out_pix
    prof = rob.profiles(img, ~msk, cx, cy, r_out, float(row.noise_e))
    top = prof[2]
    azu, ru, U = erv.unwrap(np.where(msk, np.nan, img), cx, cy, 1.15 * r_out)
    kb = np.minimum((azu / 2.0).astype(int), rob.NBIN - 1)
    Un = U / top[kb][:, None]
    if edge == 'inner':
        sel = (ru > np.nanmin(g[f'r_{edge}_R0_model']) - 15) & \
              (ru < np.nanmax(g[f'r_{edge}_R0_model']) + 20)
    else:
        sel = (ru > np.nanmin(g[f'r_{edge}_R0_model']) - 20) & \
              (ru < np.nanmax(g[f'r_{edge}_R0_model']) + 15)
    ax.imshow(Un[:, sel].T, origin='lower', aspect='auto', cmap='gray', vmin=0, vmax=1.3,
              extent=(0, 360, ru[sel][0], ru[sel][-1]), interpolation='none')
    ax.contour(azu, ru[sel], Un[:, sel].T, levels=[0.5], colors='tab:orange', linewidths=0.5)
    cut = g[f'cut_{edge}'].to_numpy(bool)
    for k in np.nonzero(cut)[0]:
        ax.axvspan(g.az_ccs_deg[k] - 1, g.az_ccs_deg[k] + 1, color='yellow', alpha=0.15, lw=0)
    az = g.az_ccs_deg.to_numpy()
    ax.plot(az, g[f'r_{edge}_{rule}_model'], '.', color='cyan', ms=2.5, label='fit c model')
    ax.plot(az, g[f'r_{edge}_{rule}_data'], '.', color='w', ms=2.0, label='data')
    d = g[f'r_{edge}_{rule}_data'] - g[f'r_{edge}_{rule}_model']
    ax.set_title(f'{edge} rim, {rule}: median data − model {np.nanmedian(d):+.2f} pixels, '
                 f'{int(d.notna().sum())} bins\nnMAD of the data about its harmonic fit '
                 f'{g[f"nmad_{edge}_{rule}_data"].iloc[0]:.2f} pixels', fontsize=8)
    ax.set_ylabel('radius (pixels)', fontsize=8)
    ax.set_xlim(0, 360)


def page_rule(pdf, sides, rule):
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.6, 1.6])
    for i, (row, img, msk, model, g) in enumerate(sides):
        _stamp(fig.add_subplot(gs[i, 0]), row, img, msk, g, rule)
        for j, edge in enumerate(('inner', 'outer')):
            ax = fig.add_subplot(gs[i, 1 + j])
            _band(ax, row, img, msk, g, rule, edge)
            if i == 1:
                ax.set_xlabel('CCS azimuth about the donut center (deg)', fontsize=8)
    fig.suptitle(f'{RULE_NAMES[rule]}\nwhite: data rim points; cyan: fit c model rim points; '
                 'bands: data / bin flat top, orange 50 % contour, yellow: bins cut for that '
                 'edge (per-edge spider cut, vignetting, masks)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


def page_profiles(pdf, side):
    row, img, msk, model, g = side
    cx, cy, r_out, noise = row.cx_a, row.cy_a, row.r_out_pix, float(row.noise_e)
    prof, rad, top, *_ = rob.profiles(img, ~msk, cx, cy, r_out, noise)
    mprof = rob.profiles(model, ~msk, cx, cy, r_out, noise)[0]
    fig, axes = plt.subplots(len(PROFILE_BINS_DEG), 2, figsize=(12, 3 * len(PROFILE_BINS_DEG)))
    for i, azd in enumerate(PROFILE_BINS_DEG):
        k = int(azd / 2.0)
        for j, e in enumerate(('inner', 'outer')):
            ax = axes[i, j]
            ctr = g[f'r_{e}_R0_model'][k]
            if not np.isfinite(ctr):
                ctr = np.nanmedian(g[f'r_{e}_R0_model'])
            lo, hi = (ctr - 15, ctr + 25) if e == 'inner' else (ctr - 25, ctr + 15)
            w = (rad > lo) & (rad < hi)
            ax.plot(rad[w], prof[k][w], '.', color='k', ms=3)
            ax.plot(rad[w], rob._smooth(prof[k])[w], '-', color='k', lw=0.8, label='data')
            ax.plot(rad[w], rob._smooth(mprof[k])[w], '-', color='0.6', lw=0.8,
                    label='fit c model')
            ax.axhline(top[k], color='0.5', lw=0.5, ls=':')
            ax.axhline(0.5 * top[k], color='0.5', lw=0.5, ls='--')
            p = g[f'plateau_{e}_data'][k]
            if np.isfinite(p):
                ax.axhline(p, color=COLOURS['R1'], lw=0.5, ls=':')
            txt = []
            for rule in er.RULES:
                v = g[f'r_{e}_{rule}_data'][k]
                if np.isfinite(v):
                    ax.axvline(v, color=COLOURS[rule], lw=1.0, label=rule)
                    txt.append(f'{rule} {v:.2f}')
            cut = bool(g[f'cut_{e}'][k])
            ax.set_title(f'bin {k} (CCS az {g.az_ccs_deg[k]:.0f} deg), {e}: '
                         + (', '.join(txt) if txt else 'no edge') + ' pixels'
                         + ('; cut' if cut else ''), fontsize=7)
            ax.set_xlabel('radius (pixels)', fontsize=7)
            ax.set_ylabel('electrons per pixel', fontsize=7)
            if i == 0 and j == 0:
                ax.legend(fontsize=6, ncol=3)
    fig.suptitle(f'{row.key}: data rim radius per rule on the radial profiles; dotted grey: '
                 'flat top, dashed grey: 50 % of it, dotted green: R1 plateau', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--rules', required=True, help='edge_rules run directory under output/')
    p.add_argument('--key', action='append', default=None, help='two keys, extra then intra')
    a = p.parse_args()
    out = pl.STUDY_OUT / a.rules
    src = dict(l.split(' ', 1) for l in (out / 'source.txt').read_text().splitlines())
    bins = pd.read_parquet(out / 'bins.parquet')
    keys = a.key or KEYS
    sides = [load_side(src['run'], src['fit_dir'], k, bins) for k in keys]
    star = keys[0].split(':')[0].split('_', 1)[-1]
    path = out / f'edge_rules_compare_{star}.pdf'
    with PdfPages(path) as pdf:
        for rule in er.RULES:
            page_rule(pdf, sides, rule)
        for s in sides:
            page_profiles(pdf, s)
    print('wrote', path, flush=True)


if __name__ == '__main__':
    main()
