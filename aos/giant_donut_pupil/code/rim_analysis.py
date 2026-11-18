"""Rim analysis: R0 edges and rim contrast of the far-field sample, data against the models.

Edge rule R0 (50 % of the 0.70-0.90 r_out flat top) with the per-edge spider cut, from
`edge_rules` runs of night 1 (pass 6, ``isr_phys_v402s``), 20261107 and 20261108 (v402s).
Bins are grouped by the mask element that sets each edge in that bin (`psi_labels`), not by
psi range. Per night, side of focus, edge and element:

- data − fit c model edge (pixels), median over bins, and the side-to-side scatter;
- rim contrast (peak of the smoothed profile within 10 pixels on the bright side of the
  edge, over the 12-25 pixel plateau; dimensionless) of the data and of the model.

With an `edge_rules_batoid` run, for its sides (same keys, labels from the data runs): the
R0 edge shift Batoid v4.03 − v4.02 (the M1 roll-off's expected share of data − model) and
the rim contrast of both Batoid images.

Pages: the element table; per side of focus and edge, data − model and rim contrast
against psi (median per 2 deg bin, one colour per night) with the Batoid v4.03 − v4.02
shift and rim contrast.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/rim_analysis.py \\
        --rules 20261118_edge_rules_20261023_pupil6_isr_phys_v402s \\
        --rules 20261118_edge_rules_20261118_night2_20261107_v402s \\
        --rules 20261118_edge_rules_20261118_night2_20261108_v402s \\
        --batoid 20261118_edge_rules_batoid
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
import psi_edges_report as per  # noqa: E402
import psi_labels as pl  # noqa: E402

STUDY = 'aos/giant_donut_pupil'
NIGHT_OF = {'20261023_pupil6': 'night 1 (20251023)', '20261118_night2_20261107': '20261107',
            '20261118_night2_20261108': '20261108'}
NIGHT_COLOURS = {'night 1 (20251023)': 'k', '20261107': 'tab:orange', '20261108': 'tab:blue'}
MIN_BINS = 20


def nmad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def load_rules(dirs):
    out = []
    for d in dirs:
        d = pl.STUDY_OUT / d
        src = dict(l.split(' ', 1) for l in (d / 'source.txt').read_text().splitlines())
        b = pd.read_parquet(d / 'bins.parquet')
        b['night'] = NIGHT_OF.get(src['run'], src['run'])
        out.append(b)
    b = pd.concat(out, ignore_index=True)
    b['psi_deg'] = np.mod(b.az_ocs_deg - (b.field_az_ocs_deg + 180.0), 360.0)
    b['pbin'] = np.minimum((b.psi_deg / per.BIN_DEG).astype(int), per.NBIN - 1)
    for e in ('inner', 'outer'):
        b[f'{e}_dmm'] = b[f'r_{e}_R0_data'] - b[f'r_{e}_R0_model']
    return b


def in_sample(b):
    """Science sides at 1.6-1.725 deg and corner sides at 1.4-1.75 deg."""
    sci = (b.sensor_class == 'science') & b.field_r_deg.between(1.6, 1.725)
    cor = (b.sensor_class != 'science') & b.field_r_deg.between(1.4, 1.75)
    return b[sci | cor]


def element_table(b, bat=None):
    rows = []
    for (night, cls, side), g in b.groupby(['night', 'sensor_class', 'side']):
        for e in ('inner', 'outer'):
            for el, h in g.groupby(f'element_{e}'):
                h = h[h[f'{e}_dmm'].notna()]
                if len(h) < MIN_BINS:
                    continue
                per_side = h.groupby('key')[f'{e}_dmm'].median()
                r = dict(night=night, sensor_class=cls, side=side, edge=e,
                         element=per.NAMES.get(el, el), n_sides=len(per_side), n_bins=len(h),
                         dmm_median_pix=float(h[f'{e}_dmm'].median()),
                         dmm_side_nmad_pix=nmad(per_side),
                         rim_data=float(h[f'rim_{e}_data'].median()),
                         rim_model=float(h[f'rim_{e}_model'].median()))
                if bat is not None:
                    k = bat[(bat.side == side) & (bat[f'element_{e}'] == el)]
                    s = (k[f'r_{e}_R0_v403'] - k[f'r_{e}_R0_v402']).dropna()
                    r.update(batoid_n_bins=len(s),
                             roll_shift_pix=float(s.median()) if len(s) else np.nan,
                             rim_v402=float(k[f'rim_{e}_v402'].median()) if len(k) else np.nan,
                             rim_v403=float(k[f'rim_{e}_v403'].median()) if len(k) else np.nan)
                rows.append(r)
    return pd.DataFrame(rows)


def load_batoid(d, b):
    """Batoid bins with the element labels of the same sides' data bins."""
    t = pd.read_parquet(pl.STUDY_OUT / d / 'bins.parquet')
    lab = b[['key', 'bin', 'element_inner', 'element_outer']]
    t = t.merge(lab, on=['key', 'bin'], how='left')
    t['psi_deg'] = np.mod(t.az_ocs_deg - (t.field_az_ocs_deg + 180.0), 360.0)
    t['pbin'] = np.minimum((t.psi_deg / per.BIN_DEG).astype(int), per.NBIN - 1)
    return t


def page_table(pdf, t, cls):
    t = t[t.sensor_class == cls].drop(columns='sensor_class')
    if not len(t):
        return
    show = t.copy()
    for c in show.columns:
        if show[c].dtype.kind == 'f':
            show[c] = show[c].map(lambda v: '' if not np.isfinite(v) else f'{v:+.2f}'
                                  if 'pix' in c else f'{v:.3f}')
    fig, ax = plt.subplots(figsize=(17, 0.8 + 0.24 * (len(show) + 2)))
    ax.axis('off')
    tb = ax.table(cellText=show.to_numpy(), colLabels=list(show.columns), loc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(6.5)
    ax.set_title(f'{cls} sides: R0 data − fit c model edge (pixels), median over bins set by the '
                 'element, and the nMAD of the per-side medians; rim contrast (dimensionless) of '
                 'data, fit c model and Batoid v4.02 / v4.03; roll_shift: Batoid R0 edge '
                 'v4.03 − v4.02 (pixels)', fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def page_psi(pdf, b, bat, side):
    """Data − model and rim contrast against psi, science sides, one side of focus."""
    g = b[(b.sensor_class == 'science') & (b.side == side)]
    if not len(g):
        return
    centers = (np.arange(per.NBIN) + 0.5) * per.BIN_DEG
    fig, axes = plt.subplots(4, 1, figsize=(11, 14), sharex=True)
    for k, e in enumerate(('inner', 'outer')):
        a1, a2 = axes[2 * k], axes[2 * k + 1]
        for night, h in g.groupby('night'):
            col = NIGHT_COLOURS.get(night, '0.5')
            n = h.key.nunique()
            m = h.groupby('pbin')[f'{e}_dmm'].median().reindex(range(per.NBIN))
            a1.plot(centers, m, '.', ms=3, color=col, label=f'{night}, {n} sides')
            r = h.groupby('pbin')[f'rim_{e}_data'].median().reindex(range(per.NBIN))
            a2.plot(centers, r, '.', ms=3, color=col, label=f'data {night}')
        rm = g.groupby('pbin')[f'rim_{e}_model'].median().reindex(range(per.NBIN))
        a2.plot(centers, rm, '-', color='tab:red', lw=1, label='fit c model (all nights)')
        if bat is not None:
            q = bat[bat.side == side]
            if len(q):
                s = (q[f'r_{e}_R0_v403'] - q[f'r_{e}_R0_v402'])
                sm = s.groupby(q.pbin).median().reindex(range(per.NBIN))
                a1.plot(centers, sm, '-', color='tab:green', lw=1.2,
                        label=f'Batoid v4.03 − v4.02, {q.key.nunique()} sides')
                for v, col in (('v402', 'tab:purple'), ('v403', 'tab:green')):
                    rr = q.groupby('pbin')[f'rim_{e}_{v}'].median().reindex(range(per.NBIN))
                    a2.plot(centers, rr, '--', color=col, lw=1, label=f'Batoid {v}')
        for el, lo, hi, _ in per.ranges(per.consensus(g, e)):
            a1.axvspan(lo, hi, color=per.COLOURS.get(el, '0.9'), alpha=0.12, lw=0)
            a1.text(0.5 * (lo + hi), 0.98, per.NAMES.get(el, el).split(' (')[0], fontsize=6,
                    ha='center', va='top', transform=a1.get_xaxis_transform())
        a1.axhline(0, color='0.5', lw=0.5)
        a1.set_ylabel(f'{e} edge, data − model\n(pixels)', fontsize=8)
        a2.set_ylabel(f'{e} rim contrast\n(dimensionless)', fontsize=8)
        for a in (a1, a2):
            a.grid(alpha=0.3)
            a.legend(fontsize=6, ncol=3)
    axes[-1].set_xlabel('psi (deg; 0 toward the field center)')
    fig.suptitle(f'science {side}-focal sides, 1.6-1.725 deg: R0 edges with the per-edge spider '
                 'cut, median per 2 deg psi bin; shading: consensus defining element', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig)
    plt.close(fig)


def main():
    from rubinwork.common import study_run
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--rules', action='append', required=True)
    p.add_argument('--batoid', default=None)
    p.add_argument('--out', default=None, help='output directory; a new study run by default')
    a = p.parse_args()
    b = in_sample(load_rules(a.rules))
    bat = load_batoid(a.batoid, b) if a.batoid else None
    out = pathlib.Path(a.out) if a.out else study_run(STUDY, 'rim_analysis').path
    out.mkdir(parents=True, exist_ok=True)
    t = element_table(b, bat)
    t.to_csv(out / 'rim_elements.csv', index=False, float_format='%.4g')
    with PdfPages(out / 'rim_analysis.pdf') as pdf:
        for cls in ('science', 'SW0', 'SW1'):
            page_table(pdf, t, cls)
        for side in ('extra', 'intra'):
            page_psi(pdf, b, bat, side)
    (out / 'source.txt').write_text('rules ' + ' '.join(a.rules) + f'\nbatoid {a.batoid}\n')
    print('wrote', out / 'rim_analysis.pdf', flush=True)


if __name__ == '__main__':
    main()
