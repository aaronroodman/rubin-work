"""Data and model donut edges stacked in relative azimuth psi, with the mask element per edge.

The pass 4 step 2 page (`pupil_pass4_report.page_psi`), for any run, plus the danish mask
element that sets each model edge (`psi_labels.py`), computed per donut at its own field
and rotator angle for the mask its fit used.

psi = (OCS image azimuth - (phi + 180 deg)) mod 360, phi the side's OCS field azimuth, so
psi 0 is the point of the donut nearest the field center (pass 4).

Per group of sides (sensor class, side of focus, field band):

- page 1: inner edge, outer edge and r_inner / r_outer against psi (dots data, lines the
  fit c model, one colour per donut); under each edge panel the defining element per 2 deg
  psi bin, the commonest over donuts (opacity: fraction of donuts that agree; hatched:
  median margin under `TIGHT_PIX`); the median data - model; the median margin, the
  distance the edge moves when its element is removed (pixels);
- page 2: per element and edge, the psi ranges it sets, the median data - model there and
  the median margin.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/psi_edges_report.py --run 20261021_pupil4 \\
        --labels 20261117_psi_labels_20261021_pupil4
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
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import psi_labels as pl  # noqa: E402

NBIN = 180
BIN_DEG = 360 / NBIN
TIGHT_PIX = 1.0
GROUPS = [(('science', 'intra'), 'science intra, 1.6-1.725 deg'),
          (('science', 'extra'), 'science extra, 1.6-1.725 deg'),
          (('SW0', 'intra'), 'SW0 intra, 1.4-1.75 deg'), (('SW0', 'extra'), 'SW0 extra, 1.4-1.75 deg'),
          (('SW1', 'intra'), 'SW1 intra, 1.4-1.75 deg'), (('SW1', 'extra'), 'SW1 extra, 1.4-1.75 deg')]
COLOURS = {'M1Baffle1 outer': 'tab:cyan', 'M1 inner': 'tab:blue', 'M2 outer': 'tab:green',
           'M2 inner': 'darkgreen', 'M3 outer': 'tab:olive', 'M3 inner': 'goldenrod',
           'CameraSilhouette outer': 'tab:red', 'CameraSilhouette outer_s': 'salmon',
           'Filter_entrance outer': 'tab:purple',
           'L3_entrance outer': 'tab:brown', 'L3_exit outer': 'tab:pink', pl.TIE: '0.75'}
# Names as on the page: the mask circle, and whether it is a clear aperture or an obscuration.
NAMES = {'M1Baffle1 outer': 'M1Baffle1 (clear)', 'M1 inner': 'M1 inner (obsc.)',
         'M2 outer': 'M2 outer (clear)', 'M2 inner': 'M2 inner (obsc.)',
         'M3 outer': 'M3 outer (clear)', 'M3 inner': 'M3 inner (obsc.)',
         'CameraSilhouette outer': 'CameraSilhouette (obsc.)',
         'CameraSilhouette outer_s': 'CameraSilhouette, v4.02s refit (obsc.)',
         'Filter_entrance outer': 'Filter entrance (clear)',
         'L3_entrance outer': 'L3 entrance (clear)', 'L3_exit outer': 'L3 exit (clear)',
         pl.TIE: 'tie (< 0.3 pixels)'}
# v4_speed circles: the v4.02 colour where the element is the same, then a cycle.
_SAME = {'M1Baffle1 inner': 'M1Baffle1 outer', 'CameraBody outer': 'CameraSilhouette outer'}
_CYCLE = iter(plt.get_cmap('tab20').colors[1::2] + plt.get_cmap('Dark2').colors)
for _label, _clear in pl._V4S.items():
    if _label in _SAME:
        COLOURS[_label] = COLOURS[_SAME[_label]]
        NAMES[_label] = NAMES[_SAME[_label]].replace('CameraSilhouette', 'CameraBody')
    else:
        if _label not in COLOURS:
            COLOURS[_label] = next(_CYCLE)
        NAMES.setdefault(_label, f'{_label} ({"clear" if _clear else "obsc."})')


def load(run, fit_dir, label_runs):
    """Bins of the labelled sides with psi, data, model and element columns."""
    c, b, _ = pl._load(run, fit_dir)
    lab = pd.concat([pd.read_parquet(pl.STUDY_OUT / r / 'labels.parquet') for r in label_runs],
                    ignore_index=True).drop_duplicates(['key', 'side', 'bin'])
    lab = pl.consistent_kind(lab)
    keep = ['key', 'side', 'bin'] + [f'{w}_{e}' for w in ('element', 'second') for e in
                                     ('inner', 'outer')] + \
        [f'{w}margin_{e}_pix' for w in ('', 'second_') for e in ('inner', 'outer')]
    b = b.merge(lab[keep], on=['key', 'side', 'bin'])
    geo = c[['key', 'side', 'sensor_class', 'field_r_deg', 'field_az_ocs_deg']]
    b = b.merge(geo, on=['key', 'side'])
    b['psi_deg'] = np.mod(b.az_ocs_deg - (b.field_az_ocs_deg + 180.0), 360.0)
    b['pbin'] = np.minimum((b.psi_deg / BIN_DEG).astype(int), NBIN - 1)
    for e in ('inner', 'outer'):
        b[f'{e}_dmm'] = b[f'r_{e}_data_pix'] - b[f'r_{e}_c_pix']
        b.loc[b.spider, f'element_{e}'] = None
    b['ratio_dmm'] = b.r_inner_data_pix / b.r_outer_data_pix - b.r_inner_c_pix / b.r_outer_c_pix
    return b


def group_of(b, group):
    cls, side = group
    g = b[(b.sensor_class == cls) & (b.side == side)]
    band = (1.6, 1.725) if cls == 'science' else (1.4, 1.75)
    return g[g.field_r_deg.between(*band)]


def consensus(g, edge):
    """Per psi bin: commonest element, fraction of donuts agreeing, median margin (pixels)."""
    col = f'element_{edge}'
    rows = []
    for k, h in g.groupby('pbin'):
        v = h[col].dropna()
        if v.empty:
            continue
        counts = v.value_counts()
        rows.append(dict(pbin=k, psi_deg=(k + 0.5) * BIN_DEG, element=counts.index[0],
                         agree=counts.iloc[0] / len(v), n=len(v),
                         margin_pix=float(h.loc[h[col] == counts.index[0],
                                                f'margin_{edge}_pix'].median()),
                         dmm_pix=float(h[f'{edge}_dmm'].median())))
    return pd.DataFrame(rows)


def ranges(cons):
    """Contiguous psi runs of one element (deg), wrapping through 360."""
    out = []
    if cons.empty:
        return out
    cons = cons.sort_values('pbin').reset_index(drop=True)
    start = 0
    for i in range(1, len(cons) + 1):
        if (i == len(cons) or cons.element[i] != cons.element[start]
                or cons.pbin[i] != cons.pbin[i - 1] + 1):
            out.append((cons.element[start], cons.psi_deg[start] - BIN_DEG / 2,
                        cons.psi_deg[i - 1] + BIN_DEG / 2, cons.iloc[start:i]))
            start = i
    if len(out) > 1 and out[0][0] == out[-1][0] and out[0][1] == 0 and out[-1][2] == 360:
        e, lo, _, a = out.pop()
        _, _, hi, b0 = out.pop(0)
        out.append((e, lo - 360, hi, pd.concat([a, b0])))
    return out


def strip(ax, cons):
    for r in cons.itertuples():
        x0 = r.psi_deg - BIN_DEG / 2
        ax.add_patch(plt.Rectangle((x0, 0), BIN_DEG, 1, color=COLOURS.get(r.element, 'k'),
                                   alpha=0.25 + 0.75 * r.agree, lw=0))
        if r.margin_pix < TIGHT_PIX:
            ax.add_patch(plt.Rectangle((x0, 0), BIN_DEG, 1, fill=False, hatch='////',
                                       edgecolor='k', lw=0))
    ax.set_ylim(0, 1)
    ax.set_yticks([])


def page_psi(pdf, g, title):
    keys = list(g[['key', 'side']].drop_duplicates().itertuples(index=False, name=None))
    cmap = plt.get_cmap('tab20' if len(keys) <= 20 else 'gist_rainbow')
    fig, axes = plt.subplots(7, 1, figsize=(12, 19), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 0.35, 3, 0.35, 2.2, 2, 1.6]))
    ax_in, st_in, ax_out, st_out, ax_q, ax_d, ax_m = axes
    for i, key in enumerate(keys):
        col = cmap(i / max(len(keys) - 1, 1)) if len(keys) > 20 else cmap(i)
        h = g[(g.key == key[0]) & (g.side == key[1])].sort_values('psi_deg')
        for ax, q in ((ax_in, 'inner'), (ax_out, 'outer'), (ax_q, 'ratio')):
            if q == 'ratio':
                yd, ym = h.r_inner_data_pix / h.r_outer_data_pix, h.r_inner_c_pix / h.r_outer_c_pix
            else:
                yd, ym = h[f'r_{q}_data_pix'], h[f'r_{q}_c_pix']
            ax.plot(h.psi_deg, yd, '.', color=col, ms=2.5)
            ax.plot(h.psi_deg, ym, '-', color=col, lw=0.8)
    cons = {e: consensus(g, e) for e in ('inner', 'outer')}
    strip(st_in, cons['inner'])
    strip(st_out, cons['outer'])
    st_in.set_ylabel('inner\nelement', fontsize=7, rotation=0, ha='right', va='center')
    st_out.set_ylabel('outer\nelement', fontsize=7, rotation=0, ha='right', va='center')
    med = g.groupby('pbin')[['inner_dmm', 'outer_dmm', 'ratio_dmm']].median()
    psi = (med.index + 0.5) * BIN_DEG
    ax_d.plot(psi, med.inner_dmm, 'k-', lw=1.2, label='inner edge')
    ax_d.plot(psi, med.outer_dmm, '-', color='tab:orange', lw=1.2, label='outer edge')
    ax_d.axhline(0, color='0.6', lw=0.6)
    ax_d.set_ylabel('median data − model\nover donuts (pixels)')
    ax_d.legend(fontsize=7, loc='upper left')
    for e, c in (('inner', 'k'), ('outer', 'tab:orange')):
        m = cons[e]
        if not m.empty:
            ax_m.semilogy(m.psi_deg, np.clip(m.margin_pix, 0.05, pl.LOST_PIX), '.-', color=c,
                          lw=0.8, ms=3, label=f'{e} edge')
    ax_m.axhline(TIGHT_PIX, color='0.5', ls=':', lw=0.8)
    ax_m.set_ylabel('median margin to the\nnext element (pixels)')
    ax_m.legend(fontsize=7, loc='upper left')
    ax_in.set_ylabel('inner edge (pixels)')
    ax_out.set_ylabel('outer edge (pixels)')
    ax_q.set_ylabel('r_inner / r_outer\n(dimensionless)')
    ax_m.set_xlim(0, 360)
    ax_m.set_xlabel('relative azimuth psi (deg): 0 nearest the field centre, anticlockwise')
    used = sorted(set(cons['inner'].get('element', [])) | set(cons['outer'].get('element', [])),
                  key=list(COLOURS).index)
    handles = [Patch(color=COLOURS[u], label=NAMES.get(u, u)) for u in used]
    handles.append(Patch(facecolor='w', edgecolor='k', hatch='////',
                         label=f'margin < {TIGHT_PIX:g} pixel'))
    fig.legend(handles=handles, loc='upper center', ncol=min(len(handles), 5), fontsize=7,
               bbox_to_anchor=(0.5, 0.955))
    ax_in.set_title(f'{title}: {len(keys)} sides. Dots: data; lines: fit c model, same colour.\n'
                    'Element strips: the danish mask circle that sets the model edge, commonest '
                    'over donuts (opacity: fraction agreeing),\nfrom each donut\'s own field, '
                    'rotator angle and mask. Spider bins left out.', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig, dpi=80)
    plt.close(fig)
    return cons


def element_table(cons, title):
    rows = []
    for e in ('inner', 'outer'):
        for el, lo, hi, part in ranges(cons[e]):
            rows.append(dict(edge=e, element=NAMES.get(el, el), psi_from_deg=lo, psi_to_deg=hi,
                             n_bins=len(part), agree=part.agree.median(),
                             dmm_median_pix=part.dmm_pix.median(),
                             margin_median_pix=part.margin_pix.median()))
    return pd.DataFrame(rows).assign(group=title)


def page_table(pdf, t, title):
    fig, ax = plt.subplots(figsize=(12, 0.6 + 0.28 * (len(t) + 2)))
    ax.axis('off')
    cols = ['edge', 'element', 'psi_from_deg', 'psi_to_deg', 'n_bins', 'agree',
            'dmm_median_pix', 'margin_median_pix']
    head = ['edge', 'element', 'psi from (deg)', 'psi to (deg)', '2 deg bins',
            'agree (fraction)', 'data − model (pixels)', 'margin (pixels)']
    cells = [[f'{v:.0f}' if c.startswith('psi') or c == 'n_bins' else
              f'{v:+.2f}' if c == 'dmm_median_pix' else f'{v:.2f}' if isinstance(v, float) else v
              for c, v in zip(cols, r)] for r in t[cols].itertuples(index=False)]
    tab = ax.table(cellText=cells, colLabels=head, loc='upper center', cellLoc='center')
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 1.3)
    ax.set_title(f'{title}: psi ranges per defining element (medians over the range; agree: '
                 'fraction of donuts with that element)', fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def report(run, fit_dir, label_runs, out_dir):
    b = load(run, fit_dir, label_runs)
    tables = []
    pdf_path = out_dir / 'psi_edges.pdf'
    with PdfPages(pdf_path) as pdf:
        for group, title in GROUPS:
            g = group_of(b, group)
            if g[['key', 'side']].drop_duplicates().shape[0] < 3:
                continue
            cons = page_psi(pdf, g, f'{run} {fit_dir or ""} {title}'.replace('  ', ' '))
            t = element_table(cons, title)
            page_table(pdf, t, title)
            tables.append(t)
    t = pd.concat(tables, ignore_index=True)
    t.to_csv(out_dir / 'psi_elements.csv', index=False, float_format='%.4g')
    print('wrote', pdf_path)
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run', required=True)
    p.add_argument('--fit-dir', default=None)
    p.add_argument('--labels', action='append', required=True,
                   help='psi_labels run(s) under output/; repeatable')
    a = p.parse_args()
    out = pl.STUDY_OUT / a.labels[0]
    report(a.run, a.fit_dir, a.labels, out)


if __name__ == '__main__':
    main()
