"""Gallery of giant-donut sides: data, fit c model and data - model, edges by mask element.

One row per side, for the sides of a `psi_labels` run: the data stamp, the fit c model
re-rendered from the stored fit (`psi_labels.fit_model`, the fit's mask, field and
rotator angle) and (data - model) / noise. On the data and model panels the model edges
(robust finder, about the fit-a center) are drawn per azimuth bin, coloured by the mask
circle that sets them (`psi_labels`), and the data edges as white dots. Stamps are CCS
pixels, origin lower left; the arrow marks psi 0, the direction of the field center.

Sides are ordered by side of focus, then by OCS field azimuth. Stamps are read from
``isr_stamps/<key>.npz`` (pupil pass 6) or ``stamps/<key>.fits`` (night2_pipeline).

Usage, from the repo root::

    python aos/giant_donut_pupil/code/psi_gallery.py --run 20261023_pupil6 \\
        --fit-dir isr_phys_v402s --labels 20261117_psi_labels_20261023_pupil6_isr_phys_v402s
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
import psi_edges_report as per  # noqa: E402
import psi_labels as pl  # noqa: E402

ROWS_PER_PAGE = 4
RESID_SIGMA = 5.0


def stamp(run_dir, key):
    name = key.replace(':', '_')
    npz = run_dir / 'isr_stamps' / f'{name}.npz'
    if npz.exists():
        d = np.load(npz)
        return np.asarray(d['img'], float), np.asarray(d['masked'], bool)
    from astropy.io import fits
    with fits.open(run_dir / 'stamps' / f'{name}.fits') as h:
        return np.asarray(h['IMAGE'].data, float), np.asarray(h['MASK'].data, bool)


def _edges(ax, g, cx, cy, col_r, colours=None, **kw):
    az = np.radians(g.az_ccs_deg.to_numpy())
    r = g[col_r].to_numpy()
    x, y = cx + r * np.cos(az), cy + r * np.sin(az)
    if colours is None:
        ax.plot(x, y, '.', **kw)
    else:
        ax.scatter(x, y, c=colours, **kw)


def row_panels(axes, row, g, img, msk, model, noise):
    cx, cy = row.cx_a, row.cy_a
    top = np.nanpercentile(img[~msk], 99.0)
    resid = np.where(msk, np.nan, (img - model) / noise)
    for ax, im, kw in ((axes[0], np.where(msk, np.nan, img), dict(cmap='gray', vmin=0, vmax=top)),
                       (axes[1], model, dict(cmap='gray', vmin=0, vmax=top)),
                       (axes[2], resid, dict(cmap='RdBu_r', vmin=-RESID_SIGMA, vmax=RESID_SIGMA))):
        h = ax.imshow(im, origin='lower', interpolation='nearest', **kw)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.colorbar(h, ax=axes[2], fraction=0.046, pad=0.02, label='(data − model) / noise')
    for ax in axes[:2]:
        for e in ('inner', 'outer'):
            cols = [per.COLOURS.get(v, 'k') if isinstance(v, str) else (0, 0, 0, 0)
                    for v in g[f'element_{e}']]
            _edges(ax, g, cx, cy, f'r_{e}_c_pix', cols, s=3, lw=0)
        for e in ('inner', 'outer'):
            _edges(ax, g, cx, cy, f'r_{e}_data_pix', color='w', ms=1.0)
    # psi 0 points from the donut toward the field center: OCS azimuth phi + 180, in CCS
    a = np.radians(row.field_az_ocs_deg + 180.0 - row.rtp_deg)
    n = img.shape[0]
    axes[0].annotate('', xy=(cx + 0.47 * n * np.cos(a), cy + 0.47 * n * np.sin(a)),
                     xytext=(cx, cy), arrowprops=dict(arrowstyle='->', color='yellow', lw=0.8))
    axes[0].set_ylabel(f'{row.key.split(":")[0]}\n{row.side}, {row.sensor_class}\n'
                       f'field {row.field_r_deg:.3f} deg\nOCS az {row.field_az_ocs_deg:.0f} deg\n'
                       f'rtp {row.rtp_deg:.2f} deg', fontsize=7)
    d = (g[['r_inner_data_pix', 'r_outer_data_pix']].to_numpy()
         - g[['r_inner_c_pix', 'r_outer_c_pix']].to_numpy())
    axes[1].set_title(f'chi2/dof {row.chi2_dof:.2f} (dof {int(row.dof)})', fontsize=7)
    axes[2].set_title(f'median data − model edge: inner {np.nanmedian(d[:, 0]):+.2f}, '
                      f'outer {np.nanmedian(d[:, 1]):+.2f} pixels', fontsize=7)


def gallery(run, fit_dir, label_runs, groups, out_dir, fwhm=1.0):
    c, b, mask_run = pl._load(run, fit_dir)
    pl._init(*map(str, mask_run))
    lab = pd.concat([pd.read_parquet(pl.STUDY_OUT / r / 'labels.parquet') for r in label_runs],
                    ignore_index=True).drop_duplicates(['key', 'side', 'bin'])
    c = pl.select(c, groups=groups).merge(lab[['key', 'side']].drop_duplicates())
    c = c.sort_values(['side', 'field_az_ocs_deg'], ascending=[False, True])
    run_dir = pl.STUDY_OUT / run
    pdf_path = out_dir / 'gallery.pdf'
    rows = list(c.itertuples(index=False))
    with PdfPages(pdf_path) as pdf:
        for i0 in range(0, len(rows), ROWS_PER_PAGE):
            chunk = rows[i0:i0 + ROWS_PER_PAGE]
            fig, axes = plt.subplots(ROWS_PER_PAGE, 3, figsize=(12, 4.0 * ROWS_PER_PAGE),
                                     squeeze=False)
            for k, r in enumerate(chunk):
                r = pd.Series(r._asdict())
                g = b[(b.key == r.key) & (b.side == r.side)].merge(
                    lab, on=['key', 'side', 'bin']).sort_values('bin')
                img, msk = stamp(run_dir, r.key)
                model = pl.fit_model(r, pl.mask_for(r), fwhm)
                row_panels(axes[k], r, g, img, msk, model, float(r.noise_e))
                print(f'{r.key} done', flush=True)
            for k in range(len(chunk), ROWS_PER_PAGE):
                for ax in axes[k]:
                    ax.axis('off')
            for ax, t in zip(axes[0], ('data (electrons)', 'fit c model (electrons)',
                                       '(data − model) / noise')):
                ax.set_title(t + '\n' + ax.get_title(), fontsize=8)
            used = [u for u in per.COLOURS if u in set(lab.element_inner) | set(lab.element_outer)]
            handles = [Patch(color=per.COLOURS[u], label=per.NAMES.get(u, u)) for u in used]
            fig.legend(handles=handles, loc='upper center', ncol=min(len(handles), 5), fontsize=7)
            fig.suptitle(f'{run} {fit_dir or ""}: model edges coloured by mask element, data '
                         'edges white; yellow arrow: psi 0 (toward the field center); CCS, origin '
                         'lower left', fontsize=9, y=0.965)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            pdf.savefig(fig, dpi=90)
            plt.close(fig)
    print('wrote', pdf_path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run', required=True)
    p.add_argument('--fit-dir', default=None)
    p.add_argument('--labels', action='append', required=True)
    p.add_argument('--group', action='append', default=None,
                   help='CLASS:SIDE, repeatable; default science intra and extra')
    a = p.parse_args()
    groups = ([tuple(g.split(':')) for g in a.group] if a.group
              else [('science', 'intra'), ('science', 'extra')])
    gallery(a.run, a.fit_dir, a.labels, groups, pl.STUDY_OUT / a.labels[0])


if __name__ == '__main__':
    main()
