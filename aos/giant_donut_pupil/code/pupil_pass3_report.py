"""Pages and summaries of pupil pass 3 (`pupil_pass3.py`).

Reads ``fits.parquet`` and ``edges_bins.parquet`` in the run, and writes:

- ``pupil_pass3.pdf``: the sample map; two pages per donut side that has saved
  images (fits a-d: data, model, data - model; then the robust inner edge, outer
  edge and inner/outer ratio against OCS azimuth), ordered by field radius; then
  the summary pages 8a-8e;
- ``summary_ratio.csv``, ``summary_zernikes.csv`` and ``summary_edges.csv``, the
  numbers on those pages.

Called from `pupil_pass3.py` (step ``report``).
"""
import pathlib

import numpy as np
import pandas as pd

import pupil_edges as pe

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from rubinwork.common.utils import nmad  # noqa: E402

FITS = ('a', 'b', 'c', 'd')
FIT_COLOURS = {'data': 'k', 'a': 'tab:blue', 'b': 'tab:orange', 'c': 'tab:green', 'd': 'tab:red'}
SIDE_MARKERS = {'extra': '^', 'intra': 'v'}
OUTER_BAND = (1.3, 1.75)
# A fit is failed when its flux over the stamp sum is off the median of its fit and
# side by more than this factor (dimensionless).
FLUX_RATIO_FACTOR = 1.5


def _fits_of(fits, k):
    return fits[(fits.fit == k) & fits.fit_ok]


def flag_failed(fits):
    """``flux_ratio`` (fitted flux over the stamp sum) and ``fit_ok``."""
    fits['flux_ratio'] = fits.flux_e / fits.stamp_sum_e
    med = fits.groupby(['fit', 'side']).flux_ratio.transform('median')
    q = fits.flux_ratio / med
    fits['fit_ok'] = (fits.error == '') & (q < FLUX_RATIO_FACTOR) & (q > 1 / FLUX_RATIO_FACTOR)
    return fits


def page_images(pdf, d, g):
    """Fits a-d: data, model and data - model, one row per fit."""
    img, msk = d['img'].astype(float), d['masked'].astype(bool)
    data = np.where(msk, np.nan, img)
    vmax = np.nanpercentile(data, 99.5)
    res = {k: np.where(msk, np.nan, img - d[f'model_{k}']) for k in FITS}
    rl = max(np.nanpercentile(np.abs(r), 99) for r in res.values())
    fig, axes = plt.subplots(4, 3, figsize=(14, 17))
    for i, k in enumerate(FITS):
        r = g.loc[k]
        for j, (a, title, kw) in enumerate((
                (data, 'data', dict(vmin=0, vmax=vmax, cmap='gray')),
                (d[f'model_{k}'], 'model', dict(vmin=0, vmax=vmax, cmap='gray')),
                (res[k], 'data - model', dict(vmin=-rl, vmax=rl, cmap='RdBu_r')))):
            ax = axes[i, j]
            im = ax.imshow(a, origin='lower', **kw)
            if k == 'd' and j == 2:
                ax.contour(d['region'], levels=[0.5], colors='k', linewidths=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.7, label='electrons per pixel')
            if j == 0:
                ax.set_ylabel(r.fit_label, fontsize=10)
            ax.set_title(title, fontsize=9)
        zs = [c for c in g.columns if c.startswith('z') and c.endswith('_ocs_um')
              and np.isfinite(r[c])]
        text = '  '.join(f'{c[:-7].upper()} {r[c]:+.2f}' for c in zs)
        chi = f'chi2/dof {r.chi2_dof:.3f} (dof {int(r.dof)}, whole donut)'
        if k == 'd':
            chi += f'; {r.chi2_dof_fitted:.3f} (dof {int(r.dof_fitted)}, fitted pixels)'
        axes[i, 0].text(0.0, -0.08, chi + '\ndeviation, µm of wavefront, OCS: ' + text,
                        transform=axes[i, 0].transAxes, fontsize=8, va='top')
    r0 = g.iloc[0]
    fig.suptitle(f'{r0.sample_id} {r0.side}-focal {r0.detector}, {r0.mag:.2f} AB mag; field radius '
                 f'{r0.field_r_deg:.3f} deg, OCS azimuth {r0.field_az_ocs_deg:.1f} deg\n'
                 'danish v4.02g, blur FWHM fixed 1.0 arcsec, spider at the rotator angle, '
                 'unfitted Zernikes at intrinsicTA; CCS stamp. Fit d: pixels inside the black '
                 'contour only', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0)
    pdf.savefig(fig, dpi=80)
    plt.close(fig)


def _spider_spans(ax, b):
    for x in b.az_ocs_deg[b.spider]:
        ax.axvspan(x - 1, x + 1, color='0.85', lw=0, zorder=0)


def page_edges(pdf, b, g):
    """Inner edge, outer edge and inner/outer ratio against OCS azimuth."""
    b = b.sort_values('az_ocs_deg')
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=True)
    for name in ('data',) + FITS:
        col = FIT_COLOURS[name]
        label = 'data' if name == 'data' else g.loc[name].fit_label
        lw, ms = (0, 3) if name == 'data' else (1.2, 0)
        style = 'o' if name == 'data' else '-'
        for ax, edge in zip(axes[:2], ('inner', 'outer')):
            ax.plot(b.az_ocs_deg, b[f'r_{edge}_{name}_pix'], style, color=col, ms=ms, lw=lw,
                    label=label)
        q = b[f'r_inner_{name}_pix'] / b[f'r_outer_{name}_pix']
        axes[2].plot(b.az_ocs_deg, q, style, color=col, ms=ms, lw=lw, label=label)
    for ax in axes:
        _spider_spans(ax, b)
    axes[0].set_ylabel('inner edge radius (pixels)')
    axes[1].set_ylabel('outer edge radius (pixels)')
    axes[2].set_ylabel('inner / outer (dimensionless)')
    axes[2].set_xlabel('azimuth about the donut centre, OCS (deg)')
    axes[2].set_xlim(0, 360)
    axes[0].legend(fontsize=8, ncol=3)
    rows = []
    for k in FITS:
        r = g.loc[k]
        rows.append(f'{k}: data − model RMS inner {r.edge_rms_inner_pix:.2f}, outer '
                    f'{r.edge_rms_outer_pix:.2f} pixels; median inner {r.edge_median_inner_pix:+.2f}, '
                    f'outer {r.edge_median_outer_pix:+.2f} pixels; ratio {r.ratio_median:.4f}')
    r0 = g.iloc[0]
    axes[0].set_title(f'{r0.sample_id} {r0.side}-focal: robust edges (kept bins only); grey: '
                      f'spider bins. Data ratio {r0.ratio_median_data:.4f} (median, '
                      'dimensionless)\n' + '\n'.join(rows), fontsize=8)
    pdf.savefig(fig, dpi=80)
    plt.close(fig)


def _by_side(ax, df, x, y, colour, label):
    for side, m in SIDE_MARKERS.items():
        s = df[df.side == side]
        ax.plot(s[x], s[y], m, color=colour, ms=5, mfc='none' if side == 'intra' else colour,
                label=f'{label}, {side}')


def page_ratio(pdf, fits, x='field_r_deg', xlabel='field radius (deg)', sel=None, title=''):
    """8a: median inner/outer ratio of data and each fit."""
    fig, ax = plt.subplots(figsize=(11, 7))
    a = _fits_of(fits, 'a')
    if sel is not None:
        a = a[sel(a)]
    _by_side(ax, a, x, 'ratio_median_data', 'k', 'data')
    for k in FITS:
        f = _fits_of(fits, k)
        if sel is not None:
            f = f[sel(f)]
        _by_side(ax, f, x, 'ratio_median', FIT_COLOURS[k], f'fit {k}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('median r_inner / r_outer over azimuth (dimensionless)')
    ax.legend(fontsize=7, ncol=3)
    ax.set_title(title, fontsize=9)
    pdf.savefig(fig)
    plt.close(fig)


def page_zernikes(pdf, fits, k, x='field_r_deg', xlabel='field radius (deg)', sel=None,
                  title=''):
    """8b: deviations of fit `k` against `x`, one Zernike per panel."""
    f = _fits_of(fits, k)
    if sel is not None:
        f = f[sel(f)]
    zs = [c for c in f.columns if c.startswith('z') and c.endswith('_ocs_um') and f[c].notna().any()]
    if not zs:
        return
    nc = 4 if len(zs) > 6 else len(zs)
    nr = int(np.ceil(len(zs) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4 * nc, 3.2 * nr + 1), squeeze=False)
    for ax, c in zip(axes.flat, zs):
        for side, m in SIDE_MARKERS.items():
            s = f[f.side == side]
            ax.plot(s[x], s[c], m, ms=5, color='k' if side == 'extra' else 'tab:red',
                    mfc='none' if side == 'intra' else None, label=side)
        for v, ls in ((0.5, ':'), (1.0, '--')):
            for sg in (1, -1):
                ax.axhline(sg * v, color='0.5', ls=ls, lw=0.8)
        ax.axhline(0, color='0.8', lw=0.5)
        ax.set_title(f'{c[:-7].upper()} deviation, OCS', fontsize=9)
        ax.set_ylabel('µm of wavefront')
        ax.set_xlabel(xlabel)
    for ax in list(axes.flat)[len(zs):]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f'Fit {FITS_LABEL(fits, k)}: Zernike deviation from the in-focus intrinsic; '
                 f'dotted ±0.5 µm, dashed ±1 µm. {title}', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def FITS_LABEL(fits, k):
    return _fits_of(fits, k).fit_label.iloc[0]


def page_chi2(pdf, fits):
    """8c: chi2/dof against field radius per fit."""
    fig, ax = plt.subplots(figsize=(11, 7))
    for k in FITS:
        f = _fits_of(fits, k)
        _by_side(ax, f, 'field_r_deg', 'chi2_dof', FIT_COLOURS[k], f'fit {k}, whole donut')
    d = _fits_of(fits, 'd')
    for side, m in SIDE_MARKERS.items():
        s = d[d.side == side]
        ax.plot(s.field_r_deg, s.chi2_dof_fitted, m, color='tab:purple', ms=6, mfc='none',
                label=f'fit d, fitted pixels, {side}')
    ax.set_yscale('log')
    ax.set_xlabel('field radius (deg)')
    ax.set_ylabel('chi2/dof (dimensionless); dof = unmasked pixels − free parameters')
    ax.legend(fontsize=7, ncol=2)
    ax.set_title('8c. chi2/dof against field radius', fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


def page_edge_rms(pdf, fits):
    """8d: edge residual RMS (data − model) per fit, inner and outer."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    for ax, edge in zip(axes, ('inner', 'outer')):
        for k in FITS:
            _by_side(ax, _fits_of(fits, k), 'field_r_deg', f'edge_rms_{edge}_pix',
                     FIT_COLOURS[k], f'fit {k}')
        ax.set_xlabel('field radius (deg)')
        ax.set_title(f'{edge} edge', fontsize=10)
    axes[0].set_ylabel('RMS of data − model edge radius over kept bins (pixels)')
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle('8d. Edge residual RMS against field radius', fontsize=11)
    pdf.savefig(fig)
    plt.close(fig)


def summaries(fits, bins):
    """The numbers behind 8a-8e."""
    ok = fits[fits.fit_ok]
    rows = []
    for (k, side), f in ok.groupby(['fit', 'side']):
        row = dict(fit=k, side=side, n_sides=len(f),
                   ratio_median_data=f.ratio_median_data.median(), ratio_median_model=f.ratio_median.median(),
                   ratio_excess=(f.ratio_median_data - f.ratio_median).median(),
                   chi2_dof_median=f.chi2_dof.median(),
                   edge_rms_inner_median_pix=f.edge_rms_inner_pix.median(),
                   edge_rms_outer_median_pix=f.edge_rms_outer_pix.median(),
                   edge_median_inner_pix=f.edge_median_inner_pix.median(),
                   edge_median_outer_pix=f.edge_median_outer_pix.median())
        if k == 'd':
            row['chi2_dof_fitted_median'] = f.chi2_dof_fitted.median()
        for c in [c for c in f if c.endswith('_ocs_um') and f[c].notna().any()]:
            row[f'{c}_median'] = f[c].median()
            row[f'{c}_nmad'] = nmad(f[c].dropna().to_numpy())
            row[f'{c}_frac_over_1um'] = float((f[c].abs() > 1.0).mean())
        rows.append(row)
    by_fit = pd.DataFrame(rows)
    # Ratio excess against field radius: Pearson r and Spearman rho.
    from scipy.stats import pearsonr, spearmanr
    corr = []
    for k in FITS:
        f = ok[ok.fit == k].dropna(subset=['ratio_median', 'ratio_median_data'])
        ex = f.ratio_median_data - f.ratio_median
        for x in ('field_r_deg', 'field_az_ocs_deg'):
            if len(f) > 3:
                corr.append(dict(fit=k, x=x, n=len(f), pearson_r=pearsonr(f[x], ex)[0],
                                 spearman_rho=spearmanr(f[x], ex)[0]))
    return by_fit, pd.DataFrame(corr)


def text_page(pdf, title, text, fontsize=7):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0, 1, title + '\n\n' + text, family='monospace', fontsize=fontsize, va='top')
    pdf.savefig(fig)
    plt.close(fig)


def report(run, cand):
    run = pathlib.Path(run)
    fits = pd.read_parquet(run / 'fits.parquet')
    if 'error' not in fits:
        fits['error'] = ''
    fits['error'] = fits.error.fillna('')
    if 'stamp_sum_e' not in fits:
        fits['stamp_sum_e'] = np.nan
    for i, r in fits[fits.stamp_sum_e.isna()].iterrows():
        path = run / 'images' / f'{r.sample_id}_{r.side}.npz'
        if path.exists():
            d = np.load(path)
            fits.at[i, 'stamp_sum_e'] = float(d['img'][~d['masked']].sum())
    fits = flag_failed(fits)
    fits.to_parquet(run / 'fits.parquet')
    fits.to_csv(run / 'fits.csv', index=False, float_format='%.5g')
    bins = pd.read_parquet(run / 'edges_bins.parquet')
    by_fit, corr = summaries(fits, bins)
    by_fit.to_csv(run / 'summary_by_fit.csv', index=False, float_format='%.4f')
    corr.to_csv(run / 'summary_ratio_correlation.csv', index=False, float_format='%.3f')
    order = fits[fits.fit == 'a'].sort_values(['field_r_deg', 'sample_id', 'side'])
    outer = lambda f: f.field_r_deg.between(*OUTER_BAND)  # noqa: E731
    with PdfPages(run / 'pupil_pass3.pdf') as pdf:
        sample_png = run / 'sample_map.png'
        if not sample_png.exists():
            sample_png = run.parent / 'sample_map.png'
        if sample_png.exists():
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(plt.imread(sample_png))
            ax.axis('off')
            pdf.savefig(fig, dpi=120)
            plt.close(fig)
        n_pages = 0
        for _, r in order.iterrows():
            path = run / 'images' / f'{r.sample_id}_{r.side}.npz'
            if not path.exists():
                continue
            g = fits[(fits.sample_id == r.sample_id) & (fits.side == r.side)].set_index('fit')
            b = bins[(bins.sample_id == r.sample_id) & (bins.side == r.side)]
            page_images(pdf, np.load(path), g)
            page_edges(pdf, b, g)
            n_pages += 2
        cols = ['fit', 'side', 'n_sides', 'ratio_median_data', 'ratio_median_model', 'ratio_excess',
                'chi2_dof_median', 'edge_rms_inner_median_pix', 'edge_rms_outer_median_pix',
                'edge_median_inner_pix', 'edge_median_outer_pix']
        failed = fits[~fits.fit_ok][['sample_id', 'side', 'fit', 'mag', 'flux_ratio', 'chi2_dof']]
        text_page(pdf, 'Fits left out of the summaries: fitted flux over the stamp sum more than '
                  f'{FLUX_RATIO_FACTOR}x off the median of its fit and side (dimensionless)',
                  failed.round(3).to_string(index=False), 8)
        text_page(pdf, 'Summary per fit and side: medians over donut sides. Ratios '
                  'dimensionless; edge numbers data − model, pixels; chi2/dof whole donut',
                  by_fit[cols].round(4).to_string(index=False) + '\n\nRatio excess (data − model '
                  'median ratio) against field radius and OCS azimuth:\n'
                  + corr.round(3).to_string(index=False), 7)
        page_ratio(pdf, fits, title='8a. Median inner/outer edge ratio against field radius: '
                   'data (black) and fits a-d; ^ extra, v intra')
        for k in FITS:
            page_zernikes(pdf, fits, k, title='8b, against field radius.')
        page_chi2(pdf, fits)
        page_edge_rms(pdf, fits)
        az = dict(x='field_az_ocs_deg', xlabel='focal-plane azimuth, OCS (deg)', sel=outer)
        page_ratio(pdf, fits, title='8e. Median inner/outer ratio against focal-plane azimuth, '
                   'field radius 1.3-1.75 deg', **az)
        for k in FITS:
            page_zernikes(pdf, fits, k, title='8e, field radius 1.3-1.75 deg, against azimuth.',
                          **az)
    print('pages', n_pages, 'report', run / 'pupil_pass3.pdf', flush=True)
