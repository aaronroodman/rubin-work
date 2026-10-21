"""Pages and tables of pupil pass 4 step 6 (`spider_pass4.py`).

Reads ``spiders.parquet`` and ``spider_profiles.npz`` in the pass 4 run and writes
``pupil_pass4_spiders.pdf`` (6c profiles per pass 3 sample donut, 6d data − model
azimuth against psi number per donut, 6e summaries) and the tables
``spider_found``, ``spider_by_vane`` and ``spider_vane_fits`` as parquet and csv.
"""
import pathlib

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from scipy.stats import pearsonr, spearmanr  # noqa: E402

from rubinwork.common.utils import nmad  # noqa: E402

BIAS_DEG = 0.4
VANE_COLOURS = plt.get_cmap('tab10')
SIDE_MARKERS = {'extra': '^', 'intra': 'v'}
QUANTITIES = (('az_half_ocs_deg_dmm', 'azimuth at r_half, data − model (deg)'),
              ('b_pix_dmm', 'b, data − model (pixels)'),
              ('depth_dmm', 'depth, data − model (dimensionless)'),
              ('fwhm_pix_dmm', 'width (FWHM), data − model (pixels)'))


def _save(df, run, name):
    df.to_parquet(run / f'{name}.parquet')
    df.to_csv(run / f'{name}.csv', index=False, float_format='%.5g')


def text_page(pdf, title, text, fontsize=7):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0, 1, title + '\n\n' + text, family='monospace', fontsize=fontsize, va='top')
    pdf.savefig(fig)
    plt.close(fig)


def page_profiles(pdf, t, prof, pid):
    g = t[(t.pass3_id == pid)]
    fig, axes = plt.subplots(2, 8, figsize=(22, 7), sharex=True)
    r0 = g.iloc[0]
    for i, side in enumerate(('intra', 'extra')):
        s = g[g.side == side].sort_values('psi_num')
        if s.empty:
            for ax in axes[i]:
                ax.axis('off')
            axes[i, 0].text(0, 0.5, f'no {side}-focal side in the sample', fontsize=10,
                            transform=axes[i, 0].transAxes)
            continue
        for _, r in s.iterrows():
            if r.psi_num < 0:
                continue
            ax = axes[i, int(r.psi_num)]
            key = f'sample:{r.sample_id}:{side}:{int(r.model_idx)}'
            for name, col in (('data', 'k'), ('model', 'tab:green')):
                if f'{key}:{name}_off' in prof:
                    ax.plot(prof[f'{key}:{name}_off'], prof[f'{key}:{name}_flux'], '-', color=col,
                            lw=1.0, label=name if name == 'data' else 'fit c')
            ax.set_title(f'{side} psi# {int(r.psi_num)}, vane {int(r.vane_id)}\n'
                         f'depth {r.depth_data:.2f}/{r.depth_model:.2f}, FWHM '
                         f'{r.fwhm_pix_data:.1f}/{r.fwhm_pix_model:.1f} pix\n'
                         f'b {r.b_pix_data:+.1f}/{r.b_pix_model:+.1f} pix'
                         + (f'\nMISSING: {r.missing}' if r.missing else ''), fontsize=7,
                         color=VANE_COLOURS(int(r.vane_id)) if r.vane_id >= 0 else 'k')
            ax.tick_params(labelsize=6)
        axes[i, 0].set_ylabel(f'{side}: flux (electrons per pixel)', fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel('offset across the vane (pixels)', fontsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f'6c. {pid} {r0.detector}, field radius {r0.field_r_deg:.3f} deg, OCS azimuth '
                 f'{r0.field_az_ocs_deg:.1f} deg: spider shadow profiles along their chords; '
                 'black data, green fit c. Labels data/model. psi#: order in relative azimuth; '
                 'vane: pupil-frame id', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig, dpi=70)
    plt.close(fig)


def page_psi_grid(pdf, t):
    ids = t.drop_duplicates('pass3_id').sort_values('field_r_deg').pass3_id.tolist()
    nc = 6
    nr = int(np.ceil(len(ids) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(22, 3.2 * nr), sharex=True, sharey=True)
    lim = np.nanpercentile(np.abs(t.az_half_ocs_deg_dmm), 99) * 1.1
    for ax, pid in zip(axes.flat, ids):
        g = t[t.pass3_id == pid]
        for side, m in SIDE_MARKERS.items():
            s = g[(g.side == side) & (g.psi_num >= 0)]
            ax.scatter(s.psi_num + (0.12 if side == 'extra' else -0.12), s.az_half_ocs_deg_dmm,
                       c=[VANE_COLOURS(int(v)) for v in s.vane_id], marker=m, s=30,
                       edgecolors='k', linewidths=0.3)
        ax.axhline(0, color='0.5', lw=0.6)
        ax.axhspan(-BIAS_DEG, BIAS_DEG, color='0.9', zorder=0)
        r0 = g.iloc[0]
        ax.set_title(f'{pid}: {r0.field_r_deg:.2f} deg, az {r0.field_az_ocs_deg:.0f} deg', fontsize=8)
        ax.set_ylim(-lim, lim)
        ax.set_xticks(range(8))
    for ax in list(axes.flat)[len(ids):]:
        ax.axis('off')
    for ax in axes[:, 0]:
        ax.set_ylabel('az at r_half, data − model (deg)', fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel('psi spider number', fontsize=7)
    handles = [plt.Line2D([], [], marker='o', ls='', color=VANE_COLOURS(v), label=f'vane {v}')
               for v in range(8)] + [plt.Line2D([], [], marker=m, ls='', color='k', label=s)
                                     for s, m in SIDE_MARKERS.items()]
    fig.legend(handles=handles, loc='upper right', ncol=10, fontsize=8)
    fig.suptitle('6d. Spider shadow azimuth at r_half, data − fit c (deg, OCS) against psi number; '
                 f'grey band ±{BIAS_DEG} deg (the method\'s rotation bias); colour: pupil-frame vane '
                 'id; ^ extra, v intra', fontsize=11, x=0.35)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, dpi=70)
    plt.close(fig)


def by_vane(t):
    rows = []
    for (side, v), g in t[t.vane_id >= 0].groupby(['side', 'vane_id']):
        row = dict(side=side, vane_id=int(v), n=int(g.az_half_ocs_deg_dmm.notna().sum()),
                   pupil_az_median_deg=float(g.pupil_az_deg.median()))
        for q, _ in QUANTITIES + (('phi0_ocs_deg_dmm', ''),):
            x = g[q].dropna().to_numpy()
            row[f'{q}_median'] = float(np.median(x)) if len(x) else np.nan
            row[f'{q}_nmad'] = nmad(x) if len(x) > 2 else np.nan
        for q in ('depth_data', 'depth_model', 'fwhm_pix_data', 'fwhm_pix_model'):
            row[f'{q}_median'] = float(g.loc[g.missing == '', q].median())
        rows.append(row)
    return pd.DataFrame(rows)


def vane_fits(t):
    """Per side of focus: common rotation, per-vane scatter about it and field dependence."""
    import statsmodels.api as sm
    rows = []
    for side, g in t[(t.missing == '') & (t.vane_id >= 0)].groupby('side'):
        y = g.az_half_ocs_deg_dmm.to_numpy()
        common = float(np.median(y))
        per = g.groupby('vane_id').az_half_ocs_deg_dmm.median()
        # Huber fit: common offset + per-vane offsets + slope in field radius
        X = np.column_stack([np.ones_like(y), g.field_r_deg.to_numpy()])
        res = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
        side_med = g.groupby(['sample_id']).az_half_ocs_deg_dmm.median()
        fr = g.groupby('sample_id').field_r_deg.first()
        rows.append(dict(side=side, n_shadows=len(y), common_median_deg=common,
                         common_nmad_deg=nmad(y), per_vane_median_spread_deg=float(per.max() - per.min()),
                         per_vane_median_nmad_deg=nmad(per.to_numpy()),
                         slope_deg_per_deg=float(res.params[1]), slope_err=float(res.bse[1]),
                         pearson_r_side_median_vs_field=pearsonr(fr, side_med)[0],
                         spearman_rho_side_median_vs_field=spearmanr(fr, side_med)[0],
                         n_sides=len(side_med)))
    return pd.DataFrame(rows)


def page_by_vane(pdf, t, q, label):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, side in zip(axes, ('extra', 'intra')):
        g = t[(t.side == side) & (t.vane_id >= 0)]
        for v, gv in g.groupby('vane_id'):
            x = gv[q].dropna()
            ax.plot(v + np.random.default_rng(int(v)).uniform(-0.25, 0.25, len(x)), x, '.',
                    color=VANE_COLOURS(int(v)), ms=3, alpha=0.6)
            if len(x):
                ax.errorbar(v + 0.35, np.median(x), yerr=nmad(x.to_numpy()), fmt='s', color='k',
                            ms=5, capsize=3)
        ax.axhline(0, color='0.5', lw=0.6)
        if q.startswith('az'):
            ax.axhspan(-BIAS_DEG, BIAS_DEG, color='0.9', zorder=0)
        ax.set_xlabel('pupil-frame vane id')
        ax.set_title(f'{side}-focal; black: median ± nMAD', fontsize=9)
        ax.set_xticks(range(8))
    axes[0].set_ylabel(label)
    lo, hi = np.nanpercentile(t[q], [1, 99])
    axes[0].set_ylim(lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo))
    fig.suptitle(f'6e. {label} against vane id, all sample and band sides', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig, dpi=80)
    plt.close(fig)


def page_vs_field(pdf, t):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, side in zip(axes, ('extra', 'intra')):
        g = t[(t.side == side) & (t.vane_id >= 0)]
        for v, gv in g.groupby('vane_id'):
            ax.plot(gv.field_r_deg, gv.az_half_ocs_deg_dmm, '.', color=VANE_COLOURS(int(v)), ms=4,
                    label=f'vane {int(v)}')
        ax.axhline(0, color='0.5', lw=0.6)
        ax.axhspan(-BIAS_DEG, BIAS_DEG, color='0.9', zorder=0)
        ax.set_xlabel('field radius (deg)')
        ax.set_title(f'{side}-focal', fontsize=9)
    axes[0].set_ylabel('azimuth at r_half, data − model (deg)')
    lo, hi = np.nanpercentile(t.az_half_ocs_deg_dmm, [1, 99])
    axes[0].set_ylim(lo - 0.2, hi + 0.2)
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle('6e. Shadow azimuth at r_half, data − fit c, against field radius, by vane id',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig, dpi=80)
    plt.close(fig)


def report(run):
    run = pathlib.Path(run)
    t = pd.read_parquet(run / 'spiders.parquet')
    prof = dict(np.load(run / 'spider_profiles.npz'))
    t['missing'] = t.missing.fillna('')
    t['pass3_id'] = t.pass3_id.fillna('')
    found = t.groupby(['set', 'side', 'sensor_class']).apply(
        lambda g: pd.Series(dict(n_sides=g.groupby('sample_id').ngroups, n_shadows=len(g),
                                 found=int((g.missing == '').sum()),
                                 missing=int((g.missing != '').sum())))).reset_index()
    reasons = t[t.missing != ''].groupby(['set', 'missing']).size().rename('n').reset_index()
    _save(found, run, 'spider_found')
    # One row per physical shadow: a pass 3 sample side also in the bands set counts once.
    u = t[t.set == 'bands']
    s = t[(t.set == 'sample')]
    s = s[~s.set_index(['sample_id', 'side']).index.isin(u.set_index(['sample_id', 'side']).index)]
    all_t = pd.concat([u, s], ignore_index=True)
    bv = by_vane(all_t)
    vf = vane_fits(all_t)
    bv_s = by_vane(t[t.set == 'sample'])
    vf_s = vane_fits(t[t.set == 'sample'])
    _save(pd.concat([bv.assign(set='all'), bv_s.assign(set='sample')]), run, 'spider_by_vane')
    _save(pd.concat([vf.assign(set='all'), vf_s.assign(set='sample')]), run, 'spider_vane_fits')
    samp = t[t.set == 'sample']
    with PdfPages(run / 'pupil_pass4_spiders.pdf') as pdf:
        text_page(pdf, '6a. Spider shadows found and missing per set, side and sensor class',
                  found.to_string(index=False) + '\n\nReasons:\n' + reasons.to_string(index=False),
                  8)
        for pid in samp.drop_duplicates('pass3_id').sort_values('field_r_deg').pass3_id:
            page_profiles(pdf, samp, prof, pid)
        page_psi_grid(pdf, samp)
        cols = ['side', 'vane_id', 'n', 'pupil_az_median_deg'] + [
            f'{q}_{s}' for q, _ in QUANTITIES for s in ('median', 'nmad')]
        text_page(pdf, '6e. Per vane id, data − fit c: median and nMAD over all distinct sides '
                  '(sample and bands). az deg OCS, b and FWHM pixels, depth dimensionless',
                  bv[cols].round(3).to_string(index=False) + '\n\nCommon rotation, per-vane '
                  'spread and field slope (az at r_half, deg; slope deg per deg of field radius, '
                  'Huber):\n' + vf.round(3).to_string(index=False) + '\n\nPass 3 sample only:\n'
                  + vf_s.round(3).to_string(index=False) + '\n\nMedian depth and FWHM, data and '
                  'model:\n' + bv[['side', 'vane_id', 'depth_data_median', 'depth_model_median',
                                    'fwhm_pix_data_median', 'fwhm_pix_model_median']].round(3)
                  .to_string(index=False), 6)
        for q, label in QUANTITIES:
            page_by_vane(pdf, all_t, q, label)
        page_vs_field(pdf, all_t)
    print('report', run / 'pupil_pass4_spiders.pdf', flush=True)
