"""Pages and tables of pupil pass 4 (`pupil_pass4.py`).

Writes, in the run:

- ``pupil_pass4.pdf``: the sample map; step 1 (central-band edges against OCS image
  azimuth, then with the extra-focal curves turned by 180 deg), step 2 (outer-band edges
  against the relative azimuth psi), step 3 summaries, the step 3 per-donut pages, then
  steps 1 and 2 again with fit c0;
- ``counts``, ``flip_correlation``, ``psi_check``, ``psi_scatter``, ``psi_miss``,
  ``c0_vs_c`` and ``c0_vs_c_sides`` tables, each as parquet and csv.

Edge conventions: azimuth is the 2 deg bin about the donut centre, OCS image azimuth
(``az_ocs_deg``). Data and model edges are the robust edges (`robust_edges`) about the
fit-a centre; spider, vignetted and masked bins are NaN. psi is the relative azimuth,
``(az_ocs - (phi + 180)) mod 360`` with phi the donut's focal-plane OCS azimuth: 0 at the
point on the ring nearest the field centre, increasing anticlockwise.

Called from `pupil_pass4.py` (step ``report``).
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

NBIN = 180
BIN_DEG = 360 / NBIN
SIDE_COLOURS = {'extra': 'tab:blue', 'intra': 'tab:red'}
PSI_BAND = (1.6, 1.725)
MATCH_PIX = 0.3
FIT_LABELS = {'c': 'fit c (sparse set free)',
              'c0': 'fit c0 (central medians fixed; Z4, flux, centre free)'}


def _save(df, run, name):
    df.to_parquet(run / f'{name}.parquet')
    df.to_csv(run / f'{name}.csv', index=False, float_format='%.5g')


def _ok_sides(fits, k):
    return fits[(fits.fit == k) & fits.fit_ok][['sample_id', 'side']]


def text_page(pdf, title, text, fontsize=7):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis('off')
    ax.text(0, 1, title + '\n\n' + text, family='monospace', fontsize=fontsize, va='top')
    pdf.savefig(fig)
    plt.close(fig)


def sample_map(pdf, cand):
    fig, ax = plt.subplots(figsize=(10, 10))
    styles = {('central', 'science'): ('o', 'tab:green'), ('outer', 'science'): ('o', 'tab:purple'),
              ('outer', 'SW0'): ('s', 'tab:orange'), ('outer', 'SW1'): ('D', 'tab:brown')}
    for (band, cls), (m, col) in styles.items():
        for side in ('extra', 'intra'):
            s = cand[(cand.band == band) & (cand.sensor_class == cls) & (cand.side == side)]
            ax.scatter(s.field_x_ocs_deg, s.field_y_ocs_deg, marker=m, s=30 if side == 'extra' else 60,
                       facecolors=col if side == 'extra' else 'none', edgecolors=col,
                       label=f'{band} {cls} {side} ({len(s)})')
    p3 = cand[cand.pass3_id != ''].drop_duplicates('pass3_id')
    for _, r in p3.iterrows():
        ax.annotate(r.pass3_id, (r.field_x_ocs_deg, r.field_y_ocs_deg), xytext=(4, 4),
                    textcoords='offset points', fontsize=6)
    for rad in (0.8, 1.4, PSI_BAND[0], PSI_BAND[1], 1.75):
        ax.add_patch(plt.Circle((0, 0), rad, fill=False, ls='--', lw=0.7, color='0.5'))
    ax.set_aspect('equal')
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.9, 1.9)
    ax.set_xlabel('field x, OCS (deg)')
    ax.set_ylabel('field y, OCS (deg)')
    ax.legend(fontsize=7, loc='lower left')
    ax.set_title(f'Pupil pass 4 sample: {len(cand)} donut sides. Central band 0-0.8 deg, outer '
                 '1.4-1.75 deg; dashed circles 0.8, 1.4, 1.6, 1.725, 1.75 deg.\nFilled: extra-focal, '
                 'open: intra-focal. Labels: pass 3 sample ids', fontsize=9)
    pdf.savefig(fig, dpi=110)
    plt.close(fig)


def counts(cand):
    c = cand.groupby(['band', 'side', 'sensor_class']).size().rename('n_sides').reset_index()
    return c


# ---------------------------------------------------------------- step 1

def central_stack(bins, fits, k):
    """Per OCS bin medians over central science donuts of data and model k, fractions of r_out.

    Returns a frame with one row per (side, OCS bin).
    """
    sel = fits[(fits.fit == k) & fits.fit_ok & (fits.band == 'central')
               & (fits.sensor_class == 'science')][['sample_id', 'side']]
    b = bins.merge(sel, on=['sample_id', 'side'])
    b = b.copy()
    b['obin'] = np.minimum((b.az_ocs_deg / BIN_DEG).astype(int), NBIN - 1)
    r0 = b.groupby(['sample_id', 'side']).r_outer_data_pix.transform('median')
    for name in ('data', k):
        for edge in ('inner', 'outer'):
            v = b[f'r_{edge}_{name}_pix'].where(~b.spider)
            b[f'{edge}_{name}'] = v / r0
        b[f'ratio_{name}'] = b[f'r_inner_{name}_pix'].where(~b.spider) / \
            b[f'r_outer_{name}_pix'].where(~b.spider)
    for q in ('inner', 'outer', 'ratio'):
        b[f'{q}_dmm'] = b[f'{q}_data'] - b[f'{q}_{k}']
    rows = []
    cols = [f'{q}_{n}' for q in ('inner', 'outer', 'ratio') for n in ('data', k, 'dmm')]
    for (side, ob), g in b.groupby(['side', 'obin']):
        row = dict(side=side, obin=ob, az_deg=(ob + 0.5) * BIN_DEG,
                   spider_frac=float(g.spider.mean()))
        for c in cols:
            v = g[c].dropna().to_numpy()
            row[c] = np.median(v) if len(v) else np.nan
            # standard error of the median, from the nMAD
            row[f'{c}_err'] = 1.2533 * nmad(v) / np.sqrt(len(v)) if len(v) > 2 else np.nan
            row[f'{c}_n'] = len(v)
        rows.append(row)
    return pd.DataFrame(rows), sel.sample_id.nunique(), sel.groupby('side').size().to_dict()


def _turn(df, side, turn):
    d = df[df.side == side].sort_values('az_deg').copy()
    if turn and side == 'extra':
        d['az_deg'] = np.mod(d.az_deg + 180.0, 360.0)
        d = d.sort_values('az_deg')
    return d


def flip_correlation(st, k):
    """Pearson r and Spearman rho of extra against intra data − model, image and pupil frame."""
    rows = []
    e = st[st.side == 'extra'].set_index('obin')
    i = st[st.side == 'intra'].set_index('obin')
    for frame, shift in (('image', 0), ('pupil', NBIN // 2)):
        for q in ('inner', 'outer', 'ratio'):
            ev = e[f'{q}_dmm'].reindex(np.mod(np.arange(NBIN) - shift, NBIN)).to_numpy()
            iv = i[f'{q}_dmm'].reindex(np.arange(NBIN)).to_numpy()
            ok = np.isfinite(ev) & np.isfinite(iv)
            rows.append(dict(model=k, frame=frame, quantity=q, n_bins=int(ok.sum()),
                             pearson_r=pearsonr(ev[ok], iv[ok])[0] if ok.sum() > 3 else np.nan,
                             spearman_rho=spearmanr(ev[ok], iv[ok])[0] if ok.sum() > 3 else np.nan))
    return pd.DataFrame(rows)


def page_central(pdf, st, k, turn, n_donuts, n_sides, corr):
    fig, axes = plt.subplots(4, 1, figsize=(12, 15), sharex=True)
    labels = {'inner': 'inner edge / median r_outer', 'outer': 'outer edge / median r_outer',
              'ratio': 'r_inner / r_outer'}
    for side, col in SIDE_COLOURS.items():
        d = _turn(st, side, turn)
        lab = f'{side}' + (' (turned 180 deg)' if turn and side == 'extra' else '')
        for ax, q in zip(axes[:3], ('inner', 'outer', 'ratio')):
            ax.plot(d.az_deg, d[f'{q}_data'], '-', color=col, lw=1.2, label=f'data, {lab}')
            ax.fill_between(d.az_deg, d[f'{q}_data'] - d[f'{q}_data_err'],
                            d[f'{q}_data'] + d[f'{q}_data_err'], color=col, alpha=0.2, lw=0)
            ax.plot(d.az_deg, d[f'{q}_{k}'], '--', color=col, lw=1.0, label=f'fit {k}, {lab}')
        for q, ls in (('inner', '-'), ('outer', ':'), ('ratio', '-.')):
            axes[3].plot(d.az_deg, d[f'{q}_dmm'], ls, color=col, lw=1.1, label=f'{q}, {lab}')
            if q != 'outer':
                axes[3].fill_between(d.az_deg, d[f'{q}_dmm'] - d[f'{q}_dmm_err'],
                                     d[f'{q}_dmm'] + d[f'{q}_dmm_err'], color=col, alpha=0.12,
                                     lw=0)
        sp = d[d.spider_frac >= 0.5].az_deg
        for ax in axes:
            for x in sp:
                ax.axvspan(x - 1, x + 1, color=col, alpha=0.12, lw=0, zorder=0)
    for ax, q in zip(axes[:3], ('inner', 'outer', 'ratio')):
        ax.set_ylabel(f'{labels[q]} (dimensionless)')
    axes[0].legend(fontsize=7, ncol=2)
    axes[3].axhline(0, color='0.6', lw=0.6)
    axes[3].set_ylabel('data − model (dimensionless)')
    axes[3].legend(fontsize=7, ncol=2)
    axes[3].set_xlim(0, 360)
    axes[3].set_xlabel('azimuth about the donut centre, OCS image (deg)'
                       + ('; extra-focal turned by 180 deg (pupil frame)' if turn else ''))
    frame = 'pupil' if turn else 'image'
    c = corr[corr.frame == frame].set_index('quantity')
    ctext = '; '.join(f'{q}: Pearson r {c.loc[q].pearson_r:+.2f}, Spearman rho '
                      f'{c.loc[q].spearman_rho:+.2f}, n {int(c.loc[q].n_bins)} bins'
                      for q in ('inner', 'outer', 'ratio'))
    axes[0].set_title(
        f'Step {"4" if k == "c0" else "1"}, page {2 if turn else 1}: central band 0-0.8 deg, science, '
        f'{n_donuts} donuts ({n_sides.get("extra", 0)} extra, {n_sides.get("intra", 0)} intra '
        f'sides); {FIT_LABELS[k]}.\nMedian over donuts per 2 deg bin; band: standard error of the '
        'median (1.2533 nMAD / sqrt n). Edges as fractions of each donut\'s median data r_outer. '
        'Shaded: spider in half the donuts or more.\n'
        f'Extra against intra data − model, {frame} frame: {ctext}', fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------- step 2

def add_psi(bins, fits):
    phi = fits.drop_duplicates(['sample_id', 'side']).set_index(['sample_id', 'side']).field_az_ocs_deg
    b = bins.merge(phi.rename('phi_deg').reset_index(), on=['sample_id', 'side'])
    b['psi_deg'] = np.mod(b.az_ocs_deg - (b.phi_deg + 180.0), 360.0)
    b['pbin'] = np.minimum((b.psi_deg / BIN_DEG).astype(int), NBIN - 1)
    b['obin'] = np.minimum((b.az_ocs_deg / BIN_DEG).astype(int), NBIN - 1)
    return b


def psi_check(bins, fits):
    """Where the outer edge is pulled in, against psi, on every corner side and the outermost
    science sides: the minimum of an m = 1, 2 harmonic fit to the data outer edge."""
    import statsmodels.api as sm
    b = add_psi(bins, fits)
    f = fits[(fits.fit == 'c') & fits.fit_ok & ((fits.sensor_class != 'science')
                                                 | (fits.field_r_deg > 1.7))]
    rows = []
    for _, r in f.iterrows():
        g = b[(b.sample_id == r.sample_id) & (b.side == r.side)]
        y = g.r_outer_data_pix.to_numpy()
        ok = np.isfinite(y)
        if ok.sum() < 30:
            continue
        for frame, az in (('psi', g.psi_deg), ('ocs', g.az_ocs_deg)):
            a = np.deg2rad(az.to_numpy())
            X = np.column_stack([np.ones_like(a), np.cos(a), np.sin(a), np.cos(2 * a), np.sin(2 * a)])
            p = sm.RLM(y[ok], X[ok], M=sm.robust.norms.HuberT()).fit().params
            grid = np.deg2rad(np.arange(0, 360, 1.0))
            G = np.column_stack([np.ones_like(grid), np.cos(grid), np.sin(grid), np.cos(2 * grid),
                                 np.sin(2 * grid)])
            m = G @ p
            if frame == 'psi':
                psi_min, depth = float(np.degrees(grid[np.argmin(m)])), float(m.max() - m.min())
            else:
                ocs_min = float(np.degrees(grid[np.argmin(m)]))
        vig = g.psi_deg[g.vignetted].to_numpy()
        vig_psi = float(np.mod(np.degrees(np.arctan2(np.sin(np.deg2rad(vig)).mean(),
                                                     np.cos(np.deg2rad(vig)).mean())), 360)) \
            if len(vig) else np.nan
        rows.append(dict(sample_id=r.sample_id, pass3_id=r.pass3_id, side=r.side,
                         vignetted_mean_psi_deg=vig_psi, n_vignetted_bins=len(vig),
                         sensor_class=r.sensor_class, field_r_deg=r.field_r_deg,
                         phi_deg=r.field_az_ocs_deg, outer_min_psi_deg=psi_min,
                         outer_min_ocs_deg=ocs_min, outer_depth_pix=depth,
                         n_bins=int(ok.sum())))
    return pd.DataFrame(rows)


def _circ_spread(deg):
    """Circular standard deviation (deg) of angles in deg."""
    a = np.deg2rad(np.asarray(deg))
    R = np.hypot(np.cos(a).mean(), np.sin(a).mean())
    return float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-12)))))


def psi_select(fits, k, group):
    cls, side = group
    f = fits[(fits.fit == k) & fits.fit_ok & (fits.side == side) & (fits.sensor_class == cls)]
    if cls == 'science':
        f = f[f.field_r_deg.between(*PSI_BAND)]
    else:
        f = f[f.band == 'outer']
    return f


def psi_stats(b, k, label):
    """Scatter between donuts per bin, in psi and in OCS, and the psi-binned data − model."""
    b = b.copy()
    out, miss = [], []
    for edge in ('inner', 'outer'):
        for name in ('data', k):
            v = b[f'r_{edge}_{name}_pix']
            b[f'{edge}_{name}_rel'] = v - v.groupby([b.sample_id, b.side]).transform('median')
        b[f'{edge}_dmm'] = b[f'r_{edge}_data_pix'] - b[f'r_{edge}_{k}_pix']
    b['ratio_dmm'] = (b.r_inner_data_pix / b.r_outer_data_pix
                      - b[f'r_inner_{k}_pix'] / b[f'r_outer_{k}_pix'])
    for edge in ('inner', 'outer'):
        row = dict(group=label, model=k, edge=edge, n_sides=b.groupby(['sample_id', 'side']).ngroups)
        for frame, col in (('psi', 'pbin'), ('ocs', 'obin')):
            for what in ('data_rel', 'dmm'):
                c = f'{edge}_{what}' if what == 'dmm' else f'{edge}_data_rel'
                s = b.groupby(col)[c].agg(lambda v: nmad(v.dropna().to_numpy())
                                          if v.notna().sum() >= 3 else np.nan)
                row[f'scatter_{what}_{frame}_pix'] = float(np.nanmedian(s))
        out.append(row)
    g = b.groupby('pbin')
    med = pd.DataFrame({q: g[f'{q}_dmm'].median() for q in ('inner', 'outer', 'ratio')})
    med['n'] = g.inner_dmm.count()
    med['psi_deg'] = (med.index + 0.5) * BIN_DEG
    # 20 deg running median, so a single noisy bin does not pick the extreme
    for q in ('inner', 'outer', 'ratio'):
        sm = med[q].rolling(10, center=True, min_periods=4).median()
        i = int(np.nanargmax(np.abs(sm.to_numpy())))
        miss.append(dict(group=label, model=k, quantity=q, worst_psi_deg=float(med.psi_deg.iloc[i]),
                         worst_value=float(sm.iloc[i]), median_over_psi=float(med[q].median()),
                         unit='pixels' if q != 'ratio' else 'dimensionless'))
    return pd.DataFrame(out), pd.DataFrame(miss), med


def page_psi(pdf, b, k, label, f, stats, miss, med, step):
    keys = list(f[['sample_id', 'side']].itertuples(index=False, name=None))
    cmap = plt.get_cmap('tab20' if len(keys) <= 20 else 'gist_rainbow')
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    spiders = []
    for i, key in enumerate(keys):
        col = cmap(i / max(len(keys) - 1, 1)) if len(keys) > 20 else cmap(i)
        g = b[(b.sample_id == key[0]) & (b.side == key[1])].sort_values('psi_deg')
        for ax, q in zip(axes[:3], ('inner', 'outer', 'ratio')):
            if q == 'ratio':
                yd = g.r_inner_data_pix / g.r_outer_data_pix
                ym = g[f'r_inner_{k}_pix'] / g[f'r_outer_{k}_pix']
            else:
                yd, ym = g[f'r_{q}_data_pix'], g[f'r_{q}_{k}_pix']
            ax.plot(g.psi_deg, yd, '.', color=col, ms=2.5)
            ax.plot(g.psi_deg, ym, '-', color=col, lw=0.8)
        spiders.append((g.psi_deg[g.spider].to_numpy(), col))
    for ax in axes[:3]:
        y0, y1 = ax.get_ylim()
        for sp, col in spiders:
            ax.plot(sp, np.full(len(sp), y0 + 0.02 * (y1 - y0)), '|', color=col, ms=6)
    for q, ls, c in (('inner', '-', 'k'), ('outer', '-', 'tab:orange')):
        axes[3].plot(med.psi_deg, med[q], ls, color=c, lw=1.2, label=f'{q} edge')
    axes[3].axhline(0, color='0.6', lw=0.6)
    axes[3].set_ylabel('median data − model over donuts (pixels)')
    ax2 = axes[3].twinx()
    ax2.plot(med.psi_deg, med.ratio, ':', color='tab:green', lw=1.2, label='ratio')
    ax2.set_ylabel('ratio data − model (dimensionless)', color='tab:green')
    axes[3].legend(fontsize=7, loc='upper left')
    axes[0].set_ylabel('inner edge (pixels)')
    axes[1].set_ylabel('outer edge (pixels)')
    axes[2].set_ylabel('r_inner / r_outer (dimensionless)')
    axes[3].set_xlim(0, 360)
    axes[3].set_xlabel('relative azimuth psi (deg): 0 nearest the field centre, anticlockwise')
    s = stats.set_index('edge')
    m = miss.set_index('quantity')
    st = '; '.join(f'{e}: scatter between donuts {s.loc[e].scatter_data_rel_psi_pix:.2f} in psi, '
                   f'{s.loc[e].scatter_data_rel_ocs_pix:.2f} in OCS pixels' for e in ('inner', 'outer'))
    ms = '; '.join(f'{q} {m.loc[q].worst_value:+.3g} {m.loc[q].unit} at psi {m.loc[q].worst_psi_deg:.0f} deg'
                   for q in ('inner', 'outer', 'ratio'))
    axes[0].set_title(f'Step {step}: {label}, {len(keys)} sides, {FIT_LABELS[k]}. Dots: data; line: '
                      'model, same colour; ticks at the bottom: spider bins per donut.\n'
                      f'{st} (median over bins of the nMAD across donuts, each donut\'s median '
                      'edge subtracted).\nLargest median data − model (20 deg running median): '
                      f'{ms}', fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig, dpi=80)
    plt.close(fig)


PSI_GROUPS = [(('science', 'intra'), 'science intra, 1.6-1.725 deg'),
              (('science', 'extra'), 'science extra, 1.6-1.725 deg'),
              (('SW0', 'intra'), 'SW0 intra, 1.4-1.75 deg'), (('SW0', 'extra'), 'SW0 extra'),
              (('SW1', 'intra'), 'SW1 intra'), (('SW1', 'extra'), 'SW1 extra, 1.4-1.75 deg')]


def step_psi(pdf, bins, fits, k, step):
    b_all = add_psi(bins, fits)
    stats, misses = [], []
    for group, label in PSI_GROUPS:
        f = psi_select(fits, k, group)
        if len(f) < 3:
            continue
        b = b_all.merge(f[['sample_id', 'side']], on=['sample_id', 'side'])
        s, m, med = psi_stats(b, k, label)
        page_psi(pdf, b, k, label, f, s, m, med, step)
        stats.append(s)
        misses.append(m)
    return pd.concat(stats, ignore_index=True), pd.concat(misses, ignore_index=True)


# ---------------------------------------------------------------- step 3

def c0_vs_c(fits):
    c = fits[(fits.fit == 'c') & fits.fit_ok].set_index(['sample_id', 'side'])
    c0 = fits[(fits.fit == 'c0') & fits.fit_ok].set_index(['sample_id', 'side'])
    j = c.join(c0, rsuffix='_c0', how='inner')
    j = j[j.band == 'outer']
    sides = pd.DataFrame(dict(
        band=j.band, sensor_class=j.sensor_class, field_r_deg=j.field_r_deg, mag=j.mag,
        chi2_dof_c=j.chi2_dof, chi2_dof_c0=j.chi2_dof_c0, chi2_ratio=j.chi2_dof_c0 / j.chi2_dof,
        rms_inner_c=j.edge_rms_inner_pix, rms_inner_c0=j.edge_rms_inner_pix_c0,
        rms_outer_c=j.edge_rms_outer_pix, rms_outer_c0=j.edge_rms_outer_pix_c0,
        z4_c=j.z4_ocs_um, z4_c0=j.z4_ocs_um_c0, z5_c=j.z5_ocs_um, z6_c=j.z6_ocs_um,
        z7_c=j.z7_ocs_um, z8_c=j.z8_ocs_um)).reset_index()
    sides['match_inner'] = sides.rms_inner_c0 - sides.rms_inner_c <= MATCH_PIX
    sides['match_outer'] = sides.rms_outer_c0 - sides.rms_outer_c <= MATCH_PIX
    sides['match_both'] = sides.match_inner & sides.match_outer
    sides['astig_coma_c_max_um'] = sides[['z5_c', 'z6_c', 'z7_c']].abs().max(axis=1)
    rows = []
    groups = [('all', sides)] + [(f'{cls} {side}', g) for (cls, side), g in
                                 sides.groupby(['sensor_class', 'side'])]
    for name, g in groups:
        big = g[g.astig_coma_c_max_um > 1.0]
        rows.append(dict(group=name, n_sides=len(g), chi2_ratio_median=g.chi2_ratio.median(),
                         rms_inner_c_median=g.rms_inner_c.median(),
                         rms_inner_c0_median=g.rms_inner_c0.median(),
                         rms_outer_c_median=g.rms_outer_c.median(),
                         rms_outer_c0_median=g.rms_outer_c0.median(),
                         frac_match_inner=g.match_inner.mean(), frac_match_outer=g.match_outer.mean(),
                         frac_match_both=g.match_both.mean(),
                         n_c_z567_over_1um=len(big),
                         frac_match_both_if_c_z567_over_1um=big.match_both.mean() if len(big) else np.nan,
                         chi2_ratio_median_if_c_z567_over_1um=big.chi2_ratio.median() if len(big) else np.nan))
    return pd.DataFrame(rows), sides


def page_c0_summary(pdf, sides):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for side, col in SIDE_COLOURS.items():
        for cls, m in (('science', 'o'), ('SW0', 's'), ('SW1', 'D')):
            s = sides[(sides.side == side) & (sides.sensor_class == cls)]
            kw = dict(color=col, marker=m, ls='', ms=5, mfc='none' if cls != 'science' else col,
                      label=f'{cls} {side}')
            axes[0, 0].plot(s.field_r_deg, s.chi2_ratio, **kw)
            axes[0, 1].plot(s.astig_coma_c_max_um, s.chi2_ratio, **kw)
            axes[1, 0].plot(s.rms_inner_c, s.rms_inner_c0, **kw)
            axes[1, 1].plot(s.rms_outer_c, s.rms_outer_c0, **kw)
    axes[0, 0].set_xlabel('field radius (deg)')
    axes[0, 0].set_ylabel('chi2/dof c0 over chi2/dof c (dimensionless)')
    axes[0, 0].set_yscale('log')
    axes[0, 1].set_xlabel('fit c max(|Z5|, |Z6|, |Z7|), OCS (µm of wavefront)')
    axes[0, 1].set_ylabel('chi2/dof c0 over chi2/dof c (dimensionless)')
    axes[0, 1].set_yscale('log')
    for ax, edge in ((axes[1, 0], 'inner'), (axes[1, 1], 'outer')):
        lim = [0, np.nanpercentile(sides[[f'rms_{edge}_c', f'rms_{edge}_c0']].to_numpy(), 98) * 1.1]
        ax.plot(lim, lim, 'k-', lw=0.6)
        ax.plot(lim, [lim[0] + MATCH_PIX, lim[1] + MATCH_PIX], 'k:', lw=0.6)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f'{edge} edge RMS, fit c (pixels)')
        ax.set_ylabel(f'{edge} edge RMS, fit c0 (pixels)')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Step 3: fit c0 against fit c, outer band 1.4-1.75 deg. Dotted: c0 RMS = c RMS '
                 f'+ {MATCH_PIX} pixels', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, dpi=90)
    plt.close(fig)


def _ztext(r):
    zs = [c for c in r.index if c.startswith('z') and c.endswith('_ocs_um') and np.isfinite(r[c])]
    return '  '.join(f'{c[:-7].upper()} {r[c]:+.2f}' for c in zs)


def page_donut_images(pdf, run, g):
    r0 = g.loc['c']
    d = np.load(run / 'images' / f'{r0.sample_id}_{r0.side}.npz')
    d0 = np.load(run / 'images' / f'{r0.sample_id}_{r0.side}_c0.npz')
    img, msk = d['img'].astype(float), d['masked'].astype(bool)
    data = np.where(msk, np.nan, img)
    vmax = np.nanpercentile(data, 99.5)
    models = {'c': d['model_c'], 'c0': d0['model_c0']}
    res = {k: np.where(msk, np.nan, img - m) for k, m in models.items()}
    rl = max(np.nanpercentile(np.abs(r), 99) for r in res.values())
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    for i, k in enumerate(('c', 'c0')):
        r = g.loc[k]
        for j, (a, title, kw) in enumerate((
                (data, 'data', dict(vmin=0, vmax=vmax, cmap='gray')),
                (models[k], f'fit {k} model', dict(vmin=0, vmax=vmax, cmap='gray')),
                (res[k], 'data − model', dict(vmin=-rl, vmax=rl, cmap='RdBu_r')))):
            ax = axes[i, j]
            im = ax.imshow(a, origin='lower', **kw)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.7, label='electrons per pixel')
            ax.set_title(title, fontsize=9)
        axes[i, 0].text(0.0, -0.06, f'{FIT_LABELS[k]}: chi2/dof {r.chi2_dof:.3f} (dof {int(r.dof)})'
                        '\ndeviation, µm of wavefront, OCS: ' + _ztext(r),
                        transform=axes[i, 0].transAxes, fontsize=8, va='top')
    fig.suptitle(f'Step 3: {r0.pass3_id} ({r0.sample_id}) {r0.side}-focal {r0.detector}, {r0.mag:.2f} '
                 f'AB mag; field radius {r0.field_r_deg:.3f} deg, OCS azimuth '
                 f'{r0.field_az_ocs_deg:.1f} deg. CCS stamp', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=4.0)
    pdf.savefig(fig, dpi=80)
    plt.close(fig)


def page_donut_edges(pdf, b, g):
    b = b.sort_values('az_ocs_deg')
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for name, col, style in (('data', 'k', 'o'), ('c', 'tab:green', '-'), ('c0', 'tab:purple', '-')):
        kw = dict(ms=3, lw=0) if name == 'data' else dict(lw=1.2)
        label = 'data' if name == 'data' else FIT_LABELS[name]
        for ax, edge in zip(axes[:2], ('inner', 'outer')):
            ax.plot(b.az_ocs_deg, b[f'r_{edge}_{name}_pix'], style, color=col, label=label, **kw)
        axes[2].plot(b.az_ocs_deg, b[f'r_inner_{name}_pix'] / b[f'r_outer_{name}_pix'], style,
                     color=col, label=label, **kw)
    for ax in axes:
        for x in b.az_ocs_deg[b.spider]:
            ax.axvspan(x - 1, x + 1, color='gold', alpha=0.4, lw=0, zorder=0)
    axes[0].set_ylabel('inner edge radius (pixels)')
    axes[1].set_ylabel('outer edge radius (pixels)')
    axes[2].set_ylabel('r_inner / r_outer (dimensionless)')
    axes[2].set_xlabel('azimuth about the donut centre, OCS image (deg)')
    axes[2].set_xlim(0, 360)
    axes[0].legend(fontsize=8)
    r0 = g.loc['c']
    rows = [f'{k}: data − model RMS inner {g.loc[k].edge_rms_inner_pix:.2f}, outer '
            f'{g.loc[k].edge_rms_outer_pix:.2f} pixels; chi2/dof {g.loc[k].chi2_dof:.3f}'
            for k in ('c', 'c0')]
    axes[0].set_title(f'{r0.pass3_id} {r0.side}-focal, field radius {r0.field_r_deg:.3f} deg: robust '
                      'edges; gold: spider bins\n' + '\n'.join(rows), fontsize=8)
    pdf.savefig(fig, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------- driver

def step_central(pdf, bins, fits, k):
    st, n_don, n_sides = central_stack(bins, fits, k)
    corr = flip_correlation(st, k)
    for turn in (False, True):
        page_central(pdf, st, k, turn, n_don, n_sides, corr)
    return st, corr


def report(run, cand, fits, bins):
    run = pathlib.Path(run)
    cnt = counts(cand)
    _save(cnt, run, 'counts')
    med = pd.read_csv(run / 'central_medians.csv')
    failed = fits[~fits.fit_ok][['sample_id', 'pass3_id', 'side', 'fit', 'mag', 'flux_ratio',
                                 'chi2_dof', 'error']]
    corr_all, stats_all, miss_all, stacks = [], [], [], []
    with PdfPages(run / 'pupil_pass4.pdf') as pdf:
        sample_map(pdf, cand)
        text_page(pdf, 'Sample: donut sides per band, side and sensor class',
                  cnt.to_string(index=False) + '\n\nLeft out of the summaries (fitted flux over '
                  'stamp sum more than 1.5x off the median of its fit and side, or an error):\n'
                  + failed.round(3).to_string(index=False), 8)
        st, corr = step_central(pdf, bins, fits, 'c')
        corr_all.append(corr)
        stacks.append(st.assign(model='c'))
        chk = psi_check(bins, fits)
        _save(chk, run, 'psi_check')
        text_page(pdf, 'Step 2 check of psi: azimuth of the minimum of an m = 1, 2 Huber fit to the '
                  'data outer edge (pixels), in psi and in OCS image azimuth (deg).\nIf psi is right, '
                  'the minimum sits at one psi on all sides of one focus, whatever the focal-plane '
                  'azimuth phi.\nCircular spread (deg) per side and class: '
                  + '; '.join(f'{c} {s}: psi {_circ_spread(g.outer_min_psi_deg):.0f}, OCS '
                              f'{_circ_spread(g.outer_min_ocs_deg):.0f} (n {len(g)})'
                              for (c, s), g in chk.groupby(['sensor_class', 'side'])),
                  chk.round(3).to_string(index=False), 7)
        s, m = step_psi(pdf, bins, fits, 'c', 2)
        stats_all.append(s)
        miss_all.append(m)
        summ, sides = c0_vs_c(fits)
        _save(summ, run, 'c0_vs_c')
        _save(sides, run, 'c0_vs_c_sides')
        text_page(pdf, 'Step 3: central medians of fit c deviations (OCS, µm of wavefront), central '
                  'band science sides', med.round(4).to_string(index=False), 8)
        text_page(pdf, 'Step 3: fit c0 against fit c, outer band 1.4-1.75 deg. chi2_ratio: chi2/dof '
                  'c0 over c (dimensionless, median); RMS: data − model edge RMS (pixels, median); '
                  f'match: c0 RMS within {MATCH_PIX} pixels of c',
                  summ.round(3).T.to_string(), 8)
        page_c0_summary(pdf, sides)
        pages = fits[(fits.fit == 'c') & (fits.pass3_id != '') & (fits.field_r_deg > 1.4)
                     & (fits.error == '')].sort_values(['field_r_deg', 'pass3_id', 'side'])
        n_pages = 0
        for _, r in pages.iterrows():
            g = fits[(fits.sample_id == r.sample_id) & (fits.side == r.side)].set_index('fit')
            if 'c0' not in g.index or not (run / 'images' / f'{r.sample_id}_{r.side}_c0.npz').exists():
                continue
            b = bins[(bins.sample_id == r.sample_id) & (bins.side == r.side)]
            page_donut_images(pdf, run, g)
            page_donut_edges(pdf, b, g)
            n_pages += 2
        st, corr = step_central(pdf, bins, fits, 'c0')
        corr_all.append(corr)
        stacks.append(st.assign(model='c0'))
        s, m = step_psi(pdf, bins, fits, 'c0', 4)
        stats_all.append(s)
        miss_all.append(m)
    _save(pd.concat(corr_all, ignore_index=True), run, 'flip_correlation')
    _save(pd.concat(stats_all, ignore_index=True), run, 'psi_scatter')
    _save(pd.concat(miss_all, ignore_index=True), run, 'psi_miss')
    _save(pd.concat(stacks, ignore_index=True), run, 'central_stack')
    print('per-donut pages', n_pages, 'report', run / 'pupil_pass4.pdf', flush=True)
