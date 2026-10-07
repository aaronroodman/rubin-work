"""Figures for the all-v-mode thermal study.

Five pages, each a function taking an open `matplotlib.backends.backend_pdf.PdfPages`. The
tables are produced by `thermal_vmodes`; nothing here fits anything except the two per-visit
Huber lines on `figure_response_scatter`, which are drawn to show the relation the night-grouped
skill summarizes rather than to restate it.

Three of the pages exist because the prose version of the result is weak on its own:

* `figure_skill` puts the false-discovery-rate threshold on the same axis as the skill, so the
  cluster of modes below the cut is visibly below it rather than asserted to be.
* `figure_residual_scale` shows the absolute residual the cluster leaves, an order of magnitude
  above v-mode 1's. Skill is a fraction, so a mode can score moderately well and still predict
  almost nothing -- that is invisible in a skill-only view.
* `figure_noise_floor` shows the between/within ratio is not monotonic in mode index, which is
  the reason `thermal_vmodes.noise_floor_table` measures it instead of assuming a cutoff.
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT))

import thermal_vmodes as TV                                      # noqa: E402
from common.utils import nmad                                    # noqa: E402

#: Mode drawn beside v-mode 1 on `figure_response_scatter`: the highest-skill mode that the
#: false-discovery-rate cut rejects, so the comparison is signal against best-non-signal.
_COMPARISON_FALLBACK = 10

_THERMAL_COLOR = '#d62728'
_PLAIN_COLOR = '#1f77b4'
_NULL_COLOR = '#7f7f7f'


def _title(fig, text, subtext=None, sub_y=0.925):
    """Page title, with an optional second line in smaller type.

    These pages are wide and short, so the subtitle sits well below the title to avoid it.
    """
    fig.suptitle(text, fontsize=11.5, y=0.992)
    if subtext:
        fig.text(0.5, sub_y, subtext, ha='center', fontsize=8.5, color='0.3')


def figure_skill(pdf, tab, q=TV.FDR_Q):
    """Per-mode thermal skill with the multiple-comparison threshold drawn on it.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    tab : `pandas.DataFrame`
        `thermal_vmodes.mode_table` result, carrying ``mode``, ``skill`` and ``thermal``.
    q : `float`, optional
        False-discovery-rate level the table was cut at (dimensionless), for the caption.

    Notes
    -----
    Two panels on a shared mode axis. The left is skill per mode with the threshold as a line;
    the right is the same values as a sorted rank plot, which is where the empirical null's shape
    is legible -- the cut is derived from the median and nMAD of these very points, so a reader
    should see the distribution it was taken from.
    """
    t = tab.sort_values('mode')
    mode = t['mode'].to_numpy(int)
    skill = t['skill'].to_numpy(float)
    thermal = t['thermal'].to_numpy(bool)
    cut = tab.attrs.get('skill_cut', np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    ax = axes[0]
    colors = np.where(thermal, _THERMAL_COLOR, _PLAIN_COLOR)
    ax.bar(mode, skill, color=colors, width=0.72)
    if np.isfinite(cut):
        ax.axhline(cut, color=_THERMAL_COLOR, lw=1.2, ls='--',
                   label=f'FDR q={q} cut, skill {cut:+.3f} (dimensionless)')
    ax.axhline(0, color='0.6', lw=0.8)
    ax.axvline(TV.WELL_CONSTRAINED_MAX + 0.5, color='0.55', lw=1.0, ls=':',
               label=f'WELL_CONSTRAINED_MAX = v{TV.WELL_CONSTRAINED_MAX}')
    ax.set_xlabel('v-mode index')
    ax.set_ylabel('skill [dimensionless, fractional reduction in\nout-of-fold residual nMAD]')
    n_th = int(thermal.sum())
    ax.set_title(f'{n_th} of {len(t)} modes called thermal\n'
                 f'red = survives the cut', fontsize=9.5)
    ax.legend(fontsize=7.5, loc='upper right')

    ax = axes[1]
    order = np.argsort(skill)[::-1]
    rank = np.arange(1, len(order) + 1)
    ax.plot(rank, skill[order], 'o', ms=5, color=_PLAIN_COLOR)
    sel = thermal[order]
    ax.plot(rank[sel], skill[order][sel], 'o', ms=8, mfc='none', mew=1.6,
            color=_THERMAL_COLOR)
    # Labels alternate above and below: the top modes are closely spaced in rank and overlap
    # when all are placed on the same side.
    for i, (r, m) in enumerate(zip(rank[:6], mode[order][:6])):
        ax.annotate(f'v{m}', (r, skill[order][r - 1]), fontsize=7,
                    xytext=(5, 6 if i % 2 == 0 else -10), textcoords='offset points')
    if np.isfinite(cut):
        ax.axhline(cut, color=_THERMAL_COLOR, lw=1.2, ls='--')
    med, scale = np.nanmedian(skill), nmad(skill)
    ax.axhline(med, color=_NULL_COLOR, lw=1.0,
               label=f'median skill {med:+.3f} (dimensionless)')
    ax.axhspan(med - scale, med + scale, color=_NULL_COLOR, alpha=0.15,
               label=f'+-1 nMAD, {scale:.3f} (dimensionless)')
    ax.set_xlabel('rank by skill')
    ax.set_ylabel('skill [dimensionless]')
    ax.set_title('The empirical null is these same points\n'
                 'the cut comes from their median and nMAD, not an analytic distribution',
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc='upper right')

    _title(fig, 'Thermal skill of every v-mode, with the screening cut',
           f'night-grouped Huber on 5 thermal channels, variant '
           f'{tab.attrs.get("variant", "unknown")}')
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    pdf.savefig(fig)
    plt.close(fig)


def figure_residual_scale(pdf, tab):
    """Absolute residual scatter per mode, null against fit.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    tab : `pandas.DataFrame`
        `thermal_vmodes.mode_table` result.

    Notes
    -----
    Skill is a *fraction*, so a mode can post a respectable skill and still leave a residual far
    too large to be useful. This page is the absolute scale that skill divides out: the modes
    scoring +0.15 to +0.25 leave an order of magnitude more residual than v-mode 1 does. Log y,
    because the modes span decades.
    """
    t = tab.sort_values('mode')
    mode = t['mode'].to_numpy(int)
    null = t['nmad_null'].to_numpy(float)
    fit = t['nmad_fit'].to_numpy(float)
    thermal = t['thermal'].to_numpy(bool)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    ax = axes[0]
    ax.semilogy(mode, null, 'o-', ms=4, lw=1.0, color=_NULL_COLOR,
                label='median-intercept null')
    ax.semilogy(mode, fit, 'o-', ms=4, lw=1.0, color=_PLAIN_COLOR, label='Huber fit')
    for m, f in zip(mode[thermal], fit[thermal]):
        ax.plot([m], [f], 'o', ms=9, mfc='none', mew=1.6, color=_THERMAL_COLOR)
        ax.annotate(f'v{m}', (m, f), fontsize=7.5, color=_THERMAL_COLOR,
                    xytext=(5, -9), textcoords='offset points')
    ax.set_xlabel('v-mode index')
    ax.set_ylabel('out-of-fold residual nMAD\n[dimensionless v-mode amplitude]')
    ax.set_title('Absolute residual, which skill divides out\n'
                 'red circle = called thermal', fontsize=9.5)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    ax.plot(t['skill'].to_numpy(float), fit, 'o', ms=5, color=_PLAIN_COLOR)
    for _, r in t.iterrows():
        if r['thermal'] or r['skill'] > 0.15:
            ax.annotate(f'v{int(r["mode"])}', (r['skill'], r['nmad_fit']), fontsize=7,
                        color=_THERMAL_COLOR if r['thermal'] else '0.35',
                        xytext=(5, 2), textcoords='offset points')
    ax.set_yscale('log')
    ax.set_xlabel('skill [dimensionless]')
    ax.set_ylabel('residual nMAD after the fit\n[dimensionless v-mode amp.]')
    ax.set_title('Moderate skill, large residual\n'
                 'the +0.15 to +0.25 cluster predicts little of what is there', fontsize=9.5)

    _title(fig, 'What is left after the fit, in absolute terms',
           'a fractional improvement on a large scatter is still a large scatter')
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    pdf.savefig(fig)
    plt.close(fig)


def figure_noise_floor(pdf, floor, ratio_cut=2.0):
    """Within-night against between-night scatter, per mode.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    floor : `pandas.DataFrame`
        `thermal_vmodes.noise_floor_table` result.
    ratio_cut : `float`, optional
        Between/within ratio (dimensionless) below which a mode carries no more night-to-night
        structure than its own measurement noise.

    Notes
    -----
    The ratio is **not monotonic in mode index**, which is the whole reason it is measured. A
    high mode can be better determined night to night than a low one, so
    `thermal_vmodes.WELL_CONSTRAINED_MAX` -- a statement about the recovery's conditioning -- is
    drawn here for contrast but is deliberately not the same line as the ratio cut.
    """
    f = floor.sort_values('mode')
    mode = f['mode'].to_numpy(int)
    within = f['within_night'].to_numpy(float)
    between = f['between_night'].to_numpy(float)
    ratio = f['signal_to_noise'].to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    ax = axes[0]
    ax.semilogy(mode, within, 'o-', ms=4, lw=1.0, color=_NULL_COLOR,
                label='within-night, successive-visit differences / sqrt(2)')
    ax.semilogy(mode, between, 'o-', ms=4, lw=1.0, color=_PLAIN_COLOR,
                label='between-night, nMAD of night medians')
    ax.set_xlabel('v-mode index')
    ax.set_ylabel('scatter [dimensionless v-mode amplitude]')
    ax.set_title('Measurement noise against night-to-night structure', fontsize=9.5)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    below = ratio <= ratio_cut
    # Bars grow from ratio 1 -- equal between-night and within-night scatter -- rather than from
    # 0, which a log axis cannot show. A bar below the baseline is a mode whose night-to-night
    # spread is smaller than its own per-visit noise.
    floor = 1.0
    ax.bar(mode[~below], ratio[~below] - floor, bottom=floor, color=_PLAIN_COLOR, width=0.72,
           label=f'ratio > {ratio_cut:.0f} (dimensionless)')
    ax.bar(mode[below], ratio[below] - floor, bottom=floor, color=_NULL_COLOR, width=0.72,
           label=f'ratio <= {ratio_cut:.0f}, not interpretable')
    ax.axhline(floor, color='0.4', lw=0.9)
    ax.axhline(ratio_cut, color=_THERMAL_COLOR, lw=1.2, ls='--')
    ax.axvline(TV.WELL_CONSTRAINED_MAX + 0.5, color='0.55', lw=1.0, ls=':',
               label=f'WELL_CONSTRAINED_MAX = v{TV.WELL_CONSTRAINED_MAX}, a different statement')
    # Log y: v-mode 1's ratio is near 37 and the rest sit between 0.8 and 10, so a linear axis
    # compresses exactly the range where the non-monotonic structure is.
    ax.set_yscale('log')
    ax.set_xlabel('v-mode index')
    ax.set_ylabel('between-night / within-night scatter [dimensionless]')
    n_ok = int((~below).sum())
    worst = mode[below][np.argsort(ratio[below])][:3] if below.any() else []
    ax.set_title(f'{n_ok} of {len(f)} modes carry structure above their own noise\n'
                 f'not monotonic in mode index: lowest ratios at '
                 f'{", ".join(f"v{m}" for m in worst)}', fontsize=9.5)
    ax.legend(fontsize=7.5)

    _title(fig, 'Where a fitted slope stops being interpretable',
           'four corner sensors constrain 84 Zernike values, and the recovery scatter grows '
           'with mode index')
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    pdf.savefig(fig)
    plt.close(fig)


def figure_response_scatter(pdf, df, tab, feature='truss_temp_mean_c',
                            feature_label='mean TMA truss temperature [deg C]'):
    """The thermal relation itself, for v-mode 1 and for the best mode the cut rejects.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    df : `pandas.DataFrame`
        Per-visit frame carrying the ``v*_olr`` columns, `feature` and ``day_obs``.
    tab : `pandas.DataFrame`
        `thermal_vmodes.mode_table` result, used to pick the comparison mode.
    feature : `str`, optional
        Thermal channel on the x axis.
    feature_label : `str`, optional
        Axis label, carrying the unit.

    Notes
    -----
    Two panels at matched scale: v-mode 1, and the highest-skill mode the false-discovery-rate
    cut rejects. Showing v-mode 1 alone would prove nothing about the other 33 -- the comparison
    is what makes the gap visible. Per-visit points as a hexbin with night medians over them,
    because the signal is a between-night one and the per-visit cloud is dominated by
    measurement scatter.

    The Huber line is fitted on the per-visit points here, which is **not** the night-grouped
    out-of-fold number the skill column reports; it is drawn to show the relation, and the title
    gives the grouped skill alongside so the two are not confused.
    """
    import statsmodels.api as sm
    from scipy.stats import pearsonr, spearmanr

    rejected = tab[~tab['thermal'].astype(bool) & np.isfinite(tab['skill'])]
    other = (int(rejected.sort_values('skill', ascending=False)['mode'].iloc[0])
             if len(rejected) else _COMPARISON_FALLBACK)
    thermal_modes = tab[tab['thermal'].astype(bool)]['mode'].astype(int).tolist()
    modes = (thermal_modes[:1] or [1]) + [other]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, mode in zip(axes, modes):
        d = TV.attach_mode_response(df, mode)
        m = np.isfinite(d['y']) & np.isfinite(d[feature])
        x = d.loc[m, feature].to_numpy(float)
        y = d.loc[m, 'y'].to_numpy(float)

        hb = ax.hexbin(x, y, gridsize=55, bins='log', mincnt=1, cmap='Blues')
        fig.colorbar(hb, ax=ax, label='visits per cell', pad=0.02)

        pn = d.loc[m].groupby('day_obs').agg(xm=(feature, 'median'), ym=('y', 'median'))
        ax.plot(pn['xm'], pn['ym'], 'o', ms=4.5, color='#2ca02c', label='night medians')

        X = sm.add_constant(x)
        fit = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
        xs = np.linspace(x.min(), x.max(), 20)
        ax.plot(xs, fit.params[0] + fit.params[1] * xs, '-', color=_THERMAL_COLOR, lw=1.5,
                label=f'Huber per-visit {fit.params[1]:+.4f} +- {fit.bse[1]:.4f}\n'
                      f'[dimensionless v-mode amplitude per deg C]')

        row = tab[tab['mode'] == mode].iloc[0]
        flag = 'called thermal' if bool(row['thermal']) else 'rejected by the cut'
        ax.set_xlabel(feature_label)
        ax.set_ylabel(f'v-mode {mode} optical state, Trim - Deviation\n'
                      f'[dimensionless v-mode amplitude]')
        ax.set_title(f'v-mode {mode}: {flag}\n'
                     f'grouped skill {row["skill"]:+.3f} (dimensionless), Pearson r '
                     f'{pearsonr(x, y)[0]:+.3f}, Spearman rho {spearmanr(x, y)[0]:+.3f}, '
                     f'n {len(x)} visits', fontsize=9.5)
        ax.legend(fontsize=7, loc='best')

    _title(fig, 'The thermal relation, signal against best non-signal',
           'the line is a per-visit Huber fit; the skill in each title is the night-grouped '
           'out-of-fold value')
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    pdf.savefig(fig)
    plt.close(fig)


def figure_intrinsic(pdf, cmp):
    """Per-mode skill on the two intrinsic routes against each other.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    cmp : `pandas.DataFrame`
        `thermal_vmodes.intrinsic_comparison` result.

    Notes
    -----
    The measured intrinsic wavefront differs from the batoid prediction by a static offset per
    rotator angle, and a static offset moves a fitted intercept rather than a thermal slope, so
    the two routes are expected to land on the same modes. Modes where the thermal flag differs
    are labelled: the test of "threshold artefact" against "real disagreement" is whether such a
    mode sits on the 1:1 line, which it does if the skills agree and only the cut separates them.
    """
    mode = cmp['mode'].to_numpy(int)
    sb = cmp['skill_batoid'].to_numpy(float)
    sm_ = cmp['skill_miw'].to_numpy(float)
    disagree = cmp['thermal_batoid'].astype(bool) != cmp['thermal_miw'].astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    ax = axes[0]
    ax.plot(sb, sm_, 'o', ms=5, color=_PLAIN_COLOR)
    lo = float(np.nanmin([sb.min(), sm_.min()]))
    hi = float(np.nanmax([sb.max(), sm_.max()]))
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], '-', color='0.6', lw=1.0, label='1:1')
    for _, r in cmp[disagree].iterrows():
        ax.plot([r['skill_batoid']], [r['skill_miw']], 'o', ms=9, mfc='none', mew=1.6,
                color=_THERMAL_COLOR)
        ax.annotate(f'v{int(r["mode"])}', (r['skill_batoid'], r['skill_miw']), fontsize=7.5,
                    color=_THERMAL_COLOR, xytext=(6, 2), textcoords='offset points')
    ax.set_xlabel('skill, batoid intrinsic [dimensionless]')
    ax.set_ylabel('skill, measured intrinsic wavefront [dimensionless]')
    n_agree = int((~disagree).sum())
    ax.set_title(f'Flags agree on {n_agree} of {len(cmp)} modes\n'
                 f'red circle = thermal on one route only', fontsize=9.5)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    diff = cmp['skill_diff'].to_numpy(float)
    ax.bar(mode, diff, color=_PLAIN_COLOR, width=0.72)
    for _, r in cmp[disagree].iterrows():
        ax.bar([int(r['mode'])], [r['skill_diff']], color=_THERMAL_COLOR, width=0.72)
    ax.axhline(0, color='0.6', lw=0.8)
    med = np.nanmedian(np.abs(diff))
    ax.axhline(med, color=_NULL_COLOR, lw=1.0, ls='--',
               label=f'median |difference| {med:.4f} (dimensionless)')
    ax.axhline(-med, color=_NULL_COLOR, lw=1.0, ls='--')
    ax.set_xlabel('v-mode index')
    ax.set_ylabel('skill difference, MIW minus batoid [dimensionless]')
    ax.set_title('Per-mode difference between the two intrinsics\n'
                 'solver held fixed, both unconstrained 50/34', fontsize=9.5)
    ax.legend(fontsize=7.5)

    _title(fig, 'Do the two intrinsic routes agree on which modes are thermal?',
           'a static intrinsic offset moves an intercept, not a thermal slope, so they should')
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(path, df, tab, floor, cmp=None, feature='truss_temp_mean_c'):
    """Assemble every page into one document.

    Parameters
    ----------
    path : `pathlib.Path` or `str`
        Output PDF path.
    df : `pandas.DataFrame`
        Per-visit frame for the primary variant.
    tab : `pandas.DataFrame`
        `thermal_vmodes.mode_table` result.
    floor : `pandas.DataFrame`
        `thermal_vmodes.noise_floor_table` result.
    cmp : `pandas.DataFrame`, optional
        `thermal_vmodes.intrinsic_comparison` result; its page is skipped when absent.
    feature : `str`, optional
        Thermal channel for the scatter page.

    Returns
    -------
    path : `pathlib.Path`
        The written path.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    path = pathlib.Path(path)
    with PdfPages(path) as pdf:
        figure_skill(pdf, tab)
        figure_residual_scale(pdf, tab)
        figure_noise_floor(pdf, floor)
        figure_response_scatter(pdf, df, tab, feature=feature)
        if cmp is not None and len(cmp):
            figure_intrinsic(pdf, cmp)
    return path
