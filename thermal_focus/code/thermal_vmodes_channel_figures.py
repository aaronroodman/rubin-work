"""Figures for the per-channel thermal screen.

The heatmap is the whole screen in one view: 34 v-modes down, every thermal channel across,
Spearman rho as colour on a diverging scale centred at zero so the **sign** is readable -- which
is what the single-skill version of this study could not show.

Each mode clearing the threshold then gets a page: the response against its leading channel, the
out-of-fold prediction against the response, and the residual histogram before and after the
combined fit with both nMAD values. That page is what decides whether a mode has a thermal
origin, rather than merely a correlation.
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
import thermal_vmodes_channels as TVC                            # noqa: E402
from common.utils import nmad                                    # noqa: E402

_STRONG_COLOR = '#d62728'
_NULL_COLOR = '#7f7f7f'
_FIT_COLOR = '#1f77b4'


def _labels(channels):
    """Short human labels for channel columns, in the given order."""
    tab = TVC.channel_table().set_index('channel')
    return [tab['label'].get(c, c) for c in channels]


def _title(fig, text, subtext=None, sub_y=0.925):
    fig.suptitle(text, fontsize=11.5, y=0.992)
    if subtext:
        fig.text(0.5, sub_y, subtext, ha='center', fontsize=8.5, color='0.3')


def figure_heatmap(pdf, grid, rho_strong=TVC.RHO_STRONG, value='spearman_rho'):
    """The whole screen: v-mode against thermal channel, rank correlation as colour.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    grid : `pandas.DataFrame`
        `thermal_vmodes_channels.channel_grid` result.
    rho_strong : `float`, optional
        Threshold marked on the colour bar and used to annotate cells.
    value : `str`, optional
        Grid column to show.

    Notes
    -----
    Diverging colour map centred at zero, because the sign is the point: a mode that
    anti-correlates with truss temperature is not thermal in the sense the deliverable means,
    and a single-signed scale would hide that. Cells clearing `rho_strong` carry their value as
    text, so the follow-up set is readable off the figure.
    """
    wide = TVC.pivot_rho(grid, value=value)
    modes = wide.index.to_numpy(int)
    channels = list(wide.columns)
    m = wide.to_numpy(float)

    height = 2.4 + 0.26 * len(modes)
    fig, ax = plt.subplots(figsize=(1.0 + 0.52 * len(channels), height))
    lim = 1.0
    im = ax.imshow(m, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim,
                   origin='upper', interpolation='nearest')
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03)
    cb.set_label('Spearman rho, v-mode optical state against channel [dimensionless]')
    for level in (-rho_strong, rho_strong):
        cb.ax.axhline(level, color='0.1', lw=1.2)

    ax.set_xticks(np.arange(len(channels)))
    ax.set_xticklabels(_labels(channels), rotation=55, ha='right', fontsize=7.5)
    ax.set_yticks(np.arange(len(modes)))
    ax.set_yticklabels([f'v{k}' for k in modes], fontsize=7)
    ax.set_xlabel('thermal telemetry channel')
    ax.set_ylabel('v-mode')

    n_strong = 0
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            if np.isfinite(v) and abs(v) >= rho_strong:
                n_strong += 1
                ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=5.5,
                        color='white' if abs(v) > 0.65 else 'black')

    # Separator after the M1M3 shape block and before the derived differences, so the three
    # physically distinct families are visible as blocks rather than one wall of colour.
    kinds = TVC.channel_table().set_index('channel')['kind']
    for j, c in enumerate(channels[:-1]):
        if kinds.get(c) != kinds.get(channels[j + 1]):
            ax.axvline(j + 0.5, color='0.15', lw=1.4)

    # Title block is a fixed ~0.85 inch, so its figure-fraction offsets scale with the height.
    head = 0.85 / height
    _title(fig, 'Every v-mode against every thermal channel, one channel at a time',
           f'{n_strong} of {m.size} pairs reach |Spearman rho| >= {rho_strong} (dimensionless); '
           f'cells past the threshold are labelled. Vertical rule separates base channels from '
           f'derived differences.',
           sub_y=1.0 - 0.62 * head)
    fig.tight_layout(rect=(0, 0, 1, 1.0 - head))
    pdf.savefig(fig)
    plt.close(fig)


def figure_skill_heatmap(pdf, grid):
    """The same grid as out-of-fold skill, which is signless but honest.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    grid : `pandas.DataFrame`
        `thermal_vmodes_channels.channel_grid` result.

    Notes
    -----
    Shown beside the rank-correlation grid because the two disagree in an informative way. A
    per-visit correlation can be large and buy nothing out of fold, when it comes from a few
    nights; night-grouped skill cannot. A cell strong in both is the only one worth a physical
    story.
    """
    wide = TVC.pivot_rho(grid, value='skill')
    modes = wide.index.to_numpy(int)
    channels = list(wide.columns)
    m = wide.to_numpy(float)
    m = np.where(m > 0, m, np.nan)

    height = 2.4 + 0.26 * len(modes)
    fig, ax = plt.subplots(figsize=(1.0 + 0.52 * len(channels), height))
    im = ax.imshow(m, aspect='auto', cmap='viridis', vmin=0, vmax=np.nanmax(m) if
                   np.isfinite(np.nanmax(m)) else 1.0, origin='upper', interpolation='nearest')
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03)
    cb.set_label('out-of-fold skill against a median-intercept null [dimensionless]')

    ax.set_xticks(np.arange(len(channels)))
    ax.set_xticklabels(_labels(channels), rotation=55, ha='right', fontsize=7.5)
    ax.set_yticks(np.arange(len(modes)))
    ax.set_yticklabels([f'v{k}' for k in modes], fontsize=7)
    ax.set_xlabel('thermal telemetry channel')
    ax.set_ylabel('v-mode')

    best = np.nanmax(m) if np.isfinite(np.nanmax(m)) else np.nan
    where = np.unravel_index(np.nanargmax(m), m.shape) if np.isfinite(best) else None
    note = ('no channel improves on the null anywhere' if where is None else
            f'best single channel: v{modes[where[0]]} on '
            f'{_labels([channels[where[1]]])[0]}, skill {best:+.3f} (dimensionless)')
    head = 0.85 / height
    _title(fig, 'The same grid as night-grouped out-of-fold skill',
           f'blank = no improvement on the null. {note}', sub_y=1.0 - 0.62 * head)
    fig.tight_layout(rect=(0, 0, 1, 1.0 - head))
    pdf.savefig(fig)
    plt.close(fig)


def figure_mode_combined(pdf, res, grid, df=None):
    """One followed-up mode: leading channel, combined prediction, residual before and after.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    res : `dict`
        `thermal_vmodes_channels.combined_fit` result, arrays included.
    grid : `pandas.DataFrame`
        `channel_grid` result, for the per-channel panel.
    df : `pandas.DataFrame`, optional
        Per-visit frame. When given, the first panel is the response against the leading
        channel, which is the scatter the request asks for; without it the panel falls back to
        the response distribution alone.

    Notes
    -----
    Three panels, left to right: the response against the single strongest channel with its
    Huber line; the out-of-fold combined prediction against the response, where a thermal mode
    falls on the 1:1 line; and the residual histogram before and after, with both nMAD values.

    The middle panel is the one that discriminates. A mode with a real multi-channel thermal
    dependence produces a prediction that tracks the response across its whole range; a mode
    whose correlation is a night-level artefact produces a prediction clustered near the response
    median whatever the response does.
    """
    mode = int(res['mode'])
    y = np.asarray(res['response'], float)
    pred = np.asarray(res['prediction'], float)
    resid = np.asarray(res['residual'], float)
    ok = np.isfinite(y) & np.isfinite(pred)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    ax = axes[0]
    lead = res.get('lead_channel')
    g = grid[(grid['mode'] == mode) & (grid['channel'] == lead)]
    lab = _labels([lead])[0] if lead else 'leading channel'
    unit = TVC.channel_table().set_index('channel')['unit'].get(lead, '')
    r = g.iloc[0] if len(g) else None

    if df is not None and lead in df.columns:
        d = TV.attach_mode_response(df, mode)
        mm = np.isfinite(d['y']) & np.isfinite(d[lead])
        xs = d.loc[mm, lead].to_numpy(float)
        ys = d.loc[mm, 'y'].to_numpy(float)
        hb0 = ax.hexbin(xs, ys, gridsize=50, bins='log', mincnt=1, cmap='Blues')
        fig.colorbar(hb0, ax=ax, label='visits per cell', pad=0.02)
        pn = d.loc[mm].groupby('day_obs').agg(xm=(lead, 'median'), ym=('y', 'median'))
        ax.plot(pn['xm'], pn['ym'], 'o', ms=4, color='#2ca02c', label='night medians')
        if r is not None and np.isfinite(r['slope']):
            xg = np.linspace(float(xs.min()), float(xs.max()), 20)
            # The stored fit has no intercept column, so the line is anchored on the robust
            # centre of both axes -- the slope is what the panel is about.
            ax.plot(xg, np.median(ys) + r['slope'] * (xg - np.median(xs)), '-',
                    color=_STRONG_COLOR, lw=1.5,
                    label=f'Huber {r["slope"]:+.4g} +- {r["slope_err"]:.2g}\n'
                          f'[v-mode amp. per {unit}]')
        ax.set_xlabel(f'{lab} [{unit}]')
        ax.set_ylabel(f'v-mode {mode} optical state\n[dimensionless v-mode amplitude]')
        ax.legend(fontsize=7, loc='best')
    else:
        ax.hist(y[np.isfinite(y)], bins=80, color=_NULL_COLOR, alpha=0.75)
        ax.set_xlabel(f'v-mode {mode} optical state [dimensionless v-mode amplitude]')
        ax.set_ylabel('visits')

    stat = ('' if r is None else
            f'Pearson r {r["pearson_r"]:+.3f}, Spearman rho {r["spearman_rho"]:+.3f}, '
            f'single-channel skill {r["skill"]:+.3f}')
    ax.set_title(f'v-mode {mode} against its strongest channel\n{lab} -- {stat}', fontsize=9.5)

    ax = axes[1]
    hb = ax.hexbin(pred[ok], y[ok], gridsize=50, bins='log', mincnt=1, cmap='Blues')
    fig.colorbar(hb, ax=ax, label='visits per cell', pad=0.02)
    lo = float(np.nanmin([pred[ok].min(), y[ok].min()]))
    hi = float(np.nanmax([pred[ok].max(), y[ok].max()]))
    ax.plot([lo, hi], [lo, hi], '-', color=_STRONG_COLOR, lw=1.3, label='1:1')
    from scipy.stats import pearsonr, spearmanr
    ax.set_xlabel('out-of-fold combined prediction\n[dimensionless v-mode amplitude]')
    ax.set_ylabel(f'v-mode {mode} optical state\n[dimensionless v-mode amplitude]')
    ax.set_title(f'{len(res["channels"])} channels combined, nights held out\n'
                 f'Pearson r {pearsonr(pred[ok], y[ok])[0]:+.3f}, Spearman rho '
                 f'{spearmanr(pred[ok], y[ok])[0]:+.3f}, n {int(ok.sum())} visits',
                 fontsize=9.5)
    ax.legend(fontsize=7.5)

    ax = axes[2]
    rr = resid[np.isfinite(resid)]
    yy = y[np.isfinite(y)]
    span = np.nanpercentile(np.abs(np.concatenate([yy - np.median(yy), rr])), 99.5)
    bins = np.linspace(-span, span, 90)
    ax.hist(yy - np.median(yy), bins=bins, color=_NULL_COLOR, alpha=0.65,
            label=f'before: about the median, nMAD {nmad(yy):.4g}')
    ax.hist(rr, bins=bins, color=_FIT_COLOR, alpha=0.65,
            label=f'after: combined fit, nMAD {nmad(rr):.4g}')
    ax.axvline(0, color='0.4', lw=0.9)
    ax.set_xlabel('residual [dimensionless v-mode amplitude]')
    ax.set_ylabel('visits')
    gain = (np.nan if not np.isfinite(res['nmad_single']) or res['nmad_single'] == 0
            else 1.0 - res['nmad_combined'] / res['nmad_single'])
    ax.set_title(f'Residual before and after the combined fit\n'
                 f'null {res["nmad_null"]:.4g} -> single {res["nmad_single"]:.4g} -> combined '
                 f'{res["nmad_combined"]:.4g}; combination buys {gain:+.1%} over one channel',
                 fontsize=9.5)
    ax.legend(fontsize=7.5)

    _title(fig, f'V-mode {mode}: does the thermal telemetry explain it?',
           f'channels combined: {", ".join(_labels(res["channels"]))}  |  skill '
           f'{res["skill_single"]:+.3f} single -> {res["skill_combined"]:+.3f} combined '
           f'(dimensionless, night-grouped out of fold)', sub_y=0.925)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    pdf.savefig(fig)
    plt.close(fig)


def figure_expectation(pdf, exp, grid):
    """Which modes top the M1M3 shape channels, against what theory predicts.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    exp : `pandas.DataFrame`
        `thermal_vmodes_channels.expectation_check` result.
    grid : `pandas.DataFrame`
        `channel_grid` result.

    Notes
    -----
    ``smatrix/docs/plots.md`` predicts the M1M3 z-gradient drives Z4 plus Z11 spherical and the
    radial gradient drives Z4 + Z11 + Z22. The question here is whether the modes topping those
    two channels are a small consistent set -- which would support a thermal origin for them --
    or scattered across high modes, which would not.
    """
    if not len(exp):
        return
    shape = ['m1m3_z_gradient_c_per_m', 'm1m3_radial_gradient_c_per_m',
             'm1m3_r2_coeff_c', 'm1_r2_coeff_c', 'm3_r2_coeff_c']
    shape = [c for c in shape if c in set(exp['channel'])]

    fig, axes = plt.subplots(1, len(shape), figsize=(3.6 * len(shape), 4.8), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), shape):
        g = grid[(grid['channel'] == c) & np.isfinite(grid['spearman_rho'])]
        g = g.sort_values('mode')
        ax.bar(g['mode'].to_numpy(int), g['spearman_rho'].to_numpy(float),
               color=_FIT_COLOR, width=0.75)
        strong = g[g['spearman_rho'].abs() >= TVC.RHO_STRONG]
        if len(strong):
            ax.bar(strong['mode'].to_numpy(int), strong['spearman_rho'].to_numpy(float),
                   color=_STRONG_COLOR, width=0.75)
            for _, r in strong.iterrows():
                ax.annotate(f'v{int(r["mode"])}', (int(r['mode']), r['spearman_rho']),
                            fontsize=7, ha='center',
                            xytext=(0, 4 if r['spearman_rho'] > 0 else -11),
                            textcoords='offset points')
        for lev in (-TVC.RHO_STRONG, TVC.RHO_STRONG):
            ax.axhline(lev, color=_STRONG_COLOR, lw=1.0, ls='--')
        ax.axhline(0, color='0.5', lw=0.8)
        ax.set_xlabel('v-mode index')
        ax.set_title(_labels([c])[0], fontsize=9.5)
    np.atleast_1d(axes)[0].set_ylabel('Spearman rho against the channel [dimensionless]')

    _title(fig, 'The M1M3 shape channels: which v-modes respond to them',
           'theory (smatrix/docs/plots.md): z-gradient drives Z4 + Z11 spherical, radial '
           'gradient drives Z4 + Z11 + Z22', sub_y=0.915)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    pdf.savefig(fig)
    plt.close(fig)


def figure_prediction(pdf, pred, content, grid):
    """The S-matrix prediction tested on the modes that actually carry each Zernike.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    pred : `pandas.DataFrame`
        `thermal_vmodes_channels.prediction_table` result.
    content : `pandas.DataFrame`
        `thermal_vmodes_channels.vmode_zernike_content` result.
    grid : `pandas.DataFrame`
        `channel_grid` result.

    Notes
    -----
    Left: where each predicted Noll term lives across the v-modes, which is the step that cannot
    be guessed -- v-mode 12 is Z15, not spherical, and Z11 sits mostly in v21 and v16. Right: the
    measured Spearman rho of those modes against the channel the prediction names. Bars reaching
    the threshold line support the prediction; bars near zero refute it for those modes.
    """
    if pred is None or not len(pred):
        return
    nolls = sorted(set(pred['noll']))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))

    ax = axes[0]
    for noll in nolls:
        if noll not in content.index:
            continue
        row = content.loc[noll]
        modes = np.arange(1, len(row) + 1)
        ax.plot(modes, row.to_numpy(float), 'o-', ms=3.5, lw=1.0, label=f'Noll Z{noll}')
    ax.set_xlabel('v-mode index')
    ax.set_ylabel('field-averaged amplitude over the four corners\n'
                  '[dimensionless, per unit v-mode amplitude]')
    ax.set_title('Where each predicted Zernike actually lives\n'
                 'not the mode index: v12 is Z15, and Z11 sits in v21 and v16', fontsize=9.5)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    labels, vals, colors = [], [], []
    for _, r in pred.iterrows():
        labels.append(f'Z{int(r["noll"])} / v{int(r["mode"])}\n'
                      f'{"zgrad" if "z_grad" in r["channel"] else "rgrad"}')
        vals.append(r['spearman_rho'])
        colors.append(_STRONG_COLOR if abs(r['spearman_rho']) >= TVC.RHO_STRONG else _FIT_COLOR)
    pos = np.arange(len(vals))
    ax.bar(pos, vals, color=colors, width=0.72)
    for lev in (-TVC.RHO_STRONG, TVC.RHO_STRONG):
        ax.axhline(lev, color=_STRONG_COLOR, lw=1.0, ls='--')
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel('Spearman rho against the predicted channel [dimensionless]')
    n_pass = int((pred['spearman_rho'].abs() >= TVC.RHO_STRONG).sum())
    ax.set_title(f'Measured response of those modes to the named channel\n'
                 f'{n_pass} of {len(pred)} predicted pairs reach '
                 f'|rho| >= {TVC.RHO_STRONG}', fontsize=9.5)

    _title(fig, 'Testing the M1M3 thermal prediction mode by mode',
           'smatrix/docs/plots.md: z-gradient drives Z4 + Z11 spherical; radial gradient drives '
           'Z4 + Z11 + Z22', sub_y=0.915)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(path, grid, results, exp=None, df=None, pred=None, content=None):
    """Assemble the per-channel screen into one document.

    Parameters
    ----------
    path : `pathlib.Path` or `str`
        Output PDF path.
    grid : `pandas.DataFrame`
        `channel_grid` result.
    results : `list` [`dict`]
        `combined_fit` results for the followed-up modes.
    exp : `pandas.DataFrame`, optional
        `expectation_check` result.
    df : `pandas.DataFrame`, optional
        Per-visit frame, for the leading-channel scatter on each mode page.
    pred : `pandas.DataFrame`, optional
        `prediction_table` result.
    content : `pandas.DataFrame`, optional
        `vmode_zernike_content` result.

    Returns
    -------
    path : `pathlib.Path`
        The written path.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    path = pathlib.Path(path)
    with PdfPages(path) as pdf:
        figure_heatmap(pdf, grid)
        figure_skill_heatmap(pdf, grid)
        if pred is not None and content is not None:
            figure_prediction(pdf, pred, content, grid)
        if exp is not None and len(exp):
            figure_expectation(pdf, exp, grid)
        for res in results:
            if np.isfinite(res.get('nmad_combined', np.nan)):
                figure_mode_combined(pdf, res, grid, df=df)
    return path
