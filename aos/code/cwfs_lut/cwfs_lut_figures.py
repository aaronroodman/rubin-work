"""Figures for the pointing-dependence study.

Five pages, each a function taking an open `matplotlib.backends.backend_pdf.PdfPages`. The
slopes come from `cwfs_lut_lib.trend_table`; the pages that draw a line refit it on the same
per-visit points so the line shown is the line fitted, not a slope carried over from a table
built on a different cut.

Per-visit points are drawn as **hexbin, not scatter**: the sample is near 100,000 visits across
ten axes and two angles, and a scatter of that in a vector PDF is both unreadable and enormous.

Two pages exist to make results visible that prose states badly:

* `figure_solver` puts the range-bounded and unconstrained recoveries of the same term on shared
  axes. The slopes differ by a factor near 2.3 and the correlations by far more, so the
  range-bounded solution is not a scaled copy -- it is a tighter function of rotator angle.
* `figure_angle_grid` draws all ten rigid-body axes against one angle, which is how an *absence*
  of elevation dependence gets shown rather than asserted.
"""
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))

import cwfs_lut_lib as C                                         # noqa: E402

#: Axis labels for the two pointing angles, carrying the unit.
ANGLE_LABELS = {'elevation_deg': 'TMA elevation [deg]',
                'rotator_angle_deg': 'camera rotator angle [deg]'}

#: The rigid-body DOF, in ts_ofc order: M2 hexapod then camera hexapod, each (dz, dx, dy, rx, ry).
RIGID_BODY_DOF = tuple(sorted(C.DOF_LABELS))

#: Correlation above which a trend is called a LUT candidate rather than noise (dimensionless
#: Pearson r). Set from the result -- one term clears it and the rest sit below 0.53.
STRONG_R = 0.6

_HEADLINE_COLOR = '#d62728'
_BINNED_COLOR = '#2ca02c'
_RBR_COLOR = '#d62728'
_UNCON_COLOR = '#1f77b4'
_BOUNCE_COLOR = '#ff7f0e'
_SURVEY_COLOR = '#1f77b4'


def _angle_label(angle):
    return ANGLE_LABELS.get(angle, angle)


def _title(fig, text, subtext=None, sub_y=0.945):
    """Page title, with an optional second line in smaller type.

    `sub_y` is lowered on short figures, where the default sits on top of the title.
    """
    fig.suptitle(text, fontsize=11.5, y=0.992)
    if subtext:
        fig.text(0.5, sub_y, subtext, ha='center', fontsize=8.5, color='0.3')


def _robust_ylim(y, pad=0.12, k=8.0):
    """Y limits set by the bulk of `y`, not its outliers.

    Parameters
    ----------
    y : `array_like` [`float`]
        Sample, finite values only used.
    pad : `float`, optional
        Fraction of the kept range added at each end.
    k : `float`, optional
        Half-width in robust deviations (nMAD) kept around the median.

    Returns
    -------
    lo, hi : `float`
        Limits, or ``(nan, nan)`` when `y` has no robust scale.

    Notes
    -----
    The open-loop DOF carry a tail from visits whose recovery is badly conditioned -- on the
    unconstrained solver it reaches 100,000 µm against a trend of order 1,000 µm. Autoscaling to
    that tail compresses every real trend to a flat line, which defeats the comparison the page
    exists to make. Trimming the *view* is not trimming the fit: `cwfs_lut_lib.huber_trend` has
    already seen every point, and the slope in the title is fitted on all of them.
    """
    from common.utils import nmad

    a = np.asarray(y, float)
    a = a[np.isfinite(a)]
    if len(a) < 10:
        return np.nan, np.nan
    med, scale = float(np.median(a)), float(nmad(a))
    if not np.isfinite(scale) or scale == 0:
        return np.nan, np.nan
    lo = max(med - k * scale, float(a.min()))
    hi = min(med + k * scale, float(a.max()))
    if hi <= lo:
        return np.nan, np.nan
    span = hi - lo
    return lo - pad * span, hi + pad * span


def _binned_medians(x, y, n_bins=24):
    """Median `y` in equal-count bins of `x`.

    Parameters
    ----------
    x, y : `numpy.ndarray` [`float`]
        Finite paired samples.
    n_bins : `int`, optional

    Returns
    -------
    xc, yc : `numpy.ndarray` [`float`]
        Bin median of `x` and of `y`, one entry per populated bin.

    Notes
    -----
    Equal-count rather than equal-width bins, because both pointing angles are sampled very
    unevenly by the observing pattern and equal-width bins put one or two visits in the extreme
    bins and tens of thousands in the middle.
    """
    if len(x) < n_bins * 2:
        return np.array([]), np.array([])
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    xc, yc = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 10:
            xc.append(np.median(x[m]))
            yc.append(np.median(y[m]))
    return np.asarray(xc), np.asarray(yc)


def _panel_trend(ax, x, y, res, ylabel, title, color=_HEADLINE_COLOR, hexbin=True, fig=None,
                 ylim=None):
    """One angle-against-DOF panel: hexbin, binned medians, Huber line.

    `ylim` overrides the robust auto-range, for pages that share one range across panels.
    """
    if hexbin:
        hb = ax.hexbin(x, y, gridsize=55, bins='log', mincnt=1, cmap='Blues')
        if fig is not None:
            fig.colorbar(hb, ax=ax, label='visits per cell', pad=0.02)
    xc, yc = _binned_medians(x, y)
    if len(xc):
        ax.plot(xc, yc, 'o-', ms=4, lw=1.0, color=_BINNED_COLOR, label='binned medians')
    if np.isfinite(res.get('slope', np.nan)):
        xs = np.linspace(float(x.min()), float(x.max()), 20)
        ax.plot(xs, res['intercept'] + res['slope'] * xs, '-', color=color, lw=1.5,
                label=f'Huber {res["slope"]:+.4g} +- {res["slope_err"]:.2g}')
    lo, hi = ylim if ylim is not None else _robust_ylim(y)
    if np.isfinite(lo) and np.isfinite(hi):
        ax.set_ylim(lo, hi)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.0)


def figure_headline(pdf, df, angle='rotator_angle_deg', dof=1, variant=''):
    """The strongest single trend, on its own page.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    df : `pandas.DataFrame`
        Per-visit frame carrying ``dof{dof}_olr`` and `angle`.
    angle : `str`, optional
        Pointing-angle column.
    dof : `int`, optional
        Degree-of-freedom index.
    variant : `str`, optional
        Recorded in the caption for provenance.

    Notes
    -----
    Left panel is the trend; right panel is the residual against the same angle, which is the
    check that a straight line is the right model. A LUT term fitted as a slope is only honest
    if the residual carries no remaining structure in the same variable.
    """
    name, unit = C.dof_label(dof)
    col = f'dof{dof}_olr'
    m = np.isfinite(df[col]) & np.isfinite(df[angle])
    x = df.loc[m, angle].to_numpy(float)
    y = df.loc[m, col].to_numpy(float)
    res = C.huber_trend(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    _panel_trend(axes[0], x, y, res,
                 ylabel=f'open-loop {name}, Deviation - Trim [{unit}]',
                 title=f'{name} against {angle}\n'
                       f'slope {res["slope"]:+.4g} {unit}/deg, Pearson r {res["pearson_r"]:+.3f}, '
                       f'Spearman rho {res["spearman_rho"]:+.3f}, n {res["n"]} visits',
                 fig=fig)
    axes[0].set_xlabel(_angle_label(angle))
    axes[0].legend(fontsize=7.5)

    ax = axes[1]
    resid = y - (res['intercept'] + res['slope'] * x)
    hb = ax.hexbin(x, resid, gridsize=55, bins='log', mincnt=1, cmap='Blues')
    fig.colorbar(hb, ax=ax, label='visits per cell', pad=0.02)
    xc, yc = _binned_medians(x, resid)
    if len(xc):
        ax.plot(xc, yc, 'o-', ms=4, lw=1.0, color=_BINNED_COLOR, label='binned medians')
    ax.axhline(0, color=_HEADLINE_COLOR, lw=1.2)
    ax.set_xlabel(_angle_label(angle))
    ax.set_ylabel(f'residual after the linear term [{unit}]')
    ax.set_title(f'Residual against the same angle\n'
                 f'robust scatter {res["resid_nmad"]:.4g} {unit}; curvature here would mean a '
                 f'straight line is the wrong LUT form', fontsize=9.0)
    ax.legend(fontsize=7.5)

    _title(fig, f'The dominant pointing term: {name} against {_angle_label(angle)}',
           f'variant {variant}; formal slope error understates the truth, successive visits '
           f'are correlated', sub_y=0.925)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    pdf.savefig(fig)
    plt.close(fig)


def figure_solver(pdf, frames, angle='rotator_angle_deg', dof=1):
    """The same term under the range-bounded and unconstrained recoveries.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    frames : `dict` [`str`, `pandas.DataFrame`]
        Per-visit frames keyed by variant id; needs `cwfs_lut_lib.RBR_VARIANT` and the batoid
        entry of `cwfs_lut_lib.INTRINSIC_VARIANTS`.
    angle : `str`, optional
    dof : `int`, optional

    Notes
    -----
    Shared y axis, so the two can be read against each other directly. Both arms use the batoid
    intrinsic, so the only thing differing is the solver. The unconstrained 50/34 recovery asks a
    median 33x the available actuator stroke, spending amplitude on states the actuators cannot
    reach, so a weaker and noisier dependence on pointing is the expected direction -- the size
    of the gap is the result, and it means a LUT term must state which recovery produced it.
    """
    name, unit = C.dof_label(dof)
    col = f'dof{dof}_olr'
    arms = [(C.RBR_VARIANT, 'range-bounded recovery (RBR)', _RBR_COLOR),
            (C.INTRINSIC_VARIANTS['batoid'], 'unconstrained truncated SVD', _UNCON_COLOR)]
    arms = [(v, lab, c) for v, lab, c in arms if v in frames]
    if len(arms) < 2:
        return

    # One y range across both panels, from the union of their bulk: the comparison is only
    # readable if the two arms share a scale, and autoscaling to the unconstrained arm's tail
    # (which reaches 100,000 µm) would flatten both trends to lines.
    pts = {}
    for variant, _label, _color in arms:
        d = frames[variant]
        m = np.isfinite(d[col]) & np.isfinite(d[angle])
        pts[variant] = (d.loc[m, angle].to_numpy(float), d.loc[m, col].to_numpy(float))
    bounds = [_robust_ylim(y) for _, y in pts.values()]
    bounds = [b for b in bounds if np.isfinite(b[0])]
    shared = (min(b[0] for b in bounds), max(b[1] for b in bounds)) if bounds else None

    fig, axes = plt.subplots(1, len(arms), figsize=(6.6 * len(arms), 5.4), sharey=True)
    out = {}
    for ax, (variant, label, color) in zip(np.atleast_1d(axes), arms):
        x, y = pts[variant]
        res = out[variant] = C.huber_trend(x, y)
        _panel_trend(ax, x, y, res,
                     ylabel=f'open-loop {name} [{unit}]',
                     title=f'{label}\nslope {res["slope"]:+.4g} {unit}/deg, Pearson r '
                           f'{res["pearson_r"]:+.3f}, n {res["n"]} visits',
                     color=color, fig=fig, ylim=shared)
        ax.set_xlabel(_angle_label(angle))
        ax.legend(fontsize=7.5)

    # The correlation gap tracks the slope ratio, not a scatter difference: the robust residual
    # scatter is near-identical between the two arms, so the stronger correlation is a larger
    # signal over the same noise. The hexbin looks wider on the unconstrained arm only because
    # its outlier tail is longer, which nMAD ignores and the eye does not.
    a, b = arms[0][0], arms[1][0]
    ratio = (out[a]['slope'] / out[b]['slope']) if out[b]['slope'] else np.nan
    _title(fig, f'The solver changes the trend more than the intrinsic does: {name}',
           f'slope ratio {ratio:+.2f} (dimensionless, RBR over unconstrained) over near-equal '
           f'robust scatter, {out[a]["resid_nmad"]:.0f} against {out[b]["resid_nmad"]:.0f} '
           f'{unit}: Pearson r {out[a]["pearson_r"]:+.3f} against {out[b]["pearson_r"]:+.3f}',
           sub_y=0.925)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    pdf.savefig(fig)
    plt.close(fig)


def figure_angle_grid(pdf, df, angle, variant='', dof_indices=RIGID_BODY_DOF):
    """All ten rigid-body axes against one pointing angle.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    df : `pandas.DataFrame`
        Per-visit frame.
    angle : `str`
        Pointing-angle column.
    variant : `str`, optional
    dof_indices : `tuple` [`int`], optional

    Notes
    -----
    This is how a null result is shown rather than asserted: for elevation, every one of the ten
    panels is flat, and a reader can see that no panel was omitted. Panel titles carry Pearson r
    and those clearing `STRONG_R` are outlined, so the one real term does not hide in a grid.
    """
    fig, axes = plt.subplots(2, 5, figsize=(21, 8.4))
    n_strong = 0
    for ax, j in zip(axes.ravel(), dof_indices):
        name, unit = C.dof_label(j)
        col = f'dof{j}_olr'
        m = np.isfinite(df[col]) & np.isfinite(df[angle])
        x = df.loc[m, angle].to_numpy(float)
        y = df.loc[m, col].to_numpy(float)
        res = C.huber_trend(x, y)
        strong = np.isfinite(res['pearson_r']) and abs(res['pearson_r']) >= STRONG_R
        n_strong += int(strong)
        _panel_trend(ax, x, y, res,
                     ylabel=f'{name} [{unit}]',
                     title=f'{name}\nslope {res["slope"]:+.4g} {unit}/deg, r '
                           f'{res["pearson_r"]:+.3f}',
                     color=_HEADLINE_COLOR if strong else '#9467bd', hexbin=True)
        ax.set_xlabel(_angle_label(angle))
        if strong:
            for side in ax.spines.values():
                side.set_edgecolor(_HEADLINE_COLOR)
                side.set_linewidth(2.0)

    verdict = (f'{n_strong} of {len(dof_indices)} axes reach |Pearson r| >= {STRONG_R} '
               f'(dimensionless)' if n_strong else
               f'no axis reaches |Pearson r| >= {STRONG_R} (dimensionless): no dependence here')
    _title(fig, f'Every rigid-body degree of freedom against {_angle_label(angle)}',
           f'variant {variant}; {verdict}')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(fig)
    plt.close(fig)


def figure_slope_summary(pdf, tabs, angle, dof_indices=RIGID_BODY_DOF, bounce=None):
    """Slope per rigid-body axis, every variant together, with the bounce test overlaid.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    tabs : `dict` [`tuple`, `pandas.DataFrame`]
        `cwfs_lut_lib.trend_table` results keyed ``(variant, angle)``.
    angle : `str`
        Pointing-angle column.
    dof_indices : `tuple` [`int`], optional
    bounce : `pandas.DataFrame`, optional
        `bounce_compare.bounce_slope` result for this angle, drawn as a fourth marker per
        axis. Omitted when the bounce run is not on disk.

    Notes
    -----
    The two results of the study in one panel: the two intrinsic routes sit on top of each other
    while the two solvers separate. Split into a µm panel and a deg panel because mixing a
    decentre slope in µm per deg with a tilt slope in deg per deg on one axis would be
    meaningless. No unit conversion is applied to either -- the bounce test stores the same
    units this study does, which is what the module docstring of `cwfs_lut_lib` settles.

    Error bars are the **formal** RLM standard errors on the survey arms and are known to
    understate the truth -- successive visits are correlated, so the effective sample is
    smaller than ``n``. They are drawn to compare arms against each other, not as confidence
    intervals. The bounce marker's bars are the per-leg median standard error divided by the
    throw and are not the same kind of quantity; the per-DOF comparison page is where the two
    are put on a common footing.
    """
    arms = [(C.RBR_VARIANT, 'RBR, batoid', _RBR_COLOR, 'o'),
            (C.INTRINSIC_VARIANTS['batoid'], 'SVD, batoid', _UNCON_COLOR, 's'),
            (C.INTRINSIC_VARIANTS['miw'], 'SVD, MIW', '#2ca02c', '^')]
    arms = [a for a in arms if (a[0], angle) in tabs]
    if not arms:
        return
    n_markers = len(arms) + (1 if bounce is not None and len(bounce) else 0)

    tilt = [j for j in dof_indices if j in C.HEX_TILT_DOF]
    lin = [j for j in dof_indices if j not in C.HEX_TILT_DOF]
    groups = [(lin, 'µm', 'slope [µm per deg]'),
              (tilt, C.HEX_TILT_UNIT, f'slope [{C.HEX_TILT_UNIT} per deg]')]

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5.6),
        gridspec_kw={'width_ratios': [len(lin), max(len(tilt), 1)]})
    for ax, (idx, unit, ylabel) in zip(axes, groups):
        if not idx:
            ax.set_visible(False)
            continue
        pos = np.arange(len(idx), dtype=float)
        for k, (variant, label, color, marker) in enumerate(arms):
            by = tabs[(variant, angle)].set_index('dof')
            s = [by['slope'].get(j, np.nan) for j in idx]
            e = [by['slope_err'].get(j, np.nan) for j in idx]
            off = (k - (n_markers - 1) / 2) * 0.22
            ax.errorbar(pos + off, s, yerr=e, fmt=marker, ms=6, lw=1.2, capsize=2.5,
                        color=color, label=label)
        if bounce is not None and len(bounce):
            # One slope per axis. A multi-leg bounce program (the five elevation legs) carries
            # one row per leg, so the weighted fit across them is the single comparable number
            # and `slope` alone would be whichever leg sorted first.
            by = bounce.drop_duplicates('index').set_index('index')
            s = [by['slope'].get(j, np.nan) for j in idx]
            e = [by['slope_err'].get(j, np.nan) for j in idx]
            off = (len(arms) - (n_markers - 1) / 2) * 0.22
            ax.errorbar(pos + off, s, yerr=e, fmt='D', ms=6, lw=1.2, capsize=2.5,
                        color=_BOUNCE_COLOR, label='bounce test')
        ax.axhline(0, color='0.6', lw=0.8)
        ax.set_xticks(pos)
        # A star on the tick marks the axes the bounce test can be compared against, which keeps
        # the list off the title where it overflowed the panel.
        ax.set_xticklabels([C.dof_label(j)[0].replace(' hexapod ', '\n')
                            + ('\n*' if j in C.BOUNCE_COMPARABLE_DOF else '')
                            for j in idx], fontsize=7.5)
        ax.set_ylabel(ylabel)
        n_cmp = sum(1 for j in idx if j in C.BOUNCE_COMPARABLE_DOF)
        # The bounce slopes are drawn on both panels, but only the starred axes are the ones
        # the quantitative comparison claims; the tilts are shown for completeness.
        note = (f'* = bounce-test comparable ({n_cmp} of {len(idx)})' if n_cmp else
                'bounce slopes shown; these axes are not part of the quantitative comparison')
        ax.set_title(f'{unit} axes\n{note}', fontsize=9.0)
        ax.legend(fontsize=7.5)

    _title(fig, f'Slope per rigid-body axis against {_angle_label(angle)}',
           'intrinsic routes overlap, solvers separate; bars are formal RLM errors and '
           'understate the truth', sub_y=0.925)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    pdf.savefig(fig)
    plt.close(fig)


#: Survey variant the bounce-comparison pages draw. The unconstrained batoid recovery, because
#: the bounce note is explicit that for a look-up-table fit the default recovery is the
#: estimator and an RBR rigid-body amplitude is a constrained estimate -- and because RBR
#: measurably degrades the agreement (translation cosine +0.52 against +0.83).
HEADLINE_VARIANT = C.INTRINSIC_VARIANTS['batoid']


def _headline_rows(tab, headline, variant=None):
    """One row per entry: the named arm pairing on one variant.

    The comparison tables hold every (bounce arm, survey arm, variant) pairing stacked, so a
    page that forgets to select gets a silent average over three variants -- which is how the
    RBR arm hid the unconstrained result on a first pass.
    """
    if tab is None or not len(tab):
        return None
    b_arm, s_arm = headline
    sel = tab[(tab['bounce_arm'] == b_arm) & (tab['survey_arm'] == s_arm)]
    if 'variant' in sel.columns:
        want = variant or HEADLINE_VARIANT
        pick = sel[sel['variant'] == want]
        sel = pick if len(pick) else sel[sel['variant'] == sorted(sel['variant'].unique())[0]]
    return sel if len(sel) else None


def figure_bounce_subspace(pdf, subspace, angle, headline=('svd', 'deviation')):
    """Agreement per subspace, as a cosine similarity -- the headline comparison page.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    subspace : `pandas.DataFrame`
        `bounce_compare.compare_subspaces` rows, carrying ``bounce_arm``, ``survey_arm`` and
        ``variant``.
    angle : `str`
        Pointing-angle column.
    headline : `tuple` [`str`], optional
        ``(bounce_arm, survey_arm)`` drawn solid; the other pairings are drawn faint.

    Notes
    -----
    The page the quantitative scope of the whole comparison rests on. The hexapod translations
    agree and the bending modes do not, which is why the per-DOF claim is restricted to the six
    translations -- a limit now measured rather than argued from field sampling.

    Only bounce terms above `bounce_compare.MIN_SIGNIFICANCE` enter each bar, and the count that
    survived is printed on it: the bending disagreement is not an absence of signal, since most
    of those modes are individually significant on the bounce side.
    """
    if subspace is None or not len(subspace):
        return
    order = ['hexapod_translation', 'hexapod_tilt', 'bending', 'all_dof', 'vmode']
    names = [s for s in order if s in set(subspace['subspace'])]
    pairings = (subspace[['bounce_arm', 'survey_arm']].drop_duplicates()
                .itertuples(index=False, name=None))
    pairings = sorted(pairings, key=lambda p: (p != tuple(headline), p))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    pos = np.arange(len(names), dtype=float)
    counts = {}
    for k, (b_arm, s_arm) in enumerate(pairings):
        is_head = (b_arm, s_arm) == tuple(headline)
        off = (k - (len(pairings) - 1) / 2) * 0.17
        sel = _headline_rows(subspace, (b_arm, s_arm))
        if sel is None:
            continue
        by = sel.drop_duplicates('subspace').set_index('subspace')
        cos = [by['cosine_similarity'].get(s, np.nan) for s in names]
        scale = [by['scale'].get(s, np.nan) for s in names]
        style = dict(color=_BOUNCE_COLOR if is_head else '0.65',
                     marker='D' if is_head else 'o',
                     ms=8 if is_head else 5, lw=0,
                     label=f'bounce {b_arm} vs survey {s_arm}'
                           + (' (headline)' if is_head else ''),
                     zorder=3 if is_head else 2)
        axes[0].plot(cos, pos + off, **style)
        axes[1].plot(scale, pos + off, **style)
        if is_head:
            counts = {s: (by['n_significant'].get(s, np.nan), by['n_terms'].get(s, np.nan))
                      for s in names}

    for ax, (xl, ref) in zip(axes, [('cosine similarity (dimensionless)', 1.0),
                                    ('scale = survey / bounce (dimensionless)', 1.0)]):
        ax.axvline(ref, color='0.5', lw=0.9, ls='--')
        ax.axvline(0.0, color='0.75', lw=0.8)
        ax.set_yticks(pos)
        # The significant-term count rides in the tick label rather than as an annotation, so
        # it cannot drift away from the row it describes.
        ax.set_yticklabels(
            [s.replace('_', ' ')
             + (f'\n{int(counts[s][0])} of {int(counts[s][1])} significant'
                if counts.get(s) and np.isfinite(counts[s][0]) else '')
             for s in names], fontsize=8)
        ax.set_xlabel(xl, fontsize=9)
        ax.grid(axis='x', alpha=0.25)
    axes[0].set_title('do the two retrievals point the same way?', fontsize=9.5)
    axes[1].set_title('and by how much?', fontsize=9.5)
    axes[0].legend(fontsize=7, loc='lower right')

    _title(fig, f'Bounce test against survey, agreement by subspace, {_angle_label(angle)}',
           f'the hexapod translations agree and the bending modes do not, which is what limits '
           f'the comparison; survey variant {HEADLINE_VARIANT}, dashed line is exact agreement',
           sub_y=0.915)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    pdf.savefig(fig)
    plt.close(fig)


def figure_bounce_per_dof(pdf, per_dof, lateral, angle, headline=('svd', 'deviation')):
    """Per-axis slopes side by side, and the lateral sums that survive the degeneracy.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    per_dof : `pandas.DataFrame`
        `bounce_compare.compare_per_dof` rows.
    lateral : `pandas.DataFrame`
        `bounce_compare.compare_lateral_sums` rows.
    angle : `str`
        Pointing-angle column.
    headline : `tuple` [`str`], optional
        ``(bounce_arm, survey_arm)`` to draw.

    Notes
    -----
    Left panel is the literal per-axis claim and right panel is the same information summed over
    the two hexapods. The pair is the argument: individual axes disagree by factors of a few
    while their sum agrees, which is the signature of a retrieval degeneracy rather than of one
    method being wrong.
    """
    b_arm, s_arm = headline
    sel = _headline_rows(per_dof, headline)
    if sel is None:
        return
    sel = sel[sel['index'].isin(C.BOUNCE_COMPARABLE_DOF)].drop_duplicates('index')
    sel = sel.sort_values('index')
    if not len(sel):
        return

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2),
                             gridspec_kw={'width_ratios': [len(sel), 3]})
    pos = np.arange(len(sel), dtype=float)
    axes[0].errorbar(pos - 0.11, sel['slope_bounce'], yerr=sel['slope_bounce_err'],
                     fmt='D', ms=6, lw=1.2, capsize=2.5, color=_BOUNCE_COLOR,
                     label='bounce test')
    axes[0].errorbar(pos + 0.11, sel['slope_survey'], yerr=sel['slope_survey_err'],
                     fmt='o', ms=6, lw=1.2, capsize=2.5, color=_SURVEY_COLOR,
                     label='survey')
    axes[0].axhline(0, color='0.6', lw=0.8)
    axes[0].set_xticks(pos)
    axes[0].set_xticklabels([n.replace(' hexapod ', '\n') for n in sel['label']], fontsize=8)
    axes[0].set_ylabel('slope [µm per deg]')
    axes[0].set_title('per axis: the split between the two hexapods disagrees', fontsize=9.5)
    axes[0].legend(fontsize=8)

    lat = _headline_rows(lateral, headline)
    if lat is not None:
        lat = lat.drop_duplicates('axis')
        lpos = np.arange(len(lat), dtype=float)
        axes[1].errorbar(lpos - 0.11, lat['slope_bounce'], yerr=lat['slope_bounce_err'],
                         fmt='D', ms=6, lw=1.2, capsize=2.5, color=_BOUNCE_COLOR,
                         label='bounce test')
        axes[1].errorbar(lpos + 0.11, lat['slope_survey'], yerr=lat['slope_survey_err'],
                         fmt='o', ms=6, lw=1.2, capsize=2.5, color=_SURVEY_COLOR,
                         label='survey')
        axes[1].axhline(0, color='0.6', lw=0.8)
        axes[1].set_xticks(lpos)
        axes[1].set_xticklabels([f'M2 + camera\n{a}' for a in lat['axis']], fontsize=8)
        axes[1].set_ylabel('slope [µm per deg]')
        axes[1].set_title('summed over both hexapods', fontsize=9.5)
        axes[1].legend(fontsize=8)

    _title(fig, f'Bounce test against survey per rigid-body axis, {_angle_label(angle)}',
           f'bounce {b_arm} against survey {s_arm} on {HEADLINE_VARIANT}; bounce bars are the '
           f'per-leg median standard error over the throw, survey bars are formal RLM errors',
           sub_y=0.915)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    pdf.savefig(fig)
    plt.close(fig)


def figure_bounce_vmode(pdf, vmode, angle, headline=('svd', 'deviation'),
                        min_significance=3.0):
    """V-mode slopes, the space that exposes the bending disagreement.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    vmode : `pandas.DataFrame`
        `bounce_compare.compare_per_dof` rows built on ``kind='vmode'``.
    angle : `str`
        Pointing-angle column.
    headline : `tuple` [`str`], optional
    min_significance : `float`, optional
        Bounce |Δ|/error above which a mode is drawn filled.

    Notes
    -----
    Included because v-mode space looks like the natural retrieval-independent comparison and
    is not: the basis is bending-dominated, so it inherits the bending disagreement and buries
    the translation agreement. Modes scatter about the y = x line rather than following it.
    """
    sel = _headline_rows(vmode, headline)
    if sel is None:
        return
    sel = sel.drop_duplicates('index').sort_values('index')
    sig = sel['significance'].abs() >= min_significance

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    ax.errorbar(sel.loc[sig, 'slope_bounce'], sel.loc[sig, 'slope_survey'],
                xerr=sel.loc[sig, 'slope_bounce_err'], yerr=sel.loc[sig, 'slope_survey_err'],
                fmt='o', ms=5, lw=0.9, capsize=2, color=_SURVEY_COLOR,
                label=f'|significance| >= {min_significance:g}')
    ax.plot(sel.loc[~sig, 'slope_bounce'], sel.loc[~sig, 'slope_survey'], 'o', ms=4,
            mfc='none', color='0.6', label='below threshold')
    lim = np.nanmax(np.abs(np.concatenate([sel['slope_bounce'].to_numpy(float),
                                           sel['slope_survey'].to_numpy(float)])))
    if np.isfinite(lim) and lim > 0:
        ax.plot([-lim, lim], [-lim, lim], ls='--', color='0.5', lw=0.9, label='y = x')
        ax.set_xlim(-1.1 * lim, 1.1 * lim)
        ax.set_ylim(-1.1 * lim, 1.1 * lim)
    for _, r in sel[sig].iterrows():
        ax.annotate(r['label'], (r['slope_bounce'], r['slope_survey']), fontsize=6.5,
                    xytext=(3, 3), textcoords='offset points', color='0.3')
    ax.set_xlabel('bounce slope [dimensionless v-mode amplitude per deg]', fontsize=9)
    ax.set_ylabel('survey slope [dimensionless v-mode amplitude per deg]', fontsize=9)
    ax.set_title('mode by mode, against exact agreement', fontsize=9.5)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    ax.errorbar(sel['index'] + 1, sel['slope_bounce'], yerr=sel['slope_bounce_err'],
                fmt='D', ms=4, lw=0.9, color=_BOUNCE_COLOR, label='bounce test')
    ax.errorbar(sel['index'] + 1, sel['slope_survey'], yerr=sel['slope_survey_err'],
                fmt='o', ms=4, lw=0.9, color=_SURVEY_COLOR, label='survey')
    ax.axhline(0, color='0.6', lw=0.8)
    ax.set_xlabel('v-mode index', fontsize=9)
    ax.set_ylabel('slope [dimensionless v-mode amplitude per deg]', fontsize=9)
    ax.set_title('where the disagreement sits', fontsize=9.5)
    ax.legend(fontsize=7.5)

    _title(fig, f'Bounce test against survey in v-mode space, {_angle_label(angle)}',
           'the v-mode basis is bending-dominated, so this space inherits the bending '
           'disagreement rather than avoiding the hexapod degeneracy', sub_y=0.915)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    pdf.savefig(fig)
    plt.close(fig)


def figure_bounce_elevation(pdf, legs, fits, dof_indices=(1, 2, 6, 7)):
    """The five elevation legs, with the first-order linear fit through them.

    Parameters
    ----------
    pdf : `matplotlib.backends.backend_pdf.PdfPages`
        Open document.
    legs : `pandas.DataFrame`
        Per-(leg, DOF) rows from `bounce_compare.compare_elevation_legs`.
    fits : `pandas.DataFrame`
        Per-DOF fit rows from the same call.
    dof_indices : `tuple` [`int`], optional
        DOF to panel; the four lateral decentres by default.

    Notes
    -----
    A chi2/dof above 1 on these panels is **not** grounds to reject the linear form. The bounce
    test carries atmospheric turbulence and other stochastic terms the per-leg error bars do not
    capture, so a perfect chi2/dof is not expected. The linear fit is a first-order
    approximation to a dependence that is physically closer to cos(elevation), and it is much
    better than no correction -- which is why the slope is drawn as the deliverable and the
    cosine fit beside it as the comparison.
    """
    if legs is None or not len(legs) or fits is None or not len(fits):
        return
    idx = [j for j in dof_indices if j in set(fits['index'])]
    if not idx:
        return
    by_fit = fits.set_index('index')

    n = len(idx)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.4), squeeze=False)
    for ax, j in zip(axes[0], idx):
        g = legs[legs['index'] == j].sort_values('throw_deg')
        f = by_fit.loc[j]
        ax.errorbar(g['throw_deg'], g['delta'], yerr=g['delta_err'], fmt='o', ms=5,
                    lw=1.1, capsize=2.5, color=_BOUNCE_COLOR, label='bounce leg', zorder=3)
        xs = np.linspace(min(g['throw_deg'].min(), 0.0), max(g['throw_deg'].max(), 0.0), 50)
        if np.isfinite(f['slope_linear']):
            ax.plot(xs, f['slope_linear'] * xs, '-', color='0.35', lw=1.2,
                    label=(f'WLS {f["slope_linear"]:+.2f} {f["unit"]}/deg\n'
                           f'chi2/dof {f["chi2_linear"]:.2f} (dof {int(f["dof_linear"])})'))
        if np.isfinite(f['slope_survey']):
            ax.plot(xs, f['slope_survey'] * xs, '--', color=_SURVEY_COLOR, lw=1.2,
                    label=f'survey {f["slope_survey"]:+.2f} {f["unit"]}/deg')
        ax.axhline(0, color='0.75', lw=0.8)
        ax.axvline(0, color='0.75', lw=0.8)
        ax.set_xlabel('elevation throw from the 70 deg reference [deg]', fontsize=8.5)
        ax.set_ylabel(f'recovered change [{f["unit"]}]', fontsize=8.5)
        ax.set_title(f'{f["label"]}\ncos fit chi2/dof {f["chi2_cos"]:.2f}', fontsize=9)
        ax.legend(fontsize=6.5)

    _title(fig, 'Bounce elevation legs against the survey elevation trend',
           'the linear term is a first-order approximation and much better than no correction; '
           'chi2/dof above 1 reflects turbulence the per-leg errors do not capture',
           sub_y=0.905)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    pdf.savefig(fig)
    plt.close(fig)


def write_pdf(path, frames, tabs, angles, headline_dof=1, headline_angle='rotator_angle_deg',
              bounce=None):
    """Assemble every page into one document.

    Parameters
    ----------
    path : `pathlib.Path` or `str`
        Output PDF path.
    frames : `dict` [`str`, `pandas.DataFrame`]
        Per-visit frames keyed by variant id.
    tabs : `dict` [`tuple`, `pandas.DataFrame`]
        `cwfs_lut_lib.trend_table` results keyed ``(variant, angle)``.
    angles : `sequence` [`str`]
        Pointing-angle columns, one grid page each.
    headline_dof : `int`, optional
        Degree of freedom for the headline and solver pages.
    headline_angle : `str`, optional
    bounce : `dict`, optional
        Bounce comparison products, as `run_cwfs_lut` assembles them: ``slopes`` keyed by
        angle for the slope-summary overlay, plus ``per_dof``, ``lateral_sum``, ``subspace``,
        ``vmode``, ``elevation_legs`` and ``elevation_fits``. Omitted when the bounce run is
        not on disk, in which case the comparison pages are skipped.

    Returns
    -------
    path : `pathlib.Path`
        The written path.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    primary = C.RBR_VARIANT if C.RBR_VARIANT in frames else next(iter(frames))
    bounce = bounce or {}
    slopes = bounce.get('slopes', {})
    path = pathlib.Path(path)
    with PdfPages(path) as pdf:
        figure_headline(pdf, frames[primary], angle=headline_angle, dof=headline_dof,
                        variant=primary)
        figure_solver(pdf, frames, angle=headline_angle, dof=headline_dof)
        for angle in angles:
            figure_angle_grid(pdf, frames[primary], angle, variant=primary)
        for angle in angles:
            figure_slope_summary(pdf, tabs, angle, bounce=slopes.get(angle))
        if bounce:
            angle = bounce.get('angle', headline_angle)
            figure_bounce_subspace(pdf, bounce.get('subspace'), angle)
            figure_bounce_per_dof(pdf, bounce.get('per_dof'), bounce.get('lateral_sum'), angle)
            figure_bounce_vmode(pdf, bounce.get('vmode'), angle)
            figure_bounce_elevation(pdf, bounce.get('elevation_legs'),
                                    bounce.get('elevation_fits'))
    return path
