"""LUT functions fitted to the bounce paired-Δ DOF, and their pages.

Each DOF's per-(night, leg) paired Δ, in the bounce's default recovery scheme, is
fitted against the B-set position x by a polynomial with no constant term in
(x - x_ref), so the correction is zero at the reference: elevation 70 deg or camera
rotator angle 0 deg.  The fit is Huber-weighted least squares on the points
whitened by their paired-Δ errors.  A DOF is kept in the LUT when its coefficient
vector differs from zero at `nsigma` or more.

Pages: one panel per DOF with the fit and its coefficients, a summary of the kept
coefficients, and the correctable FWHM left by no correction, the per-point
correction and the LUT correction.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

HUBER_T = 1.345


def huber_poly_fit(x, y, err, order, t=HUBER_T, n_iter=100):
    """Huber-weighted polynomial fit through the origin.

    Model ``y = sum_{p=1..order} c_p x^p``.

    Parameters
    ----------
    x : `numpy.ndarray`
        Offset from the reference, deg.
    y : `numpy.ndarray`
        Paired-Δ DOF, in the DOF's unit (µm or arcsec).
    err : `numpy.ndarray`
        1-sigma error of `y`, same unit.
    order : `int`
        Highest power of x.
    t : `float`, optional
        Huber threshold, in units of the whitened residual (dimensionless).
    n_iter : `int`, optional
        Maximum reweighting iterations.

    Returns
    -------
    fit : `dict` or `None`
        ``coef`` (unit per deg^p), ``cov``, ``chi2`` (sum of squared whitened
        residuals), ``dof`` (points minus coefficients), ``scale`` (the factor the
        covariance was inflated by, sqrt of the Huber-weighted chi2/dof when that
        exceeds 1, dimensionless), ``nsig`` (Gaussian-equivalent significance of
        the coefficient vector against zero), ``n`` and the points used.  `None`
        with fewer finite points than coefficients.

    Notes
    -----
    The scale is held at 1 (the stated errors) inside the reweighting rather than
    re-estimated from the residuals: with two to nine points a MAD scale is not
    usable.  The scatter enters afterwards, through the chi2/dof inflation.
    """
    x = np.asarray(x, float); y = np.asarray(y, float); e = np.asarray(err, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    x, y, e = x[ok], y[ok], e[ok]
    if len(x) < order:
        return None
    X = np.column_stack([x ** p for p in range(1, order + 1)])
    Xw, yw = X / e[:, None], y / e
    w = np.ones(len(x))
    c = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    for _ in range(n_iter):
        r = yw - Xw @ c
        w_new = np.minimum(1.0, t / np.maximum(np.abs(r), 1e-12))
        A = Xw.T @ (w_new[:, None] * Xw)
        c_new = np.linalg.solve(A, Xw.T @ (w_new * yw))
        done = np.allclose(c_new, c, rtol=1e-10, atol=1e-14)
        c, w = c_new, w_new
        if done:
            break
    r = yw - Xw @ c
    cov = np.linalg.inv(Xw.T @ (w[:, None] * Xw))
    dof = len(x) - order
    scale = 1.0
    if dof > 0:
        scale = max(1.0, float(np.sqrt(np.sum(w * r ** 2) / dof)))
    cov = cov * scale ** 2
    chi2_c = float(c @ np.linalg.solve(cov, c))
    p = float(stats.chi2.sf(chi2_c, order))
    nsig = float(stats.norm.isf(p / 2)) if p > 0 else np.inf
    return {'coef': c, 'cov': cov, 'chi2': float(np.sum(r ** 2)), 'dof': dof,
            'scale': scale, 'nsig': nsig, 'n': int(len(x)),
            'x': x, 'y': y, 'err': e, 'huber_w': w}


def poly_eval(coef, x):
    """``sum_p coef[p-1] x^p`` for an array of offsets `x`, deg."""
    x = np.asarray(x, float)
    return sum(c * x ** (p + 1) for p, c in enumerate(coef))


def poly_sigma(cov, x):
    """1-sigma error band of `poly_eval` at offsets `x`, from the coefficient covariance."""
    x = np.asarray(x, float)
    J = np.column_stack([x ** p for p in range(1, cov.shape[0] + 1)])
    return np.sqrt(np.einsum('ij,jk,ik->i', J, cov, J))


def fit_bounce_lut(entries, dof_indices, x_ref, order, nsigma):
    """Fit every DOF of one bounce.

    Parameters
    ----------
    entries : `list` [`dict`]
        Per (night, leg): ``night``, ``b_value`` (deg) and ``dof_deltas``, the
        default scheme's ``{dof_index: {'delta', 'err'}}``.
    dof_indices : `list` [`int`]
        Global DOF indices to fit.
    x_ref : `float`
        Reference position, deg; the fit is in ``b_value - x_ref``.
    order : `int`
        Polynomial order.
    nsigma : `float`
        Significance at or above which a DOF is kept in the LUT.

    Returns
    -------
    fits : `dict`
        ``{dof_index: fit}`` from `huber_poly_fit`, each with ``significant`` and
        ``nights`` added; DOF with too few points are absent.
    """
    out = {}
    for q in dof_indices:
        xs, ys, es, nts = [], [], [], []
        for e in entries:
            v = (e.get('dof_deltas') or {}).get(q)
            if v is None:
                continue
            xs.append(e['b_value'] - x_ref); ys.append(v.get('delta', np.nan))
            es.append(v.get('err', np.nan)); nts.append(int(e['night']))
        f = huber_poly_fit(xs, ys, es, order)
        if f is None:
            continue
        ok = np.isfinite(np.asarray(xs, float)) & np.isfinite(np.asarray(ys, float)) \
            & np.isfinite(np.asarray(es, float)) & (np.asarray(es, float) > 0)
        f['nights'] = list(np.asarray(nts)[ok])
        f['significant'] = bool(f['nsig'] >= nsigma)
        out[q] = f
    return out


def lut_dof(fits, x_offset, dof_idx):
    """LUT DOF at one offset from the reference, over a scheme's DOF indices.

    Parameters
    ----------
    fits : `dict`
        From `fit_bounce_lut`.
    x_offset : `float`
        B-set position minus the reference, deg.
    dof_idx : `list` [`int`]
        The scheme's global DOF indices, in its own order.

    Returns
    -------
    d : `numpy.ndarray`
        DOF in each DOF's own unit; zero for DOF not significant.
    """
    d = np.zeros(len(dof_idx))
    for i, q in enumerate(dof_idx):
        f = fits.get(int(q))
        if f is not None and f['significant']:
            d[i] = float(poly_eval(f['coef'], x_offset))
    return d


def _coef_unit(unit, p):
    return f'{unit}/deg' if p == 1 else f'{unit}/deg^{p}'


def _coef_text(f, unit):
    lines = []
    for p, (c, s) in enumerate(zip(f['coef'], np.sqrt(np.diag(f['cov']))), 1):
        lines.append(f'c{p} = {c:.3g} ± {s:.2g} {_coef_unit(unit, p)}')
    dof_txt = f"{f['chi2']:.2f}/{f['dof']}" if f['dof'] > 0 else f"{f['chi2']:.2f}/0"
    lines.append(f'chi2/dof = {dof_txt}; joint {f["nsig"]:.1f} sigma from zero')
    return '\n'.join(lines)


def plot_lut_panels(fits, dof_labels, dof_units, *, x_ref, x_range, axis,
                    title, ncols=2, rows_per_page=5, panel_size=(5.2, 3.4)):
    """Per-DOF panels: the paired-Δ points, the LUT fit with its 1-sigma band
    and the coefficients.  Kept (significant) DOF have a red title.

    Returns
    -------
    figs : `list` [`matplotlib.figure.Figure`]
    """
    qs = sorted(fits)
    xg = np.linspace(x_range[0], x_range[1], 200)
    per_page = ncols * rows_per_page
    figs = []
    for pg0 in range(0, len(qs), per_page):
        page = qs[pg0:pg0 + per_page]
        nrows = int(np.ceil(len(page) / ncols))
        fig, axes = plt.subplots(nrows, ncols, layout='constrained', squeeze=False,
                                 figsize=(panel_size[0] * ncols + 1.2,
                                          panel_size[1] * nrows + 1.6))
        for pi, q in enumerate(page):
            ax = axes[pi // ncols][pi % ncols]
            f = fits[q]
            yg = poly_eval(f['coef'], xg - x_ref)
            sg = poly_sigma(f['cov'], xg - x_ref)
            col = '#d62728' if f['significant'] else '#1f77b4'
            ax.fill_between(xg, yg - sg, yg + sg, color=col, alpha=0.15, lw=0)
            ax.plot(xg, yg, color=col, lw=1.4)
            xs = f['x'] + x_ref
            ax.errorbar(xs, f['y'], yerr=f['err'], fmt='o', ms=5, color='#1f77b4',
                        mec='black', mew=0.4, ecolor='#1f77b4', elinewidth=0.8,
                        capsize=2, zorder=3)
            for i, (x, y, nt) in enumerate(zip(xs, f['y'], f['nights'])):
                ax.annotate(f'{nt % 10000}/{x:g}', (x, y), textcoords='offset points',
                            xytext=(4, 5 if i % 2 == 0 else -10), fontsize=6,
                            color='#333333')
            ax.axhline(0, color='gray', lw=0.5)
            ax.plot([x_ref], [0], marker='x', color='black', ms=6, zorder=4)
            ax.text(0.02, 0.97, _coef_text(f, dof_units[q]), transform=ax.transAxes,
                    va='top', fontsize=6.5,
                    bbox=dict(boxstyle='round', fc='white', alpha=0.8, lw=0.3))
            ax.set_xlim(x_range[0] - 2, x_range[1] + 2)
            ax.set_title(f'DOF {q}  {dof_labels[q]}  [{dof_units[q]}]'
                         + ('  — in LUT' if f['significant'] else ''),
                         fontsize=10, color=col)
            ax.tick_params(labelsize=9); ax.grid(alpha=0.3)
            if pi // ncols == nrows - 1:
                ax.set_xlabel(f'B set {axis} [deg]', fontsize=9)
        for pi in range(len(page), nrows * ncols):
            axes[pi // ncols][pi % ncols].axis('off')
        fig.suptitle(f'{title}\nDOF {page[0]}–{page[-1]} '
                     f'(panels {pg0 + 1}–{pg0 + len(page)} of {len(qs)}); '
                     f'x marks the reference, where the LUT is zero', fontsize=12)
        figs.append(fig)
    return figs


def plot_lut_summary(fits_by_bounce, dof_labels, dof_units, title=''):
    """Kept LUT coefficients by DOF, one row per (bounce, coefficient order),
    split into µm and arcsec DOF.

    Parameters
    ----------
    fits_by_bounce : `dict`
        ``{bounce name: (fits, axis)}``.

    Returns
    -------
    fig : `matplotlib.figure.Figure` or `None`
        `None` if no DOF is kept in any bounce.
    """
    rows = []
    for name, (fits, axis) in fits_by_bounce.items():
        sig = [q for q in sorted(fits) if fits[q]['significant']]
        if not sig:
            continue
        order = len(fits[sig[0]]['coef'])
        for p in range(1, order + 1):
            rows.append((name, axis, p, sig, fits))
    if not rows:
        return None
    units = ['um', 'arcsec']
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 3.2 * len(rows) + 1.2),
                             layout='constrained', squeeze=False)
    for ri, (name, axis, p, sig, fits) in enumerate(rows):
        for ui, u in enumerate(units):
            ax = axes[ri][ui]
            qs = [q for q in sig if dof_units[q].replace('μ', 'u').replace('µ', 'u') == u]
            if not qs:
                ax.text(0.5, 0.5, f'no kept DOF in {u}', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9, color='gray')
                ax.set_xticks([])
            else:
                c = [fits[q]['coef'][p - 1] for q in qs]
                s = [np.sqrt(fits[q]['cov'][p - 1, p - 1]) for q in qs]
                xi = np.arange(len(qs))
                ax.errorbar(xi, c, yerr=s, fmt='o', color='#d62728', mec='black',
                            mew=0.4, capsize=3)
                ax.axhline(0, color='gray', lw=0.5)
                ax.set_xticks(xi)
                ax.set_xticklabels([f'{q} {dof_labels[q]}' for q in qs], rotation=45,
                                   ha='right', fontsize=8)
            ax.set_ylabel(f'c{p} [{_coef_unit(dof_units[qs[0]] if qs else u, p)}]',
                          fontsize=9)
            ax.set_title(f'{name}: c{p} of (B {axis.lower()} − ref)^{p}, kept DOF in {u}',
                         fontsize=10)
            ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    return fig


def plot_lut_table(fits_by_bounce, dof_labels, dof_units, title=''):
    """Text page: every kept DOF with its coefficients, chi2/dof and significance."""
    lines = []
    for name, (fits, axis) in fits_by_bounce.items():
        sig = [q for q in sorted(fits) if fits[q]['significant']]
        lines.append(f'{name}  ({len(sig)} of {len(fits)} DOF kept)')
        for q in sig:
            f = fits[q]
            cs = '   '.join(f'c{p}={c:+.4g}±{s:.2g} {_coef_unit(dof_units[q], p)}'
                           for p, (c, s) in enumerate(
                               zip(f['coef'], np.sqrt(np.diag(f['cov']))), 1))
            lines.append(f'  DOF {q:2d} {dof_labels[q]:<14s} {cs}   '
                         f'chi2/dof={f["chi2"]:.2f}/{f["dof"]}  {f["nsig"]:.1f} sigma')
        lines.append('')
    fig = plt.figure(figsize=(14, 0.22 * len(lines) + 1.5))
    fig.text(0.01, 0.98, title + '\n\n' + '\n'.join(lines), va='top', ha='left',
             family='monospace', fontsize=8)
    return fig


def leg_spread(points):
    """chi2/dof of the nights at one B position about their weighted mean.

    Parameters
    ----------
    points : `list` [`tuple`]
        ``(night, b_value, delta, err)`` at one B position.

    Returns
    -------
    chi2, dof : `float`, `int`
        dof is the number of finite points minus one; ``(nan, 0)`` with fewer
        than two.
    """
    y = np.array([p[2] for p in points], float)
    e = np.array([p[3] for p in points], float)
    ok = np.isfinite(y) & np.isfinite(e) & (e > 0)
    y, e = y[ok], e[ok]
    if len(y) < 2:
        return np.nan, 0
    w = 1.0 / e ** 2
    m = np.sum(w * y) / np.sum(w)
    return float(np.sum(w * (y - m) ** 2)), int(len(y) - 1)


def plot_trend_panels(series, *, x_ref, x_range, axis, order, title, unit,
                      ncols=3, rows_per_page=2, panel_size=(4.6, 3.3),
                      night_colors=None):
    """Paired Δ of many quantities vs B-set position, one panel each, nights
    colored, with the LUT-form Huber fit and its chi2/dof for scale.

    Parameters
    ----------
    series : `dict`
        ``{panel title: [(night, b_value, delta, err), ...]}``, delta and err
        in `unit`.  Panels are drawn in dict order.
    x_ref : `float`
        Reference B position, deg; drawn as an x at zero.
    x_range : `list` [`float`]
        Fit range, deg.  Points outside are drawn but not fitted.
    axis : `str`
        Axis name for the x label.
    order : `int`
        Polynomial order of the fit, no constant term in (x - x_ref).
    unit : `str`
        Unit of delta, for the y label.
    night_colors : `dict`, optional
        ``{night: color}``; one per night if omitted.

    Returns
    -------
    figs : `list` [`matplotlib.figure.Figure`]
    """
    keys = list(series)
    nights = sorted({p[0] for v in series.values() for p in v})
    if night_colors is None:
        cmap = plt.get_cmap('tab10')
        night_colors = {n: cmap(i % 10) for i, n in enumerate(nights)}
    xg = np.linspace(x_range[0], x_range[1], 200)
    per_page = ncols * rows_per_page
    figs = []
    for pg0 in range(0, len(keys), per_page):
        page = keys[pg0:pg0 + per_page]
        nrows = int(np.ceil(len(page) / ncols))
        fig, axes = plt.subplots(nrows, ncols, layout='constrained', squeeze=False,
                                 figsize=(panel_size[0] * ncols + 0.8,
                                          panel_size[1] * nrows + 1.4))
        for pi, key in enumerate(page):
            ax = axes[pi // ncols][pi % ncols]
            pts = series[key]
            by_b = {}
            for p in pts:
                by_b.setdefault(p[1], []).append(p)
            # Nights sharing a B position are spread by 1 deg so their error
            # bars do not overlap.
            for bv, grp in by_b.items():
                grp = sorted(grp, key=lambda p: p[0])
                off = (np.arange(len(grp)) - (len(grp) - 1) / 2) * 1.0
                for (nt, _b, y, e), dx in zip(grp, off):
                    ax.errorbar(bv + dx, y, yerr=e, fmt='o', ms=4.5,
                                color=night_colors[nt], mec='black', mew=0.4,
                                elinewidth=0.9, capsize=2, zorder=3)
            sel = [p for p in pts if x_range[0] <= p[1] <= x_range[1]]
            txt = []
            f = huber_poly_fit([p[1] - x_ref for p in sel], [p[2] for p in sel],
                               [p[3] for p in sel], order)
            if f is not None:
                ax.plot(xg, poly_eval(f['coef'], xg - x_ref), color='gray', lw=1.0)
                txt.append(f"fit chi2/dof = {f['chi2']:.1f}/{f['dof']}")
            for bv, grp in sorted(by_b.items()):
                c2, nd = leg_spread(grp)
                if nd > 0:
                    txt.append(f'{bv:g} deg nights: chi2/dof = {c2:.1f}/{nd}')
            ax.axhline(0, color='gray', lw=0.5)
            ax.plot([x_ref], [0], marker='x', color='black', ms=6, zorder=4)
            if txt:
                ax.text(0.02, 0.97, '\n'.join(txt), transform=ax.transAxes,
                        va='top', fontsize=6.5,
                        bbox=dict(boxstyle='round', fc='white', alpha=0.8, lw=0.3))
            ax.set_xlim(x_range[0] - 3, max(x_range[1], x_ref) + 3)
            ax.set_title(key, fontsize=10)
            ax.set_ylabel(f'Δ [{unit}]', fontsize=8)
            ax.tick_params(labelsize=8); ax.grid(alpha=0.3)
            if pi // ncols == nrows - 1:
                ax.set_xlabel(f'B set {axis} [deg]', fontsize=8)
        for pi in range(len(page), nrows * ncols):
            axes[pi // ncols][pi % ncols].axis('off')
        handles = [plt.Line2D([], [], marker='o', ls='', color=night_colors[n],
                              mec='black', mew=0.4, label=str(n)) for n in nights]
        fig.legend(handles=handles, loc='outside lower center', ncols=len(nights),
                   fontsize=8, title='day_obs', title_fontsize=8)
        fig.suptitle(title, fontsize=11)
        figs.append(fig)
    return figs


def plot_spread_summary(rows, title):
    """Bar page: chi2/dof of the repeated-position nights and of the LUT-form
    fit, per quantity, so the DZ, v-mode and DOF views can be compared.

    Parameters
    ----------
    rows : `list` [`dict`]
        ``view``, ``name``, ``spread_chi2``, ``spread_dof``, ``fit_chi2``,
        ``fit_dof``.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
    """
    views = list(dict.fromkeys(r['view'] for r in rows))
    fig, axes = plt.subplots(len(views), 1, figsize=(16, 3.4 * len(views) + 1),
                             layout='constrained', squeeze=False)
    for vi, v in enumerate(views):
        ax = axes[vi][0]
        rs = [r for r in rows if r['view'] == v]
        xi = np.arange(len(rs))
        sp = [r['spread_chi2'] / r['spread_dof'] if r['spread_dof'] else np.nan
              for r in rs]
        ft = [r['fit_chi2'] / r['fit_dof'] if r['fit_dof'] else np.nan for r in rs]
        ax.bar(xi - 0.2, sp, 0.4, color='#1f77b4', label='repeated-position nights')
        ax.bar(xi + 0.2, ft, 0.4, color='#d62728', label='LUT-form fit')
        ax.axhline(1, color='black', lw=0.7)
        ax.set_yscale('log')
        ax.set_xticks(xi)
        ax.set_xticklabels([r['name'] for r in rs], rotation=90,
                           fontsize=6 if len(rs) > 60 else 7)
        ax.set_ylabel('chi2/dof (dimensionless)', fontsize=9)
        msp, mft = np.nanmedian(sp), np.nanmedian(ft)
        ax.set_title(f'{v}: median chi2/dof = {msp:.2f} (nights at one position), '
                     f'{mft:.2f} (fit)', fontsize=10)
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
    fig.suptitle(title, fontsize=12)
    return fig
