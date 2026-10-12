#!/usr/bin/env python3
"""Bounce-test plotters: the figures the `bounce` study renders.

FAM-triplet telescope-position bounce tests: time-ordered paired-difference Δ
(comparison − reference) per Double-Zernike (k, j), OFC v-mode, and physical
DOF, with robust (MAD) errors.  This module holds the heatmap / vs-ordinal /
night-scatter / per-DOF-panel plotters; the reduction they draw is in
`rubinwork.bounce.legs`, which this module re-exports so a caller can use one
import for both.  Driven by `run_bounce.py`.  RSP-only for the marker scheme and
DOF recovery (lsst.ts.ofc via ofc_svd)."""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from common.utils import alt_to_deg as _alt_to_deg  # noqa: E402

# The reduction, now a library so `rubinwork.products.bounce_tables` can call it
# without importing a study.  Re-exported under the names the plotters and
# `run_bounce.py` already use.
from rubinwork.bounce.legs import (  # noqa: F401,E402
    CAM_HEX_DOF, DEFAULT, DEFAULT_BOUNCES, DOF22, RBR_DEFAULTS, RBR_METHOD_NAME,
    _kj_to_array, ab_position_table, bounce_nights, bounce_program_mask,
    diff_of_deltas, diff_stats, filter_visits, form_pairs, leg_axis_name,
    leg_b_value, leg_night_coverage, leg_pointing, paired_delta,
    paired_deltas_kj, paired_deltas_matrix, passing_terms, rbr_deltas,
    rbr_dof_per_pair, rbr_module, resolve_config, run_bounce, solver_deltas,
    solver_dof_per_pair, stats_per_kj, to_long_df, visit_list_str,
)

try:
    from lsst.ts.intrinsic.wavefront.intrinsics_lib import (
        classify_visit, visit_marker_style, plot_visit_point)
    _marker_ok = True
except Exception as _e:   # pragma: no cover - RSP-only marker scheme
    print(f'(bounce_lib: intrinsics_lib marker scheme unavailable: '
          f'{type(_e).__name__}: {_e})')
    _marker_ok = False

from lsst.ts.intrinsic.wavefront.common.zernike_names import (
    NOLL_NAMES, NOLL_FORMULAS, FOCAL_NAMES, PUPIL_NAMES,
)
from lsst.ts.intrinsic.wavefront.ofc_svd import (LABELS_50DOF, DOF_UNITS_50, DOF_GROUPS,
                     recover_dof_per_visit)


def plot_kj_heatmap(stats, k_list, j_list, *, value_key='delta',
                    err_key='err', title='', cbar_label='',
                    cmap='RdBu_r', vlim=None, value_fmt='{:+.2f}',
                    err_fmt='±{:.2f}', cell_fontsize=7,
                    show_text=True):
    """Heatmap with k on rows, j on columns, value in colour, and
    optionally `value\n±err` in each cell.

    Returns the figure.

    Notes
    -----
    Not called by `run_bounce.py`: the per-(k, j) panels were dropped from
    `bounce_summary.pdf` in favour of the night-vs-night cross-scatter, the same
    numbers being available in `bounce_kj_stats.parquet`.  Kept for interactive
    use from a notebook.
    """
    Z = _kj_to_array(stats, k_list, j_list, value_key)
    Errs = (_kj_to_array(stats, k_list, j_list, err_key)
            if err_key else None)
    if vlim is None:
        finite = Z[np.isfinite(Z)]
        vlim = (float(np.nanpercentile(np.abs(finite), 95))
                if finite.size else 1.0)
        vlim = max(vlim, 1e-4)

    nk, nj = len(k_list), len(j_list)
    fig, ax = plt.subplots(
        figsize=(max(8.0, 0.55 * nj + 1.5),
                 max(2.8, 0.65 * nk + 1.5)),
        layout='constrained')
    im = ax.imshow(Z, cmap=cmap, vmin=-vlim, vmax=vlim,
                   aspect='auto')
    ax.set_xticks(range(nj))
    ax.set_xticklabels([f'Z{j}' for j in j_list], fontsize=8)
    ax.set_yticks(range(nk))
    ax.set_yticklabels([str(k) for k in k_list])
    ax.set_xlabel('Pupil Zernike index j')
    ax.set_ylabel('Field index k')
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label(cbar_label)

    if show_text:
        for ri in range(nk):
            for ci in range(nj):
                v = Z[ri, ci]
                if not np.isfinite(v):
                    continue
                txt = value_fmt.format(v)
                if Errs is not None and np.isfinite(Errs[ri, ci]):
                    txt += '\n' + err_fmt.format(Errs[ri, ci])
                # Pick black or white text depending on cell brightness.
                color = ('white' if abs(v) > 0.55 * vlim else 'black')
                ax.text(ci, ri, txt, ha='center', va='center',
                        fontsize=cell_fontsize, color=color)

    if title:
        ax.set_title(title, fontsize=12)
    return fig


def plot_kj_pass_heatmap(deltas, k_list, j_list, *,
                         nsigma_threshold=3.5,
                         delta_threshold_um=0.01,
                         sigma_only_threshold=None,
                         title='',
                         pass_color='#1a936f',
                         cell_fontsize=7,
                         show_text=True):
    """Binary pass/fail heatmap over (k, j).

    A cell passes when BOTH ``|Δ| > delta_threshold_um`` and
    ``|Δ / σ| > nsigma_threshold``.  Passing cells get the single
    ``pass_color`` background and an annotation ``Δ\n±err``; failing
    cells stay white with no annotation.

    Returns the figure.

    Notes
    -----
    Not called by `run_bounce.py` — see `plot_kj_heatmap`.  The pass/fail
    decision itself is still used, via `passing_terms`, to pick which (k, j)
    terms the night-vs-night cross-scatter draws.
    """
    from matplotlib.colors import ListedColormap

    nk, nj = len(k_list), len(j_list)
    D = np.full((nk, nj), np.nan)
    E = np.full((nk, nj), np.nan)
    S = np.full((nk, nj), np.nan)
    for (k, j), d in deltas.items():
        if k in k_list and j in j_list:
            ri = k_list.index(k); ci = j_list.index(j)
            D[ri, ci] = d.get('delta', np.nan)
            E[ri, ci] = d.get('err',   np.nan)
            S[ri, ci] = d.get('sig',   np.nan)

    finite_m = np.isfinite(D) & np.isfinite(S)
    cutA = (np.abs(D) > delta_threshold_um) & (np.abs(S) > nsigma_threshold)
    cutB = ((np.abs(S) > sigma_only_threshold)
            if sigma_only_threshold is not None
            else np.zeros_like(cutA, dtype=bool))
    passes = finite_m & (cutA | cutB)
    n_pass = int(passes.sum())
    n_total = int(np.isfinite(D).sum())

    fig, ax = plt.subplots(
        figsize=(max(8.0, 0.55 * nj + 1.5),
                 max(2.8, 0.65 * nk + 1.5)),
        layout='constrained')
    cmap = ListedColormap(['white', pass_color])
    ax.imshow(passes.astype(int), cmap=cmap, vmin=0, vmax=1,
              aspect='auto')
    # Light grid lines between cells for readability.
    ax.set_xticks(np.arange(-0.5, nj, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nk, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linewidth=0.4, alpha=0.6)
    ax.tick_params(which='minor', bottom=False, left=False)

    ax.set_xticks(range(nj))
    ax.set_xticklabels([f'Z{j}' for j in j_list], fontsize=8)
    ax.set_yticks(range(nk))
    ax.set_yticklabels([str(k) for k in k_list])
    ax.set_xlabel('Pupil Zernike index j')
    ax.set_ylabel('Field index k')

    if show_text:
        for ri in range(nk):
            for ci in range(nj):
                if not passes[ri, ci]:
                    continue
                txt = f'{D[ri, ci]:+.3f}\n±{E[ri, ci]:.3f}'
                ax.text(ci, ri, txt, ha='center', va='center',
                        fontsize=cell_fontsize, color='white')

    base = title or 'Significant (k, j) terms'
    crit = (f'(|Δ| > {delta_threshold_um:g} μm AND '
            f'|Δ/σ| > {nsigma_threshold:g})')
    if sigma_only_threshold is not None:
        crit += f'  OR  |Δ/σ| > {sigma_only_threshold:g}'
    ax.set_title(f'{base}\nPass: {crit}    '
                 f'({n_pass} / {n_total} cells pass)', fontsize=11)
    return fig


def plot_ab_position_table(rows, title='', figsize=(11, 8.5), max_rows=34):
    """Render `ab_position_table` rows as table page(s) for a PDF.

    Contiguous blocks with no clear classified position are highlighted, so a
    reader can see immediately whether every A/B set resolved.

    Parameters
    ----------
    rows : `list` of `dict`
        Output of `ab_position_table`.
    title : `str`, optional
        Figure title.
    figsize : `tuple`, optional
        Figure size in inches.
    max_rows : `int`, optional
        Rows per page; longer tables spill onto further pages.

    Returns
    -------
    figs : `list`
        Matplotlib figures, one per page (empty list if `rows` is empty).
    """
    if not rows:
        return []
    cols = ['day_obs', 'role', 'position\nelev/rot (deg)', 'seq range',
            'n visits', 'n blocks', 'band', 'elev range\n(deg)',
            'rot range\n(deg)']
    figs = []
    for pg in range(0, len(rows), max_rows):
        chunk = rows[pg:pg + max_rows]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        head = title if title else 'A/B bounce positions'
        if len(rows) > max_rows:
            head = f'{head}  (rows {pg + 1}–{pg + len(chunk)} of {len(rows)})'
        fig.suptitle(head, fontsize=12, y=0.96)
        def _rng(lo, hi):
            # A single value when the spread is below the printed precision,
            # so a constant position does not read as a range; ' to ' rather
            # than an en dash keeps negatives legible (-0.7 to -0.7).
            if abs(hi - lo) < 0.05:
                return f'{lo:.1f}'
            return f'{lo:.1f} to {hi:.1f}'

        cells, colors = [], []
        for row in chunk:
            cells.append([
                str(row['day_obs']), row.get('role', ''), row['position'],
                f"{row['seq_lo']}–{row['seq_hi']}", str(row['n']),
                str(row.get('n_blocks', '')), row['band'] or '—',
                _rng(row['alt_min_deg'], row['alt_max_deg']),
                _rng(row['rot_min_deg'], row['rot_max_deg']),
            ])
            if row['unclassified']:
                colors.append(['#ffd6d6'] * len(cols))
            elif row['short']:
                colors.append(['#fff4cc'] * len(cols))
            else:
                colors.append(['white'] * len(cols))
        # Height tracks the row count so a short table does not float in the
        # middle of an otherwise empty page.
        frac = min(0.88, 0.10 + 0.032 * (len(chunk) + 1))
        ax.set_position([0.04, max(0.06, 0.90 - frac), 0.92, frac])
        tab = ax.table(cellText=cells, colLabels=cols, cellColours=colors,
                       bbox=[0, 0, 1, 1], cellLoc='center')
        tab.auto_set_font_size(False)
        tab.set_fontsize(8)
        for ci in range(len(cols)):
            tab[0, ci].set_facecolor('#dddddd')
            tab[0, ci].set_text_props(weight='bold')
        n_un = sum(1 for row in rows if row['unclassified'])
        n_sh = sum(1 for row in rows if row['short'])
        note = ('Position is elevation/rotator in deg, snapped to the marker '
                'grid.  A = the night\'s most-populated position (reference), '
                'B = the throw.')
        if n_un:
            note += (f'  {n_un} contiguous block(s) shaded red have no clear '
                     'position — check these.')
        else:
            note += '  Every contiguous block resolved to a grid position.'
        if n_sh:
            note += f'  {n_sh} block(s) shaded amber are shorter than min_block.'
        fig.text(0.5, 0.025, note, ha='center', va='bottom', fontsize=8,
                 color='#333333', wrap=True)
        figs.append(fig)
    return figs


def plot_dz_vs_ordinal_pages(fit_table, prefix, k_list, j_list,
                             j_per_page=7, title_prefix='',
                             elev_halfwidth_deg=None):
    """Pages of DZ_kj vs ordinal image number (rows = focal k, cols =
    pupil j).  Points use the standard intrinsics_lib marker scheme
    (elevation -> colour, rotator angle -> arrow, filter band -> dot on
    the arrow shaft); dotted vertical lines mark day_obs changes.
    Returns a list of figures.

    `elev_halfwidth_deg` is the elevation bucket half-width in degrees,
    passed through to `_ordinal_setup`.
    """
    s = _ordinal_setup(fit_table, base_size=4,
                       elev_halfwidth_deg=elev_halfwidth_deg)
    ft, n, ordinal = s['ft'], s['n'], s['ordinal']
    styles, bands = s['styles'], s['bands']
    changes, day_labels = s['changes'], s['day_labels']

    figs = []
    for pg in range(0, len(j_list), j_per_page):
        jchunk = list(j_list[pg:pg + j_per_page])
        nrows, ncols = len(k_list), len(jchunk)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(2.7 * ncols + 1.0, 1.8 * nrows + 1.0),
                                 layout='constrained', sharex=True,
                                 squeeze=False)
        for ri, k in enumerate(k_list):
            for ci, j in enumerate(jchunk):
                ax = axes[ri][ci]
                col = f'{prefix}_z{j}_c{k}'
                if col not in ft.colnames:
                    ax.set_visible(False)
                    continue
                y = np.asarray(ft[col], dtype=float)
                for i in range(n):
                    if np.isfinite(y[i]):
                        draw_visit_point(ax, ordinal[i], y[i], styles[i],
                                         band=bands[i])
                for ch in changes:
                    ax.axvline(ch - 0.5, color='gray', ls=':', lw=0.6,
                               alpha=0.7)
                ax.axhline(0, color='k', lw=0.4, alpha=0.4)
                if ri == 0:
                    ax.set_title(f'Z{j}', fontsize=8)
                if ci == 0:
                    ax.set_ylabel(f'k={k}', fontsize=8)
                ax.tick_params(labelsize=6)
        # Annotate the day_obs at each day-start on the first panel so
        # the ordinal axis can be read back to calendar nights.
        ax0 = axes[0][0]
        for (start_i, day) in day_labels:
            ax0.text(start_i, 0.98, str(day),
                     transform=ax0.get_xaxis_transform(),
                     rotation=90, va='top', ha='left',
                     fontsize=5, color='dimgray', alpha=0.9)
        for ax in axes[-1]:
            ax.set_xlabel('ordinal image #', fontsize=7)
        fig.suptitle(f'{title_prefix}{prefix}  DZ_kj vs ordinal image number  '
                     f'(pupil j {jchunk[0]}..{jchunk[-1]})  '
                     f'— dotted lines = day_obs change', fontsize=12)
        figs.append(fig)
    return figs


def plot_night_cross_scatter(deltas_by_night, passing_kj, title_root='',
                             zoom_lim_um=None):
    """Night-A vs Night-B Δ DZ_kj cross-comparison, one figure per night
    pair (plus a zoomed inner-region page when `zoom_lim_um` is given).

    Each passing (k, j) is an errorbar point (per-night SEM as x/y
    errors) annotated with k,j; a y = x line is drawn.  The full page
    autoscales to the data (+ errors); the zoom page fixes the axes to
    ±`zoom_lim_um`.  Returns a list of figures
    (full[, zoom] per pair; empty if < 2 nights or no passing terms).
    """
    import itertools
    nights = sorted(deltas_by_night.keys())
    figs = []
    if len(nights) < 2 or not passing_kj:
        return figs

    def _make(A, B, xs, ys, xe, ye, labs, lim):
        fig, ax = plt.subplots(figsize=(9, 9), layout='constrained')
        ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt='o', ms=6,
                    color='steelblue', ecolor='gray', elinewidth=0.9,
                    capsize=2, alpha=0.85, zorder=3)
        for x, y, l in zip(xs, ys, labs):
            ax.annotate(l, (x, y), textcoords='offset points',
                        xytext=(5, 5), fontsize=8, color='black')
        if lim is None:
            lo = float(min((xs - xe).min(), (ys - ye).min()))
            hi = float(max((xs + xe).max(), (ys + ye).max()))
            pad = 0.10 * max(hi - lo, 1e-3)
            lo -= pad; hi += pad
            ztag = ''
        else:
            lo, hi = -float(lim), float(lim)
            ztag = f'  (zoom ±{lim:g} μm)'
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.6, zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.axhline(0, color='gray', lw=0.4)
        ax.axvline(0, color='gray', lw=0.4)
        ax.set_xlabel(f'Δ DZ_kj  night {A} [μm]', fontsize=11)
        ax.set_ylabel(f'Δ DZ_kj  night {B} [μm]', fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_title(f'{title_root}\nnight {A} vs night {B}  '
                     f'({len(xs)} terms){ztag}', fontsize=12)
        return fig

    for (A, B) in itertools.combinations(nights, 2):
        da, db = deltas_by_night[A], deltas_by_night[B]
        xs, ys, xe, ye, labs = [], [], [], [], []
        for (k, j) in passing_kj:
            a = da.get((k, j)); b = db.get((k, j))
            if a is None or b is None:
                continue
            if np.isfinite(a.get('delta', np.nan)) and np.isfinite(b.get('delta', np.nan)):
                xs.append(a['delta']); ys.append(b['delta'])
                xe.append(a.get('err', np.nan)); ye.append(b.get('err', np.nan))
                labs.append(f'{k},{j}')
        if not xs:
            continue
        xs = np.array(xs); ys = np.array(ys)
        xe = np.nan_to_num(np.array(xe)); ye = np.nan_to_num(np.array(ye))
        figs.append(_make(A, B, xs, ys, xe, ye, labs, None))
        if zoom_lim_um is not None:
            figs.append(_make(A, B, xs, ys, xe, ye, labs, zoom_lim_um))
    return figs


# ==================================================================
# Paired-difference Δ engine (time-aware, model-free)
# ==================================================================


# ==================================================================
# Range-Bounded Recovery (RBR) of the paired-difference optical state
# ==================================================================
# The default recovery inverts the measured wavefront onto DOF with a
# truncated SVD (50 DOF, 34 v-modes kept).  Truncation is its only
# regularizer, and it leaves the recovered mirror bending-mode amplitudes
# free to exceed the stroke the mirror can physically reach — on these
# bounce legs by up to a factor of 11.27 (dimensionless, recovered
# amplitude over allowed range).  RBR adds a penalty that is negligible
# while |d_j| / r_j stays well under `kappa` and rises steeply as it
# approaches and passes it, so the recovered state stays inside the
# allowed range at a small cost in corrected image quality.
#
# The solver lives in smatrix/code/regularized_inversion.py, shared code beside
# the `regularized_inversion` study that derives the method and measures its
# FWHM cost; this module only applies it to the bounce paired Δ.
#
# Why per pair and not once on the median: RBR is *nonlinear*, so it does
# not commute with the median over pairs, and inverting a single median
# wavefront would give no error bar.  Each (ref, comp) pair's Δ wavefront
# is inverted on its own and the same median / median-SEM reduction used
# everywhere else in this module is applied to the resulting DOF.  That
# keeps the RBR error definition identical to the default one and lets the
# nonlinearity show up in the scatter rather than being averaged away.


# OFC v-mode / DOF recovery (LABELS_50DOF, DOF_UNITS_50,
# recover_dof_per_visit) now live in code/ofc_svd.py — imported above.


# ==================================================================
# Generic <quantity> vs ordinal-image plotting (marker scheme shared)
# ==================================================================
def draw_visit_point(ax, x, y, style, band=None, **kwargs):
    """Draw one visit marker plus its filter-band dot.

    Thin wrapper over `intrinsics_lib.plot_visit_point` that degrades to a
    plain `ax.plot` when the marker scheme is unavailable. Every per-visit
    plot in this module goes through here, so the band encoding stays
    consistent across the vs-ordinal DZ, v-mode and DOF products.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axes to draw on.
    x, y : `float`
        Point position in data coordinates.
    style : `dict`
        Marker kwargs from `visit_marker_style`.
    band : `str`, optional
        Filter band; the dot is skipped when None or unrecognized.
    **kwargs
        Extra kwargs for the marker (e.g. `zorder`, `alpha`).

    Returns
    -------
    lines : `list`
        The Line2D objects drawn.
    """
    if _marker_ok:
        return plot_visit_point(ax, x, y, style, band=band, **kwargs)
    st = dict(style)
    st.update(kwargs)
    return list(ax.plot([x], [y], **st))


def _ordinal_setup(fit_table, base_size=4, elev_halfwidth_deg=None):
    """Time-sort a fit_table and build the shared per-visit marker
    styles, day_obs change indices and day labels for vs-ordinal plots.

    Parameters
    ----------
    fit_table : `astropy.table.Table`
        Fit table with `day_obs`, `seq_num`, and optionally `alt` (deg or
        rad), `rotator_angle` (deg) and `band`.
    base_size : `float`, optional
        Marker size in points.
    elev_halfwidth_deg : `float`, optional
        Elevation bucket half-width in degrees for the marker colors.
        Defaults to the `intrinsics_lib` value (2.0 deg).

    Returns
    -------
    setup : `dict`
        Keys order/ft/n/ordinal/styles/bands/changes/day_labels. `bands` is
        the per-visit one-character band (or None), aligned to `styles`, for
        passing to `draw_visit_point`.
    """
    dobs = np.asarray(fit_table['day_obs']).astype(int)
    snum = np.asarray(fit_table['seq_num']).astype(int)
    order = np.lexsort((snum, dobs))
    ft = fit_table[order]
    dobs = dobs[order]
    n = len(ft)
    alt = (_alt_to_deg(ft['alt']) if 'alt' in ft.colnames
           else np.full(n, np.nan))
    rot = (np.asarray(ft['rotator_angle'], dtype=float)
           if 'rotator_angle' in ft.colnames else np.full(n, np.nan))
    has_band = 'band' in ft.colnames
    band = np.asarray(ft['band']).astype(str) if has_band else None
    styles, bands = [], []
    for i in range(n):
        b = (band[i] if has_band else None)
        if _marker_ok:
            cls = classify_visit(alt_deg=alt[i], rot_deg=rot[i], band=b,
                                 elev_halfwidth_deg=elev_halfwidth_deg)
            styles.append(visit_marker_style(
                elev=cls['elev'], rot=cls['rot'], band=cls['band'],
                base_size=base_size))
            bands.append(cls['band'])
        else:
            styles.append(dict(marker='o', color='steelblue',
                               markersize=3, linestyle=''))
            bands.append(None)
    changes = [i for i in range(1, n) if dobs[i] != dobs[i - 1]]
    day_labels = [(i, int(dobs[i])) for i in [0] + changes]
    return {'order': order, 'ft': ft, 'n': n, 'ordinal': np.arange(n),
            'styles': styles, 'bands': bands,
            'changes': changes, 'day_labels': day_labels}


def plot_values_vs_ordinal_pages(fit_table, value_matrix, labels,
                                 units=None, title_root='', ncols=5,
                                 rows_per_page=7, elev_halfwidth_deg=None):
    """Pages of <quantity> vs ordinal image number, one panel per column
    of `value_matrix` (aligned row-for-row to `fit_table`).  Standard
    marker scheme (elevation -> colour, rotator angle -> arrow, filter
    band -> dot on the arrow shaft); dotted day_obs lines; day_obs
    annotated on the first panel.  Returns a list of figures.

    `elev_halfwidth_deg` is the elevation bucket half-width in degrees,
    passed through to `_ordinal_setup`.
    """
    s = _ordinal_setup(fit_table, elev_halfwidth_deg=elev_halfwidth_deg)
    order, ordinal, styles = s['order'], s['ordinal'], s['styles']
    bands = s['bands']
    changes, day_labels, n = s['changes'], s['day_labels'], s['n']
    M = np.asarray(value_matrix, dtype=float)[order]
    nq = M.shape[1]
    per_page = ncols * rows_per_page
    figs = []
    for pg in range(0, nq, per_page):
        qs = list(range(pg, min(pg + per_page, nq)))
        nrows = int(np.ceil(len(qs) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(2.7 * ncols + 1.0, 1.7 * nrows + 1.0),
            layout='constrained', sharex=True, squeeze=False)
        for cell, q in enumerate(qs):
            ax = axes[cell // ncols][cell % ncols]
            y = M[:, q]
            for i in range(n):
                if np.isfinite(y[i]):
                    draw_visit_point(ax, ordinal[i], y[i], styles[i],
                                     band=bands[i])
            for ch in changes:
                ax.axvline(ch - 0.5, color='gray', ls=':', lw=0.6, alpha=0.7)
            ax.axhline(0, color='k', lw=0.4, alpha=0.4)
            lab = labels[q] if units is None else f'{labels[q]} [{units[q]}]'
            ax.set_title(lab, fontsize=8)
            ax.tick_params(labelsize=6)
        for cell in range(len(qs), nrows * ncols):
            axes[cell // ncols][cell % ncols].set_visible(False)
        ax0 = axes[0][0]
        for (start_i, day) in day_labels:
            ax0.text(start_i, 0.98, str(day),
                     transform=ax0.get_xaxis_transform(), rotation=90,
                     va='top', ha='left', fontsize=5, color='dimgray',
                     alpha=0.9)
        for c in range(ncols):
            axes[nrows - 1][c].set_xlabel('ordinal image #', fontsize=7)
        fig.suptitle(f'{title_root}  (panels {qs[0]}..{qs[-1]})  '
                     f'— dotted lines = day_obs change', fontsize=12)
        figs.append(fig)
    return figs


# ==================================================================
# DOF night-A vs night-B 5-panel scatter
# ==================================================================
DOF_PANELS = [
    ('Cam & M2 Hex piston',   DOF_GROUPS['hex_piston']),
    ('Cam & M2 Hex decenter', DOF_GROUPS['hex_decenter']),
    ('Cam & M2 Hex tip/tilt', DOF_GROUPS['hex_tiptilt']),
    ('M1M3 bending modes',    DOF_GROUPS['m1m3_bending']),
    ('M2 bending modes',      DOF_GROUPS['m2_bending']),
]


def plot_leg_night_coverage(rows, title='', figsize=(11, 4.2)):
    """Render `leg_night_coverage` rows as a one-page table figure.

    Legs with fewer than 2 qualifying nights are shaded amber, since no
    night-vs-night scatter is drawn for them.

    Parameters
    ----------
    rows : `list` [`dict`]
        Output of `leg_night_coverage`.
    title : `str`, optional
        Page title.
    figsize : `tuple` [`float`], optional
        Figure size in inches.

    Returns
    -------
    fig : `matplotlib.figure.Figure` or `None`
        `None` if `rows` is empty.
    """
    if not rows:
        return None
    cols = ['bounce', 'leg\n(B set)', 'B value\n[deg]', 'n\nvisits', 'n\npairs',
            'n\nnights', 'nights with results', 'night-pair\nscatter pages']
    # Column widths as fractions of the page: the night list needs the room, the
    # counts do not.
    widths = [0.15, 0.09, 0.08, 0.07, 0.07, 0.07, 0.36, 0.11]
    body, shade = [], []
    for r in rows:
        bv = r['b_value_deg']
        body.append([r['bounce'], r['comparison'],
                     ('' if not np.isfinite(bv) else f'{bv:g}'),
                     str(r['n_visits']), str(r['n_pairs']),
                     str(r['n_nights']), r['nights'],
                     str(r['scatter_pages'])])
        shade.append(r['n_nights'] < 2)
    fig = plt.figure(figsize=(figsize[0],
                              max(1.6, 0.40 * (len(body) + 2)) + 0.9))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.80]); ax.axis('off')
    tab = ax.table(cellText=body, colLabels=cols, cellLoc='center',
                   colWidths=widths, bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False); tab.set_fontsize(8)
    for ci in range(len(cols)):
        tab[(0, ci)].set_facecolor('#dddddd')
        tab[(0, ci)].set_text_props(weight='bold')
    for ri, bad in enumerate(shade, start=1):
        if bad:
            for ci in range(len(cols)):
                tab[(ri, ci)].set_facecolor('#fdf0d5')
    if title:
        fig.suptitle(title, fontsize=12, y=0.985)
    fig.text(0.02, 0.86,
             'Amber: only one night on this leg, so no night-vs-night '
             'scatter page is drawn.', fontsize=8, color='#7a5c00')
    return fig


def plot_dof_night_scatter(dof_deltas_by_night, labels, units=None,
                           title_root='', night_pair=None):
    """5-panel (one page) night-A vs night-B scatter of the physical
    DOF Δ values: Cam/M2 hex pistons, hex decenters, hex tip/tilts,
    M1M3 bending (20), M2 bending (20) — all 50 DOF.  Each point is an
    errorbar (paired-Δ err per night) annotated with its DOF label; a
    y = x line is drawn per panel.  One figure per night pair (or just
    `night_pair` if given).  Returns a list of figures.
    """
    import itertools
    nights = sorted(dof_deltas_by_night.keys())
    if len(nights) < 2:
        return []
    night_pairs = ([tuple(night_pair)] if night_pair is not None
                   else list(itertools.combinations(nights, 2)))
    figs = []
    for (A, B) in night_pairs:
        da = dof_deltas_by_night.get(A)
        db = dof_deltas_by_night.get(B)
        if da is None or db is None:
            continue
        fig, axes = plt.subplots(1, 5, figsize=(24, 5.2),
                                 layout='constrained')
        for pi, (ptitle, qs) in enumerate(DOF_PANELS):
            ax = axes[pi]
            xs, ys, xe, ye, labs = [], [], [], [], []
            for q in qs:
                a = da.get(q); bb = db.get(q)
                if a is None or bb is None:
                    continue
                if (np.isfinite(a.get('delta', np.nan))
                        and np.isfinite(bb.get('delta', np.nan))):
                    xs.append(a['delta']); ys.append(bb['delta'])
                    xe.append(a.get('err', np.nan))
                    ye.append(bb.get('err', np.nan))
                    labs.append(labels[q])
            if xs:
                xs = np.array(xs); ys = np.array(ys)
                xe = np.nan_to_num(np.array(xe))
                ye = np.nan_to_num(np.array(ye))
                ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt='o', ms=5,
                            color='steelblue', ecolor='gray',
                            elinewidth=0.8, capsize=2, alpha=0.85, zorder=3)
                for x, y, l in zip(xs, ys, labs):
                    ax.annotate(l, (x, y), textcoords='offset points',
                                xytext=(4, 4), fontsize=6, color='black')
                lo = float(min((xs - xe).min(), (ys - ye).min()))
                hi = float(max((xs + xe).max(), (ys + ye).max()))
                pad = 0.10 * max(hi - lo, 1e-6)
                lo -= pad; hi += pad
                ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.6,
                        zorder=1)
                ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            unit = (f' [{units[qs[0]]}]' if units is not None else '')
            ax.axhline(0, color='gray', lw=0.4)
            ax.axvline(0, color='gray', lw=0.4)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xlabel(f'night {A}{unit}', fontsize=9)
            ax.set_ylabel(f'night {B}{unit}', fontsize=9)
            ax.set_title(ptitle, fontsize=10)
            ax.grid(alpha=0.3)
        fig.suptitle(f'{title_root}\nDOF Δ:  night {A} vs night {B}',
                     fontsize=13)
        figs.append(fig)
    return figs


# (colour, marker) per overlay series in `plot_dof_vs_b_value_panels`, in the
# order the series are given.  Marker *and* colour both vary, so the schemes are
# separable in greyscale as well as in colour; the shapes and colours match
# `FWHM_SERIES_DEFAULT` so a scheme wears the same style in every bounce PDF.
# The base series takes `BASE_STYLE`.
OVERLAY_STYLES = (('#d62728', 's'), ('#2ca02c', '^'), ('#9467bd', 'D'),
                  ('#ff7f0e', 'v'), ('#8c564b', 'P'), ('#17becf', 'X'))
BASE_STYLE = ('#1f77b4', 'o')


def plot_dof_vs_b_value_panels(entries, dof_labels, dof_units,
                               dof_indices=None, x_label='Elevation [deg]',
                               title='', ncols=2, rows_per_page=5,
                               panel_size=(5.2, 3.4),
                               annotate=True, overlay_series=(), ranges=None,
                               base_label='default 50/34'):
    """Per-DOF panels of the paired-Δ DOF against the B-set position, paginated.

    One panel per degree of freedom (DOF); within a panel each point is one
    (night, B set) entry, plotted at its B-set position on the x axis with the
    paired-Δ error as a y error bar.  This is the form in which a Look-Up Table
    (LUT) reads the bounce: how each DOF's change grows with the throw.

    Panels are laid out `ncols` by `rows_per_page` per page and the DOF are
    split across as many pages as that needs, so each panel is large enough to
    read several overlaid recovery schemes at once.

    Points are drawn in increasing B-set order, so the BLOCK-T720 elevation
    sweep reads left to right in elevation and BLOCK-T724 left to right in
    camera-rotator angle.  **Colour and marker both encode the recovery
    scheme**, which is what the eye needs to separate here — nights sharing a
    B set are instead fanned out in x and named by the per-point annotation
    (`day_obs` and the B value in deg).

    Parameters
    ----------
    entries : `list` [`dict`]
        One entry per (night, comparison leg), each with keys `night`
        (`int` day_obs), `b_value` (`float`, deg), `label` (`str`, the leg
        label) and `dof_deltas` — the `{dof_index: {'delta','err'}}` dict from
        `paired_deltas_matrix`, in µm for translations and bending-mode
        amplitudes and arcsec for hexapod rotations.  Each overlay series adds
        one further key of the same form.
    dof_labels : `list` [`str`]
        DOF names, indexed by global DOF index (`LABELS_50DOF`).
    dof_units : `list` [`str`]
        Per-DOF units, indexed the same way (`DOF_UNITS_50`) — µm or arcsec.
    dof_indices : `list` [`int`], optional
        Which DOF to panel.  Defaults to every DOF that any entry populates
        with a finite Δ in the base series *or any overlay series*, so a scheme
        that reaches a DOF the base series does not still gets a panel.
    x_label : `str`, optional
        X-axis label, including the unit.
    title : `str`, optional
        Figure title, repeated on every page with the panel span appended.
    ncols : `int`, optional
        Panels per row.
    rows_per_page : `int`, optional
        Panel rows per page; `ncols * rows_per_page` panels per page.
    panel_size : `tuple` [`float`], optional
        Per-panel (width, height) in inches.
    annotate : `bool`, optional
        Annotate each point with `day_obs` and the B value in deg.
    overlay_series : `sequence` [`tuple`], optional
        `(entry_key, label)` per overlay, each `entry_key` holding a second
        `{dof_index: {'delta','err'}}` dict in the same units as the base
        series — the other recovery schemes.  Drawn as open markers taking
        colour and shape from `OVERLAY_STYLES` in the order given.  Empty draws
        the base series alone.
    ranges : `array_like`, optional
        Allowed range `r_j` per DOF, indexed by global DOF index, in that
        DOF's own unit.  When given, each panel gets a shaded band at
        ±`r_j` so a point outside the physically reachable range is visible
        by eye.
    base_label : `str`, optional
        Legend label for the filled-circle series in `dof_deltas`. Name the
        recovery scheme it came from; on a camera-hexapod-only bounce that is
        the 5 DOF / 5 v-mode scheme while the overlays include the 50 DOF / 34
        v-mode one, so the two differ and the legend must say so.

    Returns
    -------
    figs : `list` [`matplotlib.figure.Figure`]
        One figure per page, empty if no entry carries a finite Δ.
    """
    ents = sorted(entries, key=lambda e: (e['b_value'], e['night']))
    overlays = [(k, lab) + OVERLAY_STYLES[i % len(OVERLAY_STYLES)]
                for i, (k, lab) in enumerate(overlay_series)]
    if dof_indices is None:
        seen = set()
        for e in ents:
            for key in ['dof_deltas'] + [k for k, _, _, _ in overlays]:
                for q, v in (e.get(key) or {}).items():
                    if np.isfinite(v.get('delta', np.nan)):
                        seen.add(int(q))
        dof_indices = sorted(seen)
    if not dof_indices or not ents:
        return []

    nights = sorted({int(e['night']) for e in ents})

    # Several nights can share one B set (five nights throw to elevation 40 deg),
    # which would stack their points and annotations on top of each other.  Fan
    # them out in x by a fraction of the B-set spacing; the x axis stays the
    # B-set position, so the jitter is cosmetic and small.
    bvals = sorted({e['b_value'] for e in ents})
    span = (max(bvals) - min(bvals)) if len(bvals) > 1 else 1.0
    x_jit = {}
    for bv in bvals:
        at_bv = sorted({int(e['night']) for e in ents if e['b_value'] == bv})
        step = 0.030 * span
        for i, nt in enumerate(at_bv):
            x_jit[(bv, nt)] = (i - (len(at_bv) - 1) / 2) * step

    # With a single B set the jitter has no B-set spacing to scale against, so
    # the tick labels spread over a hundredth of a degree and invite reading the
    # nights as sitting at different B values.  Pin x to the B set and say so.
    # The note goes in the suptitle rather than the x label: the per-panel x
    # labels sit side by side and a long qualifier on each would overrun its
    # neighbours.
    single_b = len(bvals) == 1
    jmax = max((abs(v) for v in x_jit.values()), default=0.0)
    if single_b:
        title = (f'{title}\nOne B set only — nights offset in x for '
                 f'legibility; all sit at {bvals[0]:g} deg')

    def _collect(e, key, q):
        """x, y, yerr and night of one entry's Δ for DOF `q`, or None."""
        v = (e.get(key) or {}).get(q)
        if v is None or not np.isfinite(v.get('delta', np.nan)):
            return None
        nt = int(e['night'])
        return (e['b_value'] + x_jit.get((e['b_value'], nt), 0.0),
                v['delta'], v.get('err', np.nan), nt)

    per_page = int(ncols) * int(rows_per_page)
    figs = []
    for pg0 in range(0, len(dof_indices), per_page):
        page = dof_indices[pg0:pg0 + per_page]
        nrows = int(np.ceil(len(page) / ncols))
        fig, axes = plt.subplots(nrows, ncols, layout='constrained',
                                 figsize=(panel_size[0] * ncols + 1.2,
                                          panel_size[1] * nrows + 1.6),
                                 squeeze=False)
        for pi, q in enumerate(page):
            ax = axes[pi // ncols][pi % ncols]
            # Allowed-range band first, so the points draw over it.  Drawn with
            # the y limits frozen to the data afterwards, because the rigid-body
            # ranges (thousands of µm) are orders of magnitude wider than their Δ
            # and would otherwise set the scale and flatten the points to a line.
            band = None
            if ranges is not None and q < len(ranges) and np.isfinite(ranges[q]):
                band = float(ranges[q])
            # Overlay series first, so the filled base series draws over them.
            for okey, _olab, ocol, omark in overlays:
                pts = [p for p in (_collect(e, okey, q) for e in ents)
                       if p is not None]
                if not pts:
                    continue
                oxs, oys, oes, _ = map(list, zip(*pts))
                ax.errorbar(oxs, oys,
                            yerr=np.nan_to_num(np.asarray(oes, float)),
                            fmt='none', ecolor=ocol, elinewidth=0.8, alpha=0.6,
                            capsize=2, zorder=4)
                ax.scatter(oxs, oys, s=44, facecolors='none', edgecolors=ocol,
                           linewidths=1.3, marker=omark, zorder=5)
            pts = [p for p in (_collect(e, 'dof_deltas', q) for e in ents)
                   if p is not None]
            if pts:
                xs, ys, es, nts = map(list, zip(*pts))
                xs = np.asarray(xs, float); ys = np.asarray(ys, float)
                es = np.nan_to_num(np.asarray(es, float))
                labs = [f"{nt % 10000}/{x:g}" for nt, x in zip(nts, xs)]
                ax.errorbar(xs, ys, yerr=es, fmt='none', ecolor=BASE_STYLE[0],
                            elinewidth=0.8, alpha=0.6, capsize=2, zorder=2)
                ax.scatter(xs, ys, s=40, c=BASE_STYLE[0],
                           marker=BASE_STYLE[1], edgecolors='black',
                           linewidths=0.4, zorder=3)
                if annotate:
                    # Alternate the label side so neighbouring points in a
                    # crowded B set do not overwrite one another, and keep
                    # labels inside the axes by flipping those near the right
                    # edge.
                    xmid = 0.5 * (xs.min() + xs.max())
                    for i, (x, y, l) in enumerate(zip(xs, ys, labs)):
                        right = x > xmid
                        ax.annotate(l, (x, y), textcoords='offset points',
                                    xytext=(-4 if right else 4,
                                            5 if i % 2 == 0 else -10),
                                    ha='right' if right else 'left',
                                    fontsize=6.0, color='#333333')
                ax.margins(x=0.18)
            ax.axhline(0, color='gray', lw=0.5, alpha=0.8)
            if band is not None:
                # Keep the data's own y scale; the band is context, not a series.
                lo, hi = ax.get_ylim()
                ax.axhspan(-band, band, color='#f2c9d4', alpha=0.45, zorder=0,
                           lw=0)
                if band > max(abs(lo), abs(hi)):
                    # Range far wider than the Δ: the band covers the panel, so
                    # say so in the corner instead of rescaling away the data.
                    ax.set_ylim(lo, hi)
                    ax.text(0.02, 0.04,
                            f'±r_j = {band:.3g} {dof_units[q]}',
                            transform=ax.transAxes, fontsize=7,
                            color='#9c3b57', va='bottom')
                else:
                    ax.set_ylim(min(lo, -1.15 * band), max(hi, 1.15 * band))
            if single_b:
                ax.set_xticks(list(bvals))
                ax.set_xlim(bvals[0] - 3.0 * max(jmax, 1e-3),
                            bvals[0] + 3.0 * max(jmax, 1e-3))
            ax.set_title(f'DOF {q}  {dof_labels[q]}  [{dof_units[q]}]',
                         fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(alpha=0.3)
            if pi // ncols == nrows - 1:
                ax.set_xlabel(x_label, fontsize=9)
        for pi in range(len(page), nrows * ncols):
            axes[pi // ncols][pi % ncols].axis('off')

        # One legend, below the panels.  It must not go at the top: the title is
        # up to four lines and an upper-corner legend overlaps it.
        style = [plt.Line2D([], [], marker=BASE_STYLE[1], ls='',
                            color=BASE_STYLE[0], markeredgecolor='black',
                            markeredgewidth=0.4, label=base_label)]
        style += [plt.Line2D([], [], marker=omark, ls='',
                             markerfacecolor='none', markeredgecolor=ocol,
                             markeredgewidth=1.3, label=olab)
                  for _okey, olab, ocol, omark in overlays]
        if ranges is not None:
            style.append(plt.Rectangle((0, 0), 1, 1, color='#f2c9d4',
                                       alpha=0.45,
                                       label='allowed range ±r_j'))
        fig.legend(handles=style, loc='outside lower center',
                   ncol=min(len(style), 3), fontsize=9,
                   title='recovery scheme', title_fontsize=9, frameon=False)
        pg_lab = (f'DOF {page[0]}–{page[-1]} '
                  f'(panels {pg0 + 1}–{pg0 + len(page)} of {len(dof_indices)})')
        night_lab = (f'{len(nights)} nights: '
                     + ', '.join(str(nt) for nt in nights)
                     + ' — named per point as day_obs/B value in deg')
        fig.suptitle(f'{title}\n{pg_lab}\n{night_lab}', fontsize=12)
        figs.append(fig)
    return figs


# Default series for `plot_fwhm_vs_b_value`: (row key, legend label, colour,
# marker, line style).  Every "after" series is the *achieved* residual
# dW - S (d / w) of its own recovery scheme, in arcsec FWHM, so the vertical gaps
# between them are directly comparable image-quality costs.
FWHM_SERIES_DEFAULT = (
    ('fwhm_before', 'no correction', '#444444', 'o', '--'),
    ('fwhm_after_default', 'default 50/34', '#1f77b4', 'o', '-'),
    ('fwhm_after_rbr', 'RBR 50/34 (range-bounded)', '#d62728', 's', '-'),
    ('fwhm_after_22_12', '22/12 reduced DOF', '#2ca02c', '^', '-'),
    ('fwhm_after_oic', 'OIC 50/34 (motion penalty)', '#9467bd', 'D', '-'),
    ('fwhm_after_5_5', '5/5 camera hexapod', '#ff7f0e', 'v', '-'),
)


def plot_fwhm_vs_b_value(rows, x_label='Elevation [deg]', title='',
                         annotate=True, ax=None, series=None):
    """Correctable FWHM against the B-set position, one point per (night, leg).

    One series per recovery scheme, all in arcsec FWHM (median over the focal
    plane): the uncorrected differential FWHM of the bounce optical-state
    change, then the residual left by each scheme.  The vertical gap between
    two "after" series is the image-quality price of the difference between
    them — for instance of keeping the recovered degrees of freedom (DOF)
    inside the range the telescope can actually apply, or of dropping from 50
    DOF to the reduced operational set.

    Nights sharing a B set are fanned out in x by a few percent of the B-set
    span so their points do not stack; the x axis is still the B-set position,
    so the offset is cosmetic.  The connecting line joins the per-B-set median
    over nights rather than the individual points.

    Parameters
    ----------
    rows : `list` [`dict`]
        One entry per (night, comparison leg), each with `night` (`int`
        day_obs), `b_value` (`float`, deg), `label` (`str`, leg label),
        `n_pairs` (`int`), and the FWHM value in arcsec under each series key.
        A missing or non-finite value is skipped for that series only, so a
        scheme that does not apply to a bounce simply does not appear.
    x_label : `str`, optional
        X-axis label including the unit — elevation or camera-rotator angle
        in deg, whichever is the B-set axis of the bounce.
    title : `str`, optional
        Axes title.
    annotate : `bool`, optional
        Annotate the uncorrected points with `day_obs`.
    ax : `matplotlib.axes.Axes`, optional
        Axes to draw on; a new figure is made when omitted.
    series : `sequence` [`tuple`], optional
        `(row_key, label, colour, marker, linestyle)` per series.  Defaults to
        `FWHM_SERIES_DEFAULT`, which covers every scheme the bounce computes.

    Returns
    -------
    fig : `matplotlib.figure.Figure` or `None`
        The figure drawn on, or `None` if no row carries a finite FWHM.
    """
    ents = sorted(rows, key=lambda r: (r['b_value'], r['night']))
    series = list(FWHM_SERIES_DEFAULT if series is None else series)
    if not ents or not any(np.isfinite(e.get(k, np.nan))
                           for e in ents for k, *_ in series):
        return None
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.2, 5.0), layout='constrained')
    else:
        fig = ax.figure

    # Several nights can throw to the same B set (three nights to elevation
    # 40 deg, both rotator nights to 60 deg), which stacks their points on one
    # x and makes a connecting line meaningless.  Fan the nights out in x by a
    # small fraction of the B-set span, and join only the night-median at each
    # B set so the trend line still reads as a trend.
    bvals = sorted({e['b_value'] for e in ents})
    span = (max(bvals) - min(bvals)) if len(bvals) > 1 else 1.0
    x_jit = {}
    for bv in bvals:
        at_bv = sorted({int(e['night']) for e in ents if e['b_value'] == bv})
        for i, nt in enumerate(at_bv):
            x_jit[(bv, nt)] = (i - (len(at_bv) - 1) / 2) * 0.035 * span

    for key, lab, col, mk, ls in series:
        pts = [(e['b_value'] + x_jit.get((e['b_value'], int(e['night'])), 0.0),
                e[key], e['b_value'])
               for e in ents if np.isfinite(e.get(key, np.nan))]
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=42, color=col,
                   marker=mk, edgecolors='black', linewidths=0.4, label=lab,
                   zorder=3, alpha=0.95)
        # Trend through the per-B-set median over nights.
        med = {}
        for _, y, bv in pts:
            med.setdefault(bv, []).append(y)
        bx = sorted(med)
        ax.plot(bx, [float(np.median(med[b])) for b in bx], ls=ls, color=col,
                lw=1.2, alpha=0.65, zorder=2)
    if annotate:
        for e in ents:
            v = e.get('fwhm_before', np.nan)
            if np.isfinite(v):
                ax.annotate(f"{int(e['night']) % 10000}",
                            (e['b_value'] + x_jit.get(
                                (e['b_value'], int(e['night'])), 0.0), v),
                            textcoords='offset points', xytext=(4, 4),
                            fontsize=6, color='#333333')
    # With a single B set the jitter span has no scale to work from, so the tick
    # labels spread over a fraction of a degree and invite reading the nights as
    # sitting at different B values.  Pin the axis to the B set and say so.
    if len(bvals) == 1:
        jmax = max((abs(v) for v in x_jit.values()), default=0.0)
        ax.set_xticks(list(bvals))
        ax.set_xlim(bvals[0] - 3.0 * max(jmax, 1e-3),
                    bvals[0] + 3.0 * max(jmax, 1e-3))
        x_label = f'{x_label} (nights offset in x for legibility only)'
    elif any(v != 0.0 for v in x_jit.values()):
        x_label = f'{x_label} (nights at one B set offset in x for legibility)'
    ax.set_xlabel(x_label)
    ax.set_ylabel('differential correctable FWHM [arcsec]\n(median over focal plane)')
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig


def plot_dof_per_night_summary(dof_by_night, dof_labels, dof_units, title=''):
    """4-panel (Hex translations / Hex rotations / M1M3 / M2) summary of the
    per-night median FAM DOF; each night a separate colour of dots.

    `dof_by_night` = {night: (n_visits, n_dof)}.  Same layout as
    build_measured_intrinsic's 'DOF median per iteration' page, with the
    iteration axis replaced by night.  `dof_units` is accepted for signature
    parity; per-panel units are fixed (μm / arcsec).
    """
    nights = sorted(dof_by_night.keys())
    medians = {nt: np.nanmedian(dof_by_night[nt], axis=0) for nt in nights}
    hex_trans_idx = [0, 1, 2, 5, 6, 7]      # M2 z/x/y, Cam z/x/y
    hex_rot_idx   = [3, 4, 8, 9]            # M2 rx/ry, Cam rx/ry
    m1m3_idx      = list(range(10, 30))
    m2_idx        = list(range(30, 50))
    n_nt = len(nights)

    fig, axes = plt.subplots(4, 1, figsize=(15, 14), layout='constrained',
                             gridspec_kw=dict(height_ratios=[1.0, 1.0, 1.5, 1.5]))
    colors = plt.get_cmap('viridis')(np.linspace(0.1, 0.9, max(n_nt, 1)))
    offsets = ((np.arange(n_nt) - (n_nt - 1) / 2) * (0.7 / n_nt)
               if n_nt > 1 else np.array([0.0]))

    def _panel(ax, idx_list, ttl, y_unit):
        x = np.arange(len(idx_list))
        for xi in range(len(idx_list)):
            if xi % 2:
                ax.axvspan(xi - 0.5, xi + 0.5, color='black', alpha=0.05)
        for ci, nt in enumerate(nights):
            ax.plot(x + offsets[ci], medians[nt][idx_list], 'o', ms=7,
                    color=colors[ci], label=str(nt))
        ax.axhline(0, color='gray', lw=0.5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([dof_labels[i] for i in idx_list],
                           rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(f'DOF Value ({y_unit})')
        ax.set_title(ttl)
        ax.grid(axis='y', alpha=0.3)

    _panel(axes[0], hex_trans_idx, 'Hexapod Translations', 'μm')
    _panel(axes[1], hex_rot_idx,   'Hexapod Rotations',   'arcsec')
    _panel(axes[2], m1m3_idx,      'M1M3 Bending Modes',   'μm')
    _panel(axes[3], m2_idx,        'M2 Bending Modes',     'μm')
    axes[0].legend(loc='upper right', fontsize=9, title='night')
    fig.suptitle(title, fontsize=13)
    return fig

