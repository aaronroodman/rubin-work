#!/usr/bin/env python3
"""Bounce-test analysis helpers (verbatim port of study_bounce.ipynb cell 8).

FAM-triplet telescope-position bounce tests: time-ordered paired-difference Δ
(comparison − reference) per Double-Zernike (k, j), OFC v-mode, and physical
DOF, with robust (MAD) errors; plus the heatmap / vs-ordinal / night-scatter
plotters.  Driven by code/run_bounce.py.  RSP-only for the marker scheme and
DOF recovery (lsst.ts.ofc via ofc_svd)."""
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from lsst.ts.intrinsic.wavefront.intrinsics_lib import classify_visit, visit_marker_style
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
def _alt_to_deg(alt_arr):
    """Auto-detect radians vs degrees in `alt`."""
    a = np.asarray(alt_arr, dtype=float)
    if np.nanmax(np.abs(a)) < 2.0 * np.pi + 1e-3:
        return np.rad2deg(a)
    return a


def filter_visits(fit_table, *, alt_range=None, rotator_range=None,
                  day_obs_range=None, seq_num_range=None,
                  program=None, mask=None):
    """Build a boolean mask for visits matching every supplied criterion.

    All ranges are inclusive.  Missing keys are unrestricted.

    `program` matches the `science_program` column (case-sensitive
    exact match).  Pass a single string or a list/tuple of strings;
    the result is the union of all matches.  The fits parquet only
    started carrying `science_program` after the upstream
    `intrinsics_lib.merge_program_reason_to_visit_info` change — when
    the column is missing the program filter is silently skipped and a
    one-time warning is printed.
    """
    n = len(fit_table)
    keep = np.ones(n, dtype=bool)
    if alt_range is not None and 'alt' in fit_table.colnames:
        alt_deg = _alt_to_deg(fit_table['alt'])
        keep &= (alt_deg >= alt_range[0]) & (alt_deg <= alt_range[1])
    if rotator_range is not None and 'rotator_angle' in fit_table.colnames:
        rot = np.asarray(fit_table['rotator_angle'], dtype=float)
        keep &= (rot >= rotator_range[0]) & (rot <= rotator_range[1])
    if day_obs_range is not None and 'day_obs' in fit_table.colnames:
        d = np.asarray(fit_table['day_obs']).astype(int)
        keep &= (d >= int(day_obs_range[0])) & (d <= int(day_obs_range[1]))
    if seq_num_range is not None and 'seq_num' in fit_table.colnames:
        s = np.asarray(fit_table['seq_num']).astype(int)
        keep &= (s >= int(seq_num_range[0])) & (s <= int(seq_num_range[1]))
    if program is not None:
        if 'science_program' not in fit_table.colnames:
            if not getattr(filter_visits, '_warned_missing_program', False):
                print("  WARNING: 'science_program' not in fit_table — "
                      "the program filter is being ignored.  Re-run mktable "
                      "with the updated intrinsics_lib to populate it.")
                filter_visits._warned_missing_program = True
        else:
            sp = np.asarray(fit_table['science_program']).astype(str)
            if isinstance(program, (list, tuple, set)):
                prog_set = {str(p) for p in program}
                keep &= np.array([s in prog_set for s in sp])
            else:
                keep &= (sp == str(program))
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    return keep


def visit_list_str(fit_table, mask, max_per_day=20):
    """Pretty-print summary of which (day_obs, seq_num) visits survive a mask.

    Returns a string with one line per day_obs.  Truncates the seq_num
    list if it has more than `max_per_day` entries.
    """
    if not np.any(mask):
        return '    (no visits)'
    sub = fit_table[mask]
    dobs = np.asarray(sub['day_obs']).astype(int)
    snum = np.asarray(sub['seq_num']).astype(int)
    order = np.lexsort((snum, dobs))
    dobs = dobs[order]; snum = snum[order]
    lines = []
    for d in sorted(set(dobs.tolist())):
        s_list = sorted(snum[dobs == d].tolist())
        if len(s_list) > max_per_day:
            shown = (', '.join(str(x) for x in s_list[:max_per_day // 2])
                     + ', …, '
                     + ', '.join(str(x) for x in s_list[-max_per_day // 2:]))
            lines.append(f'    {d}: {len(s_list):3d} seq_num — '
                         f'[{shown}]')
        else:
            lines.append(f'    {d}: {len(s_list):3d} seq_num — '
                         f'{s_list}')
    return '\n'.join(lines)


def stats_per_kj(fit_table, mask, prefix, k_range, j_range):
    """Per-(k, j) median, robust RMS (1.4826*MAD), SEM of the median, n.

    SEM (standard error of the median) for normally distributed values
    is `1.2533 * sigma / sqrt(n)`; we use the MAD-based sigma estimate.
    """
    sub = fit_table[mask]
    out = {}
    for j in j_range:
        for k in k_range:
            col = f'{prefix}_z{j}_c{k}'
            if col not in sub.colnames:
                continue
            vals = np.asarray(sub[col], dtype=float)
            vals = vals[np.isfinite(vals)]
            n = int(len(vals))
            if n < 3:
                out[(int(k), int(j))] = {
                    'median': np.nan, 'sigma_mad': np.nan,
                    'sem': np.nan, 'n': n}
                continue
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            sigma_mad = 1.4826 * mad
            sem = 1.2533 * sigma_mad / np.sqrt(n)
            out[(int(k), int(j))] = {
                'median': med, 'sigma_mad': sigma_mad,
                'sem': sem, 'n': n}
    return out


def diff_stats(stats_comp, stats_ref):
    """Difference (comparison - reference) per (k, j) with quadrature errors."""
    out = {}
    keys = set(stats_comp.keys()) & set(stats_ref.keys())
    for kj in keys:
        a = stats_comp[kj]; b = stats_ref[kj]
        if not (np.isfinite(a['median']) and np.isfinite(b['median'])):
            out[kj] = {'delta': np.nan, 'err': np.nan,
                       'sig': np.nan,
                       'n_comp': a['n'], 'n_ref': b['n']}
            continue
        delta = a['median'] - b['median']
        err = float(np.sqrt(a['sem'] ** 2 + b['sem'] ** 2))
        sig = delta / err if err > 0 else np.nan
        out[kj] = {'delta': delta, 'err': err, 'sig': sig,
                   'n_comp': a['n'], 'n_ref': b['n']}
    return out


def _kj_to_array(stats_dict, k_list, j_list, key):
    """Pack one statistic into a (n_k, n_j) array for imshow."""
    Z = np.full((len(k_list), len(j_list)), np.nan)
    for (k, j), s in stats_dict.items():
        if k in k_list and j in j_list:
            Z[k_list.index(k), j_list.index(j)] = s.get(key, np.nan)
    return Z


def plot_kj_heatmap(stats, k_list, j_list, *, value_key='delta',
                    err_key='err', title='', cbar_label='',
                    cmap='RdBu_r', vlim=None, value_fmt='{:+.2f}',
                    err_fmt='±{:.2f}', cell_fontsize=7,
                    show_text=True):
    """Heatmap with k on rows, j on columns, value in colour, and
    optionally `value\n±err` in each cell.

    Returns the figure.
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


def to_long_df(stats, bounce_name, ref_label, comp_label,
                ref_stats, comp_stats, night='all'):
    """Long-format DataFrame for one bounce comparison."""
    rows = []
    for (k, j), d in sorted(stats.items()):
        r = ref_stats.get((k, j), {})
        c = comp_stats.get((k, j), {})
        rows.append({
            'bounce':      bounce_name,
            'night':       str(night),
            'reference':   ref_label,
            'comparison':  comp_label,
            'k': int(k), 'j': int(j),
            'n_ref':       int(r.get('n', 0)),
            'ref_median':  float(r.get('median', np.nan)),
            'ref_sigma':   float(r.get('sigma_mad', np.nan)),
            'ref_sem':     float(r.get('sem', np.nan)),
            'n_comp':      int(c.get('n', 0)),
            'comp_median': float(c.get('median', np.nan)),
            'comp_sigma':  float(c.get('sigma_mad', np.nan)),
            'comp_sem':    float(c.get('sem', np.nan)),
            'delta':       float(d.get('delta', np.nan)),
            'delta_err':   float(d.get('err', np.nan)),
            'significance': float(d.get('sig', np.nan)),
        })
    return pd.DataFrame(rows)


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

def bounce_program_mask(fit_table, b):
    """Boolean mask for all visits in a bounce's science_program(s)."""
    p = b.get('program')
    if isinstance(p, (list, tuple, set)):
        progs = [str(x) for x in p]
    elif p is not None:
        progs = [str(p)]
    else:
        return np.ones(len(fit_table), dtype=bool)
    return filter_visits(fit_table, program=progs)


def run_bounce(fit_table, b, prefix, k_list, j_list, day_obs=None,
               trim_segment=None):
    """Reference/comparison per-(k, j) stats and *paired* Δ for one bounce.

    The Δ is computed with the paired-difference method (see
    `form_pairs` / `paired_delta`): reference and comparison visits are
    paired in time order (never across a day_obs boundary), Δ is the
    median of the per-pair (comp − ref) differences, and the error is
    the scaled-MAD robust RMS of those differences / √n_pairs.  This
    absorbs slow time variation without modelling it.  `stats_per_kj`
    is still computed per setting for descriptive medians/RMS in the
    long table.  If `day_obs` is given, restrict to that single night.

    Returns {'ref_stats','ref_n','ref_mask',
             'comparisons': {label: {'comp_stats','comp_n','deltas',
                                     'comp_mask','pairs'}}}.
    """
    program = b.get('program')
    ref = b['reference']
    extra = {} if day_obs is None else {'day_obs_range': (int(day_obs), int(day_obs))}

    ref_kwargs = {k: v for k, v in ref.items() if k != 'label'}
    ref_kwargs.setdefault('program', program)
    ref_kwargs.update(extra)
    ref_mask = filter_visits(fit_table, **ref_kwargs)
    ref_stats = stats_per_kj(fit_table, ref_mask, prefix, k_list, j_list)

    comps = {}
    for comp in b['comparisons']:
        ck = {k: v for k, v in comp.items() if k != 'label'}
        ck.setdefault('program', program)
        ck.update(extra)
        cm = filter_visits(fit_table, **ck)
        cs = stats_per_kj(fit_table, cm, prefix, k_list, j_list)
        pairs = form_pairs(fit_table, ref_mask, cm, segment=trim_segment)
        comps[comp['label']] = {
            'comp_stats': cs, 'comp_n': int(cm.sum()),
            'deltas': paired_deltas_kj(fit_table, prefix, k_list, j_list, pairs),
            'comp_mask': cm, 'pairs': pairs,
        }
    return {'ref_stats': ref_stats, 'ref_n': int(ref_mask.sum()),
            'ref_mask': ref_mask, 'comparisons': comps}

def bounce_nights(fit_table, b, prefix, k_list, j_list, min_visits=3,
                  trim_segment=None):
    """Distinct day_obs nights where both ref and (each) comparison have
    >= min_visits, with per-night run_bounce results.

    Returns {night: run_bounce_result} for qualifying nights (sorted).
    """
    program = b.get('program')
    pmask = filter_visits(fit_table, program=program)
    if not np.any(pmask):
        return {}
    nights = sorted(set(np.asarray(fit_table['day_obs'])[pmask]
                        .astype(int).tolist()))
    out = {}
    for d in nights:
        rb = run_bounce(fit_table, b, prefix, k_list, j_list, day_obs=d,
                        trim_segment=trim_segment)
        ok = rb['ref_n'] >= min_visits and all(
            c['comp_n'] >= min_visits for c in rb['comparisons'].values())
        if ok:
            out[d] = rb
    return out


def diff_of_deltas(deltas_a, deltas_b):
    """Difference of two Δ-DZ dicts (night A − night B) per (k, j), with
    quadrature-combined errors and significance."""
    out = {}
    for kj in set(deltas_a) & set(deltas_b):
        a, b = deltas_a[kj], deltas_b[kj]
        if not (np.isfinite(a.get('delta', np.nan))
                and np.isfinite(b.get('delta', np.nan))):
            out[kj] = {'delta': np.nan, 'err': np.nan, 'sig': np.nan}
            continue
        d = a['delta'] - b['delta']
        e = float(np.sqrt(a.get('err', np.nan) ** 2 + b.get('err', np.nan) ** 2))
        out[kj] = {'delta': d, 'err': e,
                   'sig': (d / e if e > 0 else np.nan)}
    return out


def plot_dz_vs_ordinal_pages(fit_table, prefix, k_list, j_list,
                             j_per_page=7, title_prefix=''):
    """Pages of DZ_kj vs ordinal image number (rows = focal k, cols =
    pupil j).  Points use the standard intrinsics_lib marker scheme
    (elevation -> colour, rotator angle -> arrow); dotted vertical lines
    mark day_obs changes.  Returns a list of figures.
    """
    dobs = np.asarray(fit_table['day_obs']).astype(int)
    snum = np.asarray(fit_table['seq_num']).astype(int)
    order = np.lexsort((snum, dobs))
    ft = fit_table[order]
    dobs = dobs[order]
    n = len(ft)
    ordinal = np.arange(n)
    alt = (_alt_to_deg(ft['alt']) if 'alt' in ft.colnames
           else np.full(n, np.nan))
    rot = (np.asarray(ft['rotator_angle'], dtype=float)
           if 'rotator_angle' in ft.colnames else np.full(n, np.nan))
    has_band = 'band' in ft.colnames
    band = np.asarray(ft['band']).astype(str) if has_band else None

    styles = []
    for i in range(n):
        if _marker_ok:
            cls = classify_visit(alt_deg=alt[i], rot_deg=rot[i],
                                 band=(band[i] if has_band else None))
            styles.append(visit_marker_style(elev=cls['elev'], rot=cls['rot'],
                                             band=cls['band'], base_size=4))
        else:
            styles.append(dict(marker='o', color='steelblue',
                               markersize=3, linestyle=''))

    changes = [i for i in range(1, n) if dobs[i] != dobs[i - 1]]
    day_labels = [(i, int(dobs[i])) for i in [0] + changes]

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
                        ax.plot(ordinal[i], y[i], **styles[i])
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

def passing_terms(deltas, delta_th, nsigma_th, sigma_only_th=None):
    """Set of (k, j) passing the significance cut.

    A term passes if  (|Δ| > delta_th AND |Δ/σ| > nsigma_th)  OR
    (sigma_only_th is not None AND |Δ/σ| > sigma_only_th).
    """
    out = set()
    for (k, j), d in deltas.items():
        dl = d.get('delta', np.nan); sg = d.get('sig', np.nan)
        if not (np.isfinite(dl) and np.isfinite(sg)):
            continue
        cutA = abs(dl) > delta_th and abs(sg) > nsigma_th
        cutB = sigma_only_th is not None and abs(sg) > sigma_only_th
        if cutA or cutB:
            out.add((k, j))
    return out


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
def form_pairs(fit_table, ref_mask, comp_mask, segment=None):
    """Time-ordered, non-overlapping (reference, comparison) visit pairs.

    Walks all reference+comparison visits in (day_obs, seq_num) order
    and greedily pairs each visit with the nearest following visit of
    the opposite setting, never crossing a day_obs boundary.  A run of
    same-setting visits keeps only the most recent unpaired one.

    `segment` (optional, aligned to `fit_table` rows) adds a second
    no-cross boundary: a pair is only formed when both visits share the
    same segment label.  Used to avoid pairing across an AOS Trim
    re-alignment (a change in the degreeOfFreedom event id) within a
    night.  Returns (ref_row, comp_row) integer row indices.
    """
    n = len(fit_table)
    setting = np.zeros(n, dtype=int)
    setting[np.asarray(ref_mask, bool)] = -1
    setting[np.asarray(comp_mask, bool)] = 1          # comp wins any overlap
    active = np.nonzero(setting != 0)[0]
    if active.size == 0:
        return []
    dobs = np.asarray(fit_table['day_obs']).astype(int)
    snum = np.asarray(fit_table['seq_num']).astype(int)
    seg = (np.asarray(segment) if segment is not None
           else np.zeros(n, dtype=int))
    order = active[np.lexsort((snum[active], dobs[active]))]
    pairs = []
    pend = None
    for i in order:
        if (pend is not None and dobs[i] == dobs[pend]
                and seg[i] == seg[pend]
                and setting[i] == -setting[pend]):
            r = pend if setting[pend] == -1 else i
            c = i if setting[i] == 1 else pend
            pairs.append((int(r), int(c)))
            pend = None
        else:
            pend = i
    return pairs


def paired_delta(values, pairs):
    """Paired-difference Δ for one quantity over (ref, comp) pairs.

    Δ = median(comp − ref) over pairs; err = (1.4826·MAD of the per-pair
    differences) / √n_pairs; sig = Δ / err.  Returns
    {delta, err, sig, n}  (n = number of finite pairs).
    """
    v = np.asarray(values, dtype=float)
    if not pairs:
        return {'delta': np.nan, 'err': np.nan, 'sig': np.nan, 'n': 0}
    diffs = np.array([v[c] - v[r] for (r, c) in pairs], dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = int(diffs.size)
    if n < 1:
        return {'delta': np.nan, 'err': np.nan, 'sig': np.nan, 'n': 0}
    delta = float(np.median(diffs))
    sigma_mad = 1.4826 * float(np.median(np.abs(diffs - delta)))
    err = 1.2533 * sigma_mad / np.sqrt(n)        # SEM of a *median* (not mean)
    sig = delta / err if err > 0 else np.nan
    return {'delta': delta, 'err': float(err), 'sig': sig, 'n': n}


def paired_deltas_kj(fit_table, prefix, k_list, j_list, pairs):
    """Paired Δ per (k, j) for the DZ coefficients.  {(k, j): {...}}."""
    out = {}
    for j in j_list:
        for k in k_list:
            col = f'{prefix}_z{j}_c{k}'
            if col not in fit_table.colnames:
                continue
            out[(int(k), int(j))] = paired_delta(
                np.asarray(fit_table[col], dtype=float), pairs)
    return out


def paired_deltas_matrix(value_matrix, pairs, keys=None):
    """Paired Δ for each column of `value_matrix` (n_visits, n_q),
    aligned row-for-row to the fit_table the pairs index into.
    Returns {key: {delta,err,sig,n}} with key = keys[i] or int i."""
    M = np.asarray(value_matrix, dtype=float)
    nq = M.shape[1]
    ks = list(range(nq)) if keys is None else list(keys)
    return {ks[i]: paired_delta(M[:, i], pairs) for i in range(nq)}


# OFC v-mode / DOF recovery (LABELS_50DOF, DOF_UNITS_50,
# recover_dof_per_visit) now live in code/ofc_svd.py — imported above.


# ==================================================================
# Generic <quantity> vs ordinal-image plotting (marker scheme shared)
# ==================================================================
def _ordinal_setup(fit_table, base_size=4):
    """Time-sort a fit_table and build the shared per-visit marker
    styles, day_obs change indices and day labels for vs-ordinal plots.
    Returns a dict with order/ft/n/ordinal/styles/changes/day_labels."""
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
    styles = []
    for i in range(n):
        if _marker_ok:
            cls = classify_visit(alt_deg=alt[i], rot_deg=rot[i],
                                 band=(band[i] if has_band else None))
            styles.append(visit_marker_style(
                elev=cls['elev'], rot=cls['rot'], band=cls['band'],
                base_size=base_size))
        else:
            styles.append(dict(marker='o', color='steelblue',
                               markersize=3, linestyle=''))
    changes = [i for i in range(1, n) if dobs[i] != dobs[i - 1]]
    day_labels = [(i, int(dobs[i])) for i in [0] + changes]
    return {'order': order, 'ft': ft, 'n': n, 'ordinal': np.arange(n),
            'styles': styles, 'changes': changes, 'day_labels': day_labels}


def plot_values_vs_ordinal_pages(fit_table, value_matrix, labels,
                                 units=None, title_root='', ncols=5,
                                 rows_per_page=7):
    """Pages of <quantity> vs ordinal image number, one panel per column
    of `value_matrix` (aligned row-for-row to `fit_table`).  Standard
    marker scheme; dotted day_obs lines; day_obs annotated on the first
    panel.  Returns a list of figures.
    """
    s = _ordinal_setup(fit_table)
    order, ordinal, styles = s['order'], s['ordinal'], s['styles']
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
                    ax.plot(ordinal[i], y[i], **styles[i])
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

