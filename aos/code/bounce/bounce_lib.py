#!/usr/bin/env python3
"""Bounce-test analysis helpers (verbatim port of study_bounce.ipynb cell 8).

FAM-triplet telescope-position bounce tests: time-ordered paired-difference Δ
(comparison − reference) per Double-Zernike (k, j), OFC v-mode, and physical
DOF, with robust (MAD) errors; plus the heatmap / vs-ordinal / night-scatter
plotters.  Driven by code/run_bounce.py.  RSP-only for the marker scheme and
DOF recovery (lsst.ts.ofc via ofc_svd)."""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from common.utils import alt_to_deg as _alt_to_deg  # noqa: E402

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


def ab_position_table(fit_table, b, elev_halfwidth_deg=None,
                      rot_halfwidth_deg=None, min_block=2):
    """Identify the A and B telescope positions of a bounce, per night.

    A bounce alternates between a reference position **A** (BLOCK-T720:
    elevation 70 deg, rotator 0 deg) and a comparison position **B**. This
    walks each night's visits in `seq_num` order, groups them into contiguous
    blocks of one classified (elevation, rotator) position, and reports the
    block's extent. A night may contain more than one distinct B position —
    20260713 throws to elevation 75 deg and then to 30 deg, each interleaved
    with its own elevation 70 deg reference block.

    Positions are classified onto the `intrinsics_lib` grid: elevation
    centers (30, 40, 50, 60, 70, 75) deg and rotator centers (-60 … 60) deg in
    15 deg steps. A visit further than the half-width from every center is
    reported as `unclassified`, which is what flags a contiguous block with no
    clear value for the B set.

    Parameters
    ----------
    fit_table : `astropy.table.Table`
        Fit table with `day_obs`, `seq_num`, `alt` (deg or rad),
        `rotator_angle` (deg), and optionally `band`.
    b : `dict`
        Bounce definition; only `program` is used, to select the visits.
    elev_halfwidth_deg : `float`, optional
        Elevation acceptance half-width in degrees. Defaults to the
        `intrinsics_lib` value (2.0 deg). Values for BLOCK-T720 / T724 sit
        within 0.3 deg of a center, so 2 deg is comfortable.
    rot_halfwidth_deg : `float`, optional
        Rotator acceptance half-width in degrees. Default 2.0 deg.
    min_block : `int`, optional
        A (night, position) set holding fewer than this many visits is still
        reported but marked `short`.

    Returns
    -------
    rows : `list` of `dict`
        One row per (day_obs, position) set, in (day_obs, seq_lo) order, with
        keys day_obs, position, elev_deg, rot_deg, seq_lo, seq_hi, n, n_blocks,
        band, alt_min_deg, alt_max_deg, rot_min_deg, rot_max_deg,
        unclassified, short, and `role` ('A' for the night's most-populated
        position, else 'B').

    Notes
    -----
    A bounce alternates A/B on consecutive visits, so the contiguous-block
    walk finds many single-visit blocks. Those are aggregated into one row per
    distinct (night, position); `n_blocks` records how many times the
    telescope returned to that position during the night.
    """
    hw_e = 2.0 if elev_halfwidth_deg is None else float(elev_halfwidth_deg)
    hw_r = 2.0 if rot_halfwidth_deg is None else float(rot_halfwidth_deg)
    rot_centers = np.arange(-60.0, 60.0 + 1e-9, 15.0)
    elev_centers = np.array([30.0, 40.0, 50.0, 60.0, 70.0, 75.0])

    def _snap(val, centers, hw):
        if not np.isfinite(val):
            return None
        d = np.abs(centers - val)
        i = int(np.argmin(d))
        return float(centers[i]) if d[i] <= hw else None

    pmask = bounce_program_mask(fit_table, b)
    if not np.any(pmask):
        return []
    ft = fit_table[pmask]
    dobs = np.asarray(ft['day_obs']).astype(int)
    snum = np.asarray(ft['seq_num']).astype(int)
    alt = _alt_to_deg(ft['alt'])
    rot = (np.asarray(ft['rotator_angle'], dtype=float)
           if 'rotator_angle' in ft.colnames else np.full(len(ft), np.nan))
    band = (np.asarray(ft['band']).astype(str) if 'band' in ft.colnames
            else np.full(len(ft), ''))

    rows = []
    for d in sorted(set(dobs.tolist())):
        sel = np.where(dobs == d)[0]
        sel = sel[np.argsort(snum[sel])]
        cur = None
        for i in sel:
            e = _snap(float(alt[i]), elev_centers, hw_e)
            r = _snap(float(rot[i]), rot_centers, hw_r)
            key = (e, r)
            if cur is not None and cur['key'] == key:
                cur['idx'].append(i)
                continue
            if cur is not None:
                rows.append(cur)
            cur = {'day_obs': d, 'key': key, 'idx': [i]}
        if cur is not None:
            rows.append(cur)

    # Aggregate the contiguous blocks into one row per (night, position): the
    # bounce alternates A/B every visit, so the raw blocks are the alternation
    # pattern, not the positions.
    agg = {}
    for blk in rows:
        e, r = blk['key']
        key = (blk['day_obs'], e, r)
        rec = agg.setdefault(key, {'idx': [], 'n_blocks': 0})
        rec['idx'].extend(blk['idx'])
        rec['n_blocks'] += 1

    out = []
    for (d, e, r), rec in agg.items():
        idx = np.array(sorted(rec['idx']))
        unclass = (e is None) or (r is None)
        pos = 'unclassified' if unclass else f'{int(e)}/{int(r)}'
        bands = sorted(set(band[idx].tolist()))
        out.append({
            'day_obs': d, 'position': pos,
            'elev_deg': e, 'rot_deg': r,
            'seq_lo': int(snum[idx].min()), 'seq_hi': int(snum[idx].max()),
            'n': int(len(idx)), 'n_blocks': int(rec['n_blocks']),
            'band': ','.join(b0 for b0 in bands if b0),
            'alt_min_deg': float(np.nanmin(alt[idx])),
            'alt_max_deg': float(np.nanmax(alt[idx])),
            'rot_min_deg': float(np.nanmin(rot[idx])),
            'rot_max_deg': float(np.nanmax(rot[idx])),
            'unclassified': bool(unclass),
            'short': bool(len(idx) < min_block),
        })
    out.sort(key=lambda row: (row['day_obs'], row['seq_lo']))

    # Role: per night, the position holding the most visits is the reference A.
    for d in sorted(set(row['day_obs'] for row in out)):
        nres = {}
        for row in out:
            if row['day_obs'] == d and not row['unclassified']:
                nres[row['position']] = nres.get(row['position'], 0) + row['n']
        a_pos = max(nres, key=nres.get) if nres else None
        for row in out:
            if row['day_obs'] == d:
                row['role'] = ('A' if row['position'] == a_pos
                               else ('?' if row['unclassified'] else 'B'))
    return out


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
    """Distinct day_obs nights where the reference and the comparison legs
    present that night have >= min_visits, with per-night run_bounce results.

    A night qualifies on the comparison legs it actually populates, not on
    every leg the bounce defines: a multi-leg bounce need not exercise every
    leg every night (e.g. the BLOCK-T720 elevation sweep throws to 40 deg in
    April/May and to 60 / 50 / 30 / 75 deg on individual July nights).  At
    least one leg must be populated, and every populated leg must clear
    min_visits; legs with no visits that night are simply absent from the
    per-night result.

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
        present = [c['comp_n'] for c in rb['comparisons'].values()
                   if c['comp_n'] > 0]
        ok = (rb['ref_n'] >= min_visits and bool(present)
              and all(n >= min_visits for n in present))
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
# The solver lives in smatrix/code/regularized_inversion/ (the
# `regularized_inversion` study, where the method is derived and its FWHM
# cost measured); this module only applies it to the bounce paired Δ.
#
# Why per pair and not once on the median: RBR is *nonlinear*, so it does
# not commute with the median over pairs, and inverting a single median
# wavefront would give no error bar.  Each (ref, comp) pair's Δ wavefront
# is inverted on its own and the same median / median-SEM reduction used
# everywhere else in this module is applied to the resulting DOF.  That
# keeps the RBR error definition identical to the default one and lets the
# nonlinearity show up in the scatter rather than being averaged away.
RBR_METHOD_NAME = 'Range-Bounded Recovery (RBR)'
RBR_DEFAULTS = {'kappa': 4.0, 'power': 3}


def rbr_module():
    """Import the RBR solver from the `regularized_inversion` study.

    Returns
    -------
    mod : `module`
        `smatrix/code/regularized_inversion/regularized_inversion.py`.

    Notes
    -----
    A cross-topic reach into `smatrix/`, done by path insert because the repo
    is not an installed package.  The solver is deliberately *not* copied
    here: the method is derived and validated in that study, and a second
    copy would be free to drift from it.
    """
    sm = Path(__file__).resolve().parents[3] / 'smatrix' / 'code'
    # Both levels: the study dir for the solver itself, and smatrix/code for
    # the bare-name `normalization_weights` the range vector imports.
    for p in (sm / 'regularized_inversion', sm):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import regularized_inversion as ri
    return ri


def rbr_dof_per_pair(W_all, pairs, svd, ranges, *, kappa=None, power=None):
    """Recover DOF from each pair's Δ wavefront with RBR.

    Parameters
    ----------
    W_all : `numpy.ndarray`, (n_visits, n_kj)
        Per-visit DZ wavefront in µm of wavefront, row-aligned to the
        fit_table the pairs index into (the `_W` from `project_dz_table`).
    pairs : `list` of `tuple`
        `(ref_row, comp_row)` index pairs.
    svd : `OFCSvd`
        The 50-DOF / 34-v-mode decomposition the default recovery uses.
    ranges : `array_like`, (n_dof,)
        Allowed range `r_j` per DOF, each in that DOF's own unit (µm for
        translations and bending-mode amplitudes, arcsec for rotations).
    kappa : `float`, optional
        Ratio `|d_j| / r_j` at which the penalty reaches unity weight.
        Defaults to `RBR_DEFAULTS['kappa']`.
    power : `int`, optional
        Penalty exponent; the penalty goes as the ratio to the `2 * power`.
        Defaults to `RBR_DEFAULTS['power']`.

    Returns
    -------
    D : `numpy.ndarray`, (n_pairs, n_dof)
        RBR-recovered DOF for each pair's Δ wavefront, in each DOF's own
        unit.  Rows whose Δ wavefront is not finite are NaN.

    Notes
    -----
    Each pair is inverted independently, so cost is one IRLS solve per pair
    (a few tens of iterations on a 126 x 50 system — negligible here).
    """
    ri = rbr_module()
    kap = float(RBR_DEFAULTS['kappa'] if kappa is None else kappa)
    pw = int(RBR_DEFAULTS['power'] if power is None else power)
    W = np.asarray(W_all, dtype=float)
    r = np.asarray(ranges, dtype=float)
    out = np.full((len(pairs), int(svd.V.shape[0])), np.nan)
    for i, (ref, comp) in enumerate(pairs):
        dW = W[comp] - W[ref]
        if not np.all(np.isfinite(dW)):
            continue
        out[i] = ri.invert_range_penalty(dW, svd, r, kappa=kap, power=pw)
    return out


def rbr_deltas(W_all, pairs, svd, ranges, *, kappa=None, power=None,
               keys=None):
    """Median / median-SEM RBR DOF Δ over pairs.

    Same return form and same error definition as `paired_deltas_matrix`, so
    an RBR Δ is directly comparable to the default recovery's Δ.

    Returns
    -------
    deltas : `dict`
        `{dof_index: {'delta', 'err', 'sig', 'n'}}`, `delta` in each DOF's own
        unit (µm or arcsec), `err` the SEM of the median over pairs.
    """
    if not pairs:
        return {}
    D = rbr_dof_per_pair(W_all, pairs, svd, ranges, kappa=kappa, power=power)
    ks = list(range(D.shape[1])) if keys is None else list(keys)
    out = {}
    for i in range(D.shape[1]):
        v = D[:, i]
        v = v[np.isfinite(v)]
        n = int(v.size)
        if n < 1:
            out[ks[i]] = {'delta': np.nan, 'err': np.nan, 'sig': np.nan, 'n': 0}
            continue
        delta = float(np.median(v))
        sigma_mad = 1.4826 * float(np.median(np.abs(v - delta)))
        err = 1.2533 * sigma_mad / np.sqrt(n)    # SEM of a median, as elsewhere
        out[ks[i]] = {'delta': delta, 'err': float(err),
                      'sig': (delta / err if err > 0 else np.nan), 'n': n}
    return out


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


def leg_b_value(label):
    """Numeric B-set position parsed out of a comparison-leg label, in deg.

    Leg labels are written `Elev=40` or `Rot=60`, i.e. the axis name and the
    nominal B position in degrees.  Returns the number so a leg can be used as
    an x-axis coordinate rather than only as a categorical label.

    Parameters
    ----------
    label : `str`
        Comparison-leg label, e.g. `'Elev=40'` or `'Rot=60'`.

    Returns
    -------
    value : `float`
        The B-set elevation or camera-rotator angle in deg, or NaN if the
        label does not carry a number.
    """
    import re
    m = re.search(r'(-?\d+(?:\.\d+)?)', str(label))
    return float(m.group(1)) if m else float('nan')


def leg_axis_name(bounce):
    """Which telescope axis a bounce throws along — 'Elevation' or
    'Rotator angle' — inferred from its comparison-leg labels.

    Parameters
    ----------
    bounce : `dict`
        One bounce config entry, with a `comparisons` list of labelled legs.

    Returns
    -------
    name : `str`
        Axis name for an x-axis label; `'B set'` if the labels are not
        recognized.  The unit is deg in every case.
    """
    labs = ' '.join(str(c.get('label', '')) for c in bounce.get('comparisons', []))
    low = labs.lower()
    if 'elev' in low or 'alt' in low:
        return 'Elevation'
    if 'rot' in low:
        return 'Rotator angle'
    return 'B set'


def leg_pointing(fit_table, mask, day_obs=None):
    """Median measured pointing of the visits selected by a mask.

    The configured leg windows give only nominal positions, so the actual
    telescope position is taken from the visits themselves.  Restricting to one
    `day_obs` gives that night's position on the leg, which is what a per-night
    row should carry.

    Parameters
    ----------
    fit_table : `astropy.table.Table`
        The DZ fit table, carrying `alt`, `rotator_angle`, `day_obs` and
        (optionally) `science_program`.
    mask : `array_like` of `bool`
        Selects the visits on one leg.
    day_obs : `int`, optional
        Restrict to this night.  `None` pools every night on the leg.

    Returns
    -------
    pointing : `dict`
        `elevation_deg` and `rot_angle_deg` as medians in deg, `n_visits` as an
        `int`, and `block` as the `science_program` string (or a `+`-joined list
        if the selection spans more than one, which should not happen).  The
        angles are NaN when the selection is empty.
    """
    m = np.asarray(mask, dtype=bool).copy()
    if day_obs is not None and 'day_obs' in fit_table.colnames:
        m &= np.asarray(fit_table['day_obs']).astype(int) == int(day_obs)
    out = {'elevation_deg': float('nan'), 'rot_angle_deg': float('nan'),
           'n_visits': int(m.sum()), 'block': ''}
    if not m.any():
        return out
    if 'alt' in fit_table.colnames:
        out['elevation_deg'] = float(np.median(_alt_to_deg(fit_table['alt'])[m]))
    if 'rotator_angle' in fit_table.colnames:
        out['rot_angle_deg'] = float(np.median(
            np.asarray(fit_table['rotator_angle'], dtype=float)[m]))
    if 'science_program' in fit_table.colnames:
        progs = sorted(set(np.asarray(fit_table['science_program']).astype(str)[m]))
        out['block'] = '+'.join(progs)
    return out


def leg_night_coverage(bounce_results):
    """Which nights back each (bounce, comparison leg), as table rows.

    Answers "which nights have results for each value of the B conditions" —
    the precondition for a night-vs-night comparison, which needs 2 or more
    nights on the same leg.

    Parameters
    ----------
    bounce_results : `dict`
        `{bounce_name: bounce_result}` as assembled by `run_bounce.py`, whose
        per-leg blocks carry `deltas_by_night` and `comp_n`.

    Returns
    -------
    rows : `list` [`dict`]
        One row per (bounce, leg) with keys `bounce`, `comparison`,
        `b_value_deg`, `n_visits`, `n_nights`, `nights`, `n_pairs` and
        `scatter_pages` — the latter the number of night pairs
        `n_nights * (n_nights - 1) / 2` that a night-vs-night scatter can
        draw, which is 0 when only one night is available.
    """
    rows = []
    for name, br in bounce_results.items():
        for clabel, cb in br['comparisons'].items():
            nts = sorted(int(d) for d in cb.get('deltas_by_night', {}))
            n_nt = len(nts)
            rows.append({
                'bounce': name, 'comparison': clabel,
                'b_value_deg': leg_b_value(clabel),
                'n_visits': int(cb.get('comp_n', 0)),
                'n_pairs': len(cb.get('pairs', [])),
                'n_nights': n_nt,
                'nights': ', '.join(str(d) for d in nts) if nts else '-',
                'scatter_pages': n_nt * (n_nt - 1) // 2,
            })
    return rows


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


def plot_dof_vs_b_value_panels(entries, dof_labels, dof_units,
                               dof_indices=None, x_label='Elevation [deg]',
                               title='', ncols=5, panel_size=(2.6, 2.1),
                               annotate=True, overlay_key=None,
                               overlay_label='RBR', ranges=None):
    """Small per-DOF panels of the paired-Δ DOF against the B-set position.

    One panel per degree of freedom (DOF); within a panel each point is one
    (night, B set) entry, plotted at its B-set position on the x axis with the
    paired-Δ error as a y error bar.  This is the form in which a Look-Up Table
    (LUT) reads the bounce: how each DOF's change grows with the throw.

    Points are drawn in increasing B-set order, so the BLOCK-T720 elevation
    sweep reads left to right in elevation and BLOCK-T724 left to right in
    camera-rotator angle.  Colour encodes the night; the optional annotation
    gives `day_obs` and the B value in deg, so a point is identifiable without
    the legend.

    Parameters
    ----------
    entries : `list` [`dict`]
        One entry per (night, comparison leg), each with keys `night`
        (`int` day_obs), `b_value` (`float`, deg), `label` (`str`, the leg
        label) and `dof_deltas` — the `{dof_index: {'delta','err'}}` dict from
        `paired_deltas_matrix`, in µm for translations and bending-mode
        amplitudes and arcsec for hexapod rotations.
    dof_labels : `list` [`str`]
        DOF names, indexed by global DOF index (`LABELS_50DOF`).
    dof_units : `list` [`str`]
        Per-DOF units, indexed the same way (`DOF_UNITS_50`) — µm or arcsec.
    dof_indices : `list` [`int`], optional
        Which DOF to panel.  Defaults to every DOF that any entry populates
        with a finite Δ, which keeps a camera-hexapod-only bounce to its five
        panels instead of drawing 45 empty ones.
    x_label : `str`, optional
        X-axis label, including the unit.
    title : `str`, optional
        Figure title.
    ncols : `int`, optional
        Panels per row.
    panel_size : `tuple` [`float`], optional
        Per-panel (width, height) in inches.
    annotate : `bool`, optional
        Annotate each point with `day_obs` and the B value in deg.
    overlay_key : `str`, optional
        Entry key holding a second `{dof_index: {'delta','err'}}` dict to
        overlay on the same panels, in the same units — used to show the
        Range-Bounded Recovery (RBR) Δ against the default recovery's.  When
        `None` only the default Δ is drawn, which is the original behaviour.
    overlay_label : `str`, optional
        Legend label for the overlay series.
    ranges : `array_like`, optional
        Allowed range `r_j` per DOF, indexed by global DOF index, in that
        DOF's own unit.  When given, each panel gets a shaded band at
        ±`r_j` so a point outside the physically reachable range is visible
        by eye.

    Returns
    -------
    fig : `matplotlib.figure.Figure` or `None`
        `None` if no entry carries a finite Δ.
    """
    ents = sorted(entries, key=lambda e: (e['b_value'], e['night']))
    if dof_indices is None:
        seen = set()
        for e in ents:
            for q, v in e['dof_deltas'].items():
                if np.isfinite(v.get('delta', np.nan)):
                    seen.add(int(q))
        dof_indices = sorted(seen)
    if not dof_indices or not ents:
        return None

    nights = sorted({int(e['night']) for e in ents})
    cmap = plt.get_cmap('viridis')(np.linspace(0.08, 0.88, max(len(nights), 1)))
    ncol = {nt: cmap[i] for i, nt in enumerate(nights)}

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

    nrows = int(np.ceil(len(dof_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, layout='constrained',
                             figsize=(panel_size[0] * ncols + 1.2,
                                      panel_size[1] * nrows + 1.2),
                             squeeze=False)
    for pi, q in enumerate(dof_indices):
        ax = axes[pi // ncols][pi % ncols]
        # Allowed-range band first, so the points draw over it.  Drawn with
        # the y limits frozen to the data afterwards, because the rigid-body
        # ranges (thousands of µm) are orders of magnitude wider than their Δ
        # and would otherwise set the scale and flatten the points to a line.
        band = None
        if ranges is not None and q < len(ranges) and np.isfinite(ranges[q]):
            band = float(ranges[q])
        xs, ys, es, cs, labs = [], [], [], [], []
        for e in ents:
            v = e['dof_deltas'].get(q)
            if v is None or not np.isfinite(v.get('delta', np.nan)):
                continue
            nt = int(e['night'])
            xs.append(e['b_value'] + x_jit.get((e['b_value'], nt), 0.0))
            ys.append(v['delta'])
            es.append(v.get('err', np.nan))
            cs.append(ncol[nt])
            labs.append(f"{nt % 10000}/{e['b_value']:g}")
        # Overlay series (RBR), same night colours, open square markers so the
        # two recoveries are distinguishable in greyscale as well as colour.
        if overlay_key is not None:
            oxs, oys, oes, ocs = [], [], [], []
            for e in ents:
                v = (e.get(overlay_key) or {}).get(q)
                if v is None or not np.isfinite(v.get('delta', np.nan)):
                    continue
                nt = int(e['night'])
                oxs.append(e['b_value'] + x_jit.get((e['b_value'], nt), 0.0))
                oys.append(v['delta'])
                oes.append(v.get('err', np.nan))
                ocs.append(ncol[nt])
            if oxs:
                ax.errorbar(oxs, oys, yerr=np.nan_to_num(np.asarray(oes, float)),
                            fmt='none', ecolor='#888888', elinewidth=0.8,
                            capsize=2, zorder=4)
                ax.scatter(oxs, oys, s=34, facecolors='none', edgecolors=ocs,
                           linewidths=1.3, marker='s', zorder=5)
        if xs:
            xs = np.asarray(xs, float); ys = np.asarray(ys, float)
            es = np.nan_to_num(np.asarray(es, float))
            ax.errorbar(xs, ys, yerr=es, fmt='none', ecolor='gray',
                        elinewidth=0.8, capsize=2, zorder=2)
            ax.scatter(xs, ys, s=30, c=cs, edgecolors='black',
                       linewidths=0.4, zorder=3)
            if annotate:
                # Alternate the label side so neighbouring points in a crowded
                # B set do not overwrite one another, and keep labels inside
                # the axes by flipping those near the right edge.
                xmid = 0.5 * (xs.min() + xs.max())
                for i, (x, y, l) in enumerate(zip(xs, ys, labs)):
                    right = x > xmid
                    ax.annotate(l, (x, y), textcoords='offset points',
                                xytext=(-4 if right else 4,
                                        4 if i % 2 == 0 else -8),
                                ha='right' if right else 'left',
                                fontsize=4.5, color='#333333')
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
                ax.text(0.02, 0.04, f'±r_j = {band:.3g}', transform=ax.transAxes,
                        fontsize=5, color='#9c3b57', va='bottom')
            else:
                ax.set_ylim(min(lo, -1.15 * band), max(hi, 1.15 * band))
        ax.set_title(f'{dof_labels[q]}  [{dof_units[q]}]', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        if pi // ncols == nrows - 1:
            ax.set_xlabel(x_label, fontsize=7)
    for pi in range(len(dof_indices), nrows * ncols):
        axes[pi // ncols][pi % ncols].axis('off')

    handles = [plt.Line2D([], [], marker='o', ls='', color=ncol[nt],
                          markeredgecolor='black', markeredgewidth=0.4,
                          label=str(nt)) for nt in nights]
    fig.legend(handles=handles, loc='outside lower center', ncol=min(len(nights), 8),
               fontsize=8, title='day_obs', title_fontsize=8, frameon=False)
    if overlay_key is not None or ranges is not None:
        style = []
        if overlay_key is not None:
            style += [
                plt.Line2D([], [], marker='o', ls='', color='0.35',
                           markeredgecolor='black', markeredgewidth=0.4,
                           label='default 50/34 (filled circle)'),
                plt.Line2D([], [], marker='s', ls='', markerfacecolor='none',
                           markeredgecolor='0.35', markeredgewidth=1.3,
                           label=f'{overlay_label} (open square)')]
        if ranges is not None:
            style.append(plt.Rectangle((0, 0), 1, 1, color='#f2c9d4', alpha=0.45,
                                       label='allowed range ±r_j'))
        fig.legend(handles=style, loc='outside upper right', fontsize=7.5,
                   frameon=False)
    fig.suptitle(title, fontsize=12)
    return fig


def plot_fwhm_vs_b_value(rows, x_label='Elevation [deg]', title='',
                         annotate=True, ax=None):
    """Correctable FWHM against the B-set position, one point per (night, leg).

    Three series per panel, all in arcsec FWHM (median over the focal plane):
    the uncorrected differential FWHM of the bounce optical-state change, the
    residual after the default truncated 50-DOF / 34-v-mode recovery, and the
    residual after Range-Bounded Recovery (RBR).  The vertical gap between the
    last two is the image-quality price of keeping the recovered DOF inside
    the range the telescope can actually apply.

    Nights sharing a B set are fanned out in x by a few percent of the B-set
    span so their points do not stack; the x axis is still the B-set position,
    so the offset is cosmetic.  The connecting line joins the per-B-set median
    over nights rather than the individual points.

    Parameters
    ----------
    rows : `list` [`dict`]
        One entry per (night, comparison leg), each with `night` (`int`
        day_obs), `b_value` (`float`, deg), `label` (`str`, leg label),
        `n_pairs` (`int`), and the three FWHM values in arcsec:
        `fwhm_before`, `fwhm_after_default`, `fwhm_after_rbr`.  A missing or
        non-finite value is skipped for that series only.
    x_label : `str`, optional
        X-axis label including the unit — elevation or camera-rotator angle
        in deg, whichever is the B-set axis of the bounce.
    title : `str`, optional
        Axes title.
    annotate : `bool`, optional
        Annotate the uncorrected points with `day_obs`.
    ax : `matplotlib.axes.Axes`, optional
        Axes to draw on; a new figure is made when omitted.

    Returns
    -------
    fig : `matplotlib.figure.Figure` or `None`
        The figure drawn on, or `None` if no row carries a finite FWHM.
    """
    ents = sorted(rows, key=lambda r: (r['b_value'], r['night']))
    series = [('fwhm_before', 'no correction', '#444444', 'o', '--'),
              ('fwhm_after_default', 'default 50/34', '#1f77b4', 'o', '-'),
              ('fwhm_after_rbr', 'RBR (range-bounded)', '#d62728', 's', '-')]
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

