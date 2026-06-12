#!/usr/bin/env python3
"""DZ-to-DZ correlation analysis (script port of study_doublezernike.ipynb
§7-§10), run on a per-(param_set, mi_name) DZ-fit table.

Operates on the measured-intrinsic refit  output/<ps>/<mi>/fits.parquet  (or any
DZ-fit parquet with z{prefix}_z{j}_c{k} columns), so the correlations are over
the MI-subtracted residual Double-Zernike coefficients.  Reuses the plotting
primitives in code/dz_plotting.py.

Writes, under  output/<ps>/<mi>/plots/ :
    dz_correlations.pdf            full Pearson heatmap + top-|r| pair scatters
                                   + targeted astigmatism-symmetry pairs
                                   (+ optional exhaustive (k1,j1)x(k2,j2) scan)
    dz_correlations_pairs.parquet  every off-diagonal pair with |r| >= threshold

Knobs come from analysis_config.yaml (section ``dz_correlations``); CLI overrides.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mi_config as mc
import dz_plotting as dzp

try:
    from common.zernike_names import FOCAL_NAMES, PUPIL_NAMES
except Exception:                                         # pragma: no cover
    FOCAL_NAMES, PUPIL_NAMES = {}, {}

DEFAULT = dict(dz_prefix='z1toz6', max_coeff_um=2.0, corr_threshold=0.6,
               top_n=20, pairwise_scan=False,
               sig_threshold=5.0,       # Fisher-z significance (σ) gate for groups
               group_min_r=0.5,         # |r| floor to seed a conjugate group
               max_group_pages=80,      # cap on conjugate-group pages
               expected_astig_pairs=[[[5, 5], [6, 6]], [[6, 5], [5, 6]],
                                     [[1, 5], [1, 6]], [[4, 5], [4, 6]],
                                     [[2, 5], [3, 6]], [[3, 5], [2, 6]]])

# Azimuthal doublet partners (same radial order, m -> -m); m=0 maps to itself.
# Used to build the conjugate of a correlation pair.
_J_PARTNER = {4: 4, 5: 6, 6: 5, 7: 8, 8: 7, 9: 10, 10: 9, 11: 11,
              12: 13, 13: 12, 14: 15, 15: 14, 16: 17, 17: 16, 18: 19, 19: 18,
              22: 22, 23: 24, 24: 23, 25: 26, 26: 25}
_K_PARTNER = {1: 1, 2: 3, 3: 2, 4: 4, 5: 6, 6: 5}


def _significance(r, n):
    """(SE(r), Fisher z, significance σ) for Pearson r over n samples.
    σ = |arctanh r| · sqrt(n-3) (Fisher z-transform); SE(r) ≈ (1-r²)/sqrt(n-1)."""
    if n is None or n < 4 or not np.isfinite(r):
        return (np.nan, np.nan, np.nan)
    rc = min(max(float(r), -0.999999), 0.999999)
    se_r = (1.0 - rc ** 2) / np.sqrt(n - 1)
    z = np.arctanh(rc)
    return se_r, z, abs(z) * np.sqrt(n - 3)


def _conj(kj):
    """Conjugate (k, j) endpoint: swap each index to its doublet partner."""
    k, j = kj
    return (_K_PARTNER.get(k, k), _J_PARTNER.get(j, j))


def _endpt_name(kj):
    k, j = kj
    return f'{FOCAL_NAMES.get(k, f"k{k}")} of {PUPIL_NAMES.get(j, f"Z{j}")} (k={k},j={j})'


def dz_coeff_columns(df, prefix):
    pat = re.compile(rf'^{re.escape(prefix)}_z\d+_c\d+$')
    return [c for c in df.columns if pat.match(c)]


def parse_jk(col, prefix):
    m = re.match(rf'^{re.escape(prefix)}_z(\d+)_c(\d+)$', col)
    return (int(m.group(1)), int(m.group(2))) if m else None


def quality_cut(df, prefix, max_coeff_um):
    n0 = len(df)
    for bc in (f'{prefix}_bad_fit', 'bad_fit'):
        if bc in df.columns:
            df = df[~df[bc].astype(bool)].copy()
            break
    cols = dz_coeff_columns(df, prefix)
    df = df[~df[cols].abs().gt(max_coeff_um).any(axis=1)].copy()
    print(f'  quality cut: {len(df)}/{n0} visits (|c| < {max_coeff_um} μm)')
    return df


def _scatter_pairs(df, pairs, prefix, title, pdf):
    """Targeted scatter grid for an explicit list of ((k1,j1),(k2,j2))."""
    import matplotlib.pyplot as plt
    present = [((k1, j1), (k2, j2)) for (k1, j1), (k2, j2) in pairs
               if dzp.dz_col_name(k1, j1, prefix) in df.columns
               and dzp.dz_col_name(k2, j2, prefix) in df.columns]
    if not present:
        return
    ncols = 2
    nrows = (len(present) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows),
                             layout='constrained', squeeze=False)
    axes = axes.ravel()
    for idx, ((k1, j1), (k2, j2)) in enumerate(present):
        ax = axes[idx]
        ca, cb = dzp.dz_col_name(k1, j1, prefix), dzp.dz_col_name(k2, j2, prefix)
        m = df[ca].notna() & df[cb].notna()
        x, y = df.loc[m, ca].values, df.loc[m, cb].values
        ax.scatter(x, y, s=12, alpha=0.6, edgecolors='none')
        if len(x) > 2:
            c = np.polyfit(x, y, 1); xf = np.linspace(x.min(), x.max(), 50)
            ax.plot(xf, np.polyval(c, xf), 'r-', lw=1.4, alpha=0.85)
            r = float(np.corrcoef(x, y)[0, 1])
            ax.set_title(f'(k={k1},j={j1}) vs (k={k2},j={j2})  r={r:+.3f}', fontsize=10)
        ax.set_xlabel(f'(k={k1},j={j1}) [μm]', fontsize=9)
        ax.set_ylabel(f'(k={k2},j={j2}) [μm]', fontsize=9)
        ax.tick_params(labelsize=8)
    for idx in range(len(present), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(title, fontsize=13)
    pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--analysis-config', default=None)
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--fits', default=None, help='override fits.parquet path')
    args = ap.parse_args()

    cfg = {**DEFAULT, **mc.analysis_section(
        'dz_correlations', args.param_set, args.mi_name,
        config_path=(Path(args.analysis_config) if args.analysis_config else None))}
    prefix = cfg['dz_prefix']

    base = Path(args.output_root) / args.param_set / args.mi_name
    fits_path = Path(args.fits) if args.fits else base / 'fits.parquet'
    out_dir = base / 'plots'; out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[dz_correlations] {fits_path}')

    df = pd.read_parquet(fits_path)
    df = quality_cut(df, prefix, cfg['max_coeff_um'])
    # DZ columns sorted by (pupil j, focal k) so same-j blocks are contiguous
    dz_cols = sorted(dz_coeff_columns(df, prefix),
                     key=lambda c: parse_jk(c, prefix)[::-1])
    labels = [f'({k},{j})' for c in dz_cols for (j, k) in [parse_jk(c, prefix)]]
    print(f'  {len(dz_cols)} DZ columns, {len(df)} visits')

    corr = dzp.compute_dz_correlation_matrix(df, dz_cols)
    all_pairs = dzp.get_top_correlated_pairs(corr, dz_cols, labels,
                                             top_n=len(dz_cols) ** 2)
    # complete-case sample count (compute_dz_correlation_matrix drops any row
    # with a NaN in any DZ column) — used for the correlation significance.
    n_complete = int((~np.isnan(df[dz_cols].values).any(axis=1)).sum())
    print(f'  complete-case n = {n_complete}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(str(out_dir / 'dz_correlations.pdf')) as pdf:
        dzp.plot_dz_correlation_heatmap(corr, labels, pdf=pdf, show=False)
        dzp.plot_dz_scatter_top_pairs(df, all_pairs[:int(cfg['top_n'])],
                                      pdf=pdf, show=False)
        _scatter_pairs(df, cfg['expected_astig_pairs'], prefix,
                       'Expected astigmatism-symmetry pairs', pdf)
        n_grp = _conjugate_group_pages(df, all_pairs, prefix, n_complete, cfg, pdf)
        if cfg.get('pairwise_scan'):
            _pairwise_scan(df, prefix, pdf)
    print(f'  {n_grp} conjugate-group page(s) '
          f'(|r|>={cfg["group_min_r"]}, σ>={cfg["sig_threshold"]})')

    # all off-diagonal pairs above |r| OR significance -> parquet, with errors
    thr = float(cfg['corr_threshold']); sig_thr = float(cfg['sig_threshold'])
    rows = []
    for ci, cj, li, lj, r in all_pairs:
        se_r, z, sig = _significance(r, n_complete)
        if abs(r) >= thr or (np.isfinite(sig) and sig >= sig_thr):
            rows.append(dict(col_i=ci, col_j=cj, label_i=li, label_j=lj,
                             r=r, n=n_complete, se_r=se_r, fisher_z=z, sigma=sig))
    pd.DataFrame(rows).to_parquet(out_dir / 'dz_correlations_pairs.parquet')
    print(f'  wrote dz_correlations.pdf + dz_correlations_pairs.parquet '
          f'({len(rows)} pairs: |r|>={thr} or σ>={sig_thr})')


def _conjugate_group_pages(df, all_pairs, prefix, n, cfg, pdf):
    """One page per significant correlation, showing it WITH its azimuthal
    conjugates: hold one endpoint fixed and/or flip the other's k/j doublet
    partner.  For (k3,j11)x(k3,j12) [Tilt_y Spherical x Tilt_y 2ndAstig] the page
    shows the original plus (k3,j11)x(k2,j13) [conj 2nd only], (k2,j11)x(k3,j12)
    [conj 1st only], and (k2,j11)x(k2,j13) [full conj] — so a correlation that
    survives only under a *partial* conjugation is visible (the full conjugate
    can wash it out).  Seeded by |r| >= group_min_r AND σ >= sig_threshold;
    deduped by the conjugation orbit so the 4 variants emit a single page."""
    rmin = float(cfg['group_min_r']); sig_thr = float(cfg['sig_threshold'])
    cap = int(cfg['max_group_pages'])
    seen, n_pages, n_seed = set(), 0, 0
    for ci, cj, li, lj, r in all_pairs:
        if abs(r) < rmin:
            continue
        _se, _z, sig = _significance(r, n)
        if not (np.isfinite(sig) and sig >= sig_thr):
            continue
        jA, kA = parse_jk(ci, prefix); jB, kB = parse_jk(cj, prefix)
        A, B = (kA, jA), (kB, jB)
        Ac, Bc = _conj(A), _conj(B)
        # original, conjugate-2nd-only, conjugate-1st-only, full conjugate
        variants = [(A, B), (A, Bc), (Ac, B), (Ac, Bc)]
        orbit = frozenset(frozenset((P, Q)) for P, Q in variants)
        if orbit in seen:
            continue
        seen.add(orbit); n_seed += 1
        if n_pages >= cap:
            continue
        pairs, seen_pair = [], set()
        for P, Q in variants:                          # dedup identical panels
            kpq = frozenset((P, Q))
            if P != Q and kpq not in seen_pair:
                seen_pair.add(kpq); pairs.append((P, Q))
        title = f'{_endpt_name(A)}  ×  {_endpt_name(B)}   + azimuthal conjugates'
        _scatter_pairs(df, pairs, prefix, title, pdf)
        n_pages += 1
    if n_seed > cap:
        print(f'  NOTE: {n_seed} conjugate groups found; capped at {cap} pages '
              f'(raise max_group_pages to see all)')
    return n_pages


def _pairwise_scan(df, prefix, pdf):
    """Exhaustive (k1,j1) x (k2,j2) scan — one page per (j1, j2), 6x6 k-grid.
    441 pages for 21 pupil j; off by default (pairwise_scan: true to enable)."""
    import matplotlib.pyplot as plt
    js = sorted({parse_jk(c, prefix)[0] for c in dz_coeff_columns(df, prefix)})
    ks = list(range(1, 7))
    print(f'  pairwise scan: {len(js)}x{len(js)} j-pairs ...')
    for j1 in js:
        for j2 in js:
            fig, axes = plt.subplots(len(ks), len(ks), figsize=(15, 15),
                                     layout='constrained', squeeze=False)
            for a, k1 in enumerate(ks):
                for b, k2 in enumerate(ks):
                    ax = axes[a][b]
                    ca, cb = dzp.dz_col_name(k1, j1, prefix), dzp.dz_col_name(k2, j2, prefix)
                    if ca not in df.columns or cb not in df.columns:
                        ax.set_visible(False); continue
                    m = df[ca].notna() & df[cb].notna()
                    x, y = df.loc[m, ca].values, df.loc[m, cb].values
                    ax.scatter(x, y, s=4, alpha=0.5, edgecolors='none')
                    if len(x) > 2:
                        r = float(np.corrcoef(x, y)[0, 1])
                        ax.set_title(f'k{k1}-k{k2} r={r:+.2f}', fontsize=6)
                    ax.tick_params(labelsize=4)
            fig.suptitle(f'pupil j1={j1} vs j2={j2}  (focal k1 x k2)', fontsize=12)
            pdf.savefig(fig, dpi=80, bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    main()
