#!/usr/bin/env python3
"""DZ-to-DZ correlation analysis (script port of study_doublezernike.ipynb
§7-§10), run on a per-(param_set, mi_name) DZ-fit table.

Operates on the measured-intrinsic refit  output/<ps>/<mi>/fits.parquet  (or any
DZ-fit parquet with z{prefix}_z{j}_c{k} columns), so the correlations are over
the MI-subtracted residual Double-Zernike coefficients.  Reuses the plotting
primitives in code/dz_plotting.py.

The conjugate-group pages show, per significant correlation, the full
conjugation orbit as a grid of scatter panels (rows/cols = the independent
focal-k / pupil-j doublet-flips of each endpoint, up to 4x4).

A single run emits BOTH the raw analysis and an optical-corrected one from a
shared (raw) pair/group selection, so the two PDFs are page-for-page
comparable.  The corrected analysis uses the DZ that *remains* after the
n_dof/n_keep OFC correction, W_resid = (I - U_eff U_effᵀ)·W (SVD from the
mi_config entry; RSP-only, needs lsst.ts.ofc).  In the optcorr PDF every scatter
panel OVERLAYS the corrected points + fit (orange/red) on the raw points + fit
(blue/navy) at the raw axis limits — so the shrink in spread and change in slope
read off directly.  ``--no-optcorr`` skips it (raw only, off-RSP).

The raw PDF also carries the SVD-space companions to the DZ-DZ heatmap: the
DOF x DOF and v-mode x v-mode correlation matrices (across visits) plus the
top-|r| v-mode-pair scatters, from the same n_dof/n_keep OFC SVD (RSP-only).

Writes, under  output/<ps>/<mi>/plots/ :
    dz_correlations.pdf                  raw: DZ Pearson heatmap + DOF/v-mode
                                         correlation matrices + top-|r| scatters
                                         + astig pairs + conjugate-orbit grids
    dz_correlations_pairs.parquet        pairs above |r| or σ (n, se_r, fisher_z,
                                         sigma, rms_i, rms_j, cov)
    dz_correlations_optcorr.pdf          same panel set, corrected overlaid on raw,
                                         + amplitude-weighted covariance heatmap
    dz_correlations_optcorr_pairs.parquet  + r_corr / rms_*_corr / cov_corr columns

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
               mode_top_n=12,           # top v-mode-pair scatters (SVD-space pages)
               mode_annot_r=0.5,        # |r| above which DOF/v-mode cells are labeled
               scatter_dpi=80,          # raster dpi for the (many) scatter/orbit pages
               expected_astig_pairs=[[[5, 5], [6, 6]], [[6, 5], [5, 6]],
                                     [[1, 5], [1, 6]], [[4, 5], [4, 6]],
                                     [[2, 5], [3, 6]], [[3, 5], [2, 6]]])

# Azimuthal doublet partners (same radial order, m -> -m); m=0 maps to itself.
# Used to build the conjugate of a correlation pair.
_J_PARTNER = {4: 4, 5: 6, 6: 5, 7: 8, 8: 7, 9: 10, 10: 9, 11: 11,
              12: 13, 13: 12, 14: 15, 15: 14, 16: 17, 17: 16, 18: 19, 19: 18,
              22: 22, 23: 24, 24: 23, 25: 26, 26: 25}
_K_PARTNER = {1: 1, 2: 3, 3: 2, 4: 4, 5: 6, 6: 5}

# Raster dpi for the many scatter/orbit-grid pages; set from cfg in main().
# Only the rasterized scatter clouds scale with this — vector axes/text stay
# sharp, and the heatmaps keep their own (higher) dpi.
_DPI = 80


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


def _index_flips(kj):
    """All endpoints reachable by independently flipping the focal-k and/or
    pupil-j doublet partner (1, 2, or 4 distinct (k, j); m=0 indices are fixed).
    The conjugation 'family' of an endpoint."""
    k, j = kj
    return sorted({(kk, jj) for kk in {k, _K_PARTNER.get(k, k)}
                   for jj in {j, _J_PARTNER.get(j, j)}})


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


def _panel(ax, df1, df2, ca, cb):
    """Scatter + OLS line for df1 (raw, blue) and — if given — df2 (corrected,
    orange) on df1's ORIGINAL axis limits, so the corrected cloud visibly
    shrinks against the raw spread.  Returns a compact 'raw r=.. | corr r=..'
    annotation (with per-axis RMS)."""
    def _xy(df):
        m = df[ca].notna() & df[cb].notna()
        return df.loc[m, ca].values, df.loc[m, cb].values
    parts = []
    x1, y1 = _xy(df1)
    if len(x1) > 2:
        ax.scatter(x1, y1, s=7, alpha=0.35, color='steelblue',
                   edgecolors='none', rasterized=True)
        c1 = np.polyfit(x1, y1, 1)
        ax.set_xlim(float(x1.min()), float(x1.max()))
        ax.set_ylim(float(y1.min()), float(y1.max()))
        xf = np.array(ax.get_xlim())
        ax.plot(xf, np.polyval(c1, xf), '-', color='navy', lw=1.3, alpha=0.9)
        parts.append(f'raw r={float(np.corrcoef(x1, y1)[0, 1]):+.2f} '
                     f'({x1.std():.2g}×{y1.std():.2g})')
    if df2 is not None:
        x2, y2 = _xy(df2)
        if len(x2) > 2:
            ax.scatter(x2, y2, s=7, alpha=0.55, color='darkorange',
                       edgecolors='none', rasterized=True)
            c2 = np.polyfit(x2, y2, 1)
            xf = np.array(ax.get_xlim())        # corrected line across raw limits
            ax.plot(xf, np.polyval(c2, xf), '-', color='red', lw=1.3, alpha=0.9)
            parts.append(f'corr r={float(np.corrcoef(x2, y2)[0, 1]):+.2f} '
                         f'({x2.std():.2g}×{y2.std():.2g})')
    return '   '.join(parts)


def _scatter_pairs(df1, df2, pairs, prefix, title, pdf):
    """Scatter grid for an explicit list of ((k1,j1),(k2,j2)); overlays df2
    (corrected) on df1 (raw) when df2 is given."""
    import matplotlib.pyplot as plt
    present = [((k1, j1), (k2, j2)) for (k1, j1), (k2, j2) in pairs
               if dzp.dz_col_name(k1, j1, prefix) in df1.columns
               and dzp.dz_col_name(k2, j2, prefix) in df1.columns]
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
        info = _panel(ax, df1, df2, ca, cb)
        ax.set_title(f'(k{k1},j{j1}) vs (k{k2},j{j2})\n{info}', fontsize=8)
        ax.set_xlabel(f'(k{k1},j{j1}) [μm]', fontsize=9)
        ax.set_ylabel(f'(k{k2},j{j2}) [μm]', fontsize=9)
        ax.tick_params(labelsize=8)
    for idx in range(len(present), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(title, fontsize=13)
    pdf.savefig(fig, dpi=_DPI, bbox_inches='tight'); plt.close(fig)


def _top_pairs_page(df1, df2, top_pairs, prefix, pdf):
    """Top-|r| correlated pairs as an overlay scatter grid (replaces
    dzp.plot_dz_scatter_top_pairs so the corrected data can be overlaid)."""
    pairs = []
    for ci, cj, li, lj, r in top_pairs:
        ji, ki = parse_jk(ci, prefix); jj, kj = parse_jk(cj, prefix)
        pairs.append(((ki, ji), (kj, jj)))
    _scatter_pairs(df1, df2, pairs, prefix, 'Top correlated DZ pairs (raw)', pdf)


def _dz_to_W(df, prefix, svd):
    """Raw DZ matrix W (n_visits, n_kj) in svd.kj_grid order from df columns."""
    W = np.full((len(df), len(svd.kj_grid)), np.nan)
    for ci, (k, j) in enumerate(svd.kj_grid):
        col = dzp.dz_col_name(k, j, prefix)
        if col in df.columns:
            W[:, ci] = np.asarray(df[col], dtype=float)
    return W


def _build_svd(df, prefix, param_set, mi_name, mi_config_path, n_dof_ov, n_keep_ov):
    """Build the OFC sensitivity-matrix SVD (n_dof / n_keep from the mi_config
    entry, or CLI overrides).  Imports lsst.ts.ofc lazily (RSP-only)."""
    import ofc_svd as osv
    cfg_mi = mc.load_mi_config(param_set, mi_name, config_path=mi_config_path)
    b = cfg_mi['build']
    n_dof = n_dof_ov if n_dof_ov is not None else cfg_mi.get('n_dof')
    n_keep = n_keep_ov if n_keep_ov is not None else cfg_mi['n_keep']
    k_min, k_max = int(b['k_min']), int(b['k_max'])
    iZs = sorted({parse_jk(c, prefix)[0] for c in dz_coeff_columns(df, prefix)})
    return osv.build_ofc_svd(iZs, k_min, k_max, n_keep, n_dof=n_dof,
                             ofc_normalization_yaml=b.get('ofc_normalization_yaml'))


def _project_modes(df, prefix, svd):
    """Per-visit physical-DOF (n_dof) and v-mode (n_keep) amplitudes; returns
    dict(dof, vmodes, dof_labels, vmode_labels)."""
    A = svd.project_amplitudes(_dz_to_W(df, prefix, svd))
    dof_labels, _ = svd.dof_labels()
    return dict(dof=svd.dof(A), vmodes=svd.vmodes(A),
                dof_labels=list(dof_labels),
                vmode_labels=[f'v{i + 1}' for i in range(svd.n_keep_eff)])


def _apply_optical_correction(df, prefix, svd):
    """Replace the (k,j) columns the SVD spans with the residual
    W_resid = (I - U_eff U_effᵀ)·W — the DZ that remains after the n_dof/n_keep
    optical correction.  Columns outside the focal-k × pupil-j slice are kept."""
    W = _dz_to_W(df, prefix, svd)
    A = np.where(np.isfinite(W), W, 0.0) @ svd.U_eff      # u-mode amps
    W_resid = W - A @ svd.U_eff.T                          # (I - U Uᵀ) W
    out = df.copy()
    for ci, (k, j) in enumerate(svd.kj_grid):
        col = dzp.dz_col_name(k, j, prefix)
        if col in out.columns:
            out[col] = W_resid[:, ci]                      # NaN where W was NaN
    removed = np.nansum((A @ svd.U_eff.T) ** 2) / max(np.nansum(W ** 2), 1e-30)
    print(f'  optical correction: n_dof={svd.n_dof}, n_keep={svd.n_keep_eff}, '
          f'{len(svd.kj_grid)} (k,j); removed ~{removed:.1%} of DZ power')
    return out


def _selection(df, prefix):
    """Shared selection from the (raw) DZ fit table: sorted DZ columns, (k,j)
    labels, Pearson matrix, all off-diagonal pairs (|r|-sorted), complete-case n."""
    dz_cols = sorted(dz_coeff_columns(df, prefix),     # by (pupil j, focal k)
                     key=lambda c: parse_jk(c, prefix)[::-1])
    labels = [f'({k},{j})' for c in dz_cols for (j, k) in [parse_jk(c, prefix)]]
    corr = dzp.compute_dz_correlation_matrix(df, dz_cols)
    all_pairs = dzp.get_top_correlated_pairs(corr, dz_cols, labels,
                                             top_n=len(dz_cols) ** 2)
    n = int((~np.isnan(df[dz_cols].values).any(axis=1)).sum())
    return dz_cols, labels, all_pairs, n


def _write_pdf(path, df1, df2, dz_cols, labels, all_pairs, n, cfg, prefix,
               modes=None):
    """Write one correlation PDF.  Panels overlay df2 (corrected) on df1 (raw)
    when df2 is given; the panel SET is fixed by ``all_pairs`` (raw selection)
    so the raw and optcorr PDFs are page-for-page comparable.  The heatmap(s)
    reflect this PDF's primary data (df2 if given, else df1).  When ``modes`` is
    given (raw PDF), the DOF / v-mode correlation pages are appended."""
    prim = df2 if df2 is not None else df1
    corr = dzp.compute_dz_correlation_matrix(prim, dz_cols)
    top = all_pairs[:int(cfg['top_n'])]
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(str(path)) as pdf:
        dzp.plot_dz_correlation_heatmap(corr, labels, pdf=pdf, show=False)
        if df2 is not None:                            # amplitude-weighted view
            std = {c: float(np.nanstd(prim[c].values)) for c in dz_cols}
            _cov_heatmap(corr, dz_cols, labels, std, pdf)
        if modes is not None:                          # SVD-space companions
            _mode_pages(modes, cfg, pdf)
        _top_pairs_page(df1, df2, top, prefix, pdf)
        _scatter_pairs(df1, df2, cfg['expected_astig_pairs'], prefix,
                       'Expected astigmatism-symmetry pairs', pdf)
        n_grp = _conjugate_group_pages(df1, df2, all_pairs, prefix, n, cfg, pdf)
        if cfg.get('pairwise_scan') and df2 is None:   # exhaustive scan: raw only
            _pairwise_scan(df1, prefix, pdf)
    return n_grp


def _write_pairs_parquet(path, df1, df2, all_pairs, n, cfg, prefix):
    """Significant-pairs table.  Raw columns always; when df2 is given, append
    the corrected r / rms / cov for the same pairs (before/after comparison)."""
    thr = float(cfg['corr_threshold']); sig_thr = float(cfg['sig_threshold'])
    rows = []
    for ci, cj, li, lj, r in all_pairs:
        se_r, z, sig = _significance(r, n)
        if not (abs(r) >= thr or (np.isfinite(sig) and sig >= sig_thr)):
            continue
        si, sj = float(np.nanstd(df1[ci])), float(np.nanstd(df1[cj]))
        row = dict(col_i=ci, col_j=cj, label_i=li, label_j=lj, r=r, n=n,
                   se_r=se_r, fisher_z=z, sigma=sig, rms_i=si, rms_j=sj,
                   cov=abs(r) * si * sj)
        if df2 is not None:
            m = df2[ci].notna() & df2[cj].notna()
            x, y = df2.loc[m, ci].values, df2.loc[m, cj].values
            rc = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else np.nan
            sic, sjc = float(np.nanstd(df2[ci])), float(np.nanstd(df2[cj]))
            row.update(r_corr=rc, rms_i_corr=sic, rms_j_corr=sjc,
                       cov_corr=abs(rc) * sic * sjc)
        rows.append(row)
    pd.DataFrame(rows).to_parquet(path)
    return len(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--analysis-config', default=None)
    ap.add_argument('--config', default=None,
                    help='mi_config.yaml path (n_dof/n_keep for the optical correction)')
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--fits', default=None, help='override fits.parquet path')
    ap.add_argument('--no-optcorr', action='store_true',
                    help='skip the optical-correction PDF (raw only; off-RSP)')
    ap.add_argument('--n-dof', type=int, default=None, help='override mi-config n_dof')
    ap.add_argument('--n-keep', type=int, default=None, help='override mi-config n_keep')
    args = ap.parse_args()

    cfg = {**DEFAULT, **mc.analysis_section(
        'dz_correlations', args.param_set, args.mi_name,
        config_path=(Path(args.analysis_config) if args.analysis_config else None))}
    prefix = cfg['dz_prefix']
    global _DPI
    _DPI = int(cfg['scatter_dpi'])

    base = Path(args.output_root) / args.param_set / args.mi_name
    fits_path = Path(args.fits) if args.fits else base / 'fits.parquet'
    out_dir = base / 'plots'; out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[dz_correlations] {fits_path}')

    df_raw = quality_cut(pd.read_parquet(fits_path), prefix, cfg['max_coeff_um'])
    dz_cols, labels, all_pairs, n = _selection(df_raw, prefix)
    print(f'  {len(dz_cols)} DZ columns, {len(df_raw)} visits, complete-case n={n}')

    # OFC SVD (shared): projected DOF/v-modes for the SVD-space correlation
    # pages, and the W_resid for the optical-corrected PDF.  Skipped gracefully
    # off-RSP (needs lsst.ts.ofc) or via --no-optcorr.
    df_corr, modes = None, None
    if not args.no_optcorr:
        try:
            svd = _build_svd(df_raw, prefix, args.param_set, args.mi_name,
                             Path(args.config) if args.config else None,
                             args.n_dof, args.n_keep)
            modes = _project_modes(df_raw, prefix, svd)
            df_corr = _apply_optical_correction(df_raw, prefix, svd)
        except Exception as e:
            print(f'  SVD / optical correction skipped ({type(e).__name__}: {e})')

    # raw PDF + parquet (DOF / v-mode correlation pages appended when available)
    g = _write_pdf(out_dir / 'dz_correlations.pdf', df_raw, None,
                   dz_cols, labels, all_pairs, n, cfg, prefix, modes=modes)
    npair = _write_pairs_parquet(out_dir / 'dz_correlations_pairs.parquet',
                                 df_raw, None, all_pairs, n, cfg, prefix)
    print(f'  wrote dz_correlations.pdf ({g} orbit pages) + _pairs.parquet ({npair})')

    # optcorr PDF + parquet: SAME panel set (raw selection), corrected overlaid
    if df_corr is not None:
        g = _write_pdf(out_dir / 'dz_correlations_optcorr.pdf', df_raw, df_corr,
                       dz_cols, labels, all_pairs, n, cfg, prefix)
        npair = _write_pairs_parquet(
            out_dir / 'dz_correlations_optcorr_pairs.parquet',
            df_raw, df_corr, all_pairs, n, cfg, prefix)
        print(f'  wrote dz_correlations_optcorr.pdf ({g} orbit pages, raw+corrected '
              f'overlay) + _pairs.parquet ({npair})')


def _cov_heatmap(corr, dz_cols, labels, std, pdf):
    """Amplitude-weighted companion to the Pearson heatmap: covariance
    cov_ij = r_ij·σ_i·σ_j [μm²], off-diagonal only.  For the optical-corrected
    DZ this shows where *real* residual structure remains — small everywhere the
    correction emptied the coefficient — instead of r→1 at noise amplitude."""
    import matplotlib.pyplot as plt
    s = np.array([std.get(c, np.nan) for c in dz_cols])
    cov = np.asarray(corr) * np.outer(s, s)
    np.fill_diagonal(cov, np.nan)                       # hide variances
    vmax = float(np.nanpercentile(np.abs(cov), 99)) or 1e-9
    n = len(labels); fs = max(12, n * 0.15)
    fig, ax = plt.subplots(figsize=(fs, fs * 0.9))
    im = ax.imshow(cov, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                   interpolation='nearest', interpolation_stage='rgba')
    plt.colorbar(im, ax=ax, shrink=0.7, label='covariance r·σ_i·σ_j [μm²]')
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=5)
    ax.set_title(f'DZ covariance (amplitude-weighted; colour ±{vmax:.2g} μm²)',
                 fontsize=14)
    fig.tight_layout()
    pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)


def _corr_matrix_heatmap(M, labels, title, pdf, annot_r=0.5):
    """Crisp Pearson-r heatmap of the correlations AMONG the columns of M
    (n_visits, n_mode) — the SVD-space analog of the DZ-DZ heatmap.  Pairwise
    complete (pandas .corr); diagonal blanked."""
    import matplotlib.pyplot as plt
    C = pd.DataFrame(M, columns=labels).corr().to_numpy()
    np.fill_diagonal(C, np.nan)
    n = len(labels); fs = max(9, n * 0.2)
    fig, ax = plt.subplots(figsize=(fs, fs * 0.92), layout='constrained')
    im = ax.imshow(C, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto',
                   interpolation='nearest', interpolation_stage='rgba')
    fig.colorbar(im, ax=ax, shrink=0.7, label='Pearson r')
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=5)
    for i in range(n):
        for t in range(n):
            if i != t and np.isfinite(C[i, t]) and abs(C[i, t]) > annot_r:
                ax.text(t, i, f'{C[i, t]:.1f}', ha='center', va='center',
                        fontsize=4, color='white' if abs(C[i, t]) > 0.8 else 'black')
    ax.set_title(title, fontsize=14)
    pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)
    return C


def _mode_top_scatter(M, labels, C, title, pdf, top_n=12, ncols=3):
    """Scatter the top-|r| off-diagonal pairs among the columns of M."""
    import matplotlib.pyplot as plt
    n = len(labels)
    pairs = sorted(((abs(C[i, t]), i, t, C[i, t])
                    for i in range(n) for t in range(i + 1, n)
                    if np.isfinite(C[i, t])), reverse=True)[:top_n]
    if not pairs:
        return
    nrows = (len(pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.3 * nrows),
                             layout='constrained', squeeze=False)
    axes = axes.ravel()
    for idx, (_, i, t, r) in enumerate(pairs):
        ax = axes[idx]
        x, y = M[:, i], M[:, t]
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=7, alpha=0.4, color='steelblue', edgecolors='none')
        if int(m.sum()) > 2:
            sl, off = np.polyfit(x[m], y[m], 1)
            xf = np.array([x[m].min(), x[m].max()])
            ax.plot(xf, sl * xf + off, 'r-', lw=1.2)
        ax.set_title(f'{labels[i]} × {labels[t]}   r={r:+.2f}', fontsize=8)
        ax.set_xlabel(labels[i], fontsize=8); ax.set_ylabel(labels[t], fontsize=8)
        ax.tick_params(labelsize=7)
    for idx in range(len(pairs), len(axes)):
        axes[idx].set_visible(False)
    fig.suptitle(title, fontsize=13)
    pdf.savefig(fig, dpi=_DPI, bbox_inches='tight'); plt.close(fig)


def _mode_pages(modes, cfg, pdf):
    """DOF×DOF and v-mode×v-mode correlation heatmaps + top-|r| v-mode-pair
    scatter — the SVD-space companions to the DZ-DZ pages."""
    ar = float(cfg['mode_annot_r'])
    _corr_matrix_heatmap(modes['dof'], modes['dof_labels'],
                         f'{modes["dof"].shape[1]}-DOF correlation matrix '
                         f'(across visits)', pdf, ar)
    Cv = _corr_matrix_heatmap(modes['vmodes'], modes['vmode_labels'],
                              f'{modes["vmodes"].shape[1]} v-mode correlation '
                              f'matrix (across visits)', pdf, ar)
    _mode_top_scatter(modes['vmodes'], modes['vmode_labels'], Cv,
                      'Top correlated v-mode pairs', pdf,
                      top_n=int(cfg['mode_top_n']))


def _orbit_grid_page(df1, df2, A, B, prefix, title, pdf):
    """One page: a grid of scatter panels over the full conjugation orbit —
    rows = the independent k/j doublet-flips of endpoint A, cols = those of
    endpoint B (up to 4x4 = 16).  Overlays df2 (corrected) on df1 (raw) when
    df2 is given, on df1's axis limits; self panels (A-variant == B-variant)
    are blanked.  Captures every single/double index flip."""
    import matplotlib.pyplot as plt
    rows, cols = _index_flips(A), _index_flips(B)
    nr, nc = len(rows), len(cols)
    fig, axes = plt.subplots(nr, nc, figsize=(4.0 * nc, 3.5 * nr),
                             layout='constrained', squeeze=False)
    for ia, Av in enumerate(rows):
        for ib, Bv in enumerate(cols):
            ax = axes[ia][ib]
            ca, cb = dzp.dz_col_name(*Av, prefix), dzp.dz_col_name(*Bv, prefix)
            if Av == Bv or ca not in df1.columns or cb not in df1.columns:
                ax.axis('off'); continue
            info = _panel(ax, df1, df2, ca, cb)
            ax.set_title(f'(k{Av[0]},j{Av[1]})×(k{Bv[0]},j{Bv[1]})\n{info}',
                         fontsize=6)
            ax.set_xlabel(f'(k{Av[0]},j{Av[1]})', fontsize=6)
            ax.set_ylabel(f'(k{Bv[0]},j{Bv[1]})', fontsize=6)
            ax.tick_params(labelsize=5)
    fig.suptitle(title, fontsize=13)
    pdf.savefig(fig, dpi=_DPI, bbox_inches='tight'); plt.close(fig)


def _conjugate_group_pages(df1, df2, all_pairs, prefix, n, cfg, pdf):
    """One orbit-grid page per significant (raw) correlation: the full
    conjugation orbit (independent focal-k / pupil-j flips of each endpoint) as
    a grid of scatter panels — e.g. (k2,j8)x(k3,j7) [Coma] -> 4x4 over the coma
    family, (k3,j11)x(k3,j12) -> 2x4 (Spherical is m=0).  Seeded by raw
    |r| >= group_min_r AND σ >= sig_threshold; deduped by the orbit.  df2
    (corrected) is overlaid on each panel when given."""
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
        key = frozenset({frozenset(_index_flips(A)), frozenset(_index_flips(B))})
        if key in seen:
            continue
        seen.add(key); n_seed += 1
        if n_pages >= cap:
            continue
        title = (f'{_endpt_name(A)} (rows)   ×   {_endpt_name(B)} (cols)'
                 f'   — conjugation orbit')
        _orbit_grid_page(df1, df2, A, B, prefix, title, pdf)
        n_pages += 1
    if n_seed > cap:
        print(f'  NOTE: {n_seed} conjugate-orbit groups found; capped at {cap} '
              f'pages (raise max_group_pages to see all)')
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
