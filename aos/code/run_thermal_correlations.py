#!/usr/bin/env python3
"""DZ / DOF / v-mode -to-temperature correlation analysis, run on a
per-(param_set, mi_name) DZ-fit table.

Operates on  output/<ps>/<mi>/fits.parquet  (MI refit), whose visit join carries
the full temperature suite (cam/m2/m1m3 air temps, gradients, deltas, truss
temps), so the correlations are between the MI-subtracted residual state and the
thermal state.

Pages written to  output/<ps>/<mi>/plots/thermal_correlations.pdf :
  1. DZ(k,j) x thermal-variable Pearson-r heatmap (crisp, no smoothing).
  2. Per thermal variable: all DZ(k,j) scatter, k=1..6 down each column and
     7 pupil-j per page (3 pages covering the 21 j), shared x-axis.
  3. 50-DOF x thermal and n_keep-v-mode x thermal Pearson-r heatmaps.
  4. Per thermal variable: the n_keep v-modes scatter (7-column grid, one page).

Steps 3-4 need the OFC SVD (lsst.ts.ofc + TS_CONFIG_MTTCS_DIR, RSP-only); if it
is unavailable they are skipped with a note and steps 1-2 still render.

    thermal_correlations_summary.parquet  r, n for every (kind, term, thermal_var)

Knobs come from analysis_config.yaml (section ``thermal_correlations``); n_dof /
n_keep follow the build's mi_config (same as the LUT).
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
import ofc_svd as osv

# Thermal-variable suite carried on the visit join (mirrors
# dz_plotting.DEFAULT_THERMAL_VARS; inlined to avoid the heavy intrinsics_lib
# import chain, since this script does its own plotting).
DEFAULT_THERMAL_VARS = [
    "cam_air_temp", "m2_air_temp", "m1m3_air_temp", "outside_temp",
    "m2_delta_t", "cam_m1m3_delta_t", "dome_delta_t",
    "x_gradient", "y_gradient", "z_gradient", "radial_gradient",
    "tma_truss_temp_pxpy", "tma_truss_temp_mxmy",
]

DEFAULT = dict(
    dz_prefix='z1toz6', max_coeff_um=2.0,
    thermal_vars=list(DEFAULT_THERMAL_VARS),
    scatter_ncols=7,        # pupil-j per page (DZ) / v-modes per row
    annot_r=0.4)            # |r| above which a heatmap cell is annotated


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


def _corr(x, y):
    """Pearson r and N over jointly-finite samples."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return np.nan, int(m.sum())
    return float(np.corrcoef(x[m], y[m])[0, 1]), int(m.sum())


# ----------------------------------------------------------------------
# crisp r-heatmap (rows = terms, cols = thermal vars) — no interpolation
# ----------------------------------------------------------------------
def _heatmap(R, row_labels, col_labels, title, pdf, annot_r=0.4,
             row_fontsize=4):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(col_labels) + 3),
                                    max(6, 0.16 * len(row_labels) + 2)),
                           layout='constrained')
    im = ax.imshow(R, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto',
                   interpolation='nearest', interpolation_stage='rgba')
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=row_fontsize)
    for i in range(R.shape[0]):
        for t in range(R.shape[1]):
            if np.isfinite(R[i, t]) and abs(R[i, t]) > annot_r:
                ax.text(t, i, f'{R[i, t]:.2f}', ha='center', va='center',
                        fontsize=4, color='white' if abs(R[i, t]) > 0.7 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.6, label='Pearson r')
    ax.set_title(title, fontsize=13)
    pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)


def _dz_heatmap(df, dz_cols, thermal_vars, prefix, pdf, annot_r):
    """DZ(k,j) rows (k-major) x thermal cols; returns long summary rows."""
    R = np.full((len(dz_cols), len(thermal_vars)), np.nan)
    rows, labels = [], []
    for i, c in enumerate(dz_cols):
        j, k = parse_jk(c, prefix)
        labels.append(f'({k},{j})')
        for t, tv in enumerate(thermal_vars):
            if tv not in df.columns:
                continue
            r, n = _corr(df[tv], df[c])
            R[i, t] = r
            rows.append(dict(kind='dz', k=k, j=j, dof_index=np.nan,
                             mode_index=np.nan, label=f'({k},{j})',
                             thermal_var=tv, r=r, n=n))
    _heatmap(R, labels, thermal_vars,
             'DZ coefficient (k,j) vs thermal-variable correlation', pdf, annot_r)
    return rows


# ----------------------------------------------------------------------
# DZ scatter: per thermal var, k=1..6 rows x 7 pupil-j cols, 3 pages
# ----------------------------------------------------------------------
def _panel(ax, x, y, annotate=True):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[m], y[m], s=4, alpha=0.45, edgecolors='none')
    ax.axhline(0, color='k', lw=0.3, alpha=0.4)
    if int(m.sum()) > 2:
        sl, off = np.polyfit(x[m], y[m], 1)
        xf = np.array([x[m].min(), x[m].max()])
        ax.plot(xf, sl * xf + off, 'r-', lw=0.9, alpha=0.85)
        if annotate:
            r = float(np.corrcoef(x[m], y[m])[0, 1])
            ax.text(0.04, 0.92, f'r={r:+.2f}', transform=ax.transAxes,
                    fontsize=6, va='top',
                    color='crimson' if abs(r) > 0.4 else 'dimgray')


def _dz_scatter_pages(df, present_tv, prefix, pdf, ncols, ks=range(1, 7)):
    import matplotlib.pyplot as plt
    try:
        from common.zernike_names import NOLL_NAMES
    except Exception:
        NOLL_NAMES = {}
    ks = list(ks)
    js = sorted({parse_jk(c, prefix)[0] for c in dz_coeff_columns(df, prefix)})
    pages = [js[i:i + ncols] for i in range(0, len(js), ncols)]
    for tv in present_tv:
        for pi, page_js in enumerate(pages):
            fig, axes = plt.subplots(len(ks), ncols, figsize=(2.0 * ncols, 1.7 * len(ks)),
                                     layout='constrained', sharex=True, squeeze=False)
            for col in range(ncols):
                for row, k in enumerate(ks):
                    ax = axes[row][col]
                    if col >= len(page_js):
                        ax.set_visible(False); continue
                    j = page_js[col]
                    c = f'{prefix}_z{j}_c{k}'
                    if c in df.columns:
                        _panel(ax, df[tv], df[c])
                    ax.tick_params(labelsize=6)
                    if row == 0:
                        ax.set_title(f'j={j} {NOLL_NAMES.get(j, "")}', fontsize=7)
                    if col == 0:
                        ax.set_ylabel(f'k={k}', fontsize=7)
                    if row == len(ks) - 1:
                        ax.set_xlabel(tv, fontsize=7)
            fig.suptitle(f'DZ(k,j) vs {tv}   (k=1..6 rows, pupil-j cols; '
                         f'page {pi + 1}/{len(pages)})', fontsize=12)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


# ----------------------------------------------------------------------
# DOF / v-mode projection (OFC SVD) + their thermal correlations
# ----------------------------------------------------------------------
def _project(df, prefix, svd):
    """Per-visit v-mode (n_keep) and DOF (n_dof) amplitudes for df's rows."""
    W = np.full((len(df), len(svd.kj_grid)), np.nan)
    for ci, (k, j) in enumerate(svd.kj_grid):
        col = f'{prefix}_z{j}_c{k}'
        if col in df.columns:
            W[:, ci] = df[col].to_numpy(dtype=float)
    nv = int(np.isnan(W).any(axis=1).sum())          # A2: NaN-fill transparency
    print(f'  DZ NaN before projection: {100 * np.isnan(W).mean():.2f}% of cells, '
          f'{nv} visits with ≥1 NaN (zero-filled by project_amplitudes)')
    A = svd.project_amplitudes(W)
    return svd.vmodes(A), svd.dof(A)


def _mode_heatmap(M, thermal_df, thermal_vars, row_labels, kind, title, pdf,
                  annot_r):
    """Correlate each column of M (n_visits, n_mode) with each thermal var."""
    R = np.full((M.shape[1], len(thermal_vars)), np.nan)
    rows = []
    for i in range(M.shape[1]):
        for t, tv in enumerate(thermal_vars):
            r, n = _corr(thermal_df[tv].to_numpy(float), M[:, i])
            R[i, t] = r
            rows.append(dict(kind=kind, k=np.nan, j=np.nan,
                             dof_index=(i if kind == 'dof' else np.nan),
                             mode_index=(i if kind == 'vmode' else np.nan),
                             label=row_labels[i], thermal_var=tv, r=r, n=n))
    _heatmap(R, row_labels, thermal_vars, title, pdf, annot_r,
             row_fontsize=5 if M.shape[1] <= 50 else 4)
    return rows


def _vmode_scatter_pages(V, thermal_df, present_tv, pdf, ncols):
    import matplotlib.pyplot as plt
    n_mode = V.shape[1]
    nrows = int(np.ceil(n_mode / ncols))
    for tv in present_tv:
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 1.7 * nrows),
                                 layout='constrained', sharex=True, squeeze=False)
        x = thermal_df[tv].to_numpy(float)
        for idx in range(nrows * ncols):
            ax = axes[idx // ncols][idx % ncols]
            if idx >= n_mode:
                ax.set_visible(False); continue
            _panel(ax, x, V[:, idx])
            ax.tick_params(labelsize=6)
            ax.set_title(f'v{idx + 1}', fontsize=7)
            if idx // ncols == nrows - 1:
                ax.set_xlabel(tv, fontsize=7)
        fig.suptitle(f'v-modes (n_keep={n_mode}) vs {tv}', fontsize=12)
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)


def _mode_section(df, present_tv, prefix, cfg, base, pdf, annot_r, ncols):
    """Build the SVD, project, and emit DOF/v-mode heatmaps + v-mode scatter.
    Returns summary rows; returns [] (with a printed note) if the SVD can't be
    built (e.g. off-RSP where lsst.ts.ofc is unavailable)."""
    from astropy.table import QTable
    b = cfg['build']
    k_min, k_max = int(b['k_min']), int(b['k_max'])
    n_dof_spec = cfg.get('n_dof')
    n_keep_spec = cfg['n_keep']
    ofc_norm_yaml = b.get('ofc_normalization_yaml')
    visits = QTable.read(str(base.parent / 'visits.parquet'))
    if 'nollIndices' not in visits.colnames:
        raise RuntimeError('visits.parquet has no nollIndices column')
    iZs = [int(j) for j in np.asarray(visits['nollIndices'][0]).tolist()]

    svd = osv.build_ofc_svd(iZs, k_min, k_max, n_keep_spec, n_dof=n_dof_spec,
                            ofc_normalization_yaml=ofc_norm_yaml)
    dof_labels, _ = svd.dof_labels()
    print(f'  SVD: U_eff={svd.U_eff.shape}, n_dof={svd.n_dof}, '
          f'n_keep_eff={svd.n_keep_eff}')
    V, D = _project(df, prefix, svd)
    vlabels = [f'v{i + 1}' for i in range(V.shape[1])]
    rows = []
    rows += _mode_heatmap(D, df, present_tv, list(dof_labels), 'dof',
                          f'{svd.n_dof}-DOF vs thermal-variable correlation',
                          pdf, annot_r)
    rows += _mode_heatmap(V, df, present_tv, vlabels, 'vmode',
                          f'{svd.n_keep_eff} v-modes vs thermal-variable correlation',
                          pdf, annot_r)
    _vmode_scatter_pages(V, df, present_tv, pdf, ncols)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--analysis-config', default=None)
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--fits', default=None)
    args = ap.parse_args()

    sec = {**DEFAULT, **mc.analysis_section(
        'thermal_correlations', args.param_set, args.mi_name,
        config_path=(Path(args.analysis_config) if args.analysis_config else None))}
    prefix = sec['dz_prefix']
    thermal_vars = list(sec['thermal_vars'])
    ncols = int(sec['scatter_ncols'])
    annot_r = float(sec['annot_r'])
    cfg = mc.load_mi_config(args.param_set, args.mi_name,
                            config_path=(Path(args.config) if args.config else None))

    base = Path(args.output_root) / args.param_set / args.mi_name
    fits_path = Path(args.fits) if args.fits else base / 'fits.parquet'
    out_dir = base / 'plots'; out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[thermal_correlations] {fits_path}')

    df = pd.read_parquet(fits_path)
    present_tv = [tv for tv in thermal_vars if tv in df.columns]
    missing = [tv for tv in thermal_vars if tv not in df.columns]
    if missing:
        print(f'  thermal vars absent from fits.parquet: {missing}')
    if not present_tv:
        raise RuntimeError('No thermal variables present in the fit table.')
    df = quality_cut(df, prefix, sec['max_coeff_um'])
    dz_cols = sorted(dz_coeff_columns(df, prefix),
                     key=lambda c: parse_jk(c, prefix)[::-1])   # k-major
    print(f'  {len(dz_cols)} DZ columns, {len(present_tv)} thermal vars, '
          f'{len(df)} visits')

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    rows = []
    with PdfPages(str(out_dir / 'thermal_correlations.pdf')) as pdf:
        rows += _dz_heatmap(df, dz_cols, present_tv, prefix, pdf, annot_r)   # 1
        _dz_scatter_pages(df, present_tv, prefix, pdf, ncols)               # 2
        try:                                                               # 3 + 4
            rows += _mode_section(df, present_tv, prefix, cfg, base, pdf, annot_r, ncols)
        except Exception as e:
            print(f'  DOF/v-mode section skipped ({type(e).__name__}: {e}) — '
                  f'needs lsst.ts.ofc (RSP)')
    pd.DataFrame(rows).to_parquet(out_dir / 'thermal_correlations_summary.parquet')
    print(f'  wrote thermal_correlations.pdf + thermal_correlations_summary.parquet '
          f'({len(rows)} rows)')


if __name__ == '__main__':
    main()
