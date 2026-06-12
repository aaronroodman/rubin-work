#!/usr/bin/env python3
"""DZ-to-temperature correlation analysis (script port of
intrinsics_thermal_correlations.ipynb / study_doublezernike §6), run on a
per-(param_set, mi_name) DZ-fit table.

Operates on  output/<ps>/<mi>/fits.parquet  (MI refit), whose visit join carries
the full temperature suite (cam/m2/m1m3 air temps, gradients, deltas, truss
temps), so the correlations are between the MI-subtracted residual DZ
coefficients and the thermal state.  Reuses code/dz_plotting.py.

Writes, under  output/<ps>/<mi>/plots/ :
    thermal_correlations.pdf            DZ(k,j) x thermal-var Pearson heatmap +
                                        per-DZ-term scatter pages vs each thermal var
    thermal_correlations_summary.parquet  r and n for every (k, j, thermal_var)

Knobs come from analysis_config.yaml (section ``thermal_correlations``).
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

DEFAULT = dict(
    dz_prefix='z1toz6', max_coeff_um=2.0,
    thermal_vars=list(dzp.DEFAULT_THERMAL_VARS),
    dz_terms=[[1, 4], [2, 4], [3, 4], [4, 4], [5, 4], [6, 4],
              [5, 5], [6, 6], [5, 10], [6, 9]])


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


def _heatmap(df, dz_cols, labels, thermal_vars, prefix, pdf):
    """DZ(k,j) rows x thermal-var cols Pearson-r heatmap; returns long rows."""
    import matplotlib.pyplot as plt
    R = np.full((len(dz_cols), len(thermal_vars)), np.nan)
    rows = []
    for i, c in enumerate(dz_cols):
        jk = parse_jk(c, prefix)
        for t, tv in enumerate(thermal_vars):
            if tv not in df.columns:
                continue
            m = df[c].notna() & df[tv].notna()
            if int(m.sum()) > 2:
                r = float(np.corrcoef(df.loc[m, tv], df.loc[m, c])[0, 1])
                R[i, t] = r
                rows.append(dict(k=jk[1], j=jk[0], thermal_var=tv,
                                 r=r, n=int(m.sum())))
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(thermal_vars) + 3),
                                    max(8, 0.18 * len(dz_cols) + 2)),
                           layout='constrained')
    im = ax.imshow(R, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto',
                   interpolation='none')
    ax.set_xticks(range(len(thermal_vars)))
    ax.set_xticklabels(thermal_vars, rotation=90, fontsize=7)
    ax.set_yticks(range(len(dz_cols))); ax.set_yticklabels(labels, fontsize=4)
    for i in range(len(dz_cols)):
        for t in range(len(thermal_vars)):
            if np.isfinite(R[i, t]) and abs(R[i, t]) > 0.4:
                ax.text(t, i, f'{R[i, t]:.2f}', ha='center', va='center',
                        fontsize=4, color='white' if abs(R[i, t]) > 0.7 else 'black')
    fig.colorbar(im, ax=ax, shrink=0.6, label='Pearson r')
    ax.set_title('DZ coefficient vs thermal-variable correlation', fontsize=13)
    pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--analysis-config', default=None)
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--fits', default=None)
    args = ap.parse_args()

    cfg = {**DEFAULT, **mc.analysis_section(
        'thermal_correlations', args.param_set, args.mi_name,
        config_path=(Path(args.analysis_config) if args.analysis_config else None))}
    prefix = cfg['dz_prefix']
    thermal_vars = list(cfg['thermal_vars'])
    dz_terms = [tuple(t) for t in cfg['dz_terms']]

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
    df = quality_cut(df, prefix, cfg['max_coeff_um'])
    dz_cols = sorted(dz_coeff_columns(df, prefix),
                     key=lambda c: parse_jk(c, prefix)[::-1])
    labels = [f'({k},{j})' for c in dz_cols for (j, k) in [parse_jk(c, prefix)]]
    print(f'  {len(dz_cols)} DZ columns, {len(present_tv)} thermal vars, '
          f'{len(df)} visits')

    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(str(out_dir / 'thermal_correlations.pdf')) as pdf:
        rows = _heatmap(df, dz_cols, labels, present_tv, prefix, pdf)
        dzp.plot_thermal_scatter_grid(df, dz_terms, thermal_vars=present_tv,
                                      dz_prefix=prefix, pdf=pdf, show=False)
    pd.DataFrame(rows).to_parquet(out_dir / 'thermal_correlations_summary.parquet')
    print(f'  wrote thermal_correlations.pdf + thermal_correlations_summary.parquet '
          f'({len(rows)} (k,j,thermal) rows)')


if __name__ == '__main__':
    main()
