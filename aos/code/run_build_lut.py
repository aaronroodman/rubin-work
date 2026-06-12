#!/usr/bin/env python3
"""Build an averaged DOF look-up table (LUT) for one (param_set, MI config).

New-pipeline replacement for the LUT-building functionality behind
study_50dofLUT.ipynb: given ``n_dof`` and ``n_keep``, project the Phase-1
per-visit Double-Zernike fits onto the OFC sensitivity-matrix SVD
(ofc_svd.build_ofc_svd), recover the physical DOF per visit, and **collapse
over all elevation and rotator angle** into a single averaged LUT.

Unlike the measured-intrinsic build, the LUT deliberately does NOT bin by
rotator and (by default) does NOT apply an elevation window — it averages over
every visit that passes the band / program / good-fit cuts.  ``n_dof`` and
``n_keep`` are settable in mi_config.yaml (the ``lut`` block overrides the
top-level values, so the LUT can use a different mode count than the build).

Writes, under  output/<param_set>/<mi_name>/lut/ :

    lut.parquet       one row per recovered DOF: index, label, unit, value
                      (median by default), mean, robust scatter, n_visits
    lut_dz.parquet    one row per (focal k, pupil j): averaged raw DZ, the
                      n_keep-mode SVD reconstruction, and the residual
    lut_config.yaml   frozen provenance

RSP-only (build_ofc_svd needs lsst.ts.ofc + TS_CONFIG_MTTCS_DIR).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from astropy.table import QTable

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import ofc_svd as osv
import mi_config as mc
from measured_intrinsic import apply_visit_filters

DEFAULT_LUT = dict(n_dof=None, n_keep=None, reduce='median',
                   use_alt_window=False, drop_bad_fit=True, prefix='z1toz6')


def _filter_band(visits, allowed_bands):
    if not allowed_bands or 'band' not in visits.colnames:
        return visits
    b = np.asarray(visits['band']).astype(str)
    return visits[np.array([x in set(allowed_bands) for x in b])]


def _filter_program(visits, programs):
    if not programs or 'science_program' not in visits.colnames:
        return visits
    p = np.asarray(visits['science_program']).astype(str)
    return visits[np.array([x in set(programs) for x in p])]


def _good_fit_mask(fits_table, prefix):
    """Boolean mask of visits NOT flagged as bad (all True if no flag)."""
    for cand in (f'{prefix}_bad_fit', 'bad_fit'):
        if cand in fits_table.colnames:
            return ~np.asarray(fits_table[cand]).astype(bool)
    return np.ones(len(fits_table), dtype=bool)


def _reduce(arr, how):
    """Column-wise reduction ignoring NaNs: 'median' or 'mean'."""
    if how == 'mean':
        return np.nanmean(arr, axis=0)
    return np.nanmedian(arr, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True,
                    help='MI config entry name in mi_config.yaml')
    ap.add_argument('--config', default=None,
                    help='mi_config.yaml path (default: ../mi_config.yaml)')
    ap.add_argument('--analysis-config', default=None,
                    help='analysis_config.yaml path (default: ../analysis_config.yaml)')
    ap.add_argument('--n-dof', type=int, default=None,
                    help='override lut.n_dof / top-level n_dof')
    ap.add_argument('--n-keep', type=int, default=None,
                    help='override lut.n_keep / top-level n_keep')
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()

    cfg = mc.load_mi_config(args.param_set, args.mi_name,
                            config_path=(Path(args.config) if args.config else None))
    b = cfg['build']
    # LUT knobs come from analysis_config.yaml (separate from mi_config so they
    # don't re-trigger the build); fall back to code-level DEFAULT_LUT.
    lut = {**DEFAULT_LUT, **mc.analysis_section(
        'lut', args.param_set, args.mi_name,
        config_path=(Path(args.analysis_config) if args.analysis_config else None))}

    # n_dof / n_keep: CLI > lut block > top-level config entry
    n_dof_spec = (args.n_dof if args.n_dof is not None
                  else lut['n_dof'] if lut['n_dof'] is not None
                  else cfg.get('n_dof'))
    n_keep_spec = (args.n_keep if args.n_keep is not None
                   else lut['n_keep'] if lut['n_keep'] is not None
                   else cfg['n_keep'])
    k_min, k_max = int(b['k_min']), int(b['k_max'])
    prefix = lut.get('prefix', 'z1toz6')
    reduce_how = lut.get('reduce', 'median')
    coord_sys = cfg.get('coord_sys', 'OCS')
    allowed_bands = mc.as_band_list(cfg.get('filter'))
    programs = cfg.get('programs')
    ofc_norm_yaml = b.get('ofc_normalization_yaml')

    base = Path(args.output_root) / args.param_set
    out_dir = Path(args.out_dir) if args.out_dir else base / args.mi_name / 'lut'
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[build_lut] {args.param_set} / {args.mi_name}  -> {out_dir}')
    print(f'  n_dof={n_dof_spec}  n_keep={n_keep_spec}  reduce={reduce_how}  '
          f'k={k_min}..{k_max}')

    # ---- visits: filter by band/program (and optional elevation window) ----
    visits = QTable.read(str(base / 'visits.parquet'))
    n_all = len(visits)
    if lut.get('use_alt_window'):
        visits = apply_visit_filters(visits, alt_min_deg=cfg.get('alt_min_deg'),
                                     alt_max_deg=cfg.get('alt_max_deg'))
    visits = _filter_band(visits, allowed_bands)
    visits = _filter_program(visits, programs)
    kept = {(int(d), int(s)) for d, s in
            zip(np.asarray(visits['day_obs']), np.asarray(visits['seq_num']))}
    alt_note = (f'alt=[{cfg.get("alt_min_deg")},{cfg.get("alt_max_deg")}]'
                if lut.get('use_alt_window') else 'alt=ALL')
    print(f'  visits passing band/program: {len(kept)}/{n_all} '
          f'(bands={allowed_bands}, programs={programs}, {alt_note}, rotator=ALL)')

    # ---- pupil Noll list (canonical from the visits sidecar) ----
    if 'nollIndices' not in visits.colnames:
        raise RuntimeError('visits.parquet has no nollIndices column')
    iZs = [int(j) for j in np.asarray(visits['nollIndices'][0]).tolist()]

    # ---- per-visit DZ fits, restricted to kept + good-fit visits ----
    fits = QTable.read(str(base / 'fits.parquet'))
    fd = np.asarray(fits['day_obs']).astype(int)
    fs = np.asarray(fits['seq_num']).astype(int)
    in_kept = np.array([(int(d), int(s)) in kept for d, s in zip(fd, fs)])
    good = _good_fit_mask(fits, prefix) if lut.get('drop_bad_fit', True) \
        else np.ones(len(fits), dtype=bool)
    fits = fits[in_kept & good]
    print(f'  DZ-fit visits used: {len(fits)} '
          f'({int((~good).sum())} bad-fit dropped before kept-cut)')
    if len(fits) == 0:
        raise RuntimeError('No visits survive the LUT filters.')

    # ---- OFC SVD + per-visit DOF / DZ projection ----
    svd = osv.build_ofc_svd(iZs, k_min, k_max, n_keep_spec, n_dof=n_dof_spec,
                            ofc_normalization_yaml=ofc_norm_yaml)
    dof_labels, dof_units = svd.dof_labels()
    print(f'  SVD: U_eff={svd.U_eff.shape}, n_dof={svd.n_dof}, '
          f'n_keep_eff={svd.n_keep_eff}')
    _vmodes, dof, A, W = osv.project_dz_table(fits, prefix, svd)
    W_fit = A @ svd.U_eff.T                       # n_keep-mode DZ reconstruction
    W_resid = W - W_fit

    # ---- collapse over all visits (single averaged LUT) ----
    dof_val = _reduce(dof, reduce_how)
    dof_mean = np.nanmean(dof, axis=0)
    dof_mad = 1.4826 * np.nanmedian(np.abs(dof - np.nanmedian(dof, axis=0)), axis=0)
    n_vis = int(len(fits))

    dof_tbl = pa.table({
        'dof_index': pa.array(np.asarray(svd.dof_idx, dtype=int)),
        'dof_label': pa.array([str(x) for x in dof_labels]),
        'dof_unit': pa.array([str(x) for x in dof_units]),
        'value': pa.array(np.asarray(dof_val, dtype=float)),
        'mean': pa.array(np.asarray(dof_mean, dtype=float)),
        'scatter_mad': pa.array(np.asarray(dof_mad, dtype=float)),
        'n_visits': pa.array(np.full(len(dof_val), n_vis, dtype=int)),
    })
    meta = {b'param_set': args.param_set.encode(), b'mi_name': args.mi_name.encode(),
            b'reduce': str(reduce_how).encode(), b'n_dof': str(svd.n_dof).encode(),
            b'n_keep': str(svd.n_keep_eff).encode(), b'n_visits': str(n_vis).encode(),
            b'k_min': str(k_min).encode(), b'k_max': str(k_max).encode()}
    pq.write_table(dof_tbl.replace_schema_metadata(meta),
                   str(out_dir / 'lut.parquet'), compression='snappy')

    # ---- per-(k,j) averaged DZ: raw, n_keep reconstruction, residual ----
    kk = np.array([k for (k, j) in svd.kj_grid], dtype=int)
    jj = np.array([j for (k, j) in svd.kj_grid], dtype=int)
    dz_tbl = pa.table({
        'k': pa.array(kk), 'j': pa.array(jj),
        'dz_raw': pa.array(_reduce(W, reduce_how)),
        'dz_fit': pa.array(_reduce(W_fit, reduce_how)),
        'dz_resid': pa.array(_reduce(W_resid, reduce_how)),
        'n_visits': pa.array(np.full(len(kk), n_vis, dtype=int)),
    })
    pq.write_table(dz_tbl.replace_schema_metadata(meta),
                   str(out_dir / 'lut_dz.parquet'), compression='snappy')

    # ---- provenance ----
    with open(out_dir / 'lut_config.yaml', 'w') as fh:
        yaml.safe_dump({'param_set': args.param_set, 'mi_name': args.mi_name,
                        'n_dof': int(svd.n_dof), 'n_keep': int(svd.n_keep_eff),
                        'reduce': reduce_how, 'n_visits': n_vis,
                        'k_min': k_min, 'k_max': k_max,
                        'bands': allowed_bands, 'programs': programs,
                        'use_alt_window': bool(lut.get('use_alt_window'))},
                       fh, sort_keys=False)
    # ---- one-page DOF summary PDF (4-panel style of the MI validation) ----
    _write_lut_pdf(out_dir / 'lut.pdf', svd.dof_idx, dof_val, dof_mad,
                   n_vis, svd.n_dof, svd.n_keep_eff, reduce_how)

    print(f'  wrote lut.parquet ({len(dof_val)} DOF) + lut_dz.parquet '
          f'({len(kk)} (k,j)) + lut.pdf over {n_vis} visits')
    print('[build_lut] done.')


def _write_lut_pdf(path, dof_idx, values, scatter, n_visits, n_dof, n_keep,
                   reduce_how):
    """Single-page, 4-panel DOF summary (Hex translations / Hex tip-tilt /
    M1M3 / M2), mirroring intrinsic_build_plots.plot_dof_median_summary.
    Points are the LUT values; error bars are the robust (MAD) scatter."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    pos = {int(d): i for i, d in enumerate(dof_idx)}     # abs DOF index -> row
    buckets = [
        ('Hexapod Translations (M2, Cam: dz/dx/dy)', [0, 1, 2, 5, 6, 7], 'μm'),
        ('Hexapod Rotations / tip-tilt (M2, Cam: rx/ry)', [3, 4, 8, 9], 'arcsec'),
        ('M1M3 Bending Modes', list(range(10, 30)), 'μm'),
        ('M2 Bending Modes', list(range(30, 50)), 'μm'),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(15, 14), layout='constrained',
                             gridspec_kw=dict(height_ratios=[1.0, 1.0, 1.5, 1.5]))
    for ax, (title, absidx, unit) in zip(axes, buckets):
        present = [a for a in absidx if a in pos]
        x = np.arange(len(present))
        for xi in range(len(present)):
            if xi % 2:
                ax.axvspan(xi - 0.5, xi + 0.5, color='black', alpha=0.05)
        vals = np.array([values[pos[a]] for a in present])
        errs = np.array([scatter[pos[a]] for a in present])
        ax.errorbar(x, vals, yerr=errs, fmt='o', ms=7, color='steelblue',
                    ecolor='gray', elinewidth=1, capsize=3, lw=0)
        ax.axhline(0, color='gray', lw=0.5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([osv.LABELS_50DOF[a] for a in present],
                           rotation=45, ha='right', fontsize=8)
        ax.set_ylabel(f'LUT value ({unit})'); ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle(f'Averaged-DOF LUT — {reduce_how} over {n_visits} visits  '
                 f'(n_dof={n_dof}, n_keep={n_keep}); error bars = robust MAD',
                 fontsize=13)
    with PdfPages(str(path)) as pdf:
        pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    print('  wrote lut.pdf')


if __name__ == '__main__':
    main()
