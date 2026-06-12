#!/usr/bin/env python3
"""Build the Path-A (U-mode-constrained) *measured intrinsic* for one
(param_set, MI config, rotator bin) and write its grid + DZ fits + diagnostics.

Faithful script port of build_measured_intrinsic.ipynb (Path A only): reads the
Phase-1 combined ``output/<param_set>/{donuts,visits}.parquet``, filters visits
to the rotator bin / bands, builds the OFC SVD (ofc_svd.build_ofc_svd), attaches
per-donut CCD-height Z4, iterates the U-mode-constrained DZ removal
(measured_intrinsic.build_measured_intrinsic_uconstrained), and writes:

    <out>/intrinsic_grid.parquet   measured-intrinsic focal grid (+ z4_optical_OCS)
    <out>/dz_fits.parquet          per-visit DZ fits + u/v-mode + DOF + FWHM cols
    <out>/*.pdf                    validation, comparison, final-maps, CCS-binned

where <out> defaults to
    output/<param_set>/<mi_name>/build/rot_<rot_min>_<rot_max>/

Config (n_keep, path, filter, rotator bins, ...) comes from mi_config.yaml,
selected by --param-set + --mi-name; explicit flags override.  RSP-only.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
import pyarrow.parquet as pq
from astropy.table import QTable

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import ofc_svd as osv
import mi_config as mc
from dz_fitting import derive_noll_indices
from measured_intrinsic import (
    apply_visit_filters, build_measured_intrinsic_uconstrained,
    assemble_intrinsic_table, save_dz_fits)
import intrinsic_build_plots as ibp


# ----------------------------------------------------------------------
# data loading (memory-safe: only kept visits' donut row groups)
# ----------------------------------------------------------------------
def _visit_row_groups(donuts_parquet):
    """(day_obs, seq_num) -> row_group_idx from per-visit column stats."""
    pf = pq.ParquetFile(str(donuts_parquet))
    lut = {}
    for i in range(pf.num_row_groups):
        meta = pf.metadata.row_group(i)
        d = s = None
        for ci in range(meta.num_columns):
            cm = meta.column(ci)
            if cm.path_in_schema == 'day_obs' and cm.statistics is not None:
                d = cm.statistics.min
            elif cm.path_in_schema == 'seq_num' and cm.statistics is not None:
                s = cm.statistics.min
        if d is not None and s is not None:
            lut[(int(d), int(s))] = i
    return pf, lut


def load_kept_donuts(donuts_parquet, visits_kept):
    """Concatenate (as a pandas DataFrame) only the donut row groups whose
    (day_obs, seq_num) is in visits_kept."""
    pf, lut = _visit_row_groups(donuts_parquet)
    keys = list(zip(np.asarray(visits_kept['day_obs']).astype(int).tolist(),
                    np.asarray(visits_kept['seq_num']).astype(int).tolist()))
    import pandas as pd
    frames = [pf.read_row_group(lut[k]).to_pandas() for k in keys if k in lut]
    if not frames:
        raise RuntimeError('No donut row groups matched the kept visits.')
    return pd.concat(frames, ignore_index=True)


def filter_visits_by_band(visits, allowed_bands):
    if not allowed_bands or 'band' not in visits.colnames:
        return visits
    sp = np.asarray(visits['band']).astype(str)
    keep = np.array([b in set(allowed_bands) for b in sp])
    return visits[keep]


# ----------------------------------------------------------------------
def filter_visits_by_program(visits, programs):
    """Keep only visits whose science_program is in the whitelist (e.g. the
    T614 triplets; excludes the bounce blocks that slew within a triplet)."""
    if not programs or 'science_program' not in visits.colnames:
        return visits
    sp = np.asarray(visits['science_program']).astype(str)
    keep = np.array([p in set(programs) for p in sp])
    return visits[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True,
                    help='MI config entry name in mi_config.yaml')
    ap.add_argument('--config', default=None,
                    help='mi_config.yaml path (default: ../mi_config.yaml)')
    ap.add_argument('--rot-min', type=float, default=None,
                    help='Rotator-bin lower edge (deg) for this build')
    ap.add_argument('--rot-max', type=float, default=None)
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--out-dir', default=None,
                    help='Explicit output dir (default: '
                         'output/<ps>/<mi>/build/rot_<lo>_<hi>)')
    args = ap.parse_args()

    cfg = mc.load_mi_config(args.param_set, args.mi_name,
                            config_path=(Path(args.config) if args.config else None))
    b = cfg['build']

    # ---- config (defaults merged from mi_config.yaml 'defaults' block) ----
    coord_sys = cfg.get('coord_sys', 'OCS')
    n_keep_spec = cfg['n_keep']            # scalar or list -> build_ofc_svd
    n_dof_spec = cfg.get('n_dof')          # scalar or list (None = all 50)
    k_min, k_max = int(b['k_min']), int(b['k_max'])
    n_iter = int(b['n_iter'])
    n_bins = int(b['n_bins'])
    fp_radius_basis = float(b['fp_radius_basis'])
    fp_radius_grid = float(b['fp_radius_grid'])
    min_donuts = int(b['min_donuts'])
    bad_fit_threshold = float(b['bad_fit_threshold'])
    allowed_bands = mc.as_band_list(cfg.get('filter'))
    programs = cfg.get('programs')
    alt_min, alt_max = cfg.get('alt_min_deg'), cfg.get('alt_max_deg')
    rot_min, rot_max = args.rot_min, args.rot_max
    max_visits = b.get('max_visits')
    ofc_norm_yaml = b.get('ofc_normalization_yaml')
    height_source = b.get('height_source', 'batoid_rubin')
    height_map_dir = b.get('batoid_rubin_height_map_dir')
    height_map_fits = b.get('height_map_fits')
    height_to_z4_factor = b.get('height_to_z4_factor')
    wfs_edge_cut_deg = float(b.get('wfs_edge_cut_deg', 1.75))
    n_radial_bins_azimuth = int(b.get('n_radial_bins_azimuth', 4))
    azimuth_bin_deg = float(b.get('azimuth_bin_deg', 15.0))

    base = Path(args.output_root) / args.param_set
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        rtag = (f'rot_{rot_min:g}_{rot_max:g}'
                if rot_min is not None and rot_max is not None else 'rot_all')
        out_dir = base / args.mi_name / 'build' / rtag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[build_intrinsic] {args.param_set} / {args.mi_name}  '
          f'rot=[{rot_min},{rot_max}]  -> {out_dir}')

    # ---- data ----
    donuts_pq = base / 'donuts.parquet'
    visits_full = QTable.read(str(base / 'visits.parquet'))
    visits_kept = apply_visit_filters(
        visits_full, alt_min_deg=alt_min, alt_max_deg=alt_max,
        rotator_min_deg=rot_min, rotator_max_deg=rot_max)
    visits_kept = filter_visits_by_band(visits_kept, allowed_bands)
    visits_kept = filter_visits_by_program(visits_kept, programs)
    if max_visits and len(visits_kept) > max_visits:
        idx = np.unique(np.round(
            np.linspace(0, len(visits_kept) - 1, max_visits)).astype(int))
        visits_kept = visits_kept[idx]
    print(f'  visits kept: {len(visits_kept)}/{len(visits_full)} '
          f'(bands={allowed_bands}, programs={programs}, alt=[{alt_min},{alt_max}])')
    if len(visits_kept) == 0:
        raise RuntimeError('No visits pass the filters for this bin.')

    donut_df = load_kept_donuts(donuts_pq, visits_kept)
    print(f'  donuts: {len(donut_df)}')

    nZk = np.stack(donut_df[f'zk_{coord_sys}'].values).shape[1]
    noll_arr = (np.array(visits_kept['nollIndices'][0])
                if 'nollIndices' in visits_kept.colnames else None)
    iZs, iZidx = derive_noll_indices(nZk, noll_arr)
    iZs_arr = np.asarray(iZs, dtype=int)
    n_k, n_j = k_max - k_min + 1, len(iZs_arr)
    print(f'  pupil Noll j ({n_j}): {iZs};  focal k = {k_min}..{k_max}')

    # ---- OFC SVD (ofc_svd replaces the notebook inline build) ----
    svd = osv.build_ofc_svd(iZs, k_min, k_max, n_keep_spec, n_dof=n_dof_spec,
                            ofc_normalization_yaml=ofc_norm_yaml)
    U_eff, V, Sigma = svd.U_eff, svd.V, svd.Sigma
    kj_grid, n_keep_eff = svd.kj_grid, svd.n_keep_eff
    normalization_weights = svd.normalization_weights
    keep_idx = svd._keep()
    dof_labels, dof_units = svd.dof_labels()
    frac_2d = (U_eff ** 2).sum(axis=1).reshape(n_k, n_j)
    vmode_scale = osv.vmode_fwhm_scale(svd)
    label = f'nkeep_{n_keep_eff}'
    print(f'  SVD: U_eff={U_eff.shape}, n_dof={svd.n_dof}, n_keep_eff={n_keep_eff}')

    # ---- per-donut CCD-height Z4 ----
    Z4hgt = None
    if 4 in iZs:
        try:
            from lsst.obs.lsst import LsstCam
            from ccd_height import compute_ccd_heights, HEIGHT_TO_Z4_UM_PER_MM
            fac = (height_to_z4_factor if height_to_z4_factor is not None
                   else HEIGHT_TO_Z4_UM_PER_MM)
            hcols = compute_ccd_heights(
                donut_df, LsstCam.getCamera(), source=height_source,
                height_map_dir=height_map_dir, metrology_fits=height_map_fits,
                factor=fac)
            for k, v in hcols.items():
                donut_df[k] = v
            if 'Z4_height' in donut_df.columns:
                Z4hgt = np.asarray(donut_df['Z4_height'], dtype=float)
                print(f'  Z4_height: mean={np.nanmean(Z4hgt):+.4f} μm')
        except Exception as e:
            print(f'  CCD heights skipped: {type(e).__name__}: {e}')

    # ---- Path A build ----
    result = build_measured_intrinsic_uconstrained(
        donut_df, visits_kept, coord_sys, iZs, kj_grid=kj_grid, U_eff=U_eff,
        n_iter=n_iter, n_bins=n_bins, fp_radius_basis=fp_radius_basis,
        fp_radius_grid=fp_radius_grid, min_donuts=min_donuts,
        bad_fit_threshold=bad_fit_threshold)
    final = result['iter_results'][-1]
    xbins, ybins = result['xbins'], result['ybins']
    print(f'  Path A: {n_iter} iters, {len(final["fit_rows"])} visits')

    # ---- Z4 optical / height grids (per-donut subtraction before binning) ----
    z4_optical_grid = z4_height_ccs_grid = None
    if Z4hgt is not None and 4 in iZidx:
        gm = final.get('good_donut_mask', np.ones(len(donut_df), dtype=bool))
        txo = np.rad2deg(np.asarray(donut_df['thx_OCS'], dtype=float))
        tyo = np.rad2deg(np.asarray(donut_df['thy_OCS'], dtype=float))
        txc = np.rad2deg(np.asarray(donut_df['thx_CCS'], dtype=float))
        tyc = np.rad2deg(np.asarray(donut_df['thy_CCS'], dtype=float))
        z4col = iZidx[4]
        z4_opt = final['wfd_subtracted'][:, z4col] - Z4hgt
        z4_optical_grid = ibp.bin_single_focal(txo[gm], tyo[gm], z4_opt[gm],
                                               n_bins, fp_radius_grid)
        z4_height_ccs_grid = ibp.bin_single_focal(txc[gm], tyc[gm], Z4hgt[gm],
                                                  n_bins, fp_radius_grid)

    # ---- diagnostics PDFs ----
    _write_validation_pdf(out_dir / f'measured_intrinsic_{label}_pathA_validation.pdf',
                          result, donut_df, visits_kept, coord_sys, iZs, iZidx,
                          iZs_arr, k_min, k_max, kj_grid, U_eff, V, Sigma,
                          n_keep_eff, normalization_weights, vmode_scale,
                          frac_2d, n_k, n_j, n_iter, Z4hgt,
                          z4_optical_grid, z4_height_ccs_grid, xbins, ybins,
                          wfs_edge_cut_deg, n_radial_bins_azimuth, azimuth_bin_deg)
    _write_comparison_pdf(out_dir / f'measured_intrinsic_{label}_pathA_comparison.pdf',
                          result, iZs, coord_sys, xbins, ybins)
    _write_final_maps_pdf(out_dir / f'measured_intrinsic_{label}_pathA_final.pdf',
                          final['measured_grid'], iZs, xbins, ybins,
                          z4_optical_grid, z4_height_ccs_grid)

    # ---- DOF recovery + augment final fit rows, then save ----
    A_last = np.where(np.isfinite(ibp.stack_per_visit_coeffs(final['fit_rows_raw'], kj_grid)),
                      ibp.stack_per_visit_coeffs(final['fit_rows_raw'], kj_grid), 0.0) @ U_eff
    V_last = A_last / np.asarray(Sigma[keep_idx])[None, :]
    dof_last = svd.dof(A_last)
    W_raw = ibp.stack_per_visit_coeffs(final['fit_rows_raw'], kj_grid)
    W_fit = ibp.stack_per_visit_coeffs(final['fit_rows'], kj_grid)
    ibp.augment_fit_rows_with_modes(
        final['fit_rows'], A_last, V_last, dof_last, dof_labels,
        {}, V_fwhm=(V_last * vmode_scale[None, :] if vmode_scale is not None else None),
        W_resid=W_raw - W_fit, kj_list=list(kj_grid), W_raw=W_raw, W_corr=W_fit)

    save_dz_fits(final['fit_rows'], out_dir / 'dz_fits.parquet')

    extra = {'z4_optical_OCS': z4_optical_grid} if z4_optical_grid is not None else None
    tbl = assemble_intrinsic_table(
        grid=final['measured_grid'], iZs=iZs,
        xcent=result['xcent'], ycent=result['ycent'],
        coord_sys_grid=coord_sys, alt_coord_xform=None, extra_cols=extra)
    tbl.write(str(out_dir / 'intrinsic_grid.parquet'), format='parquet', overwrite=True)
    print(f'  wrote intrinsic_grid.parquet ({len(tbl)} bins) + dz_fits.parquet')

    # frozen config for provenance
    with open(out_dir / 'mi_config.yaml', 'w') as fh:
        yaml.safe_dump({'param_set': args.param_set, 'mi_name': args.mi_name,
                        'rot_min': rot_min, 'rot_max': rot_max,
                        'n_keep_eff': int(n_keep_eff), **mi}, fh, sort_keys=False)
    print('[build_intrinsic] done.')


# ----------------------------------------------------------------------
# PDF writers (port the notebook's validation / comparison / final cells)
# ----------------------------------------------------------------------
def _write_validation_pdf(path, result, donut_df, visits_kept, coord_sys, iZs,
                          iZidx, iZs_arr, k_min, k_max, kj_grid, U_eff, V, Sigma,
                          n_keep_eff, normalization_weights, vmode_scale,
                          frac_2d, n_k, n_j, n_iter, Z4hgt, z4_optical_grid,
                          z4_height_ccs_grid, xbins, ybins, wfs_edge_cut_deg,
                          n_radial_bins_azimuth, azimuth_bin_deg):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    kj_list = list(kj_grid)
    sig_keep = np.asarray(Sigma[:n_keep_eff], dtype=float)
    wfs_inner = ibp.resolve_wfs_inner_edge(None)
    with PdfPages(str(path)) as pdf:
        def emit(fig):
            if fig is not None:
                pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        A_modes, V_modes = {}, {}
        for it_idx, it in enumerate(result['iter_results']):
            W_raw = ibp.stack_per_visit_coeffs(it['fit_rows_raw'], kj_list)
            W_fit = ibp.stack_per_visit_coeffs(it['fit_rows'], kj_list)
            emit(ibp.plot_w_heatmap(W_raw, kj_list, n_k, n_j,
                                    f'Path A w_raw — iter {it_idx+1}', k_min=k_min))
            emit(ibp.plot_w_heatmap(W_fit, kj_list, n_k, n_j,
                                    f'Path A w_fit — iter {it_idx+1}', k_min=k_min))
            emit(ibp.plot_w_heatmap(W_raw - W_fit, kj_list, n_k, n_j,
                                    f'Path A w_residual — iter {it_idx+1}', k_min=k_min))
            A = np.where(np.isfinite(W_raw), W_raw, 0.0) @ U_eff
            A_modes[it_idx] = A
            V_modes[it_idx] = A / sig_keep[None, :]
        for it_idx, A in A_modes.items():
            emit(ibp._plot_modes_heatmap(A, f'Path A U-mode amps — iter {it_idx+1} (μm)'))
            emit(ibp._plot_modes_lines_all(A, n_keep_eff,
                 f'Path A all {n_keep_eff} U-mode amps — iter {it_idx+1}'))
        for it_idx in range(n_iter):
            emit(ibp.plot_per_kj_vs_visit_page(
                ibp.stack_per_visit_coeffs(result['iter_results'][it_idx]['fit_rows'], kj_list),
                kj_list, iZs_arr, k_min, k_max,
                title_root='Path A per-(k,j) w_fit vs visit',
                iter_label=f'iter {it_idx+1}'))
        # residual sigmas + example visit + sigma grids + azimuth
        residuals = ibp.compute_validation_residual(donut_df, result, coord_sys, iZs)
        sigmas = ibp.compute_per_visit_sigmas(donut_df, residuals,
                                              result['iter_results'][-1]['fit_rows'],
                                              list(iZs), iZidx)
        ex = ibp.example_visit_index(sigmas)
        emit(ibp.plot_example_visit_histograms(donut_df, residuals,
             result['iter_results'][-1]['fit_rows'], list(iZs), iZidx, ex))
        emit(ibp.plot_sigma_vs_visit_grid(sigmas, list(iZs), iZidx, visits_kept,
             which='sigma', title_root='Path A residual σ per j vs visit'))
        gm = result['iter_results'][-1].get('good_donut_mask',
                                            np.ones(len(donut_df), dtype=bool))
        txo = np.rad2deg(np.asarray(donut_df['thx_OCS'], dtype=float))
        tyo = np.rad2deg(np.asarray(donut_df['thy_OCS'], dtype=float))
        emit(ibp.plot_intrinsic_vs_azimuth(
            txo[gm], tyo[gm], result['iter_results'][-1]['wfd_subtracted'][gm],
            iZidx, list(iZs), wfs_inner, wfs_edge_cut_deg,
            n_radial_bins_azimuth, azimuth_bin_deg,
            title_root='Path A measured intrinsic vs azimuth (OCS)'))
        if z4_optical_grid is not None:
            emit(ibp.plot_z4_optical_page(
                result['iter_results'][-1]['measured_grid'].get(4),
                z4_height_ccs_grid, z4_optical_grid, xbins, ybins))
        # DOF recovery pages
        N_diag = np.asarray(normalization_weights, dtype=float)
        dof_per_iter = {}
        for it_idx, A in A_modes.items():
            dof = osv.recover_dof_per_visit(A, V, Sigma, N_diag, n_keep_eff)
            dof_per_iter[it_idx] = dof
            it = result['iter_results'][it_idx]
            for p in ibp.plot_dof_vs_visit_pages(
                    dof, osv.LABELS_50DOF, osv.DOF_UNITS_50, visits_kept,
                    [int(r['day_obs']) for r in it['fit_rows']],
                    [int(r['seq_num']) for r in it['fit_rows']],
                    title_root=f'Path A DOF vs visit — iter {it_idx+1}'):
                emit(p)
        emit(ibp.plot_dof_median_summary(dof_per_iter, osv.LABELS_50DOF,
             osv.DOF_UNITS_50, title='Path A DOF median per iteration'))
    print(f'  wrote {path.name}')


def _write_comparison_pdf(path, result, iZs, coord_sys, xbins, ybins):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    iter_grids = ibp._build_iter_grids_list(result)
    with PdfPages(str(path)) as pdf:
        for j in iZs:
            fig = ibp.plot_iter_progression_for_j(
                j, result['original_median'].get(j),
                [(lbl, g.get(j)) for lbl, g in iter_grids],
                result['tabulated_median'].get(j), xbins, ybins, coord_sys,
                path_tag=' — Path A')
            if fig is not None:
                pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    print(f'  wrote {path.name}')


def _write_final_maps_pdf(path, grid, iZs, xbins, ybins, z4_optical_grid,
                          z4_height_ccs_grid):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    panels = []
    for j in [int(x) for x in iZs]:
        if j == 4 and z4_optical_grid is not None:
            panels.append(('Z4 height (CCS)', z4_height_ccs_grid))
            panels.append(('Z4 optical (OCS)', z4_optical_grid))
        else:
            panels.append((f'Z{j}', grid.get(j)))
    per_page, ncols = 8, 4
    n_pages = (len(panels) + per_page - 1) // per_page
    with PdfPages(str(path)) as pdf:
        for pg in range(n_pages):
            chunk = panels[pg * per_page:(pg + 1) * per_page]
            nrows = (len(chunk) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows),
                                     layout='constrained')
            axes = np.atleast_1d(axes).ravel()
            for ax, (title, g) in zip(axes, chunk):
                if g is None:
                    ax.set_visible(False); continue
                finite = g[np.isfinite(g)]
                vlim = (np.nanpercentile(np.abs(finite), 95) if finite.size else 1.0) or 1.0
                pcm = ax.pcolormesh(xbins, ybins, g.T, cmap='RdBu_r',
                                    vmin=-vlim, vmax=vlim, shading='flat')
                ax.set_aspect('equal'); ax.set_title(title, fontsize=9)
                plt.colorbar(pcm, ax=ax, shrink=0.75)
            for ax in axes[len(chunk):]:
                ax.set_visible(False)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
    print(f'  wrote {path.name}')


if __name__ == '__main__':
    main()
