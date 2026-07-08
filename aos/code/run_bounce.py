#!/usr/bin/env python3
"""Bounce-test analysis (script port of study_bounce.ipynb), per (param_set,
mi_name) on the measured-intrinsic refit  output/<ps>/<mi>/fits.parquet.

FAM-triplet telescope-position bounce tests (e.g. BLOCK-T720 elevation,
BLOCK-T724 rotator): time-ordered paired-difference Δ (comparison − reference)
per Double-Zernike (k, j), OFC v-mode and physical DOF, with robust errors;
significance / pass heatmaps; DZ / v-mode / DOF vs ordinal-image pages;
per-night cross-scatter; and DOF night-vs-night scatter.  All analysis/plot
logic lives in code/bounce_lib.py (verbatim from the notebook).

Writes, under  output/<ps>/<mi>/ :
    plots/bounce_summary.pdf             Δ / significance / pass heatmaps + cross-scatter
    plots/bounce_dz_vs_ordinal.pdf       DZ_kj vs ordinal image (per bounce)
    plots/bounce_vmode_vs_ordinal.pdf    v-mode amplitude vs ordinal
    plots/bounce_dof_vs_ordinal.pdf      physical DOF vs ordinal (+ Trim sum if enabled)
    plots/bounce_dof_night_scatter.pdf   DOF night-A vs night-B 5-panel scatter
    plots/bounce_dof_night_values.pdf    FAM DOF median per night
    plots/bounce_5x5_camera_hexapod.pdf  5/5 Camera-hexapod-only v-mode + DOF plots
                                         (camera_hexapod_only bounces, e.g. rotator)
    plots/bounce_fwhm_metric.pdf         differential correctable-FWHM bar (before vs
                                         50/34 [vs 5/5]) per bounce
    bounce_kj_stats.parquet              long-format Δ table (combined + per night)
    bounce_fwhm_metric.parquet           correctable-FWHM metric per bounce comparison

Knobs (bounce specs, thresholds, n_dof/n_keep, add_dof_trim) come from
analysis_config.yaml (section ``bounce``).  RSP-only (DOF recovery via
lsst.ts.ofc; optional EFD/ConsDB Trim).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import QTable

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lsst.ts.intrinsic.wavefront import mi_config as mc
import bounce_lib as bl
from lsst.ts.intrinsic.wavefront.ofc_svd import LABELS_50DOF, DOF_UNITS_50

DEFAULT_BOUNCES = [
    {'name': 'T720_elevation', 'description': 'Elevation 40 - 70 deg, rotator ~ 0',
     'program': 'BLOCK-T720',
     'reference': {'label': 'Elev=70', 'alt_range': [67.0, 73.0], 'rotator_range': [-3.0, 3.0]},
     'comparisons': [{'label': 'Elev=40', 'alt_range': [37.0, 43.0], 'rotator_range': [-3.0, 3.0]}]},
    {'name': 'T724_rotator', 'description': 'Rotator 60 - 0 deg, elevation ~ 70',
     'program': 'BLOCK-T724',
     # rotator bounce: only the camera hexapod moves, so also evaluate a 5-DOF /
     # 5-v-mode (Camera-hexapod-only) correction alongside the full 50/34.
     'camera_hexapod_only': True,
     'reference': {'label': 'Rot=0', 'rotator_range': [-3.0, 3.0], 'alt_range': [67.0, 73.0]},
     'comparisons': [{'label': 'Rot=60', 'rotator_range': [57.0, 63.0], 'alt_range': [67.0, 73.0]}]},
]
# Camera-hexapod DOF indices in LABELS_50DOF (Cam_dz/dx/dy/rx/ry) for the 5/5 scheme.
CAM_HEX_DOF = [5, 6, 7, 8, 9]
DEFAULT = dict(
    fit_prefix='z1toz6', focal_k_range=[1, 2, 3, 4, 5, 6], pupil_j_range=None,
    bounces=DEFAULT_BOUNCES, n_dof=50, n_keep=34, ofc_normalization_yaml=None,
    heatmap_vlim_um=None, heatmap_cell_fontsize=7, sig_vlim=5.0,
    pass_nsigma_threshold=3.5, pass_delta_threshold_um=0.1,
    pass_sigma_only_threshold=5.0, cross_scatter_zoom_um=0.1,
    ordinal_j_per_page=7, night_min_visits=3,
    vmode_ncols=5, vmode_rows_per_page=7, dof_ncols=5, dof_rows_per_page=10,
    add_dof_trim=False, trim_efd_topic='lsst.sal.MTAOS.logevent_degreeOfFreedom',
    trim_consdb_url='http://consdb-pq.consdb:8080/consdb',
    trim_time_col='mjd', trim_mjd_scale='tai')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--analysis-config', default=None)
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--fits', default=None)
    ap.add_argument('--min-detectors', type=int, default=None,
                    help='Per-visit quality cut: keep visits with '
                         'n_detectors_with_min_donuts >= this (relaxes ONLY the '
                         'CCD-count cut; blur cut kept). fits.parquet now holds all '
                         'visits, so the cut is applied here. Default: None -> use '
                         'the precomputed visit_quality_pass (nd>=170).')
    args = ap.parse_args()

    cfg = {**DEFAULT, **mc.analysis_section(
        'bounce', args.param_set, args.mi_name,
        config_path=(Path(args.analysis_config) if args.analysis_config else None))}
    prefix = cfg['fit_prefix']
    bounces = cfg['bounces']

    base = Path(args.output_root) / args.param_set / args.mi_name
    fits_path = Path(args.fits) if args.fits else base / 'fits.parquet'
    out_dir = base / 'plots'; out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[bounce] {fits_path}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        from lsst.ts.intrinsic.wavefront.intrinsics_lib import markers_legend_figure
        _marker_ok = True
    except Exception:
        _marker_ok = False

    # ---- load + drop bad fits + resolve j/k lists (cell 10) ----
    fit_table = QTable.read(str(fits_path))
    for bf in (f'{prefix}_bad_fit', 'bad_fit'):
        if bf in fit_table.colnames:
            bad = np.asarray(fit_table[bf]).astype(bool)
            if int(bad.sum()):
                print(f'  dropping {int(bad.sum())} bad-flagged visits ({bf})')
            fit_table = fit_table[~bad]
            break

    # ---- per-visit quality cut (fits.parquet now holds ALL visits) ----
    # --min-detectors relaxes ONLY the CCD-count cut (bounce uses 160 to recover
    # marginal low-CCD elevation-40 points); without it, fall back to the
    # precomputed visit_quality_pass (nd>=170), preserving the old selection.
    if args.min_detectors is not None and \
            'n_detectors_with_min_donuts' in fit_table.colnames:
        from lsst.ts.intrinsic.wavefront.intrinsics_lib import quality_visit_mask
        keep = np.asarray(quality_visit_mask(
            fit_table, min_detectors_per_visit=args.min_detectors, verbose=False),
            dtype=bool)
        print(f'  quality cut (min_detectors={args.min_detectors}): '
              f'{int(keep.sum())}/{len(fit_table)} visits kept')
        fit_table = fit_table[keep]
    elif 'visit_quality_pass' in fit_table.colnames:
        keep = np.asarray(fit_table['visit_quality_pass'], dtype=bool)
        print(f'  quality cut (visit_quality_pass): '
              f'{int(keep.sum())}/{len(fit_table)} visits kept')
        fit_table = fit_table[keep]
    if cfg['pupil_j_range'] is None:
        if 'nollIndices' not in fit_table.colnames:
            raise ValueError('no nollIndices column; set pupil_j_range in config')
        iZs = [int(j) for j in np.asarray(fit_table['nollIndices'][0]).tolist()]
    else:
        iZs = list(cfg['pupil_j_range'])
    k_list = list(cfg['focal_k_range'])
    print(f'  {len(fit_table)} visits; pupil j={iZs}; focal k={k_list}')

    # ---- OFC SVD + per-visit v-mode / DOF projection (cell 12) ----
    C_all = DOF_all = None
    svd = svd5 = C5_all = DOF5_all = None
    vmode_labels = vmode5_labels = []
    try:
        from lsst.ts.intrinsic.wavefront.ofc_svd import build_ofc_svd, project_dz_table
        svd = build_ofc_svd(iZs, int(min(k_list)), int(max(k_list)),
                            cfg['n_keep'], n_dof=cfg['n_dof'],
                            ofc_normalization_yaml=cfg['ofc_normalization_yaml'])
        _svd_ok = True
        vmode_labels = svd.vmode_labels
        C_all, DOF_all, _A, _W = project_dz_table(fit_table, prefix, svd)
        print(f'  OFC SVD: n_keep={svd.n_keep_eff}, n_dof={svd.n_dof}; '
              f'projected C_all{C_all.shape}, DOF_all{DOF_all.shape}')
        # 5/5 Camera-hexapod-only SVD (shares kj_grid) for the rotator bounce.
        svd5 = build_ofc_svd(iZs, int(min(k_list)), int(max(k_list)),
                             5, n_dof=CAM_HEX_DOF,
                             ofc_normalization_yaml=cfg['ofc_normalization_yaml'])
        vmode5_labels = svd5.vmode_labels
        C5_all, DOF5_all, _A5, _W5 = project_dz_table(fit_table, prefix, svd5)
        print(f'  5/5 Camera-hex SVD: n_keep={svd5.n_keep_eff}, n_dof={svd5.n_dof}')
    except Exception as e:
        print(f'  (OFC SVD unavailable [{type(e).__name__}: {e}]; '
              f'v-mode/DOF sections skipped)')
        _svd_ok = False

    # ---- differential correctable-FWHM metric tooling (RSS of the median Δ-DZ
    # aberrations -> PSF FWHM; residual after each scheme's OFC projection) ----
    fwhm_conv = fwhm_grid = None
    try:
        from lsst.ts.wep.utils import convertZernikesToPsfWidth as fwhm_conv
        import aos_fwhm as _afw
        fwhm_grid = _afw.fp_grid()
    except Exception as e:
        print(f'  (correctable-FWHM metric disabled: {type(e).__name__}: {e})')
    fwhm_rows = []

    # ---- optional AOS Trim (aggregatedDoF) (cell 14) ----
    DOFSUM_all = TRIM_segment = None
    if cfg['add_dof_trim'] and _svd_ok and DOF_all is not None:
        try:
            from aos_trim import fetch_aggregated_dof_for_visits
            TRIM_all, info = fetch_aggregated_dof_for_visits(
                fit_table, consdb_url=cfg['trim_consdb_url'],
                topic=cfg['trim_efd_topic'], n_dof=DOF_all.shape[1],
                mjd_fallback_col=cfg['trim_time_col'], mjd_scale=cfg['trim_mjd_scale'])
            DOFSUM_all = DOF_all + TRIM_all
            TRIM_segment = pd.factorize(info['event_id'])[0]
            print(f'  AOS Trim: {info["n_dof"]}/{len(fit_table)} visits resolved')
        except Exception as e:
            print(f'  (AOS Trim unavailable [{type(e).__name__}: {e}])')

    # ---- bounce analysis (cell 16) ----
    bounce_results, long_dfs = {}, []
    for b in bounces:
        name = b['name']
        combined = bl.run_bounce(fit_table, b, prefix, k_list, iZs,
                                 day_obs=None, trim_segment=TRIM_segment)
        nights = bl.bounce_nights(fit_table, b, prefix, k_list, iZs,
                                  min_visits=cfg['night_min_visits'],
                                  trim_segment=TRIM_segment)
        br = {'description': b.get('description', ''),
              'reference_label': b['reference']['label'],
              'reference_stats': combined['ref_stats'],
              'reference_n': combined['ref_n'], 'comparisons': {}}
        print(f'  === {name} ===  ref "{b["reference"]["label"]}" '
              f'n={combined["ref_n"]}; nights={sorted(nights.keys())}')
        for comp in b['comparisons']:
            label = comp['label']
            cblock = combined['comparisons'][label]
            pairs_all = cblock['pairs']
            deltas_by_night = {d: nights[d]['comparisons'][label]['deltas'] for d in nights}
            cam_only = bool(b.get('camera_hexapod_only', False))
            vmode_deltas = dof_deltas = None
            vmode_deltas_by_night = dof_deltas_by_night = {}
            vmode5_deltas = dof5_deltas = None
            vmode5_deltas_by_night = dof5_deltas_by_night = {}
            if _svd_ok and C_all is not None:
                vmode_deltas = bl.paired_deltas_matrix(C_all, pairs_all)
                dof_deltas = bl.paired_deltas_matrix(DOF_all, pairs_all)
                vmode_deltas_by_night = {
                    d: bl.paired_deltas_matrix(C_all, nights[d]['comparisons'][label]['pairs'])
                    for d in nights}
                dof_deltas_by_night = {
                    d: bl.paired_deltas_matrix(DOF_all, nights[d]['comparisons'][label]['pairs'])
                    for d in nights}
                if cam_only and C5_all is not None:      # 5/5 Camera-hexapod-only
                    # DOF5_all has 5 cols (Cam hex); key them by their global DOF
                    # indices so the DOF plots' 50-DOF panel layout places them right.
                    vmode5_deltas = bl.paired_deltas_matrix(C5_all, pairs_all)
                    dof5_deltas = bl.paired_deltas_matrix(DOF5_all, pairs_all, keys=CAM_HEX_DOF)
                    vmode5_deltas_by_night = {
                        d: bl.paired_deltas_matrix(C5_all, nights[d]['comparisons'][label]['pairs'])
                        for d in nights}
                    dof5_deltas_by_night = {
                        d: bl.paired_deltas_matrix(DOF5_all, nights[d]['comparisons'][label]['pairs'],
                                                   keys=CAM_HEX_DOF)
                        for d in nights}
            br['comparisons'][label] = {
                'comp_stats': cblock['comp_stats'], 'comp_n': cblock['comp_n'],
                'deltas': cblock['deltas'], 'pairs': pairs_all,
                'deltas_by_night': deltas_by_night,
                'vmode_deltas': vmode_deltas, 'dof_deltas': dof_deltas,
                'vmode_deltas_by_night': vmode_deltas_by_night,
                'dof_deltas_by_night': dof_deltas_by_night,
                'cam_only': cam_only,
                'vmode5_deltas': vmode5_deltas, 'dof5_deltas': dof5_deltas,
                'vmode5_deltas_by_night': vmode5_deltas_by_night,
                'dof5_deltas_by_night': dof5_deltas_by_night}
            print(f'      comp "{label}": n={cblock["comp_n"]}, '
                  f'{len(pairs_all)} pairs')

            # ---- differential correctable-FWHM metric ----
            # median Δ-DZ (comp-ref) over pairs -> FWHM before; residual after the
            # 50/34 (and, for the rotator, 5/5) OFC projection -> FWHM after.
            if fwhm_conv is not None and _svd_ok and _W is not None and len(pairs_all):
                pd_dz = bl.paired_deltas_matrix(_W, pairs_all)
                med_dW = np.array([pd_dz[i]['delta'] for i in range(_W.shape[1])], float)
                row = {'bounce': name, 'comparison': label, 'n_pairs': len(pairs_all),
                       'fwhm_before': _afw.fp_fwhm(svd, iZs, med_dW, fwhm_grid, fwhm_conv),
                       'fwhm_after_50_34': _afw.fp_fwhm(
                           svd, iZs, _afw.residual_dW(svd, med_dW), fwhm_grid, fwhm_conv)}
                if cam_only and svd5 is not None:
                    row['fwhm_after_5_5'] = _afw.fp_fwhm(
                        svd5, iZs, _afw.residual_dW(svd5, med_dW), fwhm_grid, fwhm_conv)
                fwhm_rows.append(row)
                _extra = (f", 5/5={row['fwhm_after_5_5']:.4f}" if 'fwhm_after_5_5' in row else "")
                print(f"        correctable FWHM [arcsec]: before={row['fwhm_before']:.4f}, "
                      f"after 50/34={row['fwhm_after_50_34']:.4f}{_extra}")
            long_dfs.append(bl.to_long_df(
                cblock['deltas'], name, br['reference_label'], label,
                combined['ref_stats'], cblock['comp_stats'], night='all'))
            for d in deltas_by_night:
                long_dfs.append(bl.to_long_df(
                    deltas_by_night[d], name, br['reference_label'], label,
                    nights[d]['ref_stats'],
                    nights[d]['comparisons'][label]['comp_stats'], night=d))
        bounce_results[name] = br
    df_kj = pd.concat(long_dfs, ignore_index=True) if long_dfs else pd.DataFrame()

    # ---- DZ vs ordinal (cell 18) ----
    with PdfPages(str(out_dir / 'bounce_dz_vs_ordinal.pdf')) as pdf:
        if _marker_ok:
            leg = markers_legend_figure(show_iter_distinction=False)
            pdf.savefig(leg, bbox_inches='tight'); plt.close(leg)
        for b in bounces:
            ftb = fit_table[bl.bounce_program_mask(fit_table, b)]
            if len(ftb) == 0:
                continue
            for f in bl.plot_dz_vs_ordinal_pages(ftb, prefix, k_list, iZs,
                                                 j_per_page=cfg['ordinal_j_per_page'],
                                                 title_prefix=f'{b["name"]}: '):
                pdf.savefig(f, bbox_inches='tight'); plt.close(f)

    # ---- v-mode + DOF vs ordinal (cell 20) ----
    if _svd_ok and C_all is not None:
        with PdfPages(str(out_dir / 'bounce_vmode_vs_ordinal.pdf')) as pdf:
            if _marker_ok:
                leg = markers_legend_figure(show_iter_distinction=False)
                pdf.savefig(leg, bbox_inches='tight'); plt.close(leg)
            for b in bounces:
                m = bl.bounce_program_mask(fit_table, b)
                if int(m.sum()) == 0:
                    continue
                for f in bl.plot_values_vs_ordinal_pages(
                        fit_table[m], C_all[m], vmode_labels, units=None,
                        title_root=f'{b["name"]}: OFC v-mode amplitude c_i',
                        ncols=cfg['vmode_ncols'], rows_per_page=cfg['vmode_rows_per_page']):
                    pdf.savefig(f, bbox_inches='tight'); plt.close(f)
        with PdfPages(str(out_dir / 'bounce_dof_vs_ordinal.pdf')) as pdf:
            if _marker_ok:
                leg = markers_legend_figure(show_iter_distinction=False)
                pdf.savefig(leg, bbox_inches='tight'); plt.close(leg)
            for b in bounces:
                m = bl.bounce_program_mask(fit_table, b)
                if int(m.sum()) == 0:
                    continue
                for f in bl.plot_values_vs_ordinal_pages(
                        fit_table[m], DOF_all[m], LABELS_50DOF, units=DOF_UNITS_50,
                        title_root=f'{b["name"]}: Physical DOF (FAM analysis)',
                        ncols=cfg['dof_ncols'], rows_per_page=cfg['dof_rows_per_page']):
                    pdf.savefig(f, bbox_inches='tight'); plt.close(f)
            if DOFSUM_all is not None:
                for b in bounces:
                    m = bl.bounce_program_mask(fit_table, b)
                    if int(m.sum()) == 0:
                        continue
                    for f in bl.plot_values_vs_ordinal_pages(
                            fit_table[m], DOFSUM_all[m], LABELS_50DOF, units=DOF_UNITS_50,
                            title_root=f'{b["name"]}: Physical DOF + AOS Trim',
                            ncols=cfg['dof_ncols'], rows_per_page=cfg['dof_rows_per_page']):
                        pdf.savefig(f, bbox_inches='tight'); plt.close(f)

    # ---- summary heatmaps + cross-scatter (cell 22) ----
    with PdfPages(str(out_dir / 'bounce_summary.pdf')) as pdf:
        for name, br in bounce_results.items():
            for clabel, cb in br['comparisons'].items():
                deltas = cb['deltas']
                pdf.savefig(bl.plot_kj_heatmap(
                    deltas, k_list, iZs, value_key='delta', err_key='err',
                    title=f'{name}: Δ DZ_kj = {clabel} − {br["reference_label"]}\n'
                          f'{br["description"]} (n_ref={br["reference_n"]}, n_comp={cb["comp_n"]})',
                    cbar_label='Δ DZ [μm]', cmap='RdBu_r', vlim=cfg['heatmap_vlim_um'],
                    value_fmt='{:+.3f}', err_fmt='±{:.3f}',
                    cell_fontsize=cfg['heatmap_cell_fontsize']), bbox_inches='tight')
                pdf.savefig(bl.plot_kj_heatmap(
                    deltas, k_list, iZs, value_key='sig', err_key=None,
                    title=f'{name}: significance = Δ / σ (capped ±{cfg["sig_vlim"]:g}σ)',
                    cbar_label='Δ / σ', cmap='RdBu_r', vlim=cfg['sig_vlim'],
                    value_fmt='{:+.1f}', err_fmt='',
                    cell_fontsize=cfg['heatmap_cell_fontsize']), bbox_inches='tight')
                pdf.savefig(bl.plot_kj_pass_heatmap(
                    deltas, k_list, iZs, nsigma_threshold=cfg['pass_nsigma_threshold'],
                    delta_threshold_um=cfg['pass_delta_threshold_um'],
                    sigma_only_threshold=cfg['pass_sigma_only_threshold'],
                    title=f'{name}: significant Δ DZ_kj cells (all nights)',
                    cell_fontsize=cfg['heatmap_cell_fontsize']), bbox_inches='tight')
                plt.close('all')
                dbn = cb.get('deltas_by_night', {})
                passing = bl.passing_terms(deltas, cfg['pass_delta_threshold_um'],
                                           cfg['pass_nsigma_threshold'],
                                           sigma_only_th=cfg['pass_sigma_only_threshold'])
                for _d in dbn.values():
                    passing |= bl.passing_terms(_d, cfg['pass_delta_threshold_um'],
                                                cfg['pass_nsigma_threshold'],
                                                sigma_only_th=cfg['pass_sigma_only_threshold'])
                for nt in sorted(dbn):
                    pdf.savefig(bl.plot_kj_pass_heatmap(
                        dbn[nt], k_list, iZs, nsigma_threshold=cfg['pass_nsigma_threshold'],
                        delta_threshold_um=cfg['pass_delta_threshold_um'],
                        sigma_only_threshold=cfg['pass_sigma_only_threshold'],
                        title=f'{name}: significant Δ DZ_kj cells — night {nt}',
                        cell_fontsize=cfg['heatmap_cell_fontsize']), bbox_inches='tight')
                    plt.close('all')
                for f in bl.plot_night_cross_scatter(
                        dbn, sorted(passing),
                        title_root=f'{name}: per-night Δ DZ_kj cross-comparison '
                                   f'({clabel} − {br["reference_label"]})',
                        zoom_lim_um=cfg['cross_scatter_zoom_um']) or []:
                    pdf.savefig(f, bbox_inches='tight'); plt.close(f)

    # ---- DOF night-vs-night scatter (cell 24) ----
    if _svd_ok and C_all is not None:
        with PdfPages(str(out_dir / 'bounce_dof_night_scatter.pdf')) as pdf:
            for name, br in bounce_results.items():
                for clabel, cb in br['comparisons'].items():
                    dbn = cb.get('dof_deltas_by_night', {})
                    if len(dbn) < 2:
                        continue
                    for f in bl.plot_dof_night_scatter(
                            dbn, LABELS_50DOF, units=DOF_UNITS_50,
                            title_root=f'{name}: DOF Δ ({clabel} - {br["reference_label"]})'):
                        pdf.savefig(f, bbox_inches='tight'); plt.close(f)

    # ---- DOF per-night median values (cell 26) ----
    if _svd_ok and DOF_all is not None:
        dobs = np.asarray(fit_table['day_obs']).astype(int)
        with PdfPages(str(out_dir / 'bounce_dof_night_values.pdf')) as pdf:
            for b in bounces:
                m = bl.bounce_program_mask(fit_table, b)
                if int(m.sum()) == 0:
                    continue
                dof_by_night = {int(nt): DOF_all[m & (dobs == nt)]
                                for nt in sorted(set(dobs[m].tolist()))
                                if int((m & (dobs == nt)).sum()) > 0}
                if dof_by_night:
                    pdf.savefig(bl.plot_dof_per_night_summary(
                        dof_by_night, LABELS_50DOF, DOF_UNITS_50,
                        title=f'{b["name"]}: FAM DOF median per night '
                              f'({len(dof_by_night)} nights)'), bbox_inches='tight')
                    plt.close('all')

    # ---- 5/5 Camera-hexapod-only plots (camera_hexapod_only bounces, e.g. rotator) ----
    if _svd_ok and C5_all is not None:
        cam_labels = [LABELS_50DOF[i] for i in CAM_HEX_DOF]
        cam_units = [DOF_UNITS_50[i] for i in CAM_HEX_DOF]
        cam_bounces = [b for b in bounces if b.get('camera_hexapod_only')]
        if cam_bounces:
            dobs = np.asarray(fit_table['day_obs']).astype(int)
            with PdfPages(str(out_dir / 'bounce_5x5_camera_hexapod.pdf')) as pdf:
                if _marker_ok:
                    leg = markers_legend_figure(show_iter_distinction=False)
                    pdf.savefig(leg, bbox_inches='tight'); plt.close(leg)
                for b in cam_bounces:
                    name = b['name']; br = bounce_results[name]
                    m = bl.bounce_program_mask(fit_table, b)
                    if int(m.sum()) == 0:
                        continue
                    for f in bl.plot_values_vs_ordinal_pages(          # (1) v-mode vs ordinal
                            fit_table[m], C5_all[m], vmode5_labels, units=None,
                            title_root=f'{name} [5/5 Cam-hex]: OFC v-mode amplitude c_i',
                            ncols=cfg['vmode_ncols'], rows_per_page=cfg['vmode_rows_per_page']):
                        pdf.savefig(f, bbox_inches='tight'); plt.close(f)
                    for f in bl.plot_values_vs_ordinal_pages(          # (2) DOF vs ordinal
                            fit_table[m], DOF5_all[m], cam_labels, units=cam_units,
                            title_root=f'{name} [5/5 Cam-hex]: Camera-hexapod DOF',
                            ncols=cfg['dof_ncols'], rows_per_page=cfg['dof_rows_per_page']):
                        pdf.savefig(f, bbox_inches='tight'); plt.close(f)
                    for clabel, cb in br['comparisons'].items():       # (3) DOF night-vs-night scatter
                        dbn5 = cb.get('dof5_deltas_by_night', {})
                        if len(dbn5) >= 2:
                            for f in bl.plot_dof_night_scatter(
                                    dbn5, LABELS_50DOF, units=DOF_UNITS_50,
                                    title_root=f'{name} [5/5]: Cam-hex DOF Δ '
                                               f'({clabel} - {br["reference_label"]})'):
                                pdf.savefig(f, bbox_inches='tight'); plt.close(f)
                    dof50_by_night = {}                                # (4) DOF per-night median
                    for nt in sorted(set(dobs[m].tolist())):
                        sel = m & (dobs == nt)
                        if int(sel.sum()) == 0:
                            continue
                        full = np.full((int(sel.sum()), len(LABELS_50DOF)), np.nan)
                        full[:, CAM_HEX_DOF] = DOF5_all[sel]
                        dof50_by_night[int(nt)] = full
                    if dof50_by_night:
                        pdf.savefig(bl.plot_dof_per_night_summary(
                            dof50_by_night, LABELS_50DOF, DOF_UNITS_50,
                            title=f'{name} [5/5 Cam-hex]: DOF median per night'),
                            bbox_inches='tight')
                        plt.close('all')
            print('  wrote bounce_5x5_camera_hexapod.pdf')

    # ---- differential correctable-FWHM metric ----
    if fwhm_rows:
        fdf = pd.DataFrame(fwhm_rows)
        fdf.to_parquet(out_dir.parent / 'bounce_fwhm_metric.parquet')
        print('  correctable-FWHM metric [arcsec, median over focal plane]:')
        print('   ' + fdf.to_string(index=False).replace('\n', '\n   '))
        bar_cols = [c for c in ('fwhm_before', 'fwhm_after_50_34', 'fwhm_after_5_5')
                    if c in fdf.columns]
        lab = {'fwhm_before': 'no correction', 'fwhm_after_50_34': '50/34',
               'fwhm_after_5_5': '5/5 Cam-hex'}
        with PdfPages(str(out_dir / 'bounce_fwhm_metric.pdf')) as pdf:
            fig, ax = plt.subplots(figsize=(1.8 * len(fdf) + 3, 5), constrained_layout=True)
            x = np.arange(len(fdf)); w = 0.8 / max(len(bar_cols), 1)
            for i, c in enumerate(bar_cols):
                ax.bar(x + i * w, fdf[c].to_numpy(float), w, label=lab[c])
            ax.set_xticks(x + w * (len(bar_cols) - 1) / 2)
            ax.set_xticklabels([f'{r.bounce}\n{r.comparison}' for r in fdf.itertuples()],
                               fontsize=8)
            ax.set_ylabel('differential FWHM [arcsec]  (median over FP)')
            ax.set_title('Bounce optical-state change — correctable FWHM\n'
                         '(RSS of median Δ-DZ aberrations; residual after OFC projection)')
            ax.legend(); ax.grid(axis='y', alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print('  wrote bounce_fwhm_metric.pdf + .parquet')

    # ---- long-format table (cell 28) ----
    df_kj.to_parquet(out_dir.parent / 'bounce_kj_stats.parquet')
    print(f'  wrote bounce_*.pdf + bounce_kj_stats.parquet ({len(df_kj)} rows)')
    print('[bounce] done.')


if __name__ == '__main__':
    main()
