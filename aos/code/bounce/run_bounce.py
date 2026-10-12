#!/usr/bin/env python3
"""Bounce-test figures, per (param_set, mi_name) on the measured-intrinsic refit
output/miw/<P>_<M>/fits.parquet.

The STUDY half of the bounce analysis.  FAM-triplet telescope-position bounce
tests (e.g. BLOCK-T720 elevation, BLOCK-T724 rotator): time-ordered
paired-difference Δ (comparison − reference) per Double-Zernike (k, j), OFC
v-mode and physical DOF, with robust errors; significance / pass heatmaps; DZ /
v-mode / DOF vs ordinal-image pages; per-night cross-scatter; and DOF
night-vs-night scatter.

The reduction is `rubinwork.bounce.analysis.run_analysis`, a library, and the
four tables it produces are the `bounce_tables` PRODUCT
(`rubinwork.products.bounce_tables`).  This script calls the same analysis and
renders only the nine PDFs; it writes no table.  Why both halves run the
analysis rather than this one reading the product back: three of these pages
plot the per-visit arrays `C_all`, `DOF_all` and `DOF5_all`, which no table
carries.  Aaron's call, 2026-10-12; see the product's `__init__` for the full
reasoning and the later option that would remove the duplicate run.  The
plotters stay in bounce_lib.py.

The recovered DOF are reported two ways: the default truncated 50-DOF /
34-v-mode recovery, and Range-Bounded Recovery (RBR), which adds a per-DOF
penalty that grows steeply as a recovered amplitude approaches the range the
telescope can physically apply.  The default recovery exceeds that range on
these bounce legs by up to a factor of 11.27 (dimensionless, recovered
amplitude over allowed range); RBR bounds it at a small cost in corrected
FWHM.  The method is derived in the smatrix `regularized_inversion` study and
imported from there.

Writes, under the directory given by --out-dir (the pipeline passes
output/bounce/<P>_<M>/) :
    bounce_summary.pdf             Δ / significance / pass heatmaps + cross-scatter
    bounce_dz_vs_ordinal.pdf       DZ_kj vs ordinal image (per bounce)
    bounce_vmode_vs_ordinal.pdf    v-mode amplitude vs ordinal
    bounce_dof_vs_ordinal.pdf      physical DOF vs ordinal (+ Trim sum if enabled)
    bounce_dof_night_scatter.pdf   DOF night-A vs night-B 5-panel scatter
    bounce_dof_night_values.pdf    paired-Δ DOF vs B-set position, default
                                   recovery with the RBR overlay and the
                                   allowed-range band
    bounce_5x5_camera_hexapod.pdf  5/5 Camera-hexapod-only v-mode + DOF plots
                                   (camera_hexapod_only bounces, e.g. rotator)
    bounce_fwhm_metric.pdf         differential correctable-FWHM bar (before vs
                                   50/34 [vs 5/5] [vs RBR]) per bounce
    bounce_fwhm_vs_bvalue.pdf      correctable FWHM vs B-set elevation or
                                   rotator angle — uncorrected, default
                                   50/34, and RBR — one point per (night, leg)

Knobs (bounce specs, thresholds, n_dof/n_keep, add_dof_trim) come from
analysis_config.yaml (section ``bounce``).  RSP-only (DOF recovery via
lsst.ts.ofc; optional EFD/ConsDB Trim).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))          # same-study siblings
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # aos/code (shared + other studies)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))      # repo root (common/)

import bounce_lib as bl
from lsst.ts.intrinsic.wavefront.ofc_svd import LABELS_50DOF, DOF_UNITS_50
from rubinwork.bounce import analysis as ba
from rubinwork.bounce.legs import CAM_HEX_DOF, DOF22


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--mi-name', required=True)
    ap.add_argument('--analysis-config', default=None)
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--fits', default=None)
    # The caller (the Snakefile) owns the output layout and may name directories
    # differently from the param_set / mi_name keys, so it supplies the path
    # directly rather than having it derived from those keys.
    ap.add_argument('--out-dir', default=None,
                    help='directory to write the PDFs into '
                         '(default: output/<ps>/<mi>/bounce)')
    ap.add_argument('--min-detectors', type=int, default=None,
                    help='Per-visit quality cut: keep visits with '
                         'n_detectors_with_min_donuts >= this (relaxes ONLY the '
                         'CCD-count cut; blur cut kept). fits.parquet now holds all '
                         'visits, so the cut is applied here. Default: None -> use '
                         'the precomputed visit_quality_pass (nd>=170).')
    ap.add_argument('--rbr-kappa', type=float, default=None,
                    help='Range-Bounded Recovery: ratio |d_j|/r_j '
                         '(dimensionless, recovered amplitude over allowed '
                         'range) at which the penalty reaches unit weight. '
                         'Default from config (4.0).')
    ap.add_argument('--rbr-power', type=int, default=None,
                    help='Range-Bounded Recovery: penalty exponent; the penalty '
                         'goes as the ratio to the 2*power. Default from '
                         'config (3).')
    ap.add_argument('--no-rbr', action='store_true',
                    help='Skip Range-Bounded Recovery and report only the '
                         'default truncated recovery.')
    args = ap.parse_args()

    cfg = bl.resolve_config(args.param_set, args.mi_name,
                            analysis_config=args.analysis_config,
                            rbr_kappa=args.rbr_kappa, rbr_power=args.rbr_power,
                            no_rbr=args.no_rbr)
    prefix = cfg['fit_prefix']
    bounces = cfg['bounces']

    base = Path(args.output_root) / args.param_set / args.mi_name
    fits_path = Path(args.fits) if args.fits else base / 'fits.parquet'
    out_dir = Path(args.out_dir) if args.out_dir else base / 'bounce'
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    try:
        from lsst.ts.intrinsic.wavefront.intrinsics_lib import markers_legend_figure
        _marker_ok = True
    except Exception:
        _marker_ok = False

    # The reduction, shared with the bounce_tables product builder.
    result = ba.run_analysis(fits_path, cfg, min_detectors=args.min_detectors)
    fit_table = result.fit_table
    bounce_results = result.bounce_results
    iZs, k_list = result.iZs, result.k_list
    C_all, DOF_all = result.C_all, result.DOF_all
    C5_all, DOF5_all = result.C5_all, result.DOF5_all
    DOFSUM_all = result.DOFSUM_all
    dof_ranges = result.dof_ranges
    _svd_ok = result.svd_ok
    oic_ok = result.oic_ok
    vmode_labels, vmode5_labels = result.vmode_labels, result.vmode5_labels
    fwhm_rows, fwhm_bvalue_rows = result.fwhm_rows, result.fwhm_bvalue_rows

    # ---- DZ vs ordinal (cell 18) ----
    with PdfPages(str(out_dir / 'bounce_dz_vs_ordinal.pdf')) as pdf:
        if _marker_ok:
            leg = markers_legend_figure(show_iter_distinction=False)
            pdf.savefig(leg, bbox_inches='tight'); plt.close(leg)
        # A/B position table: what elevation/rotator each night's reference (A)
        # and throw (B) sets actually sat at, and whether every contiguous
        # block resolved to a grid position.
        for b in bounces:
            ab_rows = bl.ab_position_table(
                fit_table, b,
                elev_halfwidth_deg=cfg.get('ab_elev_halfwidth_deg'),
                rot_halfwidth_deg=cfg.get('ab_rot_halfwidth_deg'))
            for f in bl.plot_ab_position_table(
                    ab_rows, title=f'{b["name"]}: A/B bounce positions'):
                pdf.savefig(f, bbox_inches='tight'); plt.close(f)
        for b in bounces:
            ftb = fit_table[bl.bounce_program_mask(fit_table, b)]
            if len(ftb) == 0:
                continue
            for f in bl.plot_dz_vs_ordinal_pages(ftb, prefix, k_list, iZs,
                                                 j_per_page=cfg['ordinal_j_per_page'],
                                                 title_prefix=f'{b["name"]}: ',
                                                 elev_halfwidth_deg=cfg.get('ab_elev_halfwidth_deg')):
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
                        ncols=cfg['vmode_ncols'], rows_per_page=cfg['vmode_rows_per_page'],
                        elev_halfwidth_deg=cfg.get('ab_elev_halfwidth_deg')):
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
                        ncols=cfg['dof_ncols'], rows_per_page=cfg['dof_rows_per_page'],
                        elev_halfwidth_deg=cfg.get('ab_elev_halfwidth_deg')):
                    pdf.savefig(f, bbox_inches='tight'); plt.close(f)
            if DOFSUM_all is not None:
                for b in bounces:
                    m = bl.bounce_program_mask(fit_table, b)
                    if int(m.sum()) == 0:
                        continue
                    for f in bl.plot_values_vs_ordinal_pages(
                            fit_table[m], DOFSUM_all[m], LABELS_50DOF, units=DOF_UNITS_50,
                            title_root=f'{b["name"]}: Physical DOF + AOS Trim',
                            ncols=cfg['dof_ncols'], rows_per_page=cfg['dof_rows_per_page'],
                        elev_halfwidth_deg=cfg.get('ab_elev_halfwidth_deg')):
                        pdf.savefig(f, bbox_inches='tight'); plt.close(f)

    # ---- summary: per-night Δ DZ_kj cross-comparison (cell 22) ----
    # The per-(k, j) Δ and significance heatmaps are dropped: the same numbers
    # are in bounce_kj_stats.parquet, and the night-vs-night cross-scatter is
    # the product that actually answers the repeatability question.  It is
    # drawn for every comparison leg that has 2 or more qualifying nights,
    # wide and zoomed.
    with PdfPages(str(out_dir / 'bounce_summary.pdf')) as pdf:
        cov = bl.leg_night_coverage(bounce_results)
        f = bl.plot_leg_night_coverage(
            cov, title='Nights with results per bounce leg (B set)')
        if f is not None:
            pdf.savefig(f, bbox_inches='tight'); plt.close(f)
        for name, br in bounce_results.items():
            for clabel, cb in br['comparisons'].items():
                dbn = cb.get('deltas_by_night', {})
                if len(dbn) < 2:
                    print(f'  (bounce_summary: {name} {clabel} has '
                          f'{len(dbn)} night(s) — no cross-scatter page)')
                    continue
                passing = bl.passing_terms(
                    cb['deltas'], cfg['pass_delta_threshold_um'],
                    cfg['pass_nsigma_threshold'],
                    sigma_only_th=cfg['pass_sigma_only_threshold'])
                for _d in dbn.values():
                    passing |= bl.passing_terms(_d, cfg['pass_delta_threshold_um'],
                                                cfg['pass_nsigma_threshold'],
                                                sigma_only_th=cfg['pass_sigma_only_threshold'])
                for f in bl.plot_night_cross_scatter(
                        dbn, sorted(passing),
                        title_root=f'{name}: per-night Δ DZ_kj cross-comparison '
                                   f'({clabel} − {br["reference_label"]})',
                        zoom_lim_um=cfg['cross_scatter_zoom_um']) or []:
                    pdf.savefig(f, bbox_inches='tight'); plt.close(f)

    # ---- DOF night-vs-night scatter (cell 24) ----
    # One page per (comparison leg, night pair), for every leg with 2 or more
    # qualifying nights; a leg exercised on a single night gets no page.
    if _svd_ok and C_all is not None:
        with PdfPages(str(out_dir / 'bounce_dof_night_scatter.pdf')) as pdf:
            f = bl.plot_leg_night_coverage(
                bl.leg_night_coverage(bounce_results),
                title='Nights with results per bounce leg (B set)')
            if f is not None:
                pdf.savefig(f, bbox_inches='tight'); plt.close(f)
            for name, br in bounce_results.items():
                for clabel, cb in br['comparisons'].items():
                    dbn = cb.get('dof_deltas_by_night', {})
                    if len(dbn) < 2:
                        print(f'  (dof_night_scatter: {name} {clabel} has '
                              f'{len(dbn)} night(s) — no page)')
                        continue
                    for f in bl.plot_dof_night_scatter(
                            dbn, LABELS_50DOF, units=DOF_UNITS_50,
                            title_root=f'{name}: DOF Δ ({clabel} - {br["reference_label"]})'):
                        pdf.savefig(f, bbox_inches='tight'); plt.close(f)

    # ---- DOF Δ per (night, B set), as per-DOF panels (cell 26) ----
    # x is the B-set position: elevation in deg for the BLOCK-T720 sweep,
    # camera-rotator angle in deg for BLOCK-T724.  Each point is one
    # (night, leg) paired Δ, labelled with its day_obs and B value.
    #
    # Every recovery scheme the run computed appears on the same panels, so the
    # DOF value a scheme asks for can be read against the allowed range r_j and
    # against what the other schemes ask for.  The base (filled-circle) series
    # is the 50/34 recovery on both bounces; on a camera_hexapod_only bounce the
    # 5/5 scheme — the one the result is actually used in, since with only the
    # rotator moving only camera-hexapod corrections are applied — is an overlay
    # on its own five panels.  A scheme that does not reach a DOF simply has no
    # marker there: 22/12 is absent on the 28 DOF outside its index set, and 5/5
    # on the 45 DOF outside the camera hexapod.
    if _svd_ok and DOF_all is not None:
        with PdfPages(str(out_dir / 'bounce_dof_night_values.pdf')) as pdf:
            for b in bounces:
                name = b['name']
                br = bounce_results.get(name)
                if br is None:
                    continue
                cam_only = bool(b.get('camera_hexapod_only', False))
                entries = []
                for clabel, cb in br['comparisons'].items():
                    bval = bl.leg_b_value(clabel)
                    _by = {k: (cb.get(f'{k}_by_night') or {})
                           for k in ('rbr_deltas', 'dof22_deltas',
                                     'dof_oic_deltas', 'dof5_deltas')}
                    for nt, dd in (cb.get('dof_deltas_by_night') or {}).items():
                        e = {'night': int(nt), 'b_value': bval,
                             'label': clabel, 'dof_deltas': dd}
                        for k, bn in _by.items():
                            e[k] = bn.get(nt)
                        entries.append(e)
                if not entries:
                    print(f'  (dof_night_values: {name} has no per-night DOF Δ)')
                    continue
                overlay = []
                if dof_ranges is not None and any(e.get('rbr_deltas')
                                                  for e in entries):
                    overlay.append(('rbr_deltas',
                                    f'RBR 50/34, kappa={cfg["rbr_kappa"]:g} '
                                    f'power={cfg["rbr_power"]} '
                                    f'(both dimensionless)'))
                if any(e.get('dof22_deltas') for e in entries):
                    overlay.append(('dof22_deltas',
                                    f'{len(DOF22)}/{cfg["n_keep_reduced"]} '
                                    f'reduced DOF set'))
                if any(e.get('dof_oic_deltas') for e in entries):
                    overlay.append(('dof_oic_deltas',
                                    f'OIC 50/34, rho={cfg["oic_rho"]:g} '
                                    f'dimensionless (rigid body suppressed, '
                                    f'see study doc)'))
                if cam_only and any(e.get('dof5_deltas') for e in entries):
                    overlay.append(('dof5_deltas', '5/5 camera hexapod'))
                axis = bl.leg_axis_name(b)
                sub = ''
                if dof_ranges is not None:
                    sub = ('\nShaded band is the allowed range ±r_j, in each '
                           'DOF\'s own unit')
                figs = bl.plot_dof_vs_b_value_panels(
                    entries, LABELS_50DOF, DOF_UNITS_50,
                    x_label=f'B set {axis} [deg]',
                    ncols=int(cfg['dof_ncols']),
                    rows_per_page=int(cfg['dof_rows_per_page']),
                    title=f'{name}: paired Δ DOF vs B-set {axis.lower()}, '
                          f'all recovery schemes\n'
                          f'Δ = comparison − {br["reference_label"]}, '
                          f'one point per (night, B set){sub}',
                    overlay_series=overlay,
                    ranges=dof_ranges,
                    base_label=f'default {cfg["n_dof"]}/{cfg["n_keep"]}')
                for fig in figs:
                    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)

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
                            ncols=cfg['vmode_ncols'], rows_per_page=cfg['vmode_rows_per_page'],
                        elev_halfwidth_deg=cfg.get('ab_elev_halfwidth_deg')):
                        pdf.savefig(f, bbox_inches='tight'); plt.close(f)
                    for f in bl.plot_values_vs_ordinal_pages(          # (2) DOF vs ordinal
                            fit_table[m], DOF5_all[m], cam_labels, units=cam_units,
                            title_root=f'{name} [5/5 Cam-hex]: Camera-hexapod DOF',
                            ncols=cfg['dof_ncols'], rows_per_page=cfg['dof_rows_per_page'],
                        elev_halfwidth_deg=cfg.get('ab_elev_halfwidth_deg')):
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
        print('  correctable-FWHM metric [arcsec, median over focal plane]:')
        print('   ' + fdf.to_string(index=False).replace('\n', '\n   '))
        # Every "after" bar is the achieved residual dW − S·(d/w) of its own
        # scheme, so the bars within a group are a like-for-like comparison.
        bar_cols = [c for c in ('fwhm_before', 'fwhm_after_50_34',
                                'fwhm_after_rbr', 'fwhm_after_22_12',
                                'fwhm_after_oic', 'fwhm_after_5_5')
                    if c in fdf.columns]
        lab = {'fwhm_before': 'no correction',
               'fwhm_after_50_34': '50/34 truncated',
               'fwhm_after_rbr': f'RBR 50/34 (kappa={cfg["rbr_kappa"]:g}, '
                                 f'power={cfg["rbr_power"]}; both dimensionless)',
               'fwhm_after_22_12': f'{len(DOF22)}/{cfg["n_keep_reduced"]} '
                                   f'reduced DOF set',
               'fwhm_after_oic': f'OIC 50/34 (rho={cfg["oic_rho"]:g} '
                                 f'dimensionless)',
               'fwhm_after_5_5': '5/5 camera hexapod'}
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
                         'RSS of median Δ-DZ aberrations; every "after" bar is '
                         'the achieved residual dW − S·(d/w) of its own scheme')
            ax.legend(); ax.grid(axis='y', alpha=0.3)
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print('  wrote bounce_fwhm_metric.pdf')

    # ---- correctable FWHM vs B-set position, per (night, leg) ----
    # The image-quality question the scheme comparison exists to answer: what
    # does each recovery cost in delivered FWHM as a function of the throw — for
    # RBR the cost of bounding the recovered DOF to the allowed range, for 22/12
    # the cost of dropping to the reduced operational DOF set.  One page per
    # bounce, one point per (night, B set), one series per scheme.
    if fwhm_bvalue_rows:
        bdf = pd.DataFrame(fwhm_bvalue_rows)
        with PdfPages(str(out_dir / 'bounce_fwhm_vs_bvalue.pdf')) as pdf:
            for b in bounces:
                name = b['name']
                sub = bdf[bdf['bounce'] == name]
                if sub.empty:
                    continue
                axis = bl.leg_axis_name(b)
                ttl = (f'{name}: correctable FWHM of the bounce optical-state '
                       f'change vs B-set {axis.lower()}\n'
                       f'one point per (night, B set); residuals are the '
                       f'achieved dW − S·(d/w)')
                if dof_ranges is not None:
                    ttl += (f'\nRBR: kappa={cfg["rbr_kappa"]:g}, '
                            f'power={cfg["rbr_power"]} '
                            f'(penalty ~ (|d_j|/(kappa·r_j))^{2 * cfg["rbr_power"]})')
                if oic_ok:
                    ttl += (f'; OIC: rho={cfg["oic_rho"]:g} (dimensionless '
                            f'motion-penalty weight)')
                fig = bl.plot_fwhm_vs_b_value(
                    sub.to_dict('records'), x_label=f'B set {axis} [deg]',
                    title=ttl)
                if fig is not None:
                    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print(f'  wrote bounce_fwhm_vs_bvalue.pdf ({len(bdf)} rows)')

    print('[bounce] done.')


if __name__ == '__main__':
    main()
