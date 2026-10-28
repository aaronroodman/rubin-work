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

The recovered DOF are reported several ways, among them the truncated 50-DOF /
34-v-mode recovery and Range-Bounded Recovery (RBR), which adds a per-DOF
penalty that grows steeply as a recovered amplitude approaches the range the
telescope can physically apply.  The default recovery exceeds that range on
these bounce legs by up to a factor of 11.27 (dimensionless, recovered
amplitude over allowed range); RBR bounds it at a small cost in corrected
FWHM.  The method is derived in the smatrix `regularized_inversion` study and
imported from there.

Writes, under the directory given by --out-dir (the pipeline passes
bounce/output/<P>_<M>/) :
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
    bounce_fwhm_vs_bvalue.pdf      page 1: correctable-FWHM bars per leg, every
                                   scheme; then FWHM vs B-set elevation or
                                   rotator angle, one point per (night, leg)
    bounce_lut.pdf                 LUT polynomial fits of the default-scheme DOF
                                   Δ, the kept coefficients, and the FWHM left
                                   by no / per-point / LUT correction
    bounce_lut_coefficients.csv, bounce_lut_fwhm.csv   the same numbers

Each bounce is read in one recovery scheme, named on the first page of the DOF
PDFs: RBR 50/50 for the elevation sweep, 5/5 camera hexapod for the rotator.

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
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))      # aos/code (flat shared modules)

import bounce_lib as bl
import lut_fit as lf
from lsst.ts.intrinsic.wavefront.ofc_svd import LABELS_50DOF, DOF_UNITS_50
from rubinwork.bounce import analysis as ba
from rubinwork.bounce.legs import CAM_HEX_DOF, SCHEME_KEYS, default_scheme, scheme_label


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
    ap.add_argument('--miw-variant', default=None,
                    help='Read fits.parquet of this miw product variant through the '
                         'catalog, instead of --fits.')
    ap.add_argument('--miw-build', default='current')
    ap.add_argument('--run-label', default=None,
                    help='Write into the dated study run directory '
                         '<studies>/aos/bounce/<YYYYMMDD>_<label>/, with run.json, '
                         'instead of --out-dir.')
    args = ap.parse_args()

    cfg = bl.resolve_config(args.param_set, args.mi_name,
                            analysis_config=args.analysis_config,
                            rbr_kappa=args.rbr_kappa, rbr_power=args.rbr_power,
                            no_rbr=args.no_rbr)
    prefix = cfg['fit_prefix']
    bounces = cfg['bounces']

    base = Path(args.output_root) / args.param_set / args.mi_name
    if args.miw_variant:
        from rubinwork.products import catalog
        fits_path = catalog.path('miw', args.miw_variant,
                                 build=args.miw_build) / 'fits.parquet'
    else:
        fits_path = Path(args.fits) if args.fits else base / 'fits.parquet'
    if args.run_label:
        from rubinwork.common import study_run
        out_dir = study_run('aos/bounce', args.run_label).path
        if args.miw_variant:
            # The catalog read above ran before the run existed; read again so
            # run.json records the build.
            catalog.path('miw', args.miw_variant, build=args.miw_build)
    else:
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

    # Each bounce is read in one recovery scheme: RBR 50/50 for the elevation
    # sweep, 5/5 camera hexapod for the rotator (only the camera hexapod moves).
    # The per-visit DOF and labels of each scheme, for the ordinal pages.
    cam_labels = [LABELS_50DOF[i] for i in CAM_HEX_DOF]
    cam_units = [DOF_UNITS_50[i] for i in CAM_HEX_DOF]
    per_visit = {'50_34': (DOF_all, LABELS_50DOF, DOF_UNITS_50),
                 'rbr': (result.DOFRBR_all, LABELS_50DOF, DOF_UNITS_50),
                 '5_5': (DOF5_all, cam_labels, cam_units)}
    schemes = {b['name']: default_scheme(b) for b in bounces}

    def scheme_page(what):
        """First page of a PDF: which recovery scheme each bounce is shown in."""
        lines = [what, '', 'Recovery scheme per bounce:']
        for b in bounces:
            lines.append(f'  {b["name"]}: {scheme_label(schemes[b["name"]], cfg)}')
        lines += ['', f'fits: {fits_path}']
        fig = plt.figure(figsize=(11, 3.5))
        fig.text(0.03, 0.92, '\n'.join(lines), va='top', ha='left', fontsize=11,
                 family='monospace')
        return fig

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
            f = scheme_page('Physical DOF vs ordinal image, per visit')
            pdf.savefig(f, bbox_inches='tight'); plt.close(f)
            if _marker_ok:
                leg = markers_legend_figure(show_iter_distinction=False)
                pdf.savefig(leg, bbox_inches='tight'); plt.close(leg)
            for b in bounces:
                m = bl.bounce_program_mask(fit_table, b)
                sk = schemes[b['name']]
                vals, labs, units = per_visit.get(sk, per_visit['50_34'])
                if int(m.sum()) == 0 or vals is None:
                    continue
                for f in bl.plot_values_vs_ordinal_pages(
                        fit_table[m], vals[m], labs, units=units,
                        title_root=f'{b["name"]}: Physical DOF, '
                                   f'{scheme_label(sk, cfg)}',
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
            f = scheme_page('DOF paired Δ, night against night')
            pdf.savefig(f, bbox_inches='tight'); plt.close(f)
            f = bl.plot_leg_night_coverage(
                bl.leg_night_coverage(bounce_results),
                title='Nights with results per bounce leg (B set)')
            if f is not None:
                pdf.savefig(f, bbox_inches='tight'); plt.close(f)
            for name, br in bounce_results.items():
                sk = schemes[name]
                for clabel, cb in br['comparisons'].items():
                    dbn = cb.get(f'{SCHEME_KEYS[sk][0]}_by_night') or {}
                    if len(dbn) < 2:
                        print(f'  (dof_night_scatter: {name} {clabel} has '
                              f'{len(dbn)} night(s) — no page)')
                        continue
                    for f in bl.plot_dof_night_scatter(
                            dbn, LABELS_50DOF, units=DOF_UNITS_50,
                            title_root=f'{name}: DOF Δ ({clabel} - {br["reference_label"]}), '
                                       f'{scheme_label(sk, cfg)}'):
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
    # The base (filled-circle) series is the bounce's own scheme; every other
    # scheme the run computed is an open-marker overlay, so the DOF a scheme asks
    # for can be read against the allowed range r_j and against the others.  A
    # scheme that does not reach a DOF has no marker there: 22/12 is absent on
    # the 28 DOF outside its index set, 5/5 on the 45 outside the camera hexapod.
    lut_entries = {}
    if _svd_ok and DOF_all is not None:
        with PdfPages(str(out_dir / 'bounce_dof_night_values.pdf')) as pdf:
            f = scheme_page('Paired Δ DOF vs B-set position; the scheme listed '
                            'is the filled series')
            pdf.savefig(f, bbox_inches='tight'); plt.close(f)
            for b in bounces:
                name = b['name']
                br = bounce_results.get(name)
                if br is None:
                    continue
                sk = schemes[name]
                cam_only = bool(b.get('camera_hexapod_only', False))
                entries = []
                for clabel, cb in br['comparisons'].items():
                    bval = bl.leg_b_value(clabel)
                    _by = {k: (cb.get(f'{dk}_by_night') or {})
                           for k, (dk, _fk) in SCHEME_KEYS.items()}
                    for nt in (_by[sk] or {}):
                        e = {'night': int(nt), 'b_value': bval, 'label': clabel,
                             'dof_deltas': _by[sk].get(nt)}
                        for k, bn in _by.items():
                            e[k] = bn.get(nt)
                        entries.append(e)
                lut_entries[name] = entries
                if not entries:
                    print(f'  (dof_night_values: {name} has no per-night DOF Δ)')
                    continue
                overlay = [(k, scheme_label(k, cfg)) for k in SCHEME_KEYS
                           if k != sk and (k != '5_5' or cam_only)
                           and any(e.get(k) for e in entries)]
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
                    base_label=f'default: {scheme_label(sk, cfg)}')
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
               'fwhm_after_50_34': scheme_label('50_34', cfg),
               'fwhm_after_rbr': scheme_label('rbr', cfg),
               'fwhm_after_22_12': scheme_label('22_12', cfg),
               'fwhm_after_oic': scheme_label('oic', cfg),
               'fwhm_after_5_5': scheme_label('5_5', cfg)}

    # ---- correctable FWHM vs B-set position, per (night, leg) ----
    # The image-quality question the scheme comparison exists to answer: what
    # does each recovery cost in delivered FWHM as a function of the throw — for
    # RBR the cost of bounding the recovered DOF to the allowed range, for 22/12
    # the cost of dropping to the reduced operational DOF set.  One page per
    # bounce, one point per (night, B set), one series per scheme.
    if fwhm_bvalue_rows:
        bdf = pd.DataFrame(fwhm_bvalue_rows)
        fwhm_series = [
            ('fwhm_before', 'no correction', '#444444', 'o', '--'),
            ('fwhm_after_default', scheme_label('50_34', cfg), '#1f77b4', 'o', '-'),
            ('fwhm_after_rbr', scheme_label('rbr', cfg), '#d62728', 's', '-'),
            ('fwhm_after_22_12', scheme_label('22_12', cfg), '#2ca02c', '^', '-'),
            ('fwhm_after_oic', scheme_label('oic', cfg), '#9467bd', 'D', '-'),
            ('fwhm_after_5_5', scheme_label('5_5', cfg), '#ff7f0e', 'v', '-')]
        with PdfPages(str(out_dir / 'bounce_fwhm_vs_bvalue.pdf')) as pdf:
            # Page 1: the per-leg bars, pooled over nights.
            if fwhm_rows:
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
                    title=ttl, series=fwhm_series)
                if fig is not None:
                    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        print(f'  wrote bounce_fwhm_vs_bvalue.pdf ({len(bdf)} rows)')

    # ---- LUT functions: polynomial fits of the default-scheme DOF Δ ----
    # Per bounce, each DOF's per-(night, leg) paired Δ is fitted in (x - x_ref)
    # with no constant term, so the LUT is zero at the reference.  Kept DOF are
    # those whose coefficients differ from zero at lut_nsigma or more.  Then the
    # correctable FWHM left at each (night, leg) after applying the LUT, against
    # no correction and the per-point correction in the same scheme.
    if lut_entries and result.fwhm_tools is not None:
        svd_by = {'50_34': result.svd, 'rbr': result.svd_rbr, '5_5': result.svd5}
        fits_by = {}
        lut_rows = []
        with PdfPages(str(out_dir / 'bounce_lut.pdf')) as pdf:
            f = scheme_page('LUT functions fitted to the default-scheme paired-Δ DOF')
            pdf.savefig(f, bbox_inches='tight'); plt.close(f)
            for b in bounces:
                name = b['name']
                entries = lut_entries.get(name) or []
                sk = schemes[name]
                s_ = svd_by.get(sk)
                if not entries or s_ is None:
                    continue
                axis = bl.leg_axis_name(b)
                rot = axis.lower().startswith('rot')
                tag = 'rotator' if rot else 'elevation'
                order = int(cfg[f'lut_order_{tag}'])
                x_ref = float(cfg[f'lut_ref_{tag}_deg'])
                x_rng = [float(v) for v in cfg[f'lut_range_{tag}_deg']]
                dof_idx = [int(q) for q in (s_.dof_idx or range(s_.n_dof))]
                fits = lf.fit_bounce_lut(entries, dof_idx, x_ref, order,
                                         float(cfg['lut_nsigma']))
                fits_by[name] = (fits, axis)
                n_sig = sum(f['significant'] for f in fits.values())
                print(f'  LUT {name}: order {order} in ({axis} - {x_ref:g} deg); '
                      f'{n_sig} of {len(fits)} DOF at >= {cfg["lut_nsigma"]:g} sigma')
                for fig in lf.plot_lut_panels(
                        fits, LABELS_50DOF, DOF_UNITS_50, x_ref=x_ref, x_range=x_rng,
                        axis=axis, ncols=int(cfg['dof_ncols']),
                        rows_per_page=int(cfg['dof_rows_per_page']),
                        title=f'{name}: LUT fit, Δ DOF = sum_p c_p (B {axis.lower()} '
                              f'− {x_ref:g} deg)^p, p = 1..{order}; '
                              f'{scheme_label(sk, cfg)}\n'
                              f'Huber-weighted on the paired-Δ errors; covariance '
                              f'inflated by chi2/dof when above 1; red = kept '
                              f'(>= {cfg["lut_nsigma"]:g} sigma from zero), '
                              f'{n_sig} of {len(fits)} DOF'):
                    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
                for e in entries:
                    dW = result.night_dW.get((name, e['label'], int(e['night'])))
                    if dW is None:
                        continue
                    d_lut = lf.lut_dof(fits, e['b_value'] - x_ref, dof_idx)
                    fb = bdf[(bdf['bounce'] == name) & (bdf['comparison'] == e['label'])
                             & (bdf['night'] == int(e['night']))]
                    if fb.empty:
                        continue
                    lut_rows.append({
                        'bounce': name, 'comparison': e['label'],
                        'night': int(e['night']), 'b_value': e['b_value'],
                        'n_pairs': int(fb['n_pairs'].iloc[0]),
                        'fwhm_before': float(fb['fwhm_before'].iloc[0]),
                        'fwhm_after_point': float(fb[SCHEME_KEYS[sk][1]].iloc[0]),
                        'fwhm_after_lut': ba.achieved_fwhm(result, dW, d_lut, s_)})
            fig = lf.plot_lut_summary(
                fits_by, LABELS_50DOF, DOF_UNITS_50,
                title=f'Kept LUT coefficients (>= {cfg["lut_nsigma"]:g} sigma from '
                      f'zero) with 1-sigma errors')
            if fig is not None:
                pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
            fig = lf.plot_lut_table(fits_by, LABELS_50DOF, DOF_UNITS_50,
                                    title='Kept LUT coefficients')
            pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
            ldf = pd.DataFrame(lut_rows)
            for b in bounces:
                name = b['name']
                sub = ldf[ldf['bounce'] == name] if len(ldf) else ldf
                if sub.empty:
                    continue
                sk = schemes[name]
                axis = bl.leg_axis_name(b)
                fig = bl.plot_fwhm_vs_b_value(
                    sub.to_dict('records'), x_label=f'B set {axis} [deg]',
                    title=f'{name}: correctable FWHM left by the LUT vs B-set '
                          f'{axis.lower()}\n{scheme_label(sk, cfg)}; LUT applies '
                          f'only the kept DOF, zero elsewhere',
                    series=[('fwhm_before', 'no correction', '#444444', 'o', '--'),
                            ('fwhm_after_point', 'per-point correction',
                             '#1f77b4', 'o', '-'),
                            ('fwhm_after_lut', 'LUT correction', '#d62728', 's', '-')])
                if fig is not None:
                    pdf.savefig(fig, bbox_inches='tight'); plt.close(fig)
        if len(ldf):
            print('  LUT correctable FWHM [arcsec, median over focal plane]:')
            print('   ' + ldf.to_string(index=False).replace('\n', '\n   '))
            ldf.to_csv(out_dir / 'bounce_lut_fwhm.csv', index=False)
        crow = []
        for name, (fits, axis) in fits_by.items():
            for q, f in fits.items():
                r = {'bounce': name, 'dof': q, 'label': LABELS_50DOF[q],
                     'unit': DOF_UNITS_50[q], 'n': f['n'], 'chi2': f['chi2'],
                     'ndof': f['dof'], 'nsig': f['nsig'],
                     'significant': f['significant']}
                for p, (c, e) in enumerate(zip(f['coef'], np.sqrt(np.diag(f['cov']))), 1):
                    r[f'c{p}'] = c; r[f'c{p}_err'] = e
                crow.append(r)
        pd.DataFrame(crow).to_csv(out_dir / 'bounce_lut_coefficients.csv', index=False)
        print('  wrote bounce_lut.pdf')

    print('[bounce] done.')


if __name__ == '__main__':
    main()
