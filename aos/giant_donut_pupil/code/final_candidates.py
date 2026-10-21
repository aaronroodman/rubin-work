"""Final v2 giant-donut candidates and pages, all four 20251023 exposures.

The rule is ``score_selection``'s ``final``: a donut is selected if it carries a
^ (star row classed isolated, and image-clean against every catalog neighbour,
``overlap_frac_best < 0.02``) or a v (good peak; on the corner sensors the
sky-corrected rule `score_selection.corner_good_peak`). A star and a peak within
0.3 r_out are one donut.

Writes into a new ``candidates_v2`` run:

- ``candidates_v2.parquet`` and ``.csv``: one row per selected donut, plus the
  hand-labelled missed donuts the rule does not select (``selected`` False), with
  the marker, positions, field angle, magnitude, metrics and hand label;
- ``tally.csv`` and ``tally.pdf``: selected donuts per exposure and sensor class;
- ``score_summary.csv`` and ``score_summary.pdf``: the before and after scores;
- ``candidates_<visit>_<side>.pdf``: the mosaic, the tally and score pages, and
  one page per raft with the ^ and v markers coloured by hand label.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/final_candidates.py --diag 20261020_selection_diag \\
        --refine 20261020_peak_refine
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import candidates as cd  # noqa: E402  (blitz path first)
import hand_label as hl  # noqa: E402
import score_selection as ss  # noqa: E402
import select_candidates as sc  # noqa: E402
import selection_pages as pages  # noqa: E402
import selection_v2 as sv  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE  # noqa: E402
from lsst.obs.lsst import LsstCam  # noqa: E402

from rubinwork.common import study_run  # noqa: E402

LABEL_COLOURS = dict(hl.LABEL_COLOURS, unlabelled=hl.LABEL_COLOURS['unreviewed'])
METRICS = ('annulus_ratio', 'annulus_ratio_sky', 'overlap_frac', 'overlap_w_sep', 'ring_snr',
           'ring_level_e', 'ring_p99_e', 'snr')


def candidate_table(stars, peaks, marked, lab, camera):
    """Selected donuts plus unselected missed labels, with positions and metrics."""
    t = marked.copy()
    t['selected'] = True
    t['hand_label'] = ''
    for i, r in lab[lab.mark >= 0].iterrows():
        t.at[r.mark, 'hand_label'] = r.label
    missed = lab[(lab.mark < 0) & (lab.label != 'bad')]
    extra = pd.DataFrame(dict(visit=missed.visit, detector=missed.detector, x_pix=missed.x_pix,
                              y_pix=missed.y_pix, fp_x_mm=missed.fp_x_mm, fp_y_mm=missed.fp_y_mm,
                              refcat_id=-1, peak_id=-1, star_isolated=False, good_peak=False,
                              marker='', sensor_class=missed.sensor_class, selected=False,
                              hand_label=missed.label))
    t = pd.concat([t, extra], ignore_index=True)
    t['side'] = t.visit.map(sv.VISITS)
    t['marker_star'] = t.star_isolated
    t['marker_peak'] = t.good_peak
    fx, fy = [], []
    for det, x, y in zip(t.detector, t.x_pix, t.y_pix):
        ax, ay = cd.pixels_to(camera[det], [x], [y], FIELD_ANGLE)
        fx.append(np.degrees(ax[0]))
        fy.append(np.degrees(ay[0]))
    t['field_x_deg'], t['field_y_deg'] = fx, fy
    t['field_r_deg'] = np.hypot(t.field_x_deg, t.field_y_deg)
    t['field_azimuth_deg'] = np.degrees(np.arctan2(t.field_y_deg, t.field_x_deg))
    s = stars.set_index(['visit', 'refcat_id'])
    p = peaks.set_index(['visit', 'peak_id'])
    t['mag'] = np.nan
    t['mag_source'] = ''
    for c in METRICS:
        t[c] = np.nan
    for i, r in t.iterrows():
        if r.peak_id >= 0 and (r.visit, r.peak_id) in p.index:
            q = p.loc[(r.visit, r.peak_id)]
            for c in METRICS:
                t.at[i, c] = q[c]
            t.at[i, 'mag'], t.at[i, 'mag_source'] = q.mag_used, q.mag_source
        if r.refcat_id >= 0 and (r.visit, r.refcat_id) in s.index:
            q = s.loc[(r.visit, r.refcat_id)]
            t.at[i, 'mag'], t.at[i, 'mag_source'] = q.mag_best, q.mag_source
            if r.peak_id < 0:
                for c in ('annulus_ratio', 'overlap_frac', 'ring_level_e', 'ring_p99_e'):
                    t.at[i, c] = q[c]
    t['r_out_pix'] = t.r_out_mm * 100.0
    return t.drop(columns=['star_isolated', 'good_peak', 'key'])


def tally(t):
    rows = []
    for (visit, cls), g in t.groupby(['visit', 'sensor_class']):
        s = g[g.selected]
        rows.append(dict(visit=visit, side=sv.VISITS[visit], sensor_class=cls, selected=len(s),
                         both=int((s.marker == 'both').sum()), star_only=int((s.marker == '^').sum()),
                         peak_only=int((s.marker == 'v').sum()),
                         within_1p75_deg=int((s.field_r_deg <= 1.75).sum()),
                         labelled_good=int((s.hand_label == 'good').sum()),
                         labelled_bad=int((s.hand_label == 'bad').sum()),
                         missed_not_selected=int((~g.selected).sum())))
    return pd.DataFrame(rows)


def draw_markers(ax, t, size=70):
    """^ and v markers coloured by hand label; missed donuts as dashed circles."""
    lab = np.where(t.hand_label == '', 'unlabelled', t.hand_label)
    colours = np.array([LABEL_COLOURS.get(x, 'white') for x in lab], dtype=object)
    both = (t.marker == 'both').to_numpy()
    dy = np.where(both, hl.BOTH_OFFSET_ROUT * t.r_out_mm.fillna(3.4), 0.0)
    star, peak = t.marker_star.to_numpy(bool), t.marker_peak.to_numpy(bool)
    if star.any():
        ax.scatter(t.fp_x_mm[star], (t.fp_y_mm + dy)[star], s=size, marker='^',
                   c=list(colours[star]), edgecolors='black', lw=0.5)
    if peak.any():
        ax.scatter(t.fp_x_mm[peak], (t.fp_y_mm - dy)[peak], s=size, marker='v',
                   c=list(colours[peak]), edgecolors='black', lw=0.5)
    for _, r in t[~t.selected].iterrows():
        ax.add_patch(Circle((r.fp_x_mm, r.fp_y_mm), 3.4, fill=False, lw=1.2, ls='--',
                            color=LABEL_COLOURS['missed']))
    handles = [ax.scatter([], [], s=size, marker='^', c='white', edgecolors='black',
                          label=f'^ star: {int(star.sum())}'),
               ax.scatter([], [], s=size, marker='v', c='white', edgecolors='black',
                          label=f'v peak: {int(peak.sum())}  (both: {int(both.sum())})')]
    counts = pd.Series(lab[t.selected.to_numpy()]).value_counts().to_dict()
    for name in ('good', 'bad', 'unlabelled'):
        handles.append(ax.scatter([], [], s=40, marker='s', c=LABEL_COLOURS[name],
                                  label=f'{name}: {counts.get(name, 0)}'))
    handles.append(ax.plot([], [], '--', color=LABEL_COLOURS['missed'],
                           label=f'missed, not selected: {int((~t.selected).sum())}')[0])
    ax.legend(handles=handles, loc='upper right', fontsize=7, framealpha=0.85)


def write_pdf(path, visit, side, t, images, camera, tally_text, score_text):
    names = sorted(images)
    backgrounds = {n: cd.background_map(im, block=128, percentile=10.0) for n, im in images.items()}
    with PdfPages(path) as pdf:
        mosaic, geometry = pages.raft_mosaic(camera, images, backgrounds, names, sc.MOSAIC_BIN)
        fig, ax = plt.subplots(figsize=(11, 11))
        sc.show_mosaic(ax, mosaic, geometry, pages.STRETCH_E)
        pages.draw_ccd_edges(ax, camera, names, lw=0.3)
        draw_markers(ax, t, size=14)
        ax.set_title(f'{visit} ({side}-focal): selected giant donuts, final rule; markers '
                     f'coloured by hand label; mosaic binned {sc.MOSAIC_BIN}x{sc.MOSAIC_BIN}',
                     fontsize=9)
        pdf.savefig(fig, dpi=300)
        plt.close(fig)
        pages._text_page(pdf, 'Selected donuts per exposure and sensor class (donut counts)',
                         tally_text, 8)
        pages._text_page(pdf, 'Scores against the hand labels, baseline and final rule', score_text, 7)
        for raft in sorted({n[:3] for n in names}):
            rnames = [n for n in names if n.startswith(raft)]
            rmos, rgeom = pages.raft_mosaic(camera, images, backgrounds, rnames, sc.RAFT_BIN)
            fig, ax = plt.subplots(figsize=(11, 11))
            sc.show_mosaic(ax, rmos, rgeom, pages.STRETCH_E)
            pages.draw_ccd_edges(ax, camera, rnames, lw=0.8)
            for n in rnames:
                c = camera[n].getCenter(FOCAL_PLANE)
                ax.text(c.getX(), c.getY(), n[4:], color='cyan', fontsize=7, alpha=0.6,
                        ha='center')
            draw_markers(ax, t[t.detector.str.startswith(raft)])
            ax.set_title(f'{visit} ({side}-focal) raft {raft}: selected donuts; '
                         f'binned {sc.RAFT_BIN}x{sc.RAFT_BIN}', fontsize=9)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)


def score_text(summary):
    s = summary[summary.rule.isin(['baseline', 'fix_a', 'fix_b', 'final'])]
    cols = ['rule', 'group', 'marker', 'kept_good', 'kept_bad', 'purity', 'missed',
            'completeness', 'new_unlabelled']
    head = ('purity = kept good / (kept good + kept bad); completeness = kept good / '
            '(kept good + missed); both dimensionless.\nnew_unlabelled: marked donuts with no '
            'hand label on a done page (donut counts).\n'
            'fix_a: ^ also needs overlap_frac_best < 0.02 (neighbours without ComCam r count).\n'
            'fix_b: corner v uses the sky-corrected annulus ratio < 0.05, flux-weighted overlap '
            '< 0.03 and ring S/N > 3.\n\n')
    return head + s[cols].round(3).to_string(index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--diag', required=True)
    parser.add_argument('--refine', required=True)
    parser.add_argument('--no-pdf', action='store_true')
    args = parser.parse_args()
    study_dir = pathlib.Path(__file__).resolve().parents[1]
    diag_dir, refine_dir = study_dir / 'output' / args.diag, study_dir / 'output' / args.refine
    stars, peaks, labels, res, summary = ss.score_all(diag_dir, refine_dir)
    marked, lab, _, _ = res['final']
    camera = LsstCam.getCamera()
    t = candidate_table(stars, peaks, marked, lab, camera)

    run = study_run(sv.STUDY, 'candidates_v2')
    t.to_parquet(run.path / 'candidates_v2.parquet')
    t.to_csv(run.path / 'candidates_v2.csv', index=False, float_format='%.4f')
    counts = tally(t)
    counts.to_csv(run.path / 'tally.csv', index=False)
    summary.to_csv(run.path / 'score_summary.csv', index=False, float_format='%.3f')
    (run.path / 'selection.json').write_text(json.dumps(dict(
        diag=args.diag, refine=args.refine, rule='final',
        corner_annulus=ss.CORNER_ANNULUS, corner_overlap_w=ss.CORNER_OVERLAP_W,
        corner_ring_snr=ss.CORNER_RING_SNR), indent=1))
    tally_text = counts.to_string(index=False)
    stext = score_text(summary)
    with PdfPages(run.path / 'tally.pdf') as pdf:
        pages._text_page(pdf, 'Selected donuts per exposure and sensor class (donut counts)',
                         tally_text, 8)
    with PdfPages(run.path / 'score_summary.pdf') as pdf:
        pages._text_page(pdf, 'Scores against the hand labels', stext, 7)
    print(tally_text, flush=True)
    if not args.no_pdf:
        for visit, side in sv.VISITS.items():
            images, _ = sc.load_cache(study_dir / 'output' / sv.CACHES[visit] / str(visit), camera)
            write_pdf(run.path / f'candidates_{visit}_{side}.pdf', visit, side,
                      t[t.visit == visit], images, camera, tally_text, stext)
            print('pdf', visit, flush=True)
    print('done', run.path)


if __name__ == '__main__':
    main()
