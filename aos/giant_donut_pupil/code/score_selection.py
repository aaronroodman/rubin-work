"""Score giant-donut selection rules against the v2 hand labels.

The v2 labels (``hand_labels_v2.csv``, `hand_label.py`) are on the marked donuts
of 12 raft pages: ``good`` or ``bad`` for a marked donut, ``missed`` for a good
donut the markers left out. A rule marks donuts with ``^`` (a star row it keeps)
and ``v`` (a peak it keeps); a star and a peak within 0.3 r_out are one donut.
Per rule, sensor class and marker type:

- completeness: good or missed donuts that the rule marks, over all good or
  missed donuts;
- purity: marked donuts labelled good, over marked donuts labelled good or bad;
- ``new_unlabelled``: marked donuts on a done page with no label within 0.3 r_out.
  Aaron added every good donut he saw, so these are most likely not good.

Rules:

- ``baseline``: ^ on ``iso_class == 'isolated'``, v on ``good_peak``.
- ``fix_a``: ^ also needs ``overlap_frac_best < 0.02``: the overlap neighbours
  include stars with no ComCam r flux (``mag_best``).
- ``fix_b``: on the corner sensors, v is `corner_good_peak`.
- ``final``: ``fix_a`` and ``fix_b``.

``tests`` scores the other changes tried for the bad donuts (a flux-weighted
overlap at several limits, a cut on a brighter neighbour).

Usage, from the repo root::

    python aos/giant_donut_pupil/code/score_selection.py --diag 20261020_selection_diag \\
        --refine 20261020_peak_refine
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hand_label as hl  # noqa: E402
import select_candidates as sc  # noqa: E402
import selection_v2 as sv  # noqa: E402

from rubinwork.common import study_run  # noqa: E402

MATCH_ROUT = 0.3
# Corner good peak: sky-corrected annulus ratio, flux-weighted overlap and ring S/N.
CORNER_ANNULUS = 0.05
CORNER_OVERLAP_W = 0.03
CORNER_RING_SNR = 3.0
GROUPS = ('science', 'SW0', 'SW1', 'corner')


def corner_good_peak(p):
    """The corner-sensor v rule (fix b)."""
    return (p.full_on_silicon & ~(p.ring_p99_e > sc.SATURATION_E)
            & (p.annulus_ratio_sky < CORNER_ANNULUS) & (p.overlap_w_sep < CORNER_OVERLAP_W)
            & (p.ring_snr > CORNER_RING_SNR))


def is_corner(t):
    return (t.sensor_class != 'science').to_numpy()


def star_rule(stars, name):
    keep = (stars.iso_class == 'isolated').to_numpy()
    if name in ('fix_a', 'final'):
        keep &= (stars.overlap_frac_best < sc.OVERLAP_CLEAN).to_numpy()
    return keep


def peak_rule(peaks, name):
    keep = peaks.good_peak.to_numpy(bool).copy()
    if name in ('fix_b', 'final'):
        c = is_corner(peaks)
        keep[c] = corner_good_peak(peaks[c]).to_numpy()
    return keep


RULES = ('baseline', 'fix_a', 'fix_b', 'final')


def load(diag_dir, refine_dir):
    stars = pd.read_parquet(diag_dir / 'stars.parquet')
    peaks = pd.read_parquet(refine_dir / 'peak_metrics.parquet')
    stars = stars.merge(peaks[['visit', 'peak_id', 'overlap_w', 'overlap_w_sep',
                               'annulus_ratio_sky', 'ring_snr']], on=['visit', 'peak_id'],
                        how='left')
    labels = pd.read_csv(diag_dir / 'hand_labels_v2.csv')
    done = json.loads((diag_dir / 'hand_labels_v2_pages.json').read_text())
    return stars, peaks, labels, {k for k, v in done.items() if v}


def marked_donuts(stars, peaks, star_keep, peak_keep):
    """One row per marked donut over all visits (`hand_label.match_marked`)."""
    out = []
    for visit in sorted(stars.visit.unique()):
        s = stars[star_keep & (stars.visit == visit).to_numpy()]
        p = peaks[peak_keep & (peaks.visit == visit).to_numpy()]
        for det in sorted(set(s.detector) | set(p.detector)):
            m = hl.match_marked(s[s.detector == det], p[p.detector == det])
            out.append(m.assign(visit=visit))
    m = pd.concat(out, ignore_index=True)
    m['marker'] = np.where(m.star_isolated & m.good_peak, 'both',
                           np.where(m.star_isolated, '^', 'v'))
    m['sensor_class'] = ['science' if 'SW' not in d else d[-3:] for d in m.detector]
    return m


def _nearest(xy, x, y, r_mm):
    if len(xy) == 0:
        return -1
    d = np.hypot(xy[:, 0] - x, xy[:, 1] - y) / r_mm
    j = int(np.argmin(d))
    return j if d[j] < MATCH_ROUT else -1


def match_labels(labels, marked):
    """Per label, the marked donut it falls on (``mark``, -1 if none) and its marker."""
    lab = labels.copy()
    lab['sensor_class'] = ['science' if 'SW' not in d else d[-3:] for d in lab.detector]
    lab['mark'] = -1
    lab['marker'] = ''
    for (visit, det), g in lab.groupby(['visit', 'detector']):
        m = marked[(marked.visit == visit) & (marked.detector == det)]
        xy = m[['fp_x_mm', 'fp_y_mm']].to_numpy()
        for i, r in g.iterrows():
            j = _nearest(xy, r.fp_x_mm, r.fp_y_mm, m.r_out_mm.median() if len(m) else 3.4)
            if j >= 0:
                lab.at[i, 'mark'] = m.index[j]
                lab.at[i, 'marker'] = m.marker.iloc[j]
    return lab


def new_unlabelled(labels, marked, pages_done):
    """Marked donuts on done pages with no label within 0.3 r_out."""
    on_page = np.array([f'{v}_{d[:3]}' in pages_done for v, d in zip(marked.visit, marked.detector)])
    rows = []
    for i, r in marked[on_page].iterrows():
        lab = labels[(labels.visit == r.visit) & (labels.detector == r.detector)]
        if _nearest(lab[['fp_x_mm', 'fp_y_mm']].to_numpy(), r.fp_x_mm, r.fp_y_mm, r.r_out_mm) < 0:
            rows.append(i)
    return marked.loc[rows]


def score(lab, new):
    """Counts, purity and completeness per sensor group and per marker type."""
    rows = []
    good = (lab.label != 'bad').to_numpy()
    kept = (lab.mark >= 0).to_numpy()
    for group in GROUPS:
        sel = (lab.sensor_class != 'science').to_numpy() if group == 'corner' else \
            (lab.sensor_class == group).to_numpy()
        n_new = int(((new.sensor_class != 'science') if group == 'corner'
                     else (new.sensor_class == group)).sum())
        for marker in ('all', '^', 'v', 'both'):
            m = sel & ((lab.marker == marker).to_numpy() if marker != 'all' else True)
            kg, kb = int((m & good & kept).sum()), int((m & ~good & kept).sum())
            row = dict(group=group, marker=marker, kept_good=kg, kept_bad=kb,
                       purity=kg / (kg + kb) if kg + kb else np.nan)
            if marker == 'all':
                miss = int((sel & good & ~kept).sum())
                row.update(missed=miss, completeness=kg / (kg + miss) if kg + miss else np.nan,
                           new_unlabelled=n_new)
            rows.append(row)
    return pd.DataFrame(rows)


def run_rule(stars, peaks, labels, pages_done, star_keep, peak_keep):
    marked = marked_donuts(stars, peaks, star_keep, peak_keep)
    lab = match_labels(labels, marked)
    new = new_unlabelled(labels, marked, pages_done)
    return marked, lab, score(lab, new), new


def tests(stars, peaks, labels, pages_done):
    """The other fixes tried for the bad donuts (science rows only)."""
    out = []
    base_p = peak_rule(peaks, 'baseline')
    iso = (stars.iso_class == 'isolated').to_numpy()
    variants = {f'overlap_w_sep < {w:g}': iso & ~(stars.overlap_w_sep >= w).to_numpy()
                for w in (0.01, 0.015, 0.02, 0.03)}
    variants['no brighter neighbour within 2 r_out'] = iso & ~(
        (stars.nbr_dmag < 0) & (stars.nbr_dist_rout < 2.0)).to_numpy()
    variants.update({f'overlap_frac_d4 < {w:g}': iso & (stars.overlap_frac_d4 < w).to_numpy()
                     for w in (0.1, 0.2, 0.3)})
    for name, keep in variants.items():
        _, _, s, _ = run_rule(stars, peaks, labels, pages_done, keep, base_p)
        out.append(s[(s.group == 'science') & (s.marker == 'all')].assign(variant=name))
    return pd.concat(out, ignore_index=True)


def score_all(diag_dir, refine_dir):
    stars, peaks, labels, pages_done = load(diag_dir, refine_dir)
    res = {}
    for name in RULES:
        res[name] = run_rule(stars, peaks, labels, pages_done, star_rule(stars, name),
                             peak_rule(peaks, name))
    summary = pd.concat([r[2].assign(rule=name) for name, r in res.items()], ignore_index=True)
    return stars, peaks, labels, res, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--diag', required=True, help='selection_diag run under output/')
    parser.add_argument('--refine', required=True, help='peak_refine run under output/')
    args = parser.parse_args()
    study_dir = pathlib.Path(__file__).resolve().parents[1]
    diag_dir, refine_dir = study_dir / 'output' / args.diag, study_dir / 'output' / args.refine
    stars, peaks, labels, res, summary = score_all(diag_dir, refine_dir)
    run = study_run(sv.STUDY, 'selection_score')
    summary.to_csv(run.path / 'score_summary.csv', index=False, float_format='%.3f')
    t = tests(stars, peaks, labels, json.loads(
        (diag_dir / 'hand_labels_v2_pages.json').read_text()).keys())
    t.to_csv(run.path / 'score_tests.csv', index=False, float_format='%.3f')
    for name, (_, lab, _, new) in res.items():
        lab.to_csv(run.path / f'labels_{name}.csv', index=False, float_format='%.3f')
        new.to_csv(run.path / f'new_unlabelled_{name}.csv', index=False, float_format='%.3f')
    (run.path / 'source.json').write_text(json.dumps(vars(args), indent=1))
    pd.set_option('display.width', 200)
    print(summary.round(3).to_string(index=False))
    print(t.round(3).to_string(index=False))
    print('done', run.path)


if __name__ == '__main__':
    main()
