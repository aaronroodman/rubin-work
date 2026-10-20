"""Final v2 giant-donut candidate table and pages, from a scored selection_diag run.

Reads ``stars.parquet``, ``peaks.parquet``, ``hand_labels.csv`` and
``score_detail.json`` from the diag run. Applies the chosen variant
(``score_selection.combine``), attaches the hand labels, and writes into a new
``candidates_v2`` run:

- ``candidates_v2.parquet`` and ``.csv``: one row per star and visit, the four
  20251023 exposures;
- ``candidates_<visit>_<side>.pdf``: mosaic, tally, raft and CCD pages, the
  circles coloured by the final class;
- ``tally.pdf`` and ``score_summary.pdf``.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/final_candidates.py --diag 20261020_selection_diag
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import candidates as cd  # noqa: E402  (blitz path first)
import score_selection as ss  # noqa: E402
import select_candidates as sc  # noqa: E402
import selection_pages as pages  # noqa: E402
import selection_v2 as sv  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from lsst.obs.lsst import LsstCam  # noqa: E402

from rubinwork.common import study_run  # noqa: E402


def attach_labels(stars, truth):
    """Hand label per star row; '' where none."""
    stars['hand_label'] = ''
    for row, label in zip(truth.row, truth.label):
        if row >= 0:
            stars.at[row, 'hand_label'] = label
    return stars


def tally_final(stars, peaks):
    rows = []
    for (visit, cls), t in stars.groupby(['visit', 'sensor_class']):
        full = t[t.full_on_silicon]
        pk = peaks[(peaks.visit == visit) & (peaks.sensor_class == cls)]
        row = dict(visit=visit, side=sv.VISITS[visit], sensor_class=cls, on_silicon=len(t),
                   full_on_silicon=len(full), scored=int((t.iso_class != '').sum()))
        for c in ('isolated', 'marginal', 'blended', 'saturated'):
            row[c] = int((t.iso_class == c).sum())
        row['peaks_unmatched'] = int(pk.unmatched.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--diag', required=True)
    args = parser.parse_args()
    study_dir = pathlib.Path(__file__).resolve().parents[1]
    diag_dir = study_dir / 'output' / args.diag
    stars, truth, classes, scores, chosen, resid_max = ss.run(diag_dir)
    peaks = pd.read_parquet(diag_dir / 'peaks.parquet')
    stars['iso_class_v1'] = stars.iso_class
    final = classes['combined']
    stars['iso_class'] = np.where(final == 'unscored', '', final)
    stars['fit_resid_ok'] = stars.annulus_resid < resid_max
    stars = attach_labels(stars, truth)
    stars['selection'] = 'v2: ' + (', '.join(chosen) if chosen else 'baseline')

    run = study_run(sv.STUDY, 'candidates_v2')
    stars.to_parquet(run.path / 'candidates_v2.parquet')
    stars.to_csv(run.path / 'candidates_v2.csv', index=False, float_format='%.4f')
    counts = tally_final(stars, peaks)
    counts.to_csv(run.path / 'tally.csv', index=False)
    summary = ss.summary_frame(scores)
    summary.to_csv(run.path / 'score_summary.csv', index=False, float_format='%.3f')
    (run.path / 'selection.json').write_text(json.dumps(
        dict(diag=args.diag, chosen=chosen, annulus_resid_max=resid_max), indent=1))

    with PdfPages(run.path / 'tally.pdf') as pdf:
        pages._text_page(pdf, 'v2 tally per visit and sensor class (star and peak counts); '
                         f'selection {stars.selection.iloc[0]}', counts.to_string(index=False), 8)
    with PdfPages(run.path / 'score_summary.pdf') as pdf:
        text = summary.round(3).to_string(index=False)
        pages._text_page(pdf, 'Completeness (good donuts kept / good donuts) and purity (kept '
                         'labelled good / kept labelled), dimensionless,\nagainst the hand labels '
                         f'on the pages marked done. Chosen: {chosen}; annulus_resid cut '
                         f'{resid_max:.3f} (dimensionless)', text, 6)
        for name in ('baseline', 'combined'):
            conf = scores[name]['all']['confusion']
            frame = pd.DataFrame(conf).T.loc[list(ss.TRUTH_LABELS), list(ss.CLASSES)]
            pages._text_page(pdf, f'Confusion matrix, {name}: rows hand label, columns class '
                             '(donut counts)', frame.to_string(), 9)

    camera = LsstCam.getCamera()
    check = json.loads((diag_dir / 'checks.json').read_text())
    for visit, side in sv.VISITS.items():
        images, _ = sc.load_cache(study_dir / 'output' / sv.CACHES[visit] / str(visit), camera)
        backgrounds = {n: cd.background_map(im, block=128, percentile=10.0)
                       for n, im in images.items()}
        pages.write_diag_pdf(run.path / f'candidates_{visit}_{side}.pdf', visit, side,
                             stars[stars.visit == visit], peaks[peaks.visit == visit],
                             pd.read_csv(diag_dir / 'tally.csv').query('visit == @visit'),
                             check, images, backgrounds, camera)
        print('pdf', visit, flush=True)
    print(summary[summary.group == 'all'].round(3).to_string(index=False))
    print('done', run.path)


if __name__ == '__main__':
    main()
