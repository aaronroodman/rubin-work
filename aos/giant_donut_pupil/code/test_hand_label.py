"""Headless test of the v2 review tool on a copy of the diag run.

Draws a page, toggles a marked donut, adds and removes a missed donut, presses the
page button, checks the legend sits outside the image axes, and reloads the CSV.
Writes ``test_page.png`` into ``--out``.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/test_hand_label.py --out /tmp/hl_test
"""
import argparse
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import hand_label as hl  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')

from lsst.obs.lsst import LsstCam  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--diag', default='20261020_selection_diag')
    parser.add_argument('--metrics', default='20261020_peak_metrics')
    args = parser.parse_args()
    study = pathlib.Path(__file__).resolve().parents[1]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ('hand_labels_v2.csv', 'hand_labels_v2_pages.json'):
        (out / f).unlink(missing_ok=True)
    shutil.copy(study / 'output' / args.diag / 'stars.parquet', out / 'stars.parquet')
    camera = LsstCam.getCamera()
    lab = hl.Labeller(out, study / 'output' / args.metrics, study, camera)
    visit, raft = hl.PAGES[0]
    marked = lab.marked(visit, raft)
    both = marked[marked.star_isolated & marked.good_peak]
    print(f'{visit} {raft}: {len(marked)} marked, {len(both)} both, '
          f'{int((marked.star_isolated & ~marked.good_peak).sum())} ^ only, '
          f'{int((~marked.star_isolated & marked.good_peak).sum())} v only')
    assert len(marked)

    fig, ax = hl.make_figure()
    mosaic, geometry = lab.raft_image(visit, raft)
    lab.draw(ax, visit, raft, mosaic, geometry)

    # Toggle: unreviewed -> bad -> unreviewed.
    item = marked.iloc[0]
    print(lab.click(visit, raft, item.fp_x_mm + 0.1 * item.r_out_mm, item.fp_y_mm))
    assert lab.label_of(visit, raft, item.key) == 'bad'
    print(lab.click(visit, raft, item.fp_x_mm, item.fp_y_mm))
    assert lab.label_of(visit, raft, item.key) == 'unreviewed'
    assert len(lab.labels) == 0
    lab.click(visit, raft, item.fp_x_mm, item.fp_y_mm)

    # Add a missed donut in empty sky (far from every marked donut), then remove it.
    c = camera[lab.raft_names(raft)[4]].getCenter(hl.FOCAL_PLANE)
    grid = [(c.getX() + dx, c.getY() + dy) for dx in np.linspace(-15, 15, 13)
            for dy in np.linspace(-15, 15, 13)]
    far = max(grid, key=lambda p: np.min(np.hypot(marked.fp_x_mm - p[0], marked.fp_y_mm - p[1])))
    print(lab.click(visit, raft, *far))
    assert (lab.labels.label == 'missed').sum() == 1
    print(lab.click(visit, raft, far[0] + 0.3, far[1]))
    assert (lab.labels.label == 'missed').sum() == 0
    print(lab.click(visit, raft, *far))

    # Page button: the rest become good.
    n = lab.finish_page(visit, raft)
    labels = lab.page_labels(visit, raft)
    assert n == len(marked) - 1, (n, len(marked))
    assert (labels.label == 'good').sum() == len(marked) - 1
    assert (labels.label == 'bad').sum() == 1 and (labels.label == 'missed').sum() == 1
    assert lab.is_done(visit, raft)
    # On a done page, bad toggles back to good.
    lab.click(visit, raft, item.fp_x_mm, item.fp_y_mm)
    assert lab.label_of(visit, raft, item.key) == 'good'
    lab.click(visit, raft, item.fp_x_mm, item.fp_y_mm)
    assert lab.label_of(visit, raft, item.key) == 'bad'

    # Legend outside the image axes.
    lab.draw(ax, visit, raft)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    leg = ax.get_legend().get_window_extent(renderer)
    axbox = ax.get_window_extent(renderer)
    figbox = fig.bbox
    print(f'legend x0 {leg.x0:.0f} > axes x1 {axbox.x1:.0f}; legend x1 {leg.x1:.0f} '
          f'<= figure {figbox.x1:.0f} (display pixels)')
    assert leg.x0 > axbox.x1 and leg.x1 <= figbox.x1
    fig.savefig(out / 'test_page.png', dpi=80)

    # Next page and back, then a click on each: the overlay must redraw after clear().
    for v2, r2 in (hl.PAGES[1], hl.PAGES[0]):
        m2, g2 = lab.raft_image(v2, r2)
        lab.draw(ax, v2, r2, m2, g2)
        assert ax.get_legend() is not None, (v2, r2)
        lab.draw(ax, v2, r2)
    m1 = lab.marked(*hl.PAGES[1])
    it1 = m1.iloc[0]
    print(lab.click(*hl.PAGES[1], it1.fp_x_mm, it1.fp_y_mm))
    lab.draw(ax, *hl.PAGES[1], *lab.raft_image(*hl.PAGES[1]))
    lab.draw(ax, *hl.PAGES[1])
    assert lab.label_of(*hl.PAGES[1], it1.key) == 'bad'
    lab.click(*hl.PAGES[1], it1.fp_x_mm, it1.fp_y_mm)
    lab.draw(ax, visit, raft, mosaic, geometry)

    # CSV round trip.
    on_disk = pd.read_csv(out / 'hand_labels_v2.csv')
    assert list(on_disk.columns) == hl.COLUMNS, list(on_disk.columns)
    assert len(on_disk) == len(lab.labels) == len(marked) + 1
    again = hl.Labeller(out, study / 'output' / args.metrics, study, camera)
    for k in marked.key:
        assert again.label_of(visit, raft, k) == lab.label_of(visit, raft, k)
    assert again.is_done(visit, raft)
    print(on_disk.label.value_counts().to_dict())
    print(on_disk[['star_isolated', 'good_peak']].value_counts().to_dict())
    print('all checks passed')


if __name__ == '__main__':
    main()
