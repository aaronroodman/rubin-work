"""Sky-corrected annulus ratio and flux-weighted overlap at every peak.

Two metrics added to ``peak_metrics.parquet`` (``peak_quality.py``), for the two
fixes of the final selection (``study.md``, 2026-10-21):

- ``annulus_ratio_sky``: ``annulus_ratio`` with the background raised by the sky
  offset. The 10th-percentile background map sits below the sky by about
  1.3 sigma of a binned pixel (15-20 e), a large fraction of a faint corner ring.
  The offset is the mode of the background-subtracted detector.
- ``overlap_w``: the flux-weighted overlap. For every neighbour (the_monster star to
  18.5 AB mag within 3 r_out, or an orphan peak), the fraction of the target's ring,
  widened by ``EDGE_PAD_ROUT`` r_out on both edges, that lies inside the
  neighbour's ring, times the neighbour-over-target flux ratio
  ``10**(-0.4 (m_nbr - m_target))``; summed over neighbours (dimensionless). It is
  about the neighbour's light in the target's ring, as a fraction of the target's.
  ``overlap_w_sep`` leaves out neighbours within ``SEP_ROUT`` r_out: nearly
  concentric with the target, they cannot be told apart from it by any cut.
- ``ring_snr``: sky-corrected ring level over the binned-pixel noise
  (dimensionless).

Usage, from the repo root::

    python aos/giant_donut_pupil/code/peak_refine.py --metrics 20261020_peak_metrics
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import candidates as cd  # noqa: E402  (blitz path first)
import selection_v2 as sv  # noqa: E402

from lsst.obs.lsst import LsstCam  # noqa: E402

from rubinwork.common import study_run  # noqa: E402
from rubinwork.common.utils import nmad  # noqa: E402

PIX_MM = 0.01
EDGE_PAD_ROUT = 0.05
NBR_ROUT = 3.0
SEP_ROUT = 0.25


def sky_offset(sub):
    """Mode of a background-subtracted image and its pixel noise, e per pixel."""
    sigma = nmad((sub[:, 1:] - sub[:, :-1]).ravel()) / np.sqrt(2.0)
    edges = np.arange(-8.0, 8.0 + 0.125, 0.125) * sigma
    hist, _ = np.histogram(sub, bins=edges)
    hist = np.convolve(hist, np.ones(9) / 9.0, mode='same')
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1])), float(sigma)


def weighted_overlap(x_bin, y_bin, r_bin, rin_bin, mag, nbrs, shape):
    """Flux-weighted ring overlap of one donut; ``nbrs`` rows (x_bin, y_bin, mag).

    Returns the sum over neighbours, the largest single term, and the sum over
    neighbours at least ``SEP_ROUT`` r_out away (all dimensionless).
    """
    if len(nbrs) == 0:
        return 0.0, 0.0, 0.0
    pad = EDGE_PAD_ROUT * r_bin
    half = int(np.ceil(r_bin + pad)) + 1
    x0, y0 = int(round(x_bin)), int(round(y_bin))
    ys, xs = np.mgrid[max(0, y0 - half):min(shape[0], y0 + half + 1),
                      max(0, x0 - half):min(shape[1], x0 + half + 1)]
    r = np.hypot(xs - x_bin, ys - y_bin)
    ring = (r >= rin_bin - pad) & (r <= r_bin + pad)
    n_ring = max(int(ring.sum()), 1)
    total, worst, sep = 0.0, 0.0, 0.0
    for xn, yn, mn in nbrs:
        d = np.hypot(xn - x_bin, yn - y_bin)
        if d > 2 * r_bin + pad:
            continue
        rn = np.hypot(xs - xn, ys - yn)
        cover = (ring & (rn >= rin_bin) & (rn <= r_bin)).sum() / n_ring
        w = cover * 10 ** (-0.4 * (mn - mag))
        total += w
        worst = max(worst, w)
        if d >= SEP_ROUT * r_bin:
            sep += w
    return float(total), float(worst), float(sep)


def refine_visit(visit, images, camera, table, cat):
    """New columns for the peaks of one visit."""
    c_xy = cat[['x_mm', 'y_mm']].to_numpy()
    c_mag = cat.mag_best.to_numpy()
    c_id = cat.refcat_id.to_numpy()
    tree = cKDTree(c_xy)
    out = {k: np.full(len(table), np.nan) for k in (
        'sky_offset_e', 'pix_sigma_e', 'annulus_ratio_sky', 'ring_snr', 'overlap_w',
        'overlap_w_max', 'overlap_w_sep')}
    for name, pk in table.groupby('detector'):
        det = camera[name]
        image = images[name]
        sub = image - cd.background_map(image, block=128, percentile=10.0)
        off, sigma = sky_offset(sub)
        r_bin = pk.r_out_pix.iloc[0] / cd.CACHE_BIN
        rin_bin = pk.r_in_pix.iloc[0] / cd.CACHE_BIN
        r_mm = pk.r_out_pix.iloc[0] * PIX_MM
        orphans = pk[pk.match_refcat_id < 0]
        for ti, p in zip(pk.index, pk.itertuples()):
            k = table.index.get_loc(ti)
            level = p.ring_level_e - off
            out['sky_offset_e'][k], out['pix_sigma_e'][k] = off, sigma
            out['annulus_ratio_sky'][k] = (p.annulus_ratio * p.ring_level_e - off) / level \
                if level > 0 else np.nan
            out['ring_snr'][k] = level / sigma
            if not p.full_on_silicon:
                continue
            near = np.asarray(tree.query_ball_point([p.fp_x_mm, p.fp_y_mm], NBR_ROUT * r_mm), int)
            near = near[c_id[near] != p.match_refcat_id]
            nb = [np.column_stack([sv.bin_xy(det, c_xy[near, 0], c_xy[near, 1]), c_mag[near]])]
            o = orphans[orphans.peak_id != p.peak_id]
            if len(o):
                nb.append(o[['x_bin', 'y_bin', 'mag_used']].to_numpy())
            nb = np.vstack(nb)
            out['overlap_w'][k], out['overlap_w_max'][k], out['overlap_w_sep'][k] = weighted_overlap(
                p.x_bin, p.y_bin, r_bin, rin_bin, p.mag_used, nb, image.shape)
    for key, v in out.items():
        table[key] = v
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--metrics', required=True, help='peak_quality run under output/')
    parser.add_argument('--label', default='peak_refine')
    args = parser.parse_args()
    study_dir = pathlib.Path(__file__).resolve().parents[1]
    metrics_dir = study_dir / 'output' / args.metrics
    table = pd.read_parquet(metrics_dir / 'peak_metrics.parquet')
    cats = pd.read_parquet(metrics_dir / 'refcat_fp.parquet')
    camera = LsstCam.getCamera()
    run = study_run(sv.STUDY, args.label)
    parts = []
    for visit in sorted(table.visit.unique()):
        images = {}
        cache = study_dir / 'output' / sv.CACHES[visit] / str(visit)
        for name in table[table.visit == visit].detector.unique():
            images[name] = np.load(cache / f'{name}.npy').astype(float)
        part = table[table.visit == visit].copy()
        parts.append(refine_visit(visit, images, camera, part, cats[cats.visit == visit]))
        print(visit, 'done', flush=True)
    out = pd.concat(parts)
    out.to_parquet(run.path / 'peak_metrics.parquet')
    (run.path / 'source.json').write_text(json.dumps(dict(metrics=args.metrics), indent=1))
    print('done', run.path)


if __name__ == '__main__':
    main()
