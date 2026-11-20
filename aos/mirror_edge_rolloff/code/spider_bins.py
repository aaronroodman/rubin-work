"""Per 2 deg bin of each sample side: does a vane shadow cross the donut anywhere?

Traces the full Batoid v4.02 with vanes on a 400 x 400 stop grid (21 mm spacing; the vanes
are 50 mm wide). A bin is flagged when any ray first blocked by a vane lands in it
between 3 pixels inside the nominal inner edge and 3 pixels outside the nominal outer
edge (the ``model_profiles`` edges, about the ray centroid, as there). The per-edge cut
of ``model_profiles`` only looks within 6 pixels of an edge; this covers the whole flat
top, for the whole-profile stacks of ``compare_profiles``.

Writes ``spider_bins.parquet`` (key, bin, spider) into a new study run.

Usage, from the repo root::

    python aos/mirror_edge_rolloff/code/spider_bins.py --models 20261118_model_profiles \\
        --workers 60
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fit_rolloff as fr  # noqa: E402
from rubinwork.giant_donuts import batoid_model as bm  # noqa: E402
from rubinwork.giant_donuts import edge_profiles as ep  # noqa: E402

GRID = 400
MARGIN_PIX = 3.0


def side_task(task):
    row, models_dir = task
    t0 = time.time()
    m = np.load(pathlib.Path(models_dir) / 'profiles' / f'{row["key"]}.npz')
    full = bm.configure(bm.v402_optic(spider=True), float(row['rtp_deg']),
                        bm.offsets_m(row['side'], row['sensor_class'], row['step_mm']))
    x, y = bm.pupil_grid(2 * bm.PUPIL_OUTER_M / GRID)
    first, names, fx, fy = bm.first_clip(full, x, y, np.radians(row['field_x_ccs_deg']),
                                         np.radians(row['field_y_ccs_deg']))
    leaf = np.array([nm.split('.')[-1] for nm in names] + [''])
    spider = np.array([str(v).startswith(bm.SPIDER_PREFIXES) for v in leaf[first]])
    lit = (first < 0) | spider
    x0, y0 = fx[lit].mean(), fy[lit].mean()
    px, py = (fx - x0) / bm.PIXEL_M, (fy - y0) / bm.PIXEL_M
    r = np.hypot(px, py)
    b = np.minimum((np.mod(np.degrees(np.arctan2(py, px)), 360.0) / (360 / ep.NBIN)).astype(int),
                   ep.NBIN - 1)
    e_in, e_out = m['e_in'], m['e_out']
    lo = np.where(np.isfinite(e_in), e_in, np.nanmedian(e_in)) - MARGIN_PIX
    hi = np.where(np.isfinite(e_out), e_out, np.nanmedian(e_out)) + MARGIN_PIX
    hit = spider & (r > lo[b]) & (r < hi[b])
    flag = np.zeros(ep.NBIN, bool)
    flag[np.unique(b[hit])] = True
    print(f'{row["key"]} {flag.sum()} bins, {time.time() - t0:.0f} s', flush=True)
    return pd.DataFrame(dict(key=row['key'], bin=np.arange(ep.NBIN), spider=flag))


def main():
    import multiprocessing as mp
    from rubinwork.common import study_run
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--models', required=True, help='model_profiles run directory name')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    models_dir = fr._study_out() / a.models
    S = pd.read_csv(models_dir / 'sides.csv')
    if a.limit:
        S = S.head(a.limit)
    out = pathlib.Path(a.out) if a.out else study_run(fr.STUDY, 'spider_bins').path
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(r, str(models_dir)) for r in S.to_dict('records')]
    if a.workers > 1:
        with mp.get_context('spawn').Pool(min(a.workers, len(tasks))) as pool:
            parts = list(pool.imap_unordered(side_task, tasks))
    else:
        parts = [side_task(t) for t in tasks]
    pd.concat(parts, ignore_index=True).to_parquet(out / 'spider_bins.parquet', index=False)
    print('done', out, len(parts), 'sides', flush=True)


if __name__ == '__main__':
    main()
