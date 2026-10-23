"""Pupil pass 6, steps 1, 2 and 4b: the pass 5 chain on the ISR data stamps.

Configurations (one subdirectory of the run each, holding tables named as pass 5's, so
`pass5_analysis.run_all` reads them unchanged):

- ``isr_visitinfo``: ISR stamps, visitInfo rtp, v4.02g, blur 1.0 arcsec (step 1: only the
  ISR differs from pass 5);
- ``isr_phys``: as above with rtp = physical_rotator_angle (step 2; the pass 6 baseline);
- ``isr_phys_v402s``: as ``isr_phys`` with the v4.02s masks (step 4b).

Usage, from the repo root::

    python aos/giant_donut_pupil/code/pass6_data.py --configs isr_visitinfo,isr_phys --workers 100
"""
import argparse
import os
import pathlib
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass6_common as p6  # noqa: E402
import pass6_pipeline as pp  # noqa: E402
import pass5_analysis as pa  # noqa: E402

CONFIGS = {'isr_visitinfo': dict(rtp='visitinfo', mask='v4.02g'),
           'isr_phys': dict(rtp='phys', mask='v4.02g'),
           'isr_phys_v402s': dict(rtp='phys', mask='v4.02s')}
FWHM_ARCSEC = 1.0


def work(task):
    name, row = task
    cfg = CONFIGS[name]
    d = np.load(p6.RUN / 'isr_stamps' / f'{row.sample_id}_{row.side}.npz')
    try:
        out = pp.measure(row, d['img'], d['masked'], FWHM_ARCSEC, mask=cfg['mask'])
    except Exception as err:  # a failed side is a row, not a crashed run
        print('error', name, row.key, repr(err), flush=True)
        return name, None
    print(name, row.key, row.side, f'{out["fit"]["seconds"]:.0f} s', 'found',
          out['fit']['n_found'], flush=True)
    return name, out


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--configs', default='isr_visitinfo,isr_phys')
    ap.add_argument('--workers', type=int, default=100)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    p6.run_dir()
    S0 = p6.sides()
    names = args.configs.split(',')
    tasks, sides = [], {}
    for name in names:
        S = p6.with_rtp(S0, CONFIGS[name]['rtp'])
        if args.limit:
            S = S.head(args.limit)
        sides[name] = S
        tasks += [(name, r) for _, r in S.iterrows()]
    tasks.sort(key=lambda t: t[1].key)
    t0 = time.time()
    res = {n: [] for n in names}
    with mp.get_context('fork').Pool(min(args.workers, len(tasks))) as pool:
        for name, out in pool.imap_unordered(work, tasks):
            if out is not None:
                res[name].append(out)
    for name in names:
        tabs = pp.tables(res[name], sides[name])
        out_dir = p6.RUN / name
        pp.save(tabs, out_dir)
        a = pa.run_all(out_dir)
        print(name, 'sides', len(tabs['fits']))
        print(a['ts'].round(3).to_string(index=False))
        print(a['rs'].round(3).to_string(index=False))
    print(f'wall {time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
