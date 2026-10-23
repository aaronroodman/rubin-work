"""Pupil pass 6, step 1a: full-resolution stamps from a full ISR with calibrations.

Per (visit, detector) of the 188 sides: `lsst.ip.isr.IsrTaskLSST` with the obs_lsst
LSSTCam overrides (``isrLSST``), doing overscan, assembly, crosstalk, linearity, bias,
dark, flat, defects (interpolated) and deferred-charge (CTI) correction, with PTC gains.
Brighter-fatter correction is off; the per-exposure gain correction is off because no
``gain_correction`` calibration covers 20251023. Calibrations come from
``LSSTCam/calib`` at the exposure's time.

The stamp is then cut exactly as passes 3–5 (`fit_donuts.cut`: CCS rotation, neighbours
masked, one sigma-clipped constant outside 1.1 r_out subtracted).

Outputs in the run: ``isr_stamps/<sample_id>_<side>.npz`` (``img``, ``masked``,
``sky_e``), ``isr_stamps.parquet`` (cut info per side), ``isr_calibs.csv`` (the
calibration run of every input, per detector) and ``isr_config.py`` (the task config).

Usage, from the repo root::

    python aos/giant_donut_pupil/code/pass6_isr.py --workers 40
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
import fit_donuts as fd  # noqa: E402
import pupil_pass3 as p3  # noqa: E402

CALIBS = (('bias', 'bias'), ('dark', 'dark'), ('flat', 'flat'), ('defects', 'defects'),
          ('linearizer', 'linearizer'), ('crosstalk', 'crosstalk'),
          ('deferredChargeCalib', 'cti'), ('ptc', 'ptc'))
_CTX = {}


def isr_config():
    from lsst.ip.isr import IsrTaskLSSTConfig
    from lsst.obs.lsst import LsstCam
    cfg = IsrTaskLSSTConfig()
    LsstCam().applyConfigOverrides('isrLSST', cfg)
    cfg.doBrighterFatter = False
    cfg.doCorrectGains = False
    for name in ('doBias', 'doDark', 'doFlat', 'doDefect', 'doDeferredCharge', 'doLinearize',
                 'doCrosstalk'):
        setattr(cfg, name, True)
    return cfg


def chunk(task):
    (visit, det_name), grp, out_dir = task
    import lsst.daf.butler as dafButler
    from lsst.ip.isr import IsrTaskLSST
    if 'butler' not in _CTX:
        _CTX['butler'] = dafButler.Butler('/repo/main', collections=[p6.RAW_COLLECTION,
                                                                      p6.CALIB_COLLECTION])
    b = _CTX['butler']
    camera = p3._CTX['camera']
    t0 = time.time()
    did = dict(instrument='LSSTCam', exposure=int(visit), detector=camera[det_name].getId())
    raw_ref = b.find_dataset('raw', did, collections=p6.RAW_COLLECTION)
    raw = b.get(raw_ref)
    did = b.registry.expandDataId(raw_ref.dataId)
    inputs, runs = {}, dict(visit=int(visit), detector=det_name)
    span = did.records['exposure'].timespan
    for arg, dt in CALIBS:
        ref = b.find_dataset(dt, did, collections=p6.CALIB_COLLECTION, timespan=span)
        inputs[arg] = b.get(dt, did)
        runs[dt] = ref.run if ref is not None else ''
    exp = IsrTaskLSST(config=_CTX['config']).run(raw, camera=camera, **inputs).exposure
    image = np.asarray(exp.image.array, float)
    isr_s = time.time() - t0
    neigh = p3.neighbours(visit, det_name, camera, p3._CTX['cat'][p3._CTX['cat'].visit == visit],
                          p3._CTX['orphans'])
    n_quarter = camera[det_name].getOrientation().getNQuarter()
    rows = []
    for row in grp.itertuples():
        stamp, masked, info = fd.cut(image, pd.Series(dict(
            r_out_pix=row.r_out_pix, x_pix=row.x_pix, y_pix=row.y_pix, mag_r=row.mag)),
            neigh, n_quarter)
        np.savez_compressed(pathlib.Path(out_dir) / 'isr_stamps' / f'{row.sample_id}_{row.side}.npz',
                            img=stamp.astype(np.float32), masked=masked, sky_e=info['sky_e'])
        rows.append(dict(key=row.key, sample_id=row.sample_id, side=row.side, visit=int(visit),
                         detector=det_name, isr_seconds=isr_s, **info))
    print(det_name, visit, len(grp), f'{time.time() - t0:.0f} s', flush=True)
    return rows, runs


def main():
    import multiprocessing as mp
    from lsst.obs.lsst import LsstCam
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--workers', type=int, default=40)
    ap.add_argument('--limit', type=int, default=None, help='first N detectors only')
    args = ap.parse_args()
    run = p6.run_dir()
    (run / 'isr_stamps').mkdir(exist_ok=True)
    S = p6.sides()
    p3._CTX['camera'] = LsstCam.getCamera()
    p3._CTX['cat'] = pd.read_parquet(p3.METRICS_RUN / 'refcat_fp.parquet')
    pk = pd.read_parquet(p3.REFINE_RUN / 'peak_metrics.parquet')
    p3._CTX['orphans'] = pk[pk.match_refcat_id < 0][['visit', 'detector', 'x_pix', 'y_pix']]
    _CTX['config'] = isr_config()
    (run / 'isr_config.py').write_text(_CTX['config'].saveToString())
    tasks = [(k, g, str(run)) for k, g in S.groupby(['visit', 'detector'])]
    if args.limit:
        tasks = tasks[:args.limit]
    print('detectors', len(tasks), 'sides', sum(len(t[1]) for t in tasks), flush=True)
    t0 = time.time()
    rows, runs = [], []
    with mp.get_context('fork').Pool(min(args.workers, len(tasks))) as pool:
        for r, c in pool.imap_unordered(chunk, tasks):
            rows += r
            runs.append(c)
    pd.DataFrame(rows).to_parquet(run / 'isr_stamps.parquet')
    pd.DataFrame(runs).to_csv(run / 'isr_calibs.csv', index=False)
    print(f'wall {time.time() - t0:.0f} s', run, flush=True)


if __name__ == '__main__':
    main()
