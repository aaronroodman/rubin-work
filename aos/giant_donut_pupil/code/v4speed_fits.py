"""v4_speed refit of the pairs-review exposures: the pass 6 chain with Josh's model.

The pass 6 measurement (`pass6_pipeline.measure`: fits a and c, edges, guided shadows, δ
grid, edge centres) and vane width (`pass6_vanes.one`) with two changes: the danish mask
is the v4_speed obscuration file (`pass6_common.V4SPEED`), and the reference wavefront is
the ``batoid_intrinsic`` v4_speed intrinsicTA (`pupil_pass3.product_reference`), in the
exposure's band. Science sides only. Blur 1.0 arcsec, rtp = physical_rotator_angle.

Sources, each written to ``<run>/v4speed/`` in the layout of a `night2_pipeline` fit
directory (so `psi_labels`, `psi_edges_report` and `pairs_review` read it unchanged):

- ``night1``: run ``20261023_pupil6`` (20251023 337/340, r band), ISR stamps
  ``isr_stamps/<sample_id>_<side>.npz``;
- ``20261107``, ``20261108``: runs ``20261118_night2_<day_obs>`` (i band),
  ``stamps/<key>.fits``.

Steps (``--steps``): ``fit`` (tables, then `pass5_analysis.run_all`), ``vanes``
(central-band science sides: fit c with the blur free, then one free vane width;
``vane_width_*.parquet``) and ``per_side`` (`night2_report.per_side`, with the camera-body
miss from these fits and the vane widths; ``per_side.parquet``).

Usage, from the repo root::

    python aos/giant_donut_pupil/code/v4speed_fits.py --source night1 --workers 64
"""
import argparse
import json
import os
import pathlib
import pickle
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import night2_pipeline as nm  # noqa: E402  (blitz path first)

OUTPUT = nm.OUTPUT
FIT_DIR = 'v4speed'
MASK = 'v4_speed'
FWHM_ARCSEC = 1.0
SOURCES = {'night1': dict(run='20261023_pupil6', band='r'),
           '20261107': dict(run='20261118_night2_20261107', band='i'),
           '20261108': dict(run='20261118_night2_20261108', band='i')}


def _init(source):
    """Point the pass 3–6 modules at the v4_speed mask and reference (pool initializer)."""
    import pass6_common as p6
    import pass6_pipeline as pp
    import pupil_pass3 as p3
    src = SOURCES[source]
    run = OUTPUT / src['run']
    if source != 'night1':
        nm._init(run, src['band'])
    p3.REF_BAND = src['band']
    p3.REFERENCE_PRODUCT = p6.V4SPEED
    pp.DEFAULT_MASK = MASK


def _pool(workers, n, source):
    import multiprocessing as mp
    return mp.get_context('spawn').Pool(max(1, min(workers, n)), initializer=_init,
                                        initargs=(source,), maxtasksperchild=8)


def sides(source):
    """Science sides of a source with ``key``, ``rtp_deg`` (physical) and ``step_mm``."""
    import pass6_common as p6
    run = OUTPUT / SOURCES[source]['run']
    if source == 'night1':
        s = p6.with_rtp(p6.sides(), 'phys')
        s['step_mm'] = 4.0
        s['stamp'] = [str(run / 'isr_stamps' / f'{a}_{b}.npz') for a, b in zip(s.sample_id, s.side)]
    else:
        s = nm.sides(run).copy()
        s['stamp'] = [str(run / 'stamps' / f'{k.replace(":", "_")}.fits') for k in s.key]
    s = s[(s.sensor_class == 'science').to_numpy() & np.array([pathlib.Path(p).exists() for p in s.stamp])]
    return s.reset_index(drop=True)


def read(path):
    if path.endswith('.npz'):
        d = np.load(path)
        return np.asarray(d['img'], float), np.asarray(d['masked'], bool)
    return nm.read_stamp(pathlib.Path(path))


def _part(out, key, side):
    return out / 'fit_parts' / f'{key.replace(":", "_")}_{side}.pkl'


def fit_task(task):
    """One side; the result is kept in ``fit_parts/`` so a rerun resumes."""
    import pass6_pipeline as pp
    row, out = task
    part = _part(out, row.key, row.side)
    if part.exists():
        return
    nm._set_step(row.step_mm)
    img, msk = read(row.stamp)
    try:
        res = pp.measure(row, img, msk, FWHM_ARCSEC, mask=MASK, delta=True)
    except Exception as err:  # a failed side is a row, not a crashed run
        print('error', row.key, row.side, repr(err)[:200], flush=True)
        return
    res['fit']['step_mm'] = row.step_mm
    res['fit']['visit'] = int(row.visit)
    part.parent.mkdir(parents=True, exist_ok=True)
    tmp = part.with_suffix('.tmp')
    tmp.write_bytes(pickle.dumps(res))
    tmp.rename(part)
    print(row.key, row.side, f'{res["fit"]["seconds"]:.0f} s', 'found', res['fit']['n_found'],
          flush=True)


def step_fit(source, out, workers):
    import pass6_pipeline as pp
    import pass5_analysis as pa
    _init(source)
    S = sides(source)
    tasks = [(r, out) for _, r in S.iterrows()]
    t0 = time.time()
    with _pool(workers, len(tasks), source) as pool:
        for _ in pool.imap_unordered(fit_task, tasks):
            pass
    res = [pickle.loads(p.read_bytes()) for p in sorted((out / 'fit_parts').glob('*.pkl'))]
    tabs = pp.tables(res, S.drop(columns='stamp'))
    tabs['fits']['step_mm'] = tabs['fits'].key.map(S.drop_duplicates('key').set_index('key').step_mm)
    pp.save(tabs, out)
    pa.run_all(out)
    print('fit sides', len(tabs['fits']), f'of {len(S)}, wall {time.time() - t0:.0f} s', flush=True)


def vane_task(task):
    import pass6_vanes as pv
    row, fitc = task
    nm._set_step(row.step_mm)
    img, msk = read(row.stamp)
    try:
        fwhm = nm.free_blur(row, img, msk, fitc)
        res, chi, sh = pv.one((row, img, msk, fwhm, fitc, 'data'))
    except Exception as err:
        print('error vanes', row.key, row.side, repr(err)[:200], flush=True)
        return None
    res['step_mm'] = row.step_mm
    res['visit'] = int(row.visit)
    print('vanes', row.key, row.side, f'{res["width_best_mm"]:.1f} mm', flush=True)
    return res, chi, sh


def step_vanes(source, out, workers):
    _init(source)
    S = sides(source)
    S = S[S.field_r_deg.between(*nm.CENTRAL_BAND)].set_index(['key', 'side'])
    F = pd.read_parquet(out / 'fits.parquet')
    F = F[F.fit_ok]
    tasks = []
    for f in F.itertuples():
        if (f.key, f.side) not in S.index:
            continue
        row = S.loc[(f.key, f.side)].copy()
        row['key'], row['side'] = f.key, f.side
        tasks.append((row, f._asdict()))
    res, chi, sh = [], [], []
    t0 = time.time()
    with _pool(workers, len(tasks), source) as pool:
        for r in pool.imap_unordered(vane_task, tasks):
            if r is not None:
                res.append(r[0])
                chi += r[1]
                sh += r[2]
    for name, rows in (('vane_width_data', res), ('vane_width_chi2', chi),
                       ('vane_width_spiders', sh)):
        df = pd.DataFrame(rows)
        df.to_parquet(out / f'{name}.parquet')
        df.to_csv(out / f'{name}.csv', index=False, float_format='%.5g')
    print(f'vanes {len(res)} of {len(tasks)}, wall {time.time() - t0:.0f} s', flush=True)


def step_per_side(source, out):
    import night2_report as nr
    s = pd.read_parquet(out / 'sides.parquet')
    t = nr.per_side(out, s, out)
    w = pd.read_parquet(out / 'vane_width_data.parquet')
    if len(w):
        t['vane_width_mm'] = t.key.map(w.drop_duplicates('key').set_index('key').width_best_mm)
    t['source'] = source
    t.to_parquet(out / 'per_side.parquet')
    t.to_csv(out / 'per_side.csv', index=False, float_format='%.5g')
    print('per_side', len(t), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--source', required=True, choices=sorted(SOURCES))
    ap.add_argument('--steps', default='fit,vanes,per_side')
    ap.add_argument('--workers', type=int, default=64)
    a = ap.parse_args()
    import pass6_common as p6
    from rubinwork.products import catalog
    out = OUTPUT / SOURCES[a.source]['run'] / FIT_DIR
    out.mkdir(parents=True, exist_ok=True)
    m = catalog.manifest('batoid_intrinsic', *p6.V4SPEED)
    (out / 'reference.json').write_text(json.dumps(dict(
        product='batoid_intrinsic', variant=p6.V4SPEED[0], build=p6.V4SPEED[1],
        band=SOURCES[a.source]['band'], obscuration_file=m['config']['obscuration_file'],
        obscuration_sha256=m['config']['obscuration_sha256'], fwhm_arcsec=FWHM_ARCSEC),
        indent=1))
    t0 = time.time()
    for step in a.steps.split(','):
        print('step', step, f'{time.time() - t0:.0f} s', flush=True)
        if step == 'fit':
            step_fit(a.source, out, a.workers)
        elif step == 'vanes':
            step_vanes(a.source, out, a.workers)
        elif step == 'per_side':
            step_per_side(a.source, out)
        else:
            raise ValueError(f'unknown step {step}')
    print('done', out, f'{time.time() - t0:.0f} s', flush=True)


if __name__ == '__main__':
    main()
