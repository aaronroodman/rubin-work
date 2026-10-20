"""danish fits of pupil pass 2: Z4-only and Z4 + Z7 + Z8 with the v4.02g masks (steps 1d and 3c).

Same stamps, neighbour masks, intrinsic (v4.02 batoid without spider, the
side's offsets) and fixed 1.0 arcsec blur as `z4_fits.py`. The only changes
are the danish mask, the v4.02g file fitted at that side's (detector, camera,
M2) offsets (`obscuration_v402g.py`), and, for ``z478``, Z7 and Z8 floated
with Z4. Noll 5-6 and 9-28 stay at the intrinsic.

Model images are written as ``images/<cand>_<side>_v4.02g_<fit>.npz``.

Run, from ``code/``::

    python fits_pass2.py --fits-run ../output/20261018_fits --run ../output/20261019_pupil2
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import fit_donuts as fd  # noqa: E402  (puts the blitz checkout first on sys.path)
import obscuration_v402g as og  # noqa: E402
import z4_fits  # noqa: E402

NOMINAL_MASK = str(og.NOMINAL_MASK)
FITS = {'z4': (4,), 'z478': (4, 7, 8)}
_STAMPS = {}


def _work(task):
    key, fit, run_dir = task
    s = _STAMPS[key]
    mask = yaml.safe_load(og.mask_path(run_dir, s['offsets']).read_text())
    try:
        res = z4_fits.fit_z4(s['stamp'], s['masked'], s['thx'], s['thy'], s['offsets'], 'v4.02',
                             NOMINAL_MASK, s['rtp_deg'], s['r_out_pix'], noll=FITS[fit],
                             mask_params=mask)
    except Exception as err:  # a failed fit is a row, not a crashed run
        return dict(key=key, fit=fit, error=repr(err))
    np.savez_compressed(pathlib.Path(run_dir) / 'images' / f'{key[0]}_{key[1]}_v4.02g_{fit}.npz',
                        model=res.pop('model_img').astype(np.float32))
    res.pop('img')
    res.pop('masked')
    res.update(key=key, fit=fit, error='')
    print(json.dumps(dict(key=key, fit=fit, resid_rms_e=round(res['resid_rms_e'], 1),
                          **{k: round(v, 3) for k, v in res.items() if k.endswith('_dev_um')})),
          flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--fits-run', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--workers', type=int, default=48)
    args = ap.parse_args()
    import multiprocessing as mp
    run = pathlib.Path(args.run).resolve()
    (run / 'images').mkdir(exist_ok=True)
    meta, stamps = z4_fits.load_stamps(pathlib.Path(args.fits_run).resolve())
    _STAMPS.update(stamps)
    fd.pupil('v4.02', NOMINAL_MASK)
    tasks = [(key, fit, str(run)) for key in _STAMPS for fit in FITS]
    with mp.get_context('fork').Pool(args.workers) as pool:
        results = pool.map(_work, tasks, chunksize=1)
    out = []
    for res in results:
        m = meta[(meta.cand_id == res['key'][0]) & (meta.side == res['key'][1])].iloc[0].to_dict()
        row = dict(m, pupil='v4.02g', fit=res['fit'], error=res['error'])
        row.update({k: v for k, v in res.items() if k not in ('key', 'fit', 'error')})
        out.append(row)
    df = pd.DataFrame(out)
    df.to_parquet(run / 'fits_v402g.parquet')
    df.to_csv(run / 'fits_v402g.csv', index=False, float_format='%.5g')
    print('failures', int((df.error != '').sum()), 'of', len(df))


if __name__ == '__main__':
    main()
