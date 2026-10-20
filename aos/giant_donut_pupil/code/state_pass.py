"""Step 3b of pupil pass 2: Z4-only fits and Batoid images with the seq 324 state applied.

Two state variants, each the full v4.02 optic perturbed by 50 AOS DOF
(`giant_state.apply_dof`) before the rotation and the giant offsets:

* ``v4.02s``: the seq 324 ``optical_state`` (22_12) plus the 337/340 Trim tilts
  of M2 and the camera (rx, ry);
* ``v4.02s0``: the seq 324 ``optical_state`` alone.

Per variant and donut side: a danish Z4-only fit (v4.02g mask, blur 1.0
arcsec, Noll 5-28 fixed at the intrinsic of the perturbed optic, without
spider), then the Batoid image through `batoid_images.side_image`, its camera
piston set by that fit's Z4 and its centre by that fit's model image.

Run, from ``code/``::

    python state_pass.py --fits-run ../output/20261018_fits --run ../output/20261019_pupil2
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
import batoid_images as bi  # noqa: E402
import giant_state as gs  # noqa: E402
import obscuration_v402 as ob  # noqa: E402
import obscuration_v402g as og  # noqa: E402
import z4_fits  # noqa: E402

VARIANTS = {'v4.02s': True, 'v4.02s0': False}  # tag: add the Trim tilts
NOMINAL_MASK = str(og.NOMINAL_MASK)
_STAMPS = {}
_OPTICS = {}


def _fit(task):
    key, tag, run_dir = task
    s = _STAMPS[key]
    mask = yaml.safe_load(og.mask_path(run_dir, s['offsets']).read_text())
    try:
        res = z4_fits.fit_z4(s['stamp'], s['masked'], s['thx'], s['thy'], s['offsets'], 'v4.02',
                             NOMINAL_MASK, s['rtp_deg'], s['r_out_pix'], mask_params=mask,
                             optic=_OPTICS[tag])
    except Exception as err:  # a failed fit is a row, not a crashed run
        return dict(key=key, pupil=tag, error=repr(err))
    path = pathlib.Path(run_dir) / 'images' / f'{key[0]}_{key[1]}_{tag}_z4.npz'
    np.savez_compressed(path, model=res.pop('model_img').astype(np.float32))
    res.pop('img')
    res.pop('masked')
    r_outer, r_inner = ob._annulus(mask)
    res.update(key=key, pupil=tag, error='', r_outer_m=r_outer, r_inner_m=r_inner)
    print(json.dumps(dict(key=key, pupil=tag, z4_um=round(res['z4_um'], 2),
                          resid_rms_e=round(res['resid_rms_e'], 1))), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--fits-run', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--workers', type=int, default=64)
    ap.add_argument('--only', default=None, help='one donut side and variant, e.g. G0005_extra,v4.02s')
    ap.add_argument('--out-tag', default='', help='suffix of the output tables')
    args = ap.parse_args()
    import multiprocessing as mp
    run = pathlib.Path(args.run).resolve()
    (run / 'images').mkdir(exist_ok=True)
    meta, stamps = z4_fits.load_stamps(pathlib.Path(args.fits_run).resolve())
    _STAMPS.update(stamps)
    fd.pupil('v4.02', NOMINAL_MASK)
    dofs = {tag: gs.state_dof(run, trim_tilts=tt) for tag, tt in VARIANTS.items()}
    for tag, dof in dofs.items():  # before forking, so workers inherit them
        _OPTICS[tag] = gs.apply_dof(ob.v402_optic(spider=False), dof)
    tasks = [(key, tag, str(run)) for key in _STAMPS for tag in VARIANTS]
    if args.only:
        side, tag = args.only.split(',')
        tasks = [t for t in tasks if f'{t[0][0]}_{t[0][1]}' == side and t[1] == tag]
    with mp.get_context('fork').Pool(min(args.workers, len(tasks))) as pool:
        fits = pool.map(_fit, tasks, chunksize=1)
    rows = []
    for res in fits:
        m = meta[(meta.cand_id == res['key'][0]) & (meta.side == res['key'][1])].iloc[0].to_dict()
        row = dict(m, pupil=res['pupil'], error=res['error'])
        row.update({k: v for k, v in res.items() if k not in ('key', 'pupil', 'error')})
        rows.append(row)
    z4 = pd.DataFrame(rows)
    z4.to_parquet(run / f'z4_fits_state{args.out_tag}.parquet')
    z4.to_csv(run / f'z4_fits_state{args.out_tag}.csv', index=False, float_format='%.5g')
    print('fit failures', int((z4.error != '').sum()), 'of', len(z4), flush=True)

    tasks = []
    for r in z4[z4.error == ''].itertuples(index=False):
        row = r._asdict()
        row.update(out_dir=str(run), dof=dofs[r.pupil],
                   z4_image=str(run / 'images' / f'{r.cand_id}_{r.side}_{r.pupil}_z4.npz'))
        tasks.append(((r.cand_id, r.side), row, NOMINAL_MASK, str(run), r.pupil))
    with mp.get_context('fork').Pool(min(args.workers, len(tasks))) as pool:
        out = pool.map(bi.side_image, tasks, chunksize=1)
    df = pd.DataFrame(out)
    df.to_parquet(run / f'batoid_images_state{args.out_tag}.parquet')
    df.to_csv(run / f'batoid_images_state{args.out_tag}.csv', index=False, float_format='%.6g')
    print('done', run)


if __name__ == '__main__':
    main()
