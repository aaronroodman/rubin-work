"""List exposures taken as giant donuts: both hexapod z Trims stepped by about ±4000 µm.

Giant donuts are made by changing the Trim, not the look-up table (LUT). For each
exposure, the in-focus reference is the latest earlier exposure on the night whose
M2 z and camera z Trim (``dof0``, ``dof5`` of value_added ``visit_telemetry``) both
sit within ``--ref-tol`` µm of the median of the 10 exposures before it. An exposure
is a giant donut when both Trims differ from that reference by ``--step`` ±
``--tol`` µm, with the same sign. ConsDB adds the band, exposure time, block and
observation reason.

Writes ``giant_exposures.csv`` (one row per exposure) and ``giant_nights.csv`` (one
row per night, block, band and side) into a ``giant_nights`` run.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/find_giant_nights.py
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

from rubinwork.common import study_run
from rubinwork.common.telemetry_clients import make_consdb_client

from rubinwork.products import value_added as efd_db  # noqa: E402

STUDY = 'aos/giant_donut_pupil'


def references(night, ref_tol):
    """Index of the in-focus reference exposure for each row of one night, -1 if none."""
    z = night[['dof0', 'dof5']].to_numpy()
    quiet = np.zeros(len(night), bool)
    for i in range(len(night)):
        past = z[max(0, i - 10):i]
        if len(past) >= 3:
            quiet[i] = np.all(np.abs(z[i] - np.nanmedian(past, axis=0)) < ref_tol)
    ref = np.full(len(night), -1)
    last = -1
    for i in range(len(night)):
        ref[i] = last
        if quiet[i]:
            last = i
    return ref


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--step', type=float, default=4000.0, help='Trim step, µm')
    parser.add_argument('--tol', type=float, default=300.0, help='step tolerance, µm')
    parser.add_argument('--ref-tol', type=float, default=300.0,
                        help='in-focus reference tolerance, µm')
    args = parser.parse_args()
    con = efd_db.connect()
    trim = con.execute('select visit_id, day_obs, seq_num, dof0, dof5 from visit_telemetry '
                       'where dof0 is not null and dof5 is not null '
                       'order by day_obs, seq_num').fetchdf()
    hits = []
    for day_obs, night in trim.groupby('day_obs'):
        night = night.reset_index(drop=True)
        ref = references(night, args.ref_tol)
        ok = ref >= 0
        d0 = np.where(ok, night.dof0 - night.dof0.to_numpy()[ref], np.nan)
        d5 = np.where(ok, night.dof5 - night.dof5.to_numpy()[ref], np.nan)
        giant = (ok & (np.abs(np.abs(d0) - args.step) < args.tol)
                 & (np.abs(np.abs(d5) - args.step) < args.tol) & (np.sign(d0) == np.sign(d5)))
        if giant.any():
            sel = night[giant].copy()
            sel['ref_seq'] = night.seq_num.to_numpy()[ref[giant]]
            sel['d_m2_z_um'], sel['d_cam_z_um'] = d0[giant], d5[giant]
            hits.append(sel)
    hits = pd.concat(hits, ignore_index=True)
    # Positive camera and M2 z Trim is extra-focal (20251023: seq 337 at +4000 µm).
    hits['side'] = np.where(hits.d_cam_z_um > 0, 'extra', 'intra')

    cdb = make_consdb_client()
    ids = ','.join(str(v) for v in hits.visit_id)
    meta = cdb.query('select exposure_id as visit_id, band, exp_time, science_program, '
                     'img_type, observation_reason, s_ra, s_dec from cdb_lsstcam.exposure '
                     f'where exposure_id in ({ids})').to_pandas()
    hits = hits.merge(meta, on='visit_id', how='left')
    run = study_run(STUDY, 'giant_nights')
    hits.round(1).to_csv(run.path / 'giant_exposures.csv', index=False)
    nights = (hits.groupby(['day_obs', 'science_program', 'band', 'side'])
              .agg(n_exposures=('visit_id', 'size'), exp_time_s=('exp_time', 'median'),
                   seq_first=('seq_num', 'min'), seq_last=('seq_num', 'max'),
                   ref_seq=('ref_seq', 'first'), ra_deg=('s_ra', 'median'),
                   dec_deg=('s_dec', 'median'))
              .reset_index())
    nights.round(2).to_csv(run.path / 'giant_nights.csv', index=False)
    print(nights.round(2).to_string(index=False))
    print('done', run.path)


if __name__ == '__main__':
    main()
