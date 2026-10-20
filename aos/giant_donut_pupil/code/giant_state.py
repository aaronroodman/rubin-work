"""Telescope state for the giant donuts (step 1 of pupil pass 2), and model checksums.

Reference visit: day_obs 20251023 seq 336, the in-focus FAM reference
(BLOCK-T626) taken just before the giant pair 337 (extra) and 340 (intra).

* Optical state: value_added ``optical_state``, variant
  ``v22_12__batoid__consdb_v1`` (22 DOF, 12 v-modes, batoid intrinsic, as run
  online), current catalog build. The corner Zernike deviation (ConsDB OPD
  minus the batoid intrinsic) is recomputed with the builder's own
  `measured_deviation`. This is a reference only: none of it enters the
  Batoid or danish models.
* Trim: ``visit_telemetry`` ``dof0..9`` (``MTAOS.logevent_degreeOfFreedom``) at
  337 and 340 against 336.
* Hexapod: EFD ``MTHexapod.application`` actual position minus demand over
  each exposure, and the demand change from 336 split into the elevation LUT
  change (``logevent_compensationOffset``) and the rest.

Hexapod tilts are reported in arcsec; the DOF tables store them in deg.

Run, from ``code/``::

    python giant_state.py --run ../output/20261019_pupil2
"""
import argparse
import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

DAY_OBS = 20251023
REF_SEQ = 336
GIANT_SEQ = {'extra': 337, 'intra': 340}
VARIANT = 'v22_12__batoid__consdb_v1'
EXP_TIME_S = {336: 30.0, 337: 60.0, 340: 60.0}
HEX = {1: 'camera', 2: 'M2'}
AXES = ('x', 'y', 'z', 'u', 'v', 'w')
TRIM_NAMES = ('M2 z', 'M2 x', 'M2 y', 'M2 rx', 'M2 ry', 'camera z', 'camera x', 'camera y',
              'camera rx', 'camera ry')
TILT_DOF = (3, 4, 8, 9)
V402_YAML = pathlib.Path.home() / 'notebooks' / 'rubin-optical-model' / 'LSST_r_v4.02.yaml'


def visit_id(seq):
    return DAY_OBS * 100000 + seq


def optical_state():
    """22_12 DOF and corner Z4-Z26 deviation of the reference visit.

    Returns
    -------
    dof : `pandas.DataFrame`
        One row per DOF index (µm, or deg for the four tilts).
    zk : `pandas.DataFrame`
        Per corner and Noll index: OPD, intrinsic and deviation, µm of wavefront.
    meta : `dict`
    """
    from rubinwork import aos_state
    from rubinwork.common.telemetry_clients import make_consdb_client
    from rubinwork.products.value_added import reader
    from rubinwork.products.value_added.builders import build_optical_state as bos
    con = reader.connect()
    row = con.execute('SELECT * FROM optical_state WHERE visit_id = ? AND variant_id = ?',
                      [visit_id(REF_SEQ), VARIANT]).df().iloc[0]
    dof = np.asarray(row.dof, float)
    zk = aos_state.ZK_NOLL
    dev, _, opd = bos.measured_deviation(make_consdb_client(), [visit_id(REF_SEQ)], ['r'],
                                         [row.rotator_angle_deg], 'batoid', zk)
    rows = []
    for c, corner in enumerate(aos_state.SENSOR_NAMES):
        for k, j in enumerate(zk):
            i = c * len(zk) + k
            rows.append(dict(corner=corner, noll=j, opd_um=opd[0, i],
                             intrinsic_um=opd[0, i] - dev[0, i], deviation_um=dev[0, i]))
    meta = dict(visit_id=visit_id(REF_SEQ), variant=VARIANT, db=str(reader.default_db_path()),
                resid_rms_um=float(row.resid_rms_um), fwhm_cwfs_arcsec=float(row.fwhm_cwfs_arcsec),
                rotator_angle_deg=float(row.rotator_angle_deg),
                elevation_deg=float(row.elevation_deg),
                v_modes=[float(v) for v in row.v_modes])
    return (pd.DataFrame(dict(dof_index=np.arange(50), value=dof,
                              unit=['deg' if i in TILT_DOF else 'um' for i in range(50)])),
            pd.DataFrame(rows), meta)


def trim_check():
    """Trim at 337 and 340 minus Trim at 336, µm and arcsec."""
    from rubinwork.products.value_added import reader
    con = reader.connect()
    cols = ','.join(f'dof{i}' for i in range(10))
    df = con.execute(f'SELECT seq_num, obs_start, {cols} FROM visit_telemetry WHERE day_obs = ? '
                     f'AND seq_num IN (336, 337, 340)', [DAY_OBS]).df().set_index('seq_num')
    rows = []
    for side, seq in GIANT_SEQ.items():
        for i, name in enumerate(TRIM_NAMES):
            d = df.loc[seq, f'dof{i}'] - df.loc[REF_SEQ, f'dof{i}']
            tilt = i in TILT_DOF
            rows.append(dict(side=side, seq=seq, dof=name, unit='arcsec' if tilt else 'um',
                             trim_ref=df.loc[REF_SEQ, f'dof{i}'] * (3600 if tilt else 1),
                             trim_minus_ref=d * (3600 if tilt else 1)))
    return pd.DataFrame(rows), df.obs_start.to_dict()


def hexapod_check(obs_start):
    """EFD actual minus demand per exposure, and the demand change from 336 less the LUT change."""
    from astropy.time import Time
    from rubinwork.common.dof_telemetry import _run_coro
    from rubinwork.common.telemetry_clients import make_efd_client
    efd = make_efd_client()
    demand, lut, rows = {}, {}, []
    for seq in (REF_SEQ,) + tuple(GIANT_SEQ.values()):
        t0 = Time(obs_start[seq], scale='tai')
        t1 = t0 + EXP_TIME_S[seq] / 86400
        app = _run_coro(efd.select_time_series('lsst.sal.MTHexapod.application', ['*'], t0, t1))
        off = _run_coro(efd.select_time_series('lsst.sal.MTHexapod.logevent_compensationOffset',
                                               ['*'], t0, t1))
        for idx, hexa in HEX.items():
            a, o = app[app.salIndex == idx], off[off.salIndex == idx]
            demand[(seq, idx)] = np.array([a[f'demand{k}'].mean() for k in range(6)])
            lut[(seq, idx)] = np.array([o[ax].mean() for ax in AXES])
            for k, ax in enumerate(AXES):
                err = (a[f'position{k}'] - a[f'demand{k}']).to_numpy(float)
                scale = 3600.0 if k >= 3 else 1.0
                rows.append(dict(seq=seq, hexapod=hexa, axis=ax, n_samples=len(a),
                                 unit='arcsec' if k >= 3 else 'um',
                                 actual_minus_demand_mean=float(err.mean() * scale),
                                 actual_minus_demand_maxabs=float(np.abs(err).max() * scale)))
    act = pd.DataFrame(rows)
    rows = []
    for side, seq in GIANT_SEQ.items():
        for idx, hexa in HEX.items():
            dd = demand[(seq, idx)] - demand[(REF_SEQ, idx)]
            dl = lut[(seq, idx)] - lut[(REF_SEQ, idx)]
            for k, ax in enumerate(AXES):
                scale = 3600.0 if k >= 3 else 1.0
                rows.append(dict(side=side, seq=seq, hexapod=hexa, axis=ax,
                                 unit='arcsec' if k >= 3 else 'um',
                                 demand_change=dd[k] * scale, lut_change=dl[k] * scale,
                                 demand_change_less_lut=(dd[k] - dl[k]) * scale))
    return act, pd.DataFrame(rows)


def checksums():
    """md5 of the Batoid model files the pass reads."""
    import batoid
    files = {'v4.02': V402_YAML,
             'v1000': pathlib.Path(batoid.datadir) / 'LSST' / 'Rubin_v1000_r.yaml'}
    return {k: dict(path=str(p), md5=hashlib.md5(p.read_bytes()).hexdigest()) for k, p in files.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    run = pathlib.Path(args.run).resolve()
    dof, zk, meta = optical_state()
    trim, obs_start = trim_check()
    act, demand = hexapod_check(obs_start)
    meta['models'] = checksums()
    for name, df in (('state336_dof', dof), ('state336_zk', zk), ('trim_check', trim),
                     ('hexapod_actual', act), ('hexapod_demand', demand)):
        df.to_parquet(run / f'{name}.parquet')
        df.to_csv(run / f'{name}.csv', index=False, float_format='%.6g')
    (run / 'state336.json').write_text(json.dumps(meta, indent=1))
    pd.set_option('display.width', 200)
    print(json.dumps(meta['models'], indent=1))
    print(zk[zk.noll.isin([5, 6, 7, 8])].pivot(index='corner', columns='noll',
                                                values='deviation_um').round(3).to_string())
    print(trim[trim.trim_minus_ref.abs() > 1e-6].to_string())
    print(act.groupby(['hexapod', 'axis']).actual_minus_demand_maxabs.max().round(3).to_string())
    print(demand.round(3).to_string())


if __name__ == '__main__':
    main()
