"""Extract corner-WFS aggregate Zernikes (4 corners) for the in-focus visits
(RUN ON USDF).  Resolves the aggregateAOSVisitTableRaw dataset by ref (no
collection-name guessing) and writes cwfs_<visit>.parquet with per-corner
detector, OCS field angle, SNR, and zk_OCS (ztot) / zk_deviation_OCS (zdev).

Usage:
    python extract_cwfs.py --visits 2026051300025 2026051300028 \
        [--collection '*danish_1_2_0*'] [--repo /repo/main] [--out-dir data]
"""
import argparse

import numpy as np
import pandas as pd
from lsst.daf.butler import Butler


def extract_visit(b, visit, collection, out_dir):
    refs = list(b.registry.queryDatasets(
        'aggregateAOSVisitTableRaw', collections=collection,
        where=f"instrument='LSSTCam' and visit={visit}", findFirst=False))
    if not refs:
        refs = list(b.registry.queryDatasets(
            'aggregateAOSVisitTableRaw', collections='*',
            where=f"instrument='LSSTCam' and visit={visit}", findFirst=False))
    if not refs:
        print(visit, 'NO REF')
        return
    ref = refs[0]
    t = b.get(ref)                       # astropy Table (multidim zk columns)
    zdev = np.asarray(t['zk_deviation_OCS'])
    ztot = np.asarray(t['zk_OCS'])
    out = pd.DataFrame({'detector': np.asarray(t['detector']).astype(str),
                        'thx_OCS': np.asarray(t['thx_OCS']),
                        'thy_OCS': np.asarray(t['thy_OCS']),
                        'snr': np.asarray(t['snr'])})
    for i in range(zdev.shape[1]):
        out[f'zdev_{i}'] = zdev[:, i]
        out[f'ztot_{i}'] = ztot[:, i]
    path = f'{out_dir}/cwfs_{visit}.parquet'
    out.to_parquet(path)
    dets = np.asarray(t['detector']).astype(str)
    print(visit, len(out), 'donuts, nZk=', zdev.shape[1], 'corners:',
          sorted(set(d[:3] for d in dets)), ' run:', ref.run)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--visits', type=int, nargs='+', required=True)
    ap.add_argument('--collection', default='*danish_1_2_0*')
    ap.add_argument('--repo', default='/repo/main')
    ap.add_argument('--out-dir', default='data')
    args = ap.parse_args()

    b = Butler(args.repo)
    for v in args.visits:
        extract_visit(b, v, args.collection, args.out_dir)


if __name__ == '__main__':
    main()
