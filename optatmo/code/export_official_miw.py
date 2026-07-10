"""Export the official ip_isr ``intrinsicZernikes`` calib (MIW, CCD-height in
CCS Z4) from the Butler to two parquet files our OCS-reconstruction MIW loader
(miw.py) consumes:

    intrinsic_official_ocs.parquet   thx_deg, thy_deg, Z{j}_OCS   (shared disk)
    intrinsic_official_ccs.parquet   detector, thx_deg, thy_deg, Z{j}_CCS
                                     (per-detector footprint; CCD height already
                                      folded into Z4 at every footprint point)

Values are microns; (thx_deg, thy_deg) are CCS field angles in degrees.  This is
the exact ts_wep calibration product (v3 = per-CCD footprint grid, height at
each point).  RUN ON USDF (needs Butler + ip_isr).

Usage:
    python export_official_miw.py \
        [--collection u/gmegias/calib/DM-55048/intrinsicZernikes.v3] \
        [--filter i_39] [--repo /repo/main] [--out-dir <dir>]
"""
import argparse
import numpy as np
import pandas as pd
from lsst.daf.butler import Butler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='/repo/main')
    ap.add_argument('--collection',
                    default='u/gmegias/calib/DM-55048/intrinsicZernikes.v3')
    ap.add_argument('--filter', default='i_39')       # physical_filter
    ap.add_argument('--instrument', default='LSSTCam')
    ap.add_argument('--out-dir', default='.')
    args = ap.parse_args()

    b = Butler(args.repo)
    refs = list(b.registry.queryDatasets(
        'intrinsicZernikes', collections=args.collection,
        where=f"instrument='{args.instrument}' and physical_filter='{args.filter}'",
        findFirst=True))
    dets = sorted({r.dataId['detector'] for r in refs})
    if not dets:
        raise SystemExit(f'no intrinsicZernikes in {args.collection} '
                         f'for filter {args.filter}')
    print(f'{len(dets)} detectors in {args.collection} (filter {args.filter})')

    # --- shared OCS map (identical across detectors): take it from the first ---
    c0 = b.get('intrinsicZernikes', collections=args.collection,
               instrument=args.instrument, detector=dets[0],
               physical_filter=args.filter)
    noll = [int(j) for j in np.asarray(c0.noll_indices)]
    ocs = {'thx_deg': np.asarray(c0.field_x_ocs, float),
           'thy_deg': np.asarray(c0.field_y_ocs, float)}
    vocs = np.asarray(c0.values_ocs, float)                  # (n_ocs, n_noll)
    for k, j in enumerate(noll):
        ocs[f'Z{j}_OCS'] = vocs[:, k]
    pd.DataFrame(ocs).to_parquet(f'{args.out_dir}/intrinsic_official_ocs.parquet')
    print(f'  OCS: {vocs.shape[0]} pts, Noll {noll} -> intrinsic_official_ocs.parquet')

    # --- per-detector CCS maps (height already in Z4) --------------------------
    rows = []
    for d in dets:
        c = b.get('intrinsicZernikes', collections=args.collection,
                  instrument=args.instrument, detector=d,
                  physical_filter=args.filter)
        x = np.asarray(c.field_x, float)
        y = np.asarray(c.field_y, float)
        v = np.asarray(c.values, float)                      # (n_pts, n_noll)
        rec = {'detector': np.full(x.size, d, int),
               'thx_deg': x, 'thy_deg': y}
        for k, j in enumerate(noll):
            rec[f'Z{j}_CCS'] = v[:, k]
        rows.append(pd.DataFrame(rec))
    ccs = pd.concat(rows, ignore_index=True)
    ccs.to_parquet(f'{args.out_dir}/intrinsic_official_ccs.parquet')
    print(f'  CCS: {len(ccs)} pts across {len(dets)} detectors '
          f'-> intrinsic_official_ccs.parquet')

    # detector id<->name map (so the corner comparison can map CWFS sensor
    # names like 'R00_SW0' to the integer detector ids used above)
    try:
        from lsst.obs.lsst import LsstCam
        cam = LsstCam.getCamera()
        names = pd.DataFrame({'detector': [d.getId() for d in cam],
                              'name': [d.getName() for d in cam]})
        names.to_parquet(f'{args.out_dir}/detector_names.parquet')
        print(f'  names: {len(names)} detectors -> detector_names.parquet')
    except Exception as e:
        print(f'  (skipped detector_names: {e})')


if __name__ == '__main__':
    main()
