"""Extract PSF-star HSM moments for chosen in-focus visits (RUN ON USDF).

For each visit: select clean calib PSF stars, cut stamps from
preliminary_visit_image, recompute HSM moments (2nd/3rd/4th order, same
estimator as the JAX model) with errors, compute the CCS field angle, and record
the rotator angle.  Writes one parquet per visit: psfmoments_<visit>.parquet.

Usage:
    python extract_psf_moments.py --visits 2026051300025 2026051300028 \
        [--collection LSSTCam/runs/nightlyValidation] [--repo /repo/main] \
        [--out-dir data] [--max-per-ccd 45] [--snr-min 80]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import galsim

# moments_hsm lives beside this script in code/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moments_hsm import measure_hsm_moments               # noqa: E402
from lsst.daf.butler import Butler                          # noqa: E402
from lsst.obs.lsst import LsstCam                           # noqa: E402
from lsst.afw.cameraGeom import PIXELS, FIELD_ANGLE         # noqa: E402
import lsst.geom as geom                                    # noqa: E402

HALF = 16
MKEYS = ['e0', 'e1', 'e2', 'M21', 'M12', 'M30', 'M03',
         'M22', 'M31', 'M13', 'M40', 'M04']
EKEYS = ['e0', 'e1', 'e2', 'M21', 'M12', 'M30', 'M03', 'M22']


def extract_visit(b, cam, visit, collection, out_dir, max_per_ccd, snr_min):
    st = b.get('single_visit_star', collections=collection,
               instrument='LSSTCam', visit=visit).to_pandas()
    snr = st.psfFlux / st.psfFluxErr
    ps = st[st.detect_isPrimary & (st.extendedness < 0.5) & (snr > snr_min)
            & (~st.pixelFlags_saturated) & (~st.pixelFlags_edge)
            & (~st.pixelFlags_interpolatedCenter) & (~st.pixelFlags_bad)
            & np.isfinite(st.psfFlux) & (st.psfFlux > 0)]
    # cap per detector (brightest) for uniform full-focal-plane coverage
    ps = ps.sort_values('psfFlux', ascending=False).groupby('detector').head(max_per_ccd)
    print(f'  {visit}: {len(ps)} stars across {ps.detector.nunique()} detectors')
    vrec = list(b.registry.queryDimensionRecords(
        'visit', where=f"instrument='LSSTCam' and visit={visit}"))[0]
    rot_deg = float(np.degrees(vrec.boresight_rotation_angle.asRadians())) \
        if hasattr(vrec, 'boresight_rotation_angle') else np.nan
    rows = []
    for det in sorted(ps.detector.unique()):
        sub = ps[ps.detector == det]
        try:
            exp = b.get('preliminary_visit_image', collections=collection,
                        instrument='LSSTCam', visit=visit, detector=int(det))
        except Exception:
            continue
        img, var = exp.image.array, exp.variance.array
        ny, nx = img.shape
        tr = cam[int(det)].getTransform(PIXELS, FIELD_ANGLE)
        for _, r in sub.iterrows():
            xi, yi = int(round(r.x)), int(round(r.y))
            if xi < HALF or yi < HALF or xi >= nx - HALF or yi >= ny - HALF:
                continue
            sl = np.s_[yi - HALF:yi + HALF + 1, xi - HALF:xi + HALF + 1]
            stamp = np.ascontiguousarray(img[sl].astype(float))
            nv = float(np.nanmedian(var[sl]))
            try:
                mom, err = measure_hsm_moments(galsim.Image(stamp, scale=0.2),
                                               noise_var=nv)
            except Exception:
                mom = None
            if mom is None:
                continue
            fa = tr.applyForward(geom.Point2D(float(r.x), float(r.y)))
            rec = {'visit': visit, 'detector': int(det), 'x': r.x, 'y': r.y,
                   'thx_ccs_deg': np.degrees(fa.getX()),
                   'thy_ccs_deg': np.degrees(fa.getY()),
                   'rot_deg': rot_deg, 'psfFlux': r.psfFlux,
                   'ixx_tbl': r.ixx, 'iyy_tbl': r.iyy, 'ixy_tbl': r.ixy}
            for k in MKEYS:
                rec[k] = mom[k]
            for k in EKEYS:
                rec[k + '_err'] = np.sqrt(err[k])
            rows.append(rec)
    df = pd.DataFrame(rows)
    out = f'{out_dir}/psfmoments_{visit}.parquet'
    df.to_parquet(out)
    print(f'visit {visit}: rot={rot_deg:.2f} deg, wrote {len(df)} stars -> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--visits', type=int, nargs='+', required=True)
    ap.add_argument('--collection', default='LSSTCam/runs/nightlyValidation')
    ap.add_argument('--repo', default='/repo/main')
    ap.add_argument('--out-dir', default='data')
    ap.add_argument('--max-per-ccd', type=int, default=45)
    ap.add_argument('--snr-min', type=float, default=80.0)
    args = ap.parse_args()

    b = Butler(args.repo)
    cam = LsstCam.getCamera()
    for v in args.visits:
        extract_visit(b, cam, v, args.collection, args.out_dir,
                      args.max_per_ccd, args.snr_min)


if __name__ == '__main__':
    main()
