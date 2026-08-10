#!/usr/bin/env python
"""
Compare the simulated guider moments against on-sky data, using the SAME moment
code (guiderMoments.measureStampMoments) so the numbers are directly comparable.

Reproduces the decomposeDetector scheme on the simulated stamps: per stamp a
fixed-sigma weighted centroid + unweighted moments about it; then the flux-weighted
mean per-stamp moments (the "static" shape) and the flux-weighted covariance of the
centroids (the "image-motion" moment), both noise-floor subtracted, in arcsec**2.
These map to the data summary's Mxx_stamp/Myy_stamp/... (static) and
Mxx_motion/Myy_motion/... (motion).

  python guider_atmo_compare.py --outdir output/atmo_sim_20260706_688 \
      --data-summary output/night_20260706/moments/dayObs=20260706

Reads the sim npy/moments/meta from --outdir, and the on-sky per-(expId,detector)
summary parquet(s) from --data-summary (a dir or a file). Matches by dayObs/seqNum
stored in the sim meta.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from guiderMoments import measureStampMoments, MomentConfig, _FWHM_PER_SIGMA  # noqa: E402


def decompose_sim(outdir, visit=0, config=None):
    """Static + motion second-moment decomposition of the sim, per guider [arcsec**2]."""
    config = config or MomentConfig()
    stamps = np.load(os.path.join(outdir, f"guider_atmo_stamps_visit{visit:02d}.npy"))
    mom = pd.read_parquet(os.path.join(outdir, "guider_atmo_moments.parquet"))
    mom = mom[mom.visit == visit]
    with open(os.path.join(outdir, "guider_atmo_meta.json")) as fh:
        meta = json.load(fh)
    pixscale = meta.get("pixel_scale", 0.2)
    gain = meta.get("gain", 1.6)
    roi = int(meta.get("roi", stamps.shape[-1]))
    order = meta.get("guider_order") or list(mom[mom.stamp == 0].guider)
    s2 = pixscale**2

    rows = []
    for gi, det in enumerate(order):
        d = mom[(mom.guider == det) & mom.ok].sort_values("stamp")
        if d.empty:
            continue
        sigma_pix = float(np.median(d.fwhm_asec)) / (_FWHM_PER_SIGMA * pixscale)
        r_aper = config.apNSigma * sigma_pix
        half = int(np.ceil(2 * r_aper)) + 2
        pm = []
        for r in d.itertuples():
            img = stamps[int(r.stamp), gi].astype(float)
            ref = (roi / 2.0 + r.cen_x, roi / 2.0 + r.cen_y)
            m = measureStampMoments(img, ref, sigma_pix, r_aper, half, config.cenNIter)
            if m is not None:
                pm.append(m)
        if not pm:
            continue
        pm = pd.DataFrame(pm)
        f = pm["flux"].to_numpy()
        w = f.sum()
        # centroid noise floor per stamp: photon-limited sigma_cen^2 [pix^2]
        n_e = np.clip(f * gain, 1.0, None)
        xerr2 = (sigma_pix**2) / n_e
        nx = float(np.mean(xerr2)) if config.subtractNoiseFloor else 0.0
        # static: flux-weighted mean per-stamp moments (noise-floor subtracted)
        mxxS = (f * pm.Mxx).sum() / w - nx
        myyS = (f * pm.Myy).sum() / w - nx
        mxyS = (f * pm.Mxy).sum() / w
        # motion: flux-weighted covariance of the weighted centroids
        xb, yb = pm.xc.to_numpy(), pm.yc.to_numpy()
        xm, ym = (f * xb).sum() / w, (f * yb).sum() / w
        vxx = (f * (xb - xm) ** 2).sum() / w - nx
        vyy = (f * (yb - ym) ** 2).sum() / w - nx
        vxy = (f * (xb - xm) * (yb - ym)).sum() / w
        rows.append(dict(
            detector=det, nStamps=len(pm),
            Mxx_stamp=mxxS * s2, Myy_stamp=myyS * s2, Mxy_stamp=mxyS * s2,
            T_stamp=(mxxS + myyS) * s2, Q1_stamp=(mxxS - myyS) * s2, Q2_stamp=2 * mxyS * s2,
            Mxx_motion=vxx * s2, Myy_motion=vyy * s2, Mxy_motion=vxy * s2,
            T_motion=(vxx + vyy) * s2, Q1_motion=(vxx - vyy) * s2, Q2_motion=2 * vxy * s2))
    return pd.DataFrame(rows), meta


def load_data(data_summary, dayObs, seqNum):
    if os.path.isdir(data_summary):
        files = glob.glob(os.path.join(data_summary, "*.parquet"))
    else:
        files = [data_summary]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[df.seqNum == seqNum]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True, help="sim output dir")
    ap.add_argument("--data-summary", required=True,
                    help="on-sky summary parquet dir or file (per-(expId,detector))")
    ap.add_argument("--visit", type=int, default=0)
    args = ap.parse_args()

    sim, meta = decompose_sim(args.outdir, visit=args.visit)
    data = load_data(args.data_summary, meta.get("dayObs"), meta.get("seqNum"))
    print(f"dayObs {meta.get('dayObs')} seqNum {meta.get('seqNum')}  "
          f"(sim source={meta.get('source')}, target_fwhm={meta.get('target_fwhm')})\n")

    keep = ["T_stamp", "Q1_stamp", "Q2_stamp", "T_motion", "Q1_motion", "Q2_motion"]
    hdr = f"{'detector':9s} " + " ".join(f"{k:>10s}" for k in keep)
    for tag, df in (("SIM", sim), ("DATA", data)):
        print(tag)
        print(hdr)
        for det in sim.detector:
            r = df[df.detector == det]
            if r.empty:
                continue
            r = r.iloc[0]
            print(f"{det:9s} " + " ".join(f"{r[k]:10.4f}" if k in r else f"{'-':>10s}"
                                           for k in keep))
        print()

    # field medians + the key scalars
    print("field median   T_stamp[arcsec^2]   T_motion[arcsec^2]   sqrt(T_motion/2)['']")
    for tag, df in (("SIM", sim), ("DATA", data)):
        ts, tm = df["T_stamp"].median(), df["T_motion"].median()
        print(f"  {tag:4s}        {ts:10.4f}          {tm:10.4f}         {np.sqrt(tm/2):8.4f}")
    if "jitter_altaz" in data:
        print(f"\n  (data jitter_altaz = {data['jitter_altaz'].iloc[0]:.4f}\" "
              "[trend-removed per-axis centroid scatter])")


if __name__ == "__main__":
    main()
