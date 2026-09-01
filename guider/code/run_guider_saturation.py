#!/usr/bin/env python3
"""Guide-star peak (saturation) data per science visit per CCD.

For every science guider exposure of a night, measure each guide star's peak --
the mean of the top-3 pixels of a representative stamp after the stamp 0,1
per-column bias subtraction (3x3 median-filtered to reject cosmics) -- and write
one row per (expId, detector) to a parquet. Use it to build the per-CCD peak
(saturation) histogram and to find/revisit saturated stamps.

Batch: one job per night, e.g.
    python run_guider_saturation.py --day-obs 20260706 \
        --out output/guider_saturation_20260706.parquet
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def parseArgs(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day-obs", type=int, required=True)
    p.add_argument("--out", default=None,
                   help="Parquet path (default output/guider_saturation_<dayObs>.parquet).")
    p.add_argument("--repo", default="main")
    p.add_argument("--collections", nargs="+", default=["LSSTCam/raw/guider", "LSSTCam/raw/all"])
    p.add_argument("--view", default="dvcs")
    p.add_argument("--stamp", type=int, default=50, help="Representative stamp index.")
    p.add_argument("--nblank", type=int, default=2, help="Number of leading (pre-shutter) bias stamps.")
    p.add_argument("--image-types", nargs="+", default=["science"],
                   help="Keep only exposures with these observation_types.")
    return p.parse_args(argv)


def tflux(arrs):
    return np.array([np.nansum(x - np.nanmedian(x)) for x in arrs])


def main(argv=None):
    args = parseArgs(argv)
    out = args.out or f"output/guider_saturation_{args.day_obs}.parquet"

    from scipy.ndimage import median_filter
    from lsst.daf.butler import Butler
    from lsst.summit.utils.guiders.reading import GuiderReader
    try:
        from lsst.daf.butler import DatasetNotFoundError
    except ImportError:
        from lsst.daf.butler._exceptions import DatasetNotFoundError

    butler = Butler(args.repo, collections=args.collections)
    reader = GuiderReader(butler, view=args.view)

    # science guider exposures for the night
    exps = sorted({r.dataId["exposure"] for r in butler.registry.queryDatasets(
        "guider_raw", where="instrument='LSSTCam' AND exposure.day_obs = d", bind={"d": args.day_obs})})
    wanted = {t.lower() for t in args.image_types}
    sci = {r.id for r in butler.registry.queryDimensionRecords(
        "exposure", where="instrument='LSSTCam' AND exposure.day_obs = d", bind={"d": args.day_obs})
        if (r.observation_type or "").lower() in wanted}
    visits = [e for e in exps if e in sci]
    print(f"{args.day_obs}: {len(visits)}/{len(exps)} science guider exposures", file=sys.stderr)

    rows = []
    for i, expId in enumerate(visits):
        seqNum = expId % 100000
        try:
            raw = reader.getGuiderRawStamps(dayObs=args.day_obs, seqNum=seqNum)
        except (DatasetNotFoundError, RuntimeError):
            continue
        for det in sorted(raw):
            arrs = [s.stamp_im.image.array.astype(float) for s in raw[det]]
            n = len(arrs)
            if n < 3:
                continue
            md = raw[det].metadata
            cb = np.nanmedian(np.concatenate(arrs[:args.nblank], axis=0), axis=0)
            mx = np.nanmax(tflux(arrs))
            on = np.where(tflux(arrs) > 0.5 * mx)[0] if np.isfinite(mx) else np.array([], int)
            st = args.stamp if (args.stamp < n and (len(on) == 0 or args.stamp in on)) \
                else (int(on[len(on) // 2]) if len(on) else n // 2)
            img = arrs[st] - cb[None, :]                      # stamp 0,1 column bias only
            sm = median_filter(np.nan_to_num(img, nan=-1e9), size=3)   # cosmic-robust
            flat = np.sort(sm.ravel())
            sy, sx = np.unravel_index(int(np.argmax(sm)), sm.shape)
            rows.append(dict(
                expId=int(expId), dayObs=int(args.day_obs), seqNum=int(seqNum), detector=det,
                peak_top3=float(np.mean(flat[-3:])), peak_max=float(flat[-1]),
                n_stamps=int(n), n_on=int(len(on)), stamp_used=int(st),
                star_row=int(sy), star_col=int(sx),
                roirow=int(md["ROIROW"]) if "ROIROW" in md else -1,
                roicol=int(md["ROICOL"]) if "ROICOL" in md else -1,
                roiseg=str(md["ROISEG"]) if "ROISEG" in md else "",
            ))
        if i % 50 == 0:
            print(f"  {i}/{len(visits)}", file=sys.stderr)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} rows, {df['expId'].nunique() if len(df) else 0} visits)")


if __name__ == "__main__":
    main()
