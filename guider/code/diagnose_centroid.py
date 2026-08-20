#!/usr/bin/env python3
"""Localize the guider centroid/red-dot offset: measurement bias vs plotting.

For one (dayObs, seqNum, detector) it overlays, on the SAME DVCS coadd the movie
displays, three centroids and prints their pixel differences:

  tracker   : xroi_ref (median of the pipeline's per-stamp HSM centroids)
  com       : flux center-of-mass in a window (independent, unweighted)
  hsm_fresh : a fresh galsim FindAdaptiveMom on the coadd, 0-based origin

If com/hsm_fresh land on the star but `tracker` is several px off  -> real
centroid bias in the pipeline path (background/1-based/contaminant). If all three
agree and only the movie dot looks off -> plotting-convention (half-pixel/extent).

    python diagnose_centroid.py --day-obs 20260706 --seq-num 44 --detector R00_SG1
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "optatmo", "code"))


def com_window(img, cx, cy, half=8):
    """Background-subtracted flux center-of-mass in a +/-half window (0-based)."""
    ny, nx = img.shape
    x0, x1 = max(0, int(cx) - half), min(nx, int(cx) + half + 1)
    y0, y1 = max(0, int(cy) - half), min(ny, int(cy) + half + 1)
    sub = img[y0:y1, x0:x1].astype(float)
    sub = sub - np.nanmedian(img)
    sub[~np.isfinite(sub)] = 0.0
    sub[sub < 0] = 0.0
    ys, xs = np.mgrid[y0:y1, x0:x1]
    tot = sub.sum()
    if tot <= 0:
        return np.nan, np.nan
    return float((xs * sub).sum() / tot), float((ys * sub).sum() / tot)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day-obs", type=int, required=True)
    ap.add_argument("--seq-num", type=int, required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--stars", default=None,
                    help="stars parquet (default: output/night_<dayObs>/guider_stars_<dayObs>.parquet)")
    ap.add_argument("--repo", default="main")
    ap.add_argument("--collections", nargs="+",
                    default=["LSSTCam/raw/guider", "LSSTCam/raw/all"])
    ap.add_argument("--out", default=None, help="PNG output (default: scratch alongside)")
    args = ap.parse_args()

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import galsim
    from lsst.daf.butler import Butler
    from lsst.summit.utils.guiders.reading import GuiderReader

    det = args.detector
    starsPath = args.stars or f"output/night_{args.day_obs}/guider_stars_{args.day_obs}.parquet"
    sdf = pd.read_parquet(starsPath)
    sdf = sdf[(sdf["detector"] == det) & (sdf["seqNum"] == args.seq_num)] \
        if "seqNum" in sdf else sdf[sdf["detector"] == det]
    if sdf.empty:
        raise SystemExit(f"no stars rows for {det} seq {args.seq_num} in {starsPath}")
    refX = float(sdf["xroi_ref"].median())
    refY = float(sdf["yroi_ref"].median())

    reader = GuiderReader(Butler(args.repo, collections=args.collections), view="dvcs")
    gd = reader.get(dayObs=args.day_obs, seqNum=args.seq_num, doSubtractMedian=True)
    coadd = gd.getStampArrayCoadd(det)

    # independent centroids on the SAME dvcs coadd
    comx, comy = com_window(coadd, refX, refY)
    gimg = galsim.Image(np.ascontiguousarray(np.nan_to_num(coadd, nan=0.0)), xmin=0, ymin=0)
    try:
        hsm = galsim.hsm.FindAdaptiveMom(gimg, strict=False)
        hx, hy = float(hsm.moments_centroid.x), float(hsm.moments_centroid.y)
        hmsg = hsm.error_message or "ok"
    except Exception as exc:  # noqa: BLE001
        hx = hy = np.nan; hmsg = str(exc)

    print(f"\n{det}  dayObs={args.day_obs} seq={args.seq_num}  (0-based pixel coords)")
    print(f"  tracker  xroi_ref = ({refX:8.3f}, {refY:8.3f})   [median of pipeline HSM]")
    print(f"  com      window   = ({comx:8.3f}, {comy:8.3f})   [independent flux COM]")
    print(f"  hsm_fresh (0-based)= ({hx:8.3f}, {hy:8.3f})   [{hmsg}]")
    print(f"  tracker - com      = ({refX - comx:+.3f}, {refY - comy:+.3f}) px")
    print(f"  tracker - hsm_fresh= ({refX - hx:+.3f}, {refY - hy:+.3f}) px")
    print("  (hsm_fresh is 0-based here; the pipeline's galsim.Image is 1-based,")
    print("   so expect pipeline to read ~+1 px vs this in each axis.)")

    ny, nx = coadd.shape
    half = 15
    x0, x1 = max(0, int(refX) - half), min(nx, int(refX) + half)
    y0, y1 = max(0, int(refY) - half), min(ny, int(refY) + half)
    fig, ax = plt.subplots(figsize=(7, 7))
    reg = coadd[y0:y1, x0:x1]
    vmin, vmax = np.nanpercentile(reg, [50, 99])
    ax.imshow(reg, origin="lower", cmap="Greys", vmin=vmin, vmax=vmax,
              extent=(x0, x1, y0, y1), interpolation="nearest")
    ax.plot(refX, refY, "o", color="firebrick", ms=10, label="tracker xroi_ref")
    ax.plot(comx, comy, "+", color="dodgerblue", ms=14, mew=3, label="flux COM")
    ax.plot(hx, hy, "x", color="limegreen", ms=14, mew=3, label="fresh HSM (0-based)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"{det}  {args.day_obs}/{args.seq_num}  coadd centroids")
    out = args.out or f"centroid_diag_{args.day_obs}_{args.seq_num}_{det}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
