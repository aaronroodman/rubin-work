#!/usr/bin/env python3
"""RubinTV-style guider movies for one exposure (Snakemake runner).

Reproduces the two animations RubinTV publishes, using the same summit_utils
code path (`GuiderPlotter.makeAnimation`) with RubinTV's exact parameters:

  <seqNum>_full.mp4   full ROI      cutoutSize=-1, plo=70, phi=99, fps=5
  <seqNum>_star.mp4   star cutout   cutoutSize=20, plo=50, phi=98, fps=10

Intended for a strided subset of a night (e.g. every 50th visit) as a visual
cross-check against the original RubinTV output. Missing-data visits touch empty
outputs + a SKIPPED marker so the batch/aggregate still complete.

    python run_guider_movie.py --day-obs 20260709 --seq-num 808 --outdir <dir>
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guiderEdgeRecovery import makeTrackerConfig  # noqa: E402

# RubinTV makeAnimation parameters (rubintv_production/guiders.py).
RTV = {
    "full": dict(cutoutSize=-1, plo=70, phi=99, fps=5),
    "star": dict(cutoutSize=20, plo=50, phi=98, fps=10),
}


def parseArgs(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day-obs", type=int, required=True)
    p.add_argument("--seq-num", type=int, required=True)
    p.add_argument("--outdir", required=True, help="Output dir (per-seq movies written here).")
    p.add_argument("--repo", default="main")
    p.add_argument("--collections", nargs="+", default=["LSSTCam/raw/guider", "LSSTCam/raw/all"])
    p.add_argument("--view", default="dvcs")
    p.add_argument("--which", choices=["both", "full", "star"], default="both")
    p.add_argument("--format", choices=["mp4", "gif"], default="mp4",
                   help="mp4 needs the ffmpeg writer (RubinTV default); gif uses pillow.")
    p.add_argument("--min-snr", type=float, default=10.0)
    p.add_argument("--max-ellipticity", type=float, default=0.7)
    p.add_argument("--edge-margin", type=int, default=3)
    p.add_argument("--min-finite-fraction", type=float, default=0.5)
    p.add_argument("--no-recover-edge-stars", action="store_true")
    p.add_argument("--strict", action="store_true",
                   help="Fail on missing/insufficient data instead of skipping.")
    return p.parse_args(argv)


def outPaths(outdir, seqNum, ext):
    return {k: os.path.join(outdir, f"{seqNum}_{k}.{ext}") for k in ("full", "star")}


def touchSkipped(outdir, seqNum, dayObs, ext, reason, which):
    os.makedirs(outdir, exist_ok=True)
    for k, p in outPaths(outdir, seqNum, ext).items():
        if which in ("both", k):
            open(p, "w").close()   # empty placeholder keeps the DAG satisfied
    with open(os.path.join(outdir, f"{seqNum}_movie_SKIPPED.txt"), "w") as fh:
        fh.write(f"{dayObs * 100000 + seqNum}\t{dayObs}\t{seqNum}\t{reason}\n")


def main(argv=None):
    args = parseArgs(argv)
    expId = args.day_obs * 100000 + args.seq_num
    paths = outPaths(args.outdir, args.seq_num, args.format)

    import matplotlib
    matplotlib.use("Agg")
    from lsst.daf.butler import Butler
    from lsst.summit.utils.guiders.reading import GuiderReader
    from lsst.summit.utils.guiders.tracking import GuiderStarTracker
    from lsst.summit.utils.guiders.plotting import GuiderPlotter
    try:
        from lsst.daf.butler import DatasetNotFoundError
    except ImportError:
        from lsst.daf.butler._exceptions import DatasetNotFoundError

    butler = Butler(args.repo, collections=args.collections)
    reader = GuiderReader(butler, view=args.view)
    try:
        guiderData = reader.get(dayObs=args.day_obs, seqNum=args.seq_num, doSubtractMedian=True)
    except (DatasetNotFoundError, RuntimeError) as exc:
        if args.strict:
            raise
        print(f"{expId}: SKIPPED -- {exc}", file=sys.stderr)
        touchSkipped(args.outdir, args.seq_num, args.day_obs, args.format, str(exc), args.which)
        return

    recover = not args.no_recover_edge_stars
    config = makeTrackerConfig(
        minFiniteFraction=args.min_finite_fraction if recover else 1.0,
        minSnr=args.min_snr, maxEllipticity=args.max_ellipticity, edgeMargin=args.edge_margin,
    )
    stars = GuiderStarTracker(guiderData, config).trackGuiderStars(refCatalog=None)

    os.makedirs(args.outdir, exist_ok=True)
    stale = os.path.join(args.outdir, f"{args.seq_num}_movie_SKIPPED.txt")
    if os.path.exists(stale):
        os.remove(stale)

    # stars enable the star-cutout zoom; an empty table still makes the full movie.
    plotter = GuiderPlotter(guiderData, stars)
    for kind in ("full", "star"):
        if args.which not in ("both", kind):
            continue
        plotter.makeAnimation(saveAs=paths[kind], **RTV[kind])
        print(f"{expId}: wrote {paths[kind]}")


if __name__ == "__main__":
    main()
