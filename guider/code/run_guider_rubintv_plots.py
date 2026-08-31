#!/usr/bin/env python3
"""RubinTV-style guider plots for one exposure (Snakemake runner).

Reproduces the RubinTV GuiderPlotter output (given the guider-fixes branch) as a
single review PDF per visit: the six strip plots (centroid Alt/Az, centroid
pixel, flux, rotator, ellipticity, PSF FWHM) plus the zoomed and full-frame
mosaics. Intended for a strided subset of a night, to review the effect of the
fixes (outlier band, centroid, layout).

    python run_guider_rubintv_plots.py --day-obs 20260709 --seq-num 808 --outdir <dir>
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guiderEdgeRecovery import makeTrackerConfig  # noqa: E402

STRIP_TYPES = ["centroidAltAz", "centroidPixel", "flux", "rotator", "ellip", "psf"]


def parseArgs(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day-obs", type=int, required=True)
    p.add_argument("--seq-num", type=int, required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--repo", default="main")
    p.add_argument("--collections", nargs="+", default=["LSSTCam/raw/guider", "LSSTCam/raw/all"])
    p.add_argument("--view", default="dvcs")
    p.add_argument("--min-snr", type=float, default=10.0)
    p.add_argument("--max-ellipticity", type=float, default=0.7)
    p.add_argument("--edge-margin", type=int, default=3)
    p.add_argument("--min-finite-fraction", type=float, default=0.5)
    p.add_argument("--no-recover-edge-stars", action="store_true")
    p.add_argument("--strict", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parseArgs(argv)
    expId = args.day_obs * 100000 + args.seq_num
    out = os.path.join(args.outdir, f"{args.seq_num}_rtvplots.pdf")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from lsst.daf.butler import Butler
    from lsst.summit.utils.guiders.reading import GuiderReader
    from lsst.summit.utils.guiders.tracking import GuiderStarTracker
    from lsst.summit.utils.guiders.plotting import GuiderPlotter
    try:
        from lsst.daf.butler import DatasetNotFoundError
    except ImportError:
        from lsst.daf.butler._exceptions import DatasetNotFoundError

    reader = GuiderReader(Butler(args.repo, collections=args.collections), view=args.view)
    try:
        gd = reader.get(dayObs=args.day_obs, seqNum=args.seq_num, doSubtractMedian=True)
    except (DatasetNotFoundError, RuntimeError) as exc:
        if args.strict:
            raise
        print(f"{expId}: SKIPPED -- {exc}", file=sys.stderr)
        os.makedirs(args.outdir, exist_ok=True)
        open(out, "w").close()   # empty placeholder keeps the DAG satisfied
        return

    recover = not args.no_recover_edge_stars
    cfg = makeTrackerConfig(
        minFiniteFraction=args.min_finite_fraction if recover else 1.0,
        minSnr=args.min_snr, maxEllipticity=args.max_ellipticity, edgeMargin=args.edge_margin,
    )
    stars = GuiderStarTracker(gd, cfg).trackGuiderStars(refCatalog=None)
    plotter = GuiderPlotter(gd, stars)

    os.makedirs(args.outdir, exist_ok=True)
    with PdfPages(out) as pdf:
        for kind in STRIP_TYPES:
            try:
                fig = plotter.stripPlot(plotType=kind)
                pdf.savefig(fig); plt.close(fig)
            except Exception as exc:  # noqa: BLE001
                print(f"{expId}: stripPlot {kind} failed: {exc}", file=sys.stderr)
        for label, kw in [("zoom", dict(cutoutSize=20, plo=50, phi=98)),
                          ("full", dict(cutoutSize=-1, plo=70, phi=99))]:
            try:
                fig = plotter.plotMosaic(stampNum=-1, **kw)
                pdf.savefig(fig); plt.close(fig)
            except Exception as exc:  # noqa: BLE001
                print(f"{expId}: mosaic {label} failed: {exc}", file=sys.stderr)
    print(f"{expId}: wrote {out}")


if __name__ == "__main__":
    main()
