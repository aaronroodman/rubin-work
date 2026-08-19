#!/usr/bin/env python3
"""Find the science CCD nearest each guide sensor (run once on the RSP).

Uses the LSSTCam cameraGeom to compute, for every GUIDER detector, the nearest
SCIENCE detector by focal-plane centre distance. Prints a table (the top-N
nearest per guider, so you can sanity-check against a known pair, e.g. guiders
194/195 -> CCD 23) and writes a YAML mapping each guider to its adjacent science
CCD for the guider moment pipeline to consume.

    python guider_adjacent_ccds.py [--out guider_adjacent_ccds.yaml] [--topn 3]
"""
from __future__ import annotations

import argparse
import os


def classify(det, DetectorType):
    """Return 'science', 'guider', or 'other' for a detector."""
    name = det.getName()
    t = det.getType()
    if t == DetectorType.SCIENCE:
        return "science"
    if t == DetectorType.GUIDER or name.endswith(("SG0", "SG1")):
        return "guider"
    return "other"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=os.path.join(here, "guider_adjacent_ccds.yaml"))
    ap.add_argument("--topn", type=int, default=3, help="nearest N science CCDs to print.")
    args = ap.parse_args()

    import numpy as np
    from lsst.obs.lsst import LsstCam
    from lsst.afw.cameraGeom import DetectorType, FOCAL_PLANE

    cam = LsstCam.getCamera()
    science, guiders = [], []
    for det in cam:
        c = det.getCenter(FOCAL_PLANE)   # Point2D, mm
        rec = (det.getId(), det.getName(), c.getX(), c.getY())
        kind = classify(det, DetectorType)
        if kind == "science":
            science.append(rec)
        elif kind == "guider":
            guiders.append(rec)

    sci_xy = np.array([[r[2], r[3]] for r in science])
    mapping = {}
    print(f"{len(guiders)} guiders, {len(science)} science CCDs\n")
    print(f"{'guider':<12}{'id':>5}   nearest science CCDs (name:id:dist_mm)")
    for gid, gname, gx, gy in sorted(guiders, key=lambda r: r[0]):
        d = np.hypot(sci_xy[:, 0] - gx, sci_xy[:, 1] - gy)
        order = np.argsort(d)[: args.topn]
        near = [(science[i][1], science[i][0], float(d[i])) for i in order]
        name0, id0, dist0 = near[0]
        mapping[gname] = {
            "guider_id": int(gid),
            "adjacent_ccd": int(id0),
            "adjacent_ccd_name": name0,
            "dist_mm": round(dist0, 1),
        }
        near_str = "  ".join(f"{n}:{i}:{dd:.0f}" for n, i, dd in near)
        print(f"{gname:<12}{gid:>5}   {near_str}")

    try:
        import yaml

        with open(args.out, "w") as fh:
            yaml.safe_dump(mapping, fh, sort_keys=True)
        print(f"\nwrote {args.out}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n(could not write YAML: {exc}); paste this dict instead:\n{mapping}")


if __name__ == "__main__":
    main()
