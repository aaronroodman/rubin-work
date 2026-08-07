#!/usr/bin/env python
"""
Feed the atmospheric-PSF simulation into the summit_utils guider analysis code.

Because the simulator already produces stamps in the DVCS view (guider_atmo_sim.py
Option A), the downstream path is identical to on-sky data: this wraps the sim
output in a light `SimGuiderData` shim that quacks like
`lsst.summit.utils.guiders.reading.GuiderData` for exactly the interface the
`GuiderPlotter` (and moment code) use, so the REAL summit_utils movie/moment code
runs unchanged on simulated data.

  # on the RSP (needs summit_utils):
  python guider_atmo_to_guiderdata.py --outdir output/atmo_sim --save sim_movie.mp4

Or in a notebook:
  from guider_atmo_to_guiderdata import load_sim_guiderdata
  gd, starsDf = load_sim_guiderdata("output/atmo_sim")
  from lsst.summit.utils.guiders.plotting import GuiderPlotter
  GuiderPlotter(gd, starsDf).makeAnimation(cutoutSize=31, saveAs="sim_movie.mp4")

Reads guider_atmo_stamps_visit{v}.npy + guider_atmo_moments.parquet +
guider_atmo_meta.json from --outdir.
"""
import argparse
import json
import os
import numpy as np
import pandas as pd


class SimGuiderData:
    """Duck-typed stand-in for summit_utils GuiderData, backed by sim DVCS stamps.

    Implements only the surface GuiderPlotter touches: len, [det, i],
    getStampArrayCoadd(det), guiderNames, view, camRotAngle, alt, az, expid,
    guiderDurationSec.  All stamps are already in DVCS view, so no transform.
    """

    def __init__(self, stamps, order, meta):
        self._stamps = stamps                       # [n_stamp, n_guider, ny, nx]
        self._idx = {name: i for i, name in enumerate(order)}
        self._meta = meta
        self.view = "dvcs"
        self.camRotAngle = float(meta.get("camRotAngle", 0.0))

    def __len__(self):
        return int(self._stamps.shape[0])

    @property
    def guiderNames(self):
        return list(self._idx)

    @property
    def expid(self):
        return int(self._meta.get("dayObs", 0)) * 100000 + int(self._meta.get("seqNum", 0))

    @property
    def guiderDurationSec(self):
        return len(self) * float(self._meta.get("cadence", 0.2))

    @property
    def alt(self):
        return float(self._meta.get("alt", 90.0))

    @property
    def az(self):
        return float(self._meta.get("az", 0.0))

    def getStampArrayCoadd(self, detName):
        return np.median(self._stamps[:, self._idx[detName]], axis=0)

    def __getitem__(self, key):
        if not (isinstance(key, tuple) and len(key) == 2):
            raise TypeError("SimGuiderData supports [detName, stampIndex] access")
        detName, idx = key
        return self._stamps[idx, self._idx[detName]]

    def __iter__(self):
        return iter(self._idx)


def _stars_df(moments, roi, expid, pixel_scale=0.2):
    """Build the summit_utils starsDf from the sim moments so GuiderPlotter draws
    the tracking centroid and jitter. The star is placed at the ROI centre, so the
    reference is the centre and (cen_x, cen_y) give the per-stamp wander (DVCS pix).
    Alt/Az offsets from the DVCS axes at rotator 0 (DVCS Y=azimuth, X=-elevation)."""
    c = roi / 2.0
    df = moments.rename(columns={"guider": "detector"}).copy()
    df["expid"] = expid
    df["xroi_ref"], df["yroi_ref"] = c, c
    df["xroi"] = c + df["cen_x"]
    df["yroi"] = c + df["cen_y"]
    df["dx"], df["dy"] = df["cen_x"], df["cen_y"]
    ax = df["cen_x_asec"] if "cen_x_asec" in df else df["cen_x"] * pixel_scale
    ay = df["cen_y_asec"] if "cen_y_asec" in df else df["cen_y"] * pixel_scale
    df["daz"] = ay                                   # DVCS Y = azimuth
    df["dalt"] = -ax                                 # DVCS X = -elevation
    df["magoffset"] = 0.0
    cols = ["expid", "detector", "stamp", "xroi", "yroi", "xroi_ref", "yroi_ref",
            "dx", "dy", "dalt", "daz", "magoffset"]
    return df[cols]


def load_sim_guiderdata(outdir, visit=0):
    """Return (SimGuiderData, starsDf) built from a guider_atmo_sim.py run."""
    stamps = np.load(os.path.join(outdir, f"guider_atmo_stamps_visit{visit:02d}.npy"))
    moments = pd.read_parquet(os.path.join(outdir, "guider_atmo_moments.parquet"))
    moments = moments[moments.visit == visit]
    meta_path = os.path.join(outdir, "guider_atmo_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
    else:
        # Output predates the sidecar (older guider_atmo_sim.py). Fall back to
        # defaults derived from the data; re-run the sim to write the sidecar
        # (alt/az then reflect the real pointing instead of these defaults).
        print(f"  note: no {os.path.basename(meta_path)}; using defaults "
              "(alt=90, az=0, cadence=0.2). Re-run guider_atmo_sim.py to write it.")
        meta = {}
    order = meta.get("guider_order") or list(moments[moments.stamp == 0].guider)
    roi = int(meta.get("roi", stamps.shape[-1]))
    gd = SimGuiderData(stamps, order, meta)
    return gd, _stars_df(moments, roi, gd.expid,
                         pixel_scale=meta.get("pixel_scale", 0.2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="output/atmo_sim")
    ap.add_argument("--visit", type=int, default=0)
    ap.add_argument("--save", default=None, help="movie path (.mp4/.gif)")
    ap.add_argument("--cutout", type=int, default=31)
    ap.add_argument("--fps", type=int, default=5)
    args = ap.parse_args()

    gd, starsDf = load_sim_guiderdata(args.outdir, visit=args.visit)
    print(f"SimGuiderData: {len(gd)} stamps, guiders={gd.guiderNames}")
    from lsst.summit.utils.guiders.plotting import GuiderPlotter   # RSP only
    save = args.save or os.path.join(args.outdir, f"sim_guiderdata_movie_v{args.visit:02d}.mp4")
    GuiderPlotter(gd, starsDf).makeAnimation(cutoutSize=args.cutout, fps=args.fps, saveAs=save)
    print(f"wrote {save}  (via summit_utils GuiderPlotter)")


if __name__ == "__main__":
    main()
