"""Per-visit guider image motion and guider PSF shape in Alt/Az, cached as parquet.

From the guider_moments ``stars`` table: at each stamp, the median over tracked stars of
``daz`` (already x cos(alt)) and ``dalt`` [arcsec] is the motion common to all guiders.
Its variances and covariance over the exposure are the image-motion second moments in
the Az/El frame of the value_added ``psf_shape`` columns (+x azimuth, +y elevation).
Drift is kept: a linear drift elongates the image as much as jitter does. Also the
median over stamps and stars of the guider PSF ``e1_altaz``, ``e2_altaz`` and ``fwhm``.

Usage::

    python nightlyiq/iq_environment/code/fetch_guider.py
"""

import pathlib

import numpy as np
import pandas as pd

from rubinwork.products.guider_moments import reader as guider

VARIANT = "v4"
CACHE = pathlib.Path(__file__).resolve().parents[1] / "output" / "cache" / "guider_per_visit.parquet"
MIN_STARS = 3   # stars a stamp needs to enter the common-mode motion
MIN_STAMPS = 20


def per_visit(stars):
    s = stars.dropna(subset=["daz", "dalt"])
    stamp = s.groupby(["expid", "stamp"]).agg(daz=("daz", "median"), dalt=("dalt", "median"),
                                              n=("daz", "size"))
    stamp = stamp[stamp.n >= MIN_STARS].reset_index()
    rows = []
    for exp, g in stamp.groupby("expid"):
        if len(g) < MIN_STAMPS:
            continue
        c = np.cov(g.daz, g.dalt)
        rows.append(dict(visit_id=int(exp), gd_var_az=c[0, 0], gd_var_el=c[1, 1],
                         gd_cov_azel=c[0, 1], gd_n_stamps=len(g),
                         gd_n_stars_med=float(g.n.median())))
    out = pd.DataFrame(rows)
    shape = s.groupby("expid").agg(gd_e1_altaz=("e1_altaz", "median"),
                                   gd_e2_altaz=("e2_altaz", "median"),
                                   gd_fwhm=("fwhm", "median")).reset_index()
    shape = shape.rename(columns={"expid": "visit_id"})
    return out.merge(shape, on="visit_id", how="left")


def fetch():
    parts = []
    for night in guider.nights(VARIANT):
        stars = guider.load_table("stars", night, variant=VARIANT)
        if "dalt" not in stars.columns:
            print(f"{night}: stars table empty, skipped", flush=True)
            continue
        v = per_visit(stars)
        print(f"{night}: {len(v)} visits", flush=True)
        parts.append(v)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    df = fetch()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    print(f"{len(df)} visits written to {CACHE}")
    print(df.describe().T.round(4).to_string())
