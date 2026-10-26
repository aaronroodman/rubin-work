"""Per-visit thermal telemetry for every cwfs_science visit, cached as parquet.

Air and truss temperatures from ConsDB transformed EFD (`join_consdb`, group
``thermal``); M1M3 gradients and camera-body temperature from `visit_telemetry`; M1M3
glass terms from `m1m3_thermal_r2`. Same quantities as the FAM study `aos/intra_extra`.

Run from this study's directory::

    python code/fetch_thermal.py
"""

import pathlib

import pandas as pd

from rubinwork.products import cwfs_science
from rubinwork.products.value_added import reader as efd_db

VARIANT = "d13-blitz_rso1071"
CACHE = pathlib.Path(__file__).resolve().parents[1] / "output" / "cache" / "thermal_per_visit.parquet"

GRAD_COLS = ("m1m3_x_gradient_c_per_m", "m1m3_y_gradient_c_per_m",
             "m1m3_z_gradient_c_per_m", "m1m3_radial_gradient_c_per_m", "cam_AverageTemp")
R2_COLS = ("m1m3_r2_coeff_c", "m1m3_rms_c", "m1m3_mean_temp_c", "m1m3_std_temp_c",
           "m1m3_range_temp_c", "m1_mean_temp_c", "m3_mean_temp_c")
CONSDB_COLS = ("m1m3_air_temp", "m2_air_temp", "cam_air_temp", "outside_temp",
               "truss_temp_mean_c")


def fetch():
    """Thermal telemetry, one row per visit, keyed on ``visit``."""
    v = cwfs_science.load_table("ccd_visits", VARIANT)[["visit", "day_obs", "seq_num"]]
    req = v.drop_duplicates().rename(columns={"visit": "visit_id"}).reset_index(drop=True)
    got = efd_db.join_consdb(req, groups=("thermal",))
    got = got[["visit_id"] + [c for c in got.columns
                              if c in CONSDB_COLS or c.endswith("_interpolated")]]
    con = efd_db.open_db()
    try:
        con.register("req", req[["visit_id"]])
        grads = con.execute(f"SELECT t.visit_id, {', '.join('t.' + c for c in GRAD_COLS)} "
                            "FROM req JOIN visit_telemetry t USING (visit_id)").fetchdf()
        r2 = con.execute(f"SELECT m.visit_id, {', '.join('m.' + c for c in R2_COLS)} "
                         "FROM req JOIN m1m3_thermal_r2 m USING (visit_id)").fetchdf()
    finally:
        con.close()
    out = (req.merge(got, on="visit_id", how="left")
           .merge(grads, on="visit_id", how="left").merge(r2, on="visit_id", how="left"))
    return out.rename(columns={"visit_id": "visit"})


if __name__ == "__main__":
    df = fetch()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    print(f"{len(df)} visits written to {CACHE}")
    print(df.notna().mean().round(4).to_string())
