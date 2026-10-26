"""Per-visit wind and airflow telemetry for every cwfs_science visit, cached as parquet.

All from ConsDB: the weather tower ``wind_speed`` and ``wind_dir`` (direction the wind
comes from) in ``cdb_lsstcam.exposure``, and from the transformed EFD
``efd_lsstcam.exposure_efd`` the outside airflow sensor (ESS salIndex 301) and the five
3D sonic anemometers: 110 on the deployable platform at the top of the stairs, 123-126
on the TMA top-ring quadrants. Each EFD value is a mean over the exposure; a
``stddev_mean`` is the 30 s standard deviation averaged over the exposure. 123-126 report
from 2026-02-16 on for these visits.

Run from this study's directory::

    python code/fetch_wind.py
"""

import pathlib

from rubinwork.common.consdb_efd import _query_cols
from rubinwork.common.telemetry_clients import make_consdb_client
from rubinwork.products import cwfs_science

VARIANT = "d13-blitz_rso1071"
CACHE = pathlib.Path(__file__).resolve().parents[1] / "output" / "cache" / "wind_per_visit.parquet"
SONIC = (110, 123, 124, 125, 126)
SONIC_FIELDS = ("wind_speed_0_mean", "wind_speed_1_mean", "wind_speed_2_mean",
                "wind_speed_magnitude_mean", "wind_speed_maxmagnitude_mean",
                "wind_speed_stddev_0_mean", "wind_speed_stddev_1_mean",
                "wind_speed_stddev_2_mean", "sonic_temperature_mean",
                "sonic_temperature_stddev_mean")
EFD_COLS = ([f"mt_salindex{i}_{f}" for i in SONIC for f in SONIC_FIELDS]
            + ["mt_salindex301_airflow_speed_mean", "mt_salindex301_airflow_direction_mean",
               "mt_salindex301_airflow_stddev_mean"])


def fetch():
    """Wind telemetry, one row per visit, keyed on ``visit``; EFD names lose ``mt_``."""
    v = cwfs_science.load_table("ccd_visits", VARIANT)[["visit"]].drop_duplicates()
    cdb = make_consdb_client("auto")
    efd = _query_cols(cdb, "efd_lsstcam.exposure_efd", EFD_COLS, v["visit"], "wind")
    tower = _query_cols(cdb, "cdb_lsstcam.exposure", ["wind_speed", "wind_dir"],
                        v["visit"], "weather")
    out = v
    for g in (efd, tower):
        g = g.drop_duplicates("exposure_id").rename(columns={"exposure_id": "visit"})
        out = out.merge(g, on="visit", how="left")
    out.columns = [c.removeprefix("mt_") for c in out.columns]
    return out.astype({c: float for c in out.columns if c != "visit"})


if __name__ == "__main__":
    df = fetch()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    print(f"{len(df)} visits written to {CACHE}")
    print(df.notna().mean().round(3).to_string())
