"""M1M3 glass-minus-air temperature statistics for the table in ``docs/schema.md``.

Glass is ``m1m3_thermal_r2.m1m3_mean_temp_c`` from the value-added database. The two air
temperatures come from ConsDB ``efd_lsstcam.exposure_efd``, which this database does not copy.
Prints the Markdown table rows to paste into the ``m1m3_thermal_r2`` section of
``value_added/docs/schema.md``.

Run from the repository root::

    python -m rubinwork.products.value_added.builders.glass_minus_air --from 20260101
"""
import argparse

import numpy as np
import pandas as pd

from rubinwork.common.telemetry_clients import make_consdb_client
from rubinwork.common.utils import nmad
from rubinwork.products.value_added import reader

#: (label, ConsDB column, location) for each air sensor in the schema table.
AIR_SENSORS = [
    ("M1M3TS TS1-A", "mt_m1m3_glycol_loop_temperature_above_mirror_temp",
     "above mirror level"),
    ("ESS 113", "mt_salindex113_temperature_0_mean",
     "TMA central section, not close to M1M3"),
]

#: |glass - air| above this [degC] is a dead or unset sensor, not a temperature.
MAX_ABS_DT_C = 20.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="day_obs_min", type=int, default=20260101,
                        help="first day_obs YYYYMMDD")
    parser.add_argument("--db", default=None, help="database path; default the current build")
    args = parser.parse_args(argv)

    con = reader.open_db(args.db)
    glass = con.execute(
        "SELECT visit_id, day_obs, m1m3_mean_temp_c FROM m1m3_thermal_r2 "
        "WHERE day_obs >= ? AND m1m3_mean_temp_c IS NOT NULL",
        [args.day_obs_min]).df()
    con.close()

    cdb = make_consdb_client()
    cols = [c for _, c, _ in AIR_SENSORS]
    air = cdb.query(
        f"SELECT exposure_id, {', '.join(cols)} FROM efd_lsstcam.exposure_efd "
        f"WHERE exposure_id >= {args.day_obs_min * 100000}").to_pandas()
    air[cols] = air[cols].apply(pd.to_numeric, errors="coerce")
    df = glass.merge(air.rename(columns={"exposure_id": "visit_id"}), on="visit_id")

    print(f"day_obs {df.day_obs.min()} to {df.day_obs.max()}, "
          f"{df.day_obs.nunique()} nights, {len(df)} exposures with glass and ConsDB rows")
    medians = {}
    for label, col, where in AIR_SENSORS:
        dt = (df.m1m3_mean_temp_c - df[col]).dropna()
        dt = dt[np.abs(dt) < MAX_ABS_DT_C]
        medians[label] = dt.median()
        print(f"| {label} | `{col}` | {where} | "
              f"{dt.median():+.2f} / {nmad(dt):.2f} | {len(dt):,} |")
    (a, ma), (b, mb) = medians.items()
    print(f"{b} minus {a}, difference of the glass-minus-air medians: {ma - mb:+.2f} degC")


if __name__ == "__main__":
    main()
