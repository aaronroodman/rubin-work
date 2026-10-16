"""Fetch each FAM exposure's own midpoint time from the Consolidated Database.

`visits.parquet` carries one ``mjd``, that of the extra-focal exposure. The intra-focal
exposure's own time is needed to interpolate the intra-focal Z4 forward to the extra-focal
epoch, so the dZ4 difference is not contaminated by the intra-before-extra time ordering.

Run from the repo root::

    python aos/unpaired_intra_vs_extra/fetch_exposure_times.py \
        --variant danish_1_3_v1000 --build 20261013 \
        --out-dir aos/unpaired_intra_vs_extra/output/20261015_visit_medians
"""

import argparse
import pathlib

import pandas as pd

from rubinwork.common.telemetry_clients import make_consdb_client
from rubinwork.products import catalog

QUERY = """
SELECT exposure_id, day_obs, seq_num, exp_midpt_mjd, obs_start_mjd
FROM cdb_lsstcam.exposure
WHERE day_obs IN ({days})
"""


def fetch(variant, build):
    """Exposure midpoint times for every night the variant covers.

    Parameters
    ----------
    variant : `str`
        `fam_tables` variant name.
    build : `str`
        Build identifier.

    Returns
    -------
    times : `pandas.DataFrame`
        One row per exposure on those nights: ``exposure_id``, ``day_obs``, ``seq_num``,
        ``exp_midpt_mjd`` and ``obs_start_mjd`` [days, TAI MJD].
    """
    data_dir = catalog.path('fam_tables', variant, build)
    visits = pd.read_parquet(data_dir / 'visits.parquet', columns=['day_obs'])
    days = sorted(visits['day_obs'].unique())

    client = make_consdb_client()
    query = QUERY.format(days=', '.join(str(int(d)) for d in days))
    return client.query(query).to_pandas()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='danish_1_3_v1000')
    parser.add_argument('--build', default='20261013')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    times = fetch(args.variant, args.build)

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'exposure_times.parquet'
    times.to_parquet(out_path, index=False)

    print(f'{len(times)} exposures over {times["day_obs"].nunique()} nights')
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
