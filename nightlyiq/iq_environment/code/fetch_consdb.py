"""Per-visit ConsDB columns for every value_added visit, cached as parquet.

Exposure metadata and transformed-EFD temperatures through `join_consdb` (groups
``meta`` and ``thermal``); observation reason and image quality (``psf_sigma_median``,
``aos_fwhm``, ``donut_blur_fwhm``, ``seeing_zenith_500nm_median``) from
``visit1_quicklook``, the columns the nightly IQ report uses. Fetched a month at a time.

Usage::

    python nightlyiq/iq_environment/code/fetch_consdb.py
"""

import pathlib

import pandas as pd

from rubinwork.common.telemetry_clients import make_consdb_client
from rubinwork.products.value_added import reader

VA_BUILD = "snapshots/20261105"
CACHE = pathlib.Path(__file__).resolve().parents[1] / "output" / "cache" / "consdb_per_visit.parquet"
IQ_COLS = ("psf_sigma_median", "aos_fwhm", "donut_blur_fwhm", "seeing_zenith_500nm_median")


def fetch_iq(cdb, days, instrument="lsstcam"):
    """Observation reason and visit image quality for the given ``day_obs`` list."""
    q = (f"SELECT e.exposure_id AS visit_id, e.observation_reason, "
         f"{', '.join('q.' + c for c in IQ_COLS)} "
         f"FROM cdb_{instrument}.exposure e "
         f"LEFT JOIN cdb_{instrument}.visit1_quicklook q ON q.visit_id = e.exposure_id "
         f"WHERE e.day_obs IN ({', '.join(str(d) for d in days)})")
    iq = cdb.query(q).to_pandas()
    for c in IQ_COLS:
        iq[c] = pd.to_numeric(iq[c], errors="coerce")
    iq["visit_id"] = iq["visit_id"].astype("int64")
    return iq


def fetch():
    """ConsDB columns, one row per value_added visit, keyed on ``visit_id``."""
    con = reader.open_db(reader.catalog_db_path("v1", VA_BUILD))
    try:
        v = con.execute("SELECT visit_id, day_obs, seq_num FROM visit_telemetry").df()
    finally:
        con.close()
    cdb = make_consdb_client("auto")
    parts = []
    for month, chunk in v.groupby(v.day_obs // 100):
        got = reader.join_consdb(chunk, groups=("meta", "thermal"), cdb=cdb)
        got = got.merge(fetch_iq(cdb, sorted(chunk.day_obs.unique())), on="visit_id",
                        how="left")
        parts.append(got)
        print(f"{month}: {len(chunk)} visits")
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    df = fetch()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    print(f"{len(df)} visits written to {CACHE}")
    print(df.notna().mean().round(3).to_string())
