#!/usr/bin/env python3
"""Backfill thermal telemetry onto a FAM visits.parquet, without re-running mktable.

Thermal (the 13 ESS-temperature / M1M3-gradient / TMA-truss columns, incl.
`z_gradient`) is a VISIT-level product: run_mktable merges it into the visits
table only (never the donuts).  When a chunk's mktable ran somewhere the EFD or
ConsDB was unreachable (e.g. a Slurm compute node), the thermal step fails and
those columns are dropped -- but the expensive donut/Zernike streaming is fine.
This script fixes only the thermal part: it loads an existing visits.parquet,
re-fetches thermal via the package's own get_thermal_data + merge, and rewrites
the file.  Fast (a visit-level EFD/ConsDB fetch), no donut reprocessing.

Run on a node where BOTH the EFD (usdf_efd) and ConsDB resolve -- the RSP
terminal or a slaciana/slacrd interactive node -- NOT a batch compute node.

    python run_backfill_thermal.py --visits output/<ps>/chunks/<d>_<d>/visits.parquet

Reuses run_mktable's ConsDB-URL + token convention (default external USDF URL,
token from ~/.lsst/consdb_token; the in-pod URL also works on the RSP).
"""
import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
from astropy.table import QTable

from lsst.summit.utils import ConsDbClient
from lsst.ts.intrinsic.wavefront.intrinsics_lib import (
    get_thermal_data, merge_thermal_to_visit_info, makeEfdClient,
    _close_efd_client)

DEFAULT_CONSDB_URL = "https://usdf-rsp.slac.stanford.edu/consdb"


def make_consdb_client(consdb_url):
    """ConsDbClient with the same token convention as run_mktable: embed the
    ~/.lsst/consdb_token into an external URL; leave in-pod URLs untouched."""
    url = consdb_url
    if "@" not in url and "consdb-pq.consdb" not in url:
        tf = Path.home() / ".lsst" / "consdb_token"
        if tf.exists():
            url = url.replace("://", f"://user:{tf.read_text().strip()}@", 1)
            print("  ConsDB: external URL with token from ~/.lsst/consdb_token")
        else:
            print("  WARNING: external ConsDB URL but no ~/.lsst/consdb_token found")
    else:
        print("  ConsDB: in-pod URL")
    return ConsDbClient(url)


async def backfill(visits_path, output_path, consdb_url, temp_window, force):
    vi = QTable.read(str(visits_path))
    has = "z_gradient" in vi.colnames
    days = sorted({int(d) for d in vi["day_obs"]})
    print(f"[backfill_thermal] {visits_path}")
    print(f"  {len(vi)} visits, {len(vi.colnames)} cols, day_obs={days}, "
          f"thermal already present={has}")
    if has and not force:
        print("  thermal already present -- nothing to do (use --force to refetch)")
        return 0

    consdb = make_consdb_client(consdb_url)
    try:
        probe = consdb.query(
            f"SELECT day_obs, seq_num FROM cdb_lsstcam.exposure "
            f"WHERE day_obs IN ({', '.join(str(d) for d in days)}) LIMIT 5"
        ).to_pandas()
        print(f"  ConsDB reachable ({len(probe)} sample exposure rows)")
    except Exception as e:
        print(f"  ERROR: ConsDB query failed -> {type(e).__name__}: {e}")
        print("  Are you on an interactive node (RSP/slaciana), not a batch node?")
        return 2

    efd = None
    try:
        efd = makeEfdClient()
        print("  EFD client OK")
        kw = {} if temp_window is None else {"temp_time_window_sec": temp_window}
        thermal_df = await get_thermal_data(consdb, efd, vi, **kw)
        nz = int(thermal_df["z_gradient"].notna().sum()) if "z_gradient" in thermal_df else 0
        print(f"  thermal_df: {thermal_df.shape}, z_gradient valid {nz}/{len(thermal_df)}")
        vi = merge_thermal_to_visit_info(thermal_df, vi)
    except Exception as e:
        print(f"  ERROR: thermal fetch/merge failed -> {type(e).__name__}: {e}")
        return 2
    finally:
        if efd is not None:
            await _close_efd_client(efd)

    out = output_path or visits_path
    vi.write(str(out), overwrite=True)
    chk = QTable.read(str(out))
    nz = (int(np.isfinite(np.asarray(chk["z_gradient"], float)).sum())
          if "z_gradient" in chk.colnames else 0)
    ntc = sum(1 for c in chk.colnames if c in (
        "z_gradient", "x_gradient", "y_gradient", "radial_gradient",
        "cam_air_temp", "m2_air_temp", "m1m3_air_temp", "outside_temp",
        "m2_delta_t", "cam_m1m3_delta_t", "dome_delta_t",
        "tma_truss_temp_pxpy", "tma_truss_temp_mxmy"))
    print(f"  wrote {out}: {len(chk.colnames)} cols, {ntc}/13 thermal cols, "
          f"z_gradient valid {nz}/{len(chk)}")
    if nz == 0:
        print("  WARNING: z_gradient is all-NaN -- EFD had no M1M3 gradient data "
              "for these visits (check the EFD/date range)")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--visits", required=True,
                    help="visits.parquet to backfill (rewritten in place unless --output)")
    ap.add_argument("--output", default=None,
                    help="write here instead of overwriting --visits")
    ap.add_argument("--consdb-url", default=DEFAULT_CONSDB_URL,
                    help=f"ConsDB URL (default: {DEFAULT_CONSDB_URL})")
    ap.add_argument("--temp-window", type=float, default=None,
                    help="temp_time_window_sec (default: package default)")
    ap.add_argument("--force", action="store_true",
                    help="refetch even if thermal columns are already present")
    args = ap.parse_args()

    vp = Path(args.visits)
    if not vp.exists():
        sys.exit(f"visits file not found: {vp}")
    rc = asyncio.run(backfill(vp, args.output, args.consdb_url,
                              args.temp_window, args.force))
    sys.exit(rc)


if __name__ == "__main__":
    main()
