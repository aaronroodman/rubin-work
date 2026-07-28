#!/usr/bin/env python
"""Build a per-night full-telemetry AOS table (parquet) for one day_obs.

Unlike ``build_t539_table.py`` (which selects only converged closed-loop
in-focus images across a day_obs range), this takes EVERY science visit in a
single night so the AOS telemetry -- in particular the M1M3/M2 mirror LUT
bending modes -- can be studied across the night's full elevation range (the
wide range needed to separate the cos(E)/sin(E) gravity terms).

Reuses the shared ``telemetry_pipeline.collect_telemetry`` collector, so it
attaches exactly the same columns as the T539 table: aggregated DOF trim,
hexapod + mirror LUT, geom v-modes, per-corner Zernikes, thermal + wind.

ConsDB + EFD only (no Butler) -- runs on the USDF or Summit RSP pod.
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import telemetry_pipeline    # noqa: E402  (adds aos/code + olr/code to sys.path)
import aos_trim              # noqa: E402


def build(args):
    os.environ["no_proxy"] = os.environ.get("no_proxy", "") + ",.consdb"

    # 1. All visits in the night from ConsDB ----------------------------------
    client = aos_trim.make_consdb_client(args.consdb_url)
    q = f"""
        SELECT v1.*, ql.physical_rotator_angle, ql.psf_sigma_median,
               ql.seeing_zenith_500nm_median
        FROM cdb_{args.instrument}.visit1 v1
        LEFT JOIN cdb_{args.instrument}.visit1_quicklook ql
          ON v1.visit_id = ql.visit_id
        WHERE v1.day_obs = {args.day_obs}
    """
    visits = client.query(q).to_pandas()
    print(f"Fetched {len(visits)} visits for day_obs {args.day_obs}", flush=True)

    # Keep real on-sky science exposures (need obs_start for the EFD anchoring
    # and an elevation for the mirror-mode-vs-elevation study).
    if args.image_type and "img_type" in visits:
        keep = visits["img_type"].fillna("").str.lower().isin(
            [t.lower() for t in args.image_type])
        visits = visits[keep]
    visits = visits[visits["obs_start"].notna()]
    visits = visits.sort_values("seq_num").reset_index(drop=True)
    if "psf_sigma_median" in visits:
        visits["psf_fwhm_median"] = 2.355 * pd.to_numeric(
            visits["psf_sigma_median"], errors="coerce") * 0.2
    print(f"Kept {len(visits)} science visits", flush=True)
    if len(visits) == 0:
        raise SystemExit(f"no science visits for day_obs {args.day_obs}")

    ids = ",".join(str(v) for v in visits["visit_id"])
    for col in ("donut_blur_fwhm", "aos_fwhm"):
        try:
            extra = client.query(
                f"SELECT visit_id, {col} FROM cdb_{args.instrument}.visit1_quicklook "
                f"WHERE visit_id IN ({ids})").to_pandas()
            visits = visits.merge(extra, on="visit_id", how="left")
        except Exception as e:
            print(f"{col} not available: {type(e).__name__}", flush=True)

    # 2-5. Full AOS telemetry (DOF/LUT/v-modes/Zernikes/thermal/wind) ---------
    #      Corner Zernikes are usually absent for ordinary science visits, so
    #      skip them by default (they are only produced for AOS closed-loop runs).
    visits = telemetry_pipeline.collect_telemetry(
        visits, args, client, with_zernikes=args.with_zernikes)

    # 6. Save -----------------------------------------------------------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    visits.to_parquet(args.out, index=False)
    print(f"Wrote {len(visits)} rows x {visits.shape[1]} cols to {args.out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day-obs", type=int, required=True, dest="day_obs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--instrument", default="lsstcam")
    ap.add_argument("--consdb-url", default="auto",
                    dest="consdb_url")
    ap.add_argument("--efd", default="usdf_efd")
    ap.add_argument("--ofc-config-dir", required=True, dest="ofc_config_dir")
    ap.add_argument("--image-type", nargs="*", default=["science"],
                    dest="image_type",
                    help="img_type values to keep (default: science); "
                         "pass none/empty to keep all")
    ap.add_argument("--with-zernikes", action="store_true", dest="with_zernikes",
                    help="also fetch per-corner retrieved Zernikes (default off "
                         "for a full night -- usually absent for science visits)")
    ap.add_argument("--n-vmode", type=int, default=12, dest="n_vmode")
    ap.add_argument("--telemetry-source", default="hybrid", choices=["hybrid", "consdb", "efd"],
                    dest="telemetry_source",
                    help="hybrid = mirror/temps/wind from ConsDB + DOF/hexapod from raw EFD (default); consdb = all from ConsDB (sparse logevents); "
                         "efd = raw per-visit EFD (cross-check / fallback)")
    ap.add_argument("--no-gradients-from-efd", dest="gradients_from_efd",
                    action="store_false",
                    help="skip the raw-EFD M1M3 gradient fetch in consdb mode")
    ap.set_defaults(gradients_from_efd=True)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
