#!/usr/bin/env python
"""
Fetch ACTUAL Cerro Pachon weather (ERA5 model-level winds/temperature) for a
date range and write the forecast pickle that psf-weather-station / guider_atmo_sim
consume. Wraps psfws.get_ecmwf_data so the model levels match the bundled p_and_h.p.

Prerequisites (checked at startup):
  * cdsapi, xarray, cfgrib installed:   pip3 install xarray cfgrib   (cdsapi already present)
  * ~/.cdsapirc with a Copernicus CDS personal access token, e.g.:
        url: https://cds.climate.copernicus.eu/api
        key: <your-personal-access-token>
    and you must accept the "ERA5 complete" (model-level) licence once at
    https://cds.climate.copernicus.eu (Datasets -> ERA5 -> Download -> accept terms).

Usage:
  python fetch_cp_weather.py                       # July 2026 nights (default), dry-run preflight
  python fetch_cp_weather.py --run                 # actually download
  python fetch_cp_weather.py --d1 20260701 --d2 20260731 --run
  python fetch_cp_weather.py --outdir ~/rubin-data/guider/psfws --run

Then feed it to the simulator:
  python guider_atmo_sim.py --atmo-source psfws \
      --psfws-forecast ecmwf_-30.25_-70.75_20260701_20260713.p \
      --psfws-data-dir <outdir> --psfws-date "2026-07-09 06:00"

Notes:
  * Cerro Pachon point: lat=-30.25, lon=-70.75 (must be multiples of 0.25).
  * ERA5 latency: final ERA5 lags ~2-3 months; very recent dates come back as
    ERA5T (expver=5). psfws hardcodes expver='1'; if a recent request returns
    empty, this script says so. Fallbacks, in order of preference:
      1. same month LAST YEAR:  --d1 20250701 --d2 20250713   (final ERA5, real)
      2. the bundled 2019 July climatology (no download at all):
         python guider_atmo_sim.py --atmo-source psfws --psfws-month 7
  * No summit telemetry is needed: guider_atmo_sim runs psfws forecast-only for
    custom forecast files (ground layer interpolated from the ECMWF profile).
"""
import argparse
import os
import pathlib
import sys

LAT, LON = -30.25, -70.75            # Cerro Pachon, rounded to 0.25 deg
DEFAULT_D1, DEFAULT_D2 = "20260701", "20260713"   # July 2026 nights being analyzed


def preflight():
    problems = []
    for mod in ("cdsapi", "xarray", "cfgrib"):
        try:
            __import__(mod)
        except Exception as exc:                              # noqa: BLE001
            problems.append(f"  missing '{mod}': {exc}  ->  pip3 install {mod}")
    rc = pathlib.Path.home() / ".cdsapirc"
    if not rc.is_file():
        problems.append(
            "  no ~/.cdsapirc. Create it with your CDS token:\n"
            "      url: https://cds.climate.copernicus.eu/api\n"
            "      key: <your-personal-access-token>\n"
            "    and accept the ERA5-complete licence at https://cds.climate.copernicus.eu")
    return problems


def _dedupe_forecast(path):
    """Drop duplicate timestamps (ERA5 vs ERA5T expver overlap duplicates recent
    dates) so psfws/guider_atmo_sim see one row per time. Rewrites the pickle."""
    import pickle
    with open(path, "rb") as f:
        df = pickle.load(f)
    if df.index.has_duplicates:
        n0 = len(df)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        with open(path, "wb") as f:
            pickle.dump(df, f)
        print(f"  de-duplicated: {n0} -> {len(df)} rows (ERA5/ERA5T overlap)")


def _validate_forecast(path):
    """True if the forecast pickle has usable rows (guards against empty ERA5T)."""
    import pickle
    try:
        with open(path, "rb") as f:
            df = pickle.load(f)
    except Exception as exc:                                  # noqa: BLE001
        print(f"  could not read {path}: {exc}")
        return False
    n = df.index.nunique()
    print(f"  forecast rows: {len(df)} ({n} unique times)"
          + (f"  {df.index.min()} .. {df.index.max()}" if n else ""))
    return n > 0


def import_psfws_ecmwf():
    """psfws.get_ecmwf_data does a bare `import utils`, so add the pkg dir to path."""
    import psfws
    pkg = os.path.dirname(psfws.__file__)
    if pkg not in sys.path:
        sys.path.insert(0, pkg)
    import get_ecmwf_data as ge                                # noqa: E402
    return ge


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--d1", default=DEFAULT_D1, help="start date YYYYMMDD")
    ap.add_argument("--d2", default=DEFAULT_D2, help="end date YYYYMMDD")
    ap.add_argument("--lat", type=float, default=LAT)
    ap.add_argument("--lon", type=float, default=LON)
    ap.add_argument("--outdir", default=None,
                    help="dir for grib + forecast pickle (default: psfws pkg data dir)")
    ap.add_argument("--run", action="store_true",
                    help="actually download (default is a dry-run preflight)")
    args = ap.parse_args()

    problems = preflight()
    if problems:
        print("Preflight found issues:\n" + "\n".join(problems))
        if args.run:
            print("\nFix the above, then re-run with --run.")
            sys.exit(1)
    else:
        print("Preflight OK: cdsapi/xarray/cfgrib importable, ~/.cdsapirc present.")

    import psfws
    outdir = pathlib.Path(args.outdir) if args.outdir else \
        pathlib.Path(os.path.dirname(psfws.__file__)) / "data"
    outdir.mkdir(parents=True, exist_ok=True)
    forecast_name = f"ecmwf_{args.lat}_{args.lon}_{args.d1}_{args.d2}.p"

    print(f"\nplan: ERA5 model-level winds for lat={args.lat} lon={args.lon} "
          f"{args.d1}..{args.d2}")
    print(f"      -> {outdir / forecast_name}")

    if not args.run:
        print("\n(dry run — re-run with --run to download)")
        return
    if problems:
        return

    ge = import_psfws_ecmwf()
    cwd = os.getcwd()
    try:
        os.chdir(outdir)                    # psfws writes grib relative to cwd
        ge.get_ecmwf_data(start_date=args.d1, end_date=args.d2,
                          lat=args.lat, lon=args.lon, grib_dir=outdir, delete=False)
    finally:
        os.chdir(cwd)

    _dedupe_forecast(outdir / forecast_name)
    ok = _validate_forecast(outdir / forecast_name)
    print(f"\nwrote {outdir / forecast_name}")
    if not ok:
        yr = int(args.d1[:4]) - 1
        print("\n*** The forecast has no usable rows -- likely ERA5T/expver for very "
              "recent dates. Fallbacks:\n"
              f"    1. last year:  python fetch_cp_weather.py --d1 {yr}{args.d1[4:]} "
              f"--d2 {yr}{args.d2[4:]} --run\n"
              "    2. bundled 2019 climatology (no download):  "
              "python guider_atmo_sim.py --atmo-source psfws --psfws-month 7")
        return
    print("Use it with:\n"
          f"  python guider_atmo_sim.py --atmo-source psfws \\\n"
          f"      --psfws-forecast {forecast_name} --psfws-data-dir {outdir} \\\n"
          f"      --psfws-date \"{args.d1[:4]}-{args.d1[4:6]}-{args.d1[6:]} 06:00\"\n"
          "  (--psfws-date also sets the alt/az pointing via boresight_ra/dec in CFG;\n"
          "   use --obs-time / --alt / --az to override.)")


if __name__ == "__main__":
    main()
