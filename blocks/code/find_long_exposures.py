"""Find ConsDB visits with exp_time >= a threshold, across ALL science programs.

The blocks visits_summary parquet is filtered to survey/AOS science programs and
never selects exposure time, so it cannot answer "which visits are >= 120 s"
(e.g. the long BLOCK-T637 exposures on 20260409, and whether there are more).
This queries cdb_<instrument>.visit1 directly for the exposure-time column
(auto-detected) with no science_program cut, and summarizes by day / program.

    python code/find_long_exposures.py --min-exptime 120
    python code/find_long_exposures.py --min-exptime 120 --day-min 20260409 --day-max 20260409
    python code/find_long_exposures.py --min-exptime 120 --out ../blocks/output/long_exposures.csv

Runs on the RSP (in-pod ConsDB) or S3DF slaciana (external endpoint +
~/.lsst/consdb_token); --consdb-url defaults to 'auto' (see aos_trim).
"""
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# reuse the shared ConsDB client (rubin-work/aos/code), same as build_night_table
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "aos" / "code"))
from aos_trim import make_consdb_client                        # noqa: E402


def find_exp_column(client, instrument):
    """Return the visit1 exposure-time column name (auto-detected)."""
    df = client.query(f"SELECT * FROM cdb_{instrument}.visit1 LIMIT 1").to_pandas()
    cands = [c for c in df.columns
             if c.lower() in ('exp_time', 'exptime', 'exposure_time')
             or 'exp_time' in c.lower() or 'exposure_time' in c.lower()]
    return (cands[0] if cands else None), list(df.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-exptime', type=float, default=120.0)
    ap.add_argument('--day-min', type=int, default=None)
    ap.add_argument('--day-max', type=int, default=None)
    ap.add_argument('--instrument', default='lsstcam')
    ap.add_argument('--consdb-url', default='auto')
    ap.add_argument('--token-file', default=None)
    ap.add_argument('--out', default=None, help='optional CSV of the matching visits')
    args = ap.parse_args()

    client = make_consdb_client(url=args.consdb_url, token_file=args.token_file)
    expcol, allcols = find_exp_column(client, args.instrument)
    if expcol is None:
        sys.exit(f'no exposure-time column in visit1; columns:\n{allcols}')
    print(f'exposure-time column: {expcol}')

    where = [f'{expcol} >= {args.min_exptime}']
    if args.day_min is not None:
        where.append(f'day_obs >= {args.day_min}')
    if args.day_max is not None:
        where.append(f'day_obs <= {args.day_max}')
    q = f"""SELECT visit_id, day_obs, science_program, target_name,
                   observation_reason, band, physical_filter, {expcol} AS exp_time
            FROM cdb_{args.instrument}.visit1
            WHERE {' AND '.join(where)}
            ORDER BY day_obs, visit_id"""
    df = client.query(q).to_pandas()
    df['exp_time'] = pd.to_numeric(df['exp_time'], errors='coerce')
    print(f'\n{len(df)} visits with {expcol} >= {args.min_exptime:g} s'
          + (f' over day_obs [{args.day_min}, {args.day_max}]'
             if args.day_min or args.day_max else ' (all dates)'))
    if not len(df):
        return

    g = (df.groupby(['day_obs', 'science_program'], dropna=False)
           .agg(n=('visit_id', 'size'), exp_min=('exp_time', 'min'),
                exp_max=('exp_time', 'max'),
                seq_lo=('visit_id', 'min'), seq_hi=('visit_id', 'max'))
           .reset_index())
    print('\nby day_obs / science_program:')
    print(g.to_string(index=False))
    print('\nscience programs with long exposures:',
          sorted(df.science_program.dropna().astype(str).unique().tolist()))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f'\nwrote {args.out} ({len(df)} rows)')


if __name__ == '__main__':
    main()
