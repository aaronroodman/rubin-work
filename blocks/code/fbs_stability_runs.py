"""List the FBS-driven AOS-stability runs from a blocks visits_summary parquet,
with a mean PSF FWHM per run so the visits are easy to find.

FBS_stability visits are those with observation_reason == 'fbs_driven_aos_
stability_test' (there is also a plain 'aos_stability_test' -- excluded by
default).  A "run" = one night's execution of a block, i.e. group by
(day_obs, science_program).  Reports n_visits, visit_id range, band(s), and the
mean/median of the per-visit PSF FWHM, plus a per-block rollup.

    python code/fbs_stability_runs.py                       # uses the newest visits_summary parquet
    python code/fbs_stability_runs.py --parquet output/visits_summary_20251026-20260730.parquet
    python code/fbs_stability_runs.py --fwhm-col psf_fwhm_area_50 --out output/fbs_stability_runs.csv
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def newest_summary():
    fs = sorted(glob.glob('output/visits_summary_*.parquet'))
    return fs[-1] if fs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parquet', default=None, help='visits_summary parquet (default: newest)')
    ap.add_argument('--reason', default='fbs_driven_aos_stability_test')
    ap.add_argument('--fwhm-col', default='psf_fwhm_50',
                    help='per-visit FWHM column to average (arcsec)')
    ap.add_argument('--out', default=None, help='optional CSV of the per-run table')
    ap.add_argument('--visits-out', default=None, help='optional CSV of the matching visits')
    args = ap.parse_args()

    path = args.parquet or newest_summary()
    if not path or not os.path.exists(path):
        raise SystemExit('no visits_summary parquet found (pass --parquet)')
    df = pd.read_parquet(path)
    fcol = args.fwhm_col
    f = df[df.observation_reason.astype(str) == args.reason].copy()
    f[fcol] = pd.to_numeric(f[fcol], errors='coerce')
    print(f'{path}: {len(f)} {args.reason} visits, FWHM col = {fcol}\n')

    runs = (f.groupby(['day_obs', 'science_program'], dropna=False)
              .agg(n_visits=('visit_id', 'size'),
                   visit_min=('visit_id', 'min'), visit_max=('visit_id', 'max'),
                   mean_fwhm=(fcol, 'mean'), median_fwhm=(fcol, 'median'),
                   std_fwhm=(fcol, 'std'),
                   bands=('band', lambda s: ''.join(sorted(set(s.dropna())))))
              .reset_index()
              .sort_values(['day_obs', 'science_program']))
    for c in ('day_obs', 'visit_min', 'visit_max'):
        runs[c] = runs[c].astype('int64')
    for c in ('mean_fwhm', 'median_fwhm', 'std_fwhm'):
        runs[c] = runs[c].round(3)
    print('=== FBS_stability runs (day_obs x block) ===')
    print(runs.to_string(index=False))

    block = (f.groupby('science_program', dropna=False)
               .agg(n_runs=('day_obs', 'nunique'), n_visits=('visit_id', 'size'),
                    mean_fwhm=(fcol, 'mean'), median_fwhm=(fcol, 'median'))
               .reset_index())
    for c in ('mean_fwhm', 'median_fwhm'):
        block[c] = block[c].round(3)
    print('\n=== per-block rollup ===')
    print(block.to_string(index=False))
    print(f'\ntotal: {len(runs)} runs, {len(f)} visits, {f.day_obs.nunique()} nights')

    if args.out:
        runs.to_csv(args.out, index=False)
        print(f'wrote {args.out}')
    if args.visits_out:
        cols = ['visit_id', 'day_obs', 'science_program', 'seq_num', 'band', fcol]
        f[cols].sort_values('visit_id').to_csv(args.visits_out, index=False)
        print(f'wrote {args.visits_out}')


if __name__ == '__main__':
    main()
