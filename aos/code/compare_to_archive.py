#!/usr/bin/env python3
"""Compare the new combined pipeline outputs against the archived (pre-reorg) ones.

After a from-scratch Snakemake run, this checks that the new combined tables
  output/<param_set>/{donuts,fits,visits}.parquet
reproduce the old per-chunk flat files that were renamed into an archive dir
(e.g. output-archive-2026-06-11/<phrase>_<dmin>_<dmax>{,_fits,_visits}.parquet).

For each param_set in snake_config.yaml it builds the "expected" table by
concatenating the archived chunk files and compares:
  * fits, visits   — row count + per-(day_obs, seq_num) aligned numeric compare
                     (the DZ coefficients are the key scientific result)
  * donuts         — row count + order-independent per-column summary stats
                     (per-donut rows are hard to align 1:1)

Run on the RSP (needs pyarrow).  Examples:
  python code/compare_to_archive.py --archive /home/r/roodman/u/LSST/notebooks/rubin-work/aos/output-archive-2026-06-11
  python code/compare_to_archive.py --archive ../output-archive-2026-06-11 --param-set fam_danish_v1_triplets_bin_2x
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ALIGN_KEY = ['day_obs', 'seq_num']


def phrase(ps, param_sets):
    r = param_sets.get(ps, {})
    if r.get('collection_phrase'):
        return r['collection_phrase']
    parts = r['fam_collections'][0].split('/')
    return parts[2] if (parts[0] == 'u' and len(parts) > 2) else r['fam_collections'][0]


def archived_chunk_paths(archive, ph, chunks, suffix):
    """Old flat names: <phrase>_<dmin>_<dmax>{suffix}.parquet (suffix '' | '_fits' | '_visits')."""
    return [Path(archive) / f'{ph}_{a}_{b}{suffix}.parquet' for a, b in chunks]


def load_concat(paths):
    have = [p for p in paths if p.exists()]
    missing = [str(p) for p in paths if not p.exists()]
    if not have:
        return None, missing
    return pd.concat([pd.read_parquet(p) for p in have], ignore_index=True), missing


def numeric_scalar_cols(df):
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in ALIGN_KEY]


def compare_aligned(name, new, old, rtol, atol):
    """Sort both on (day_obs, seq_num) and compare numeric scalar columns."""
    ok = (len(new) == len(old))
    print(f'  rows: new={len(new)}  archived={len(old)}  '
          f'{"MATCH" if ok else "DIFFER"}')
    cn, co = set(new.columns), set(old.columns)
    if cn != co:
        print(f'  columns: new-only={sorted(cn - co)[:6]}  '
              f'archived-only={sorted(co - cn)[:6]}')
    key = [k for k in ALIGN_KEY if k in new.columns and k in old.columns]
    if not key or len(new) != len(old):
        print('  (skipping value compare — no key or row mismatch)')
        return ok
    n = new.sort_values(key).reset_index(drop=True)
    o = old.sort_values(key).reset_index(drop=True)
    if not n[key].equals(o[key]):
        print('  WARNING: key columns differ after sort — different visit sets')
        return False
    cols = [c for c in numeric_scalar_cols(new) if c in co]
    worst = []
    for c in cols:
        a = pd.to_numeric(n[c], errors='coerce').to_numpy(dtype=float)
        b = pd.to_numeric(o[c], errors='coerce').to_numpy(dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() == 0:
            continue
        d = np.abs(a[m] - b[m])
        nbad = int((d > atol + rtol * np.abs(b[m])).sum())
        if nbad:
            worst.append((c, float(d.max()), nbad))
    worst.sort(key=lambda x: -x[1])
    if worst:
        print(f'  {len(worst)}/{len(cols)} numeric cols differ beyond tol '
              f'(rtol={rtol}, atol={atol}); worst:')
        for c, mx, nb in worst[:12]:
            print(f'    {c:32s} max|Δ|={mx:.3g}  n_beyond_tol={nb}')
        return False
    print(f'  all {len(cols)} numeric columns match within tol '
          f'(rtol={rtol}, atol={atol})')
    return ok


def compare_summary(name, new, old, rtol, atol):
    """Order-independent: compare per-column mean/std (for the big donut table)."""
    ok = (len(new) == len(old))
    print(f'  rows: new={len(new)}  archived={len(old)}  '
          f'{"MATCH" if ok else "DIFFER"}')
    cols = [c for c in numeric_scalar_cols(new) if c in set(old.columns)]
    worst = []
    for c in cols:
        a = pd.to_numeric(new[c], errors='coerce')
        b = pd.to_numeric(old[c], errors='coerce')
        for stat, fa, fb in (('mean', a.mean(), b.mean()), ('std', a.std(), b.std())):
            if np.isfinite(fa) and np.isfinite(fb):
                if abs(fa - fb) > atol + rtol * abs(fb):
                    worst.append((f'{c}.{stat}', abs(fa - fb)))
    worst.sort(key=lambda x: -x[1])
    if worst:
        print(f'  {len(worst)} column summary-stats differ beyond tol; worst:')
        for c, d in worst[:12]:
            print(f'    {c:36s} |Δ|={d:.3g}')
        return False
    print(f'  {len(cols)} numeric columns: mean & std match within tol')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--archive', required=True,
                    help='Archive dir holding the old flat per-chunk parquets')
    ap.add_argument('--output-root', default='output',
                    help='Root of the new combined outputs (default: output)')
    ap.add_argument('--param-set', default=None,
                    help='Limit to one param_set (default: all in snake_config.yaml)')
    ap.add_argument('--config', default='snake_config.yaml')
    ap.add_argument('--param-sets-file', default='param_sets.yaml')
    ap.add_argument('--rtol', type=float, default=1e-5)
    ap.add_argument('--atol', type=float, default=1e-6)
    args = ap.parse_args()

    param_sets = yaml.safe_load(open(args.param_sets_file))
    cfg = yaml.safe_load(open(args.config))['param_sets']
    targets = [args.param_set] if args.param_set else list(cfg)

    all_ok = True
    for ps in targets:
        ph = phrase(ps, param_sets)
        chunks = [(int(a), int(b)) for a, b in cfg[ps]['chunks']]
        print(f'\n########## {ps}  (phrase={ph}, {len(chunks)} chunks) ##########')
        for name, suffix, cmp in (('donuts', '', compare_summary),
                                  ('fits', '_fits', compare_aligned),
                                  ('visits', '_visits', compare_aligned)):
            print(f'\n=== {name} ===')
            new_path = Path(args.output_root) / ps / f'{name}.parquet'
            if not new_path.exists():
                print(f'  NEW MISSING: {new_path}'); all_ok = False; continue
            old, missing = load_concat(archived_chunk_paths(args.archive, ph, chunks, suffix))
            if missing:
                print(f'  archived missing: {missing}')
            if old is None:
                print('  no archived chunks found'); all_ok = False; continue
            ok = cmp(name, pd.read_parquet(new_path), old, args.rtol, args.atol)
            all_ok = all_ok and ok

    print('\n' + ('=' * 50))
    print('OVERALL: ' + ('ALL MATCH ✓' if all_ok else 'DIFFERENCES FOUND — review above'))
    return 0 if all_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
