"""Build a fit manifest (day_obs seq_num per line) of in-focus CWFS-triplet
visits, for the fit_manifest.sbatch production array.

Source = the danish aggregateAOSVisitTableRaw `visits.parquet` tables (each row =
one FAM triplet, keyed on the FAM extra seq_num; the in-focus star visit is
seq_num + 1).  Pass one or more visits.parquet paths (globs OK); rows are kept if
visit_quality_pass and not rotator_flagged.

    python code/build_fam_manifest.py --out pipelines/prod_famcwfs.txt \
        '../aos/output/fam_danish_*/**/visits.parquet'

If your FAM triplets are enumerated some other way (e.g. a ConsDB
observation_reason / science_program), build the manifest from that instead --
the sbatch only needs 'day_obs seq_num' lines.
"""
import argparse
import glob

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('globs', nargs='+', help='visits.parquet path(s)/glob(s)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--no-quality-cut', action='store_true',
                    help='keep all triplets (default: visit_quality_pass only)')
    args = ap.parse_args()

    paths = sorted({p for g in args.globs for p in glob.glob(g, recursive=True)})
    if not paths:
        raise SystemExit(f'no visits.parquet matched: {args.globs}')
    seen, rows = set(), []
    for p in paths:
        v = pd.read_parquet(p)
        if not args.no_quality_cut and 'visit_quality_pass' in v.columns:
            v = v[v.visit_quality_pass]
            if 'rotator_flagged' in v.columns:
                v = v[~v.rotator_flagged]
        for day, sq in zip(v.day_obs.astype(int), v.seq_num.astype(int)):
            key = (day, sq + 1)                       # in-focus = FAM extra + 1
            if key not in seen:
                seen.add(key); rows.append(key)
    rows.sort()
    with open(args.out, 'w') as f:
        for day, seq in rows:
            f.write(f'{day} {seq}\n')
    print(f'wrote {args.out}: {len(rows)} in-focus visits from {len(paths)} table(s)')
    print(f'  nights: {sorted({d for d, _ in rows})}')


if __name__ == '__main__':
    main()
