#!/usr/bin/env python3
"""Check Butler provenance consistency across a param_set's date chunks.

For each chunk of a param_set (date ranges from snake_config.yaml), pick the
first visit present, then look up — via the Butler — the RUN collection that
produced its donut/Zernike dataset, that RUN's software package versions, and
the dataset's column set.  Reports whether RUN / package tags / column sets are
**consistent across all chunks**.

Motivation: a param_set points at a single chained collection, so every chunk
should resolve to the same processing RUN and code version.  If they don't, the
collection was assembled from more than one processing pass (different code
versions), which shows up downstream as schema differences between chunks
(e.g. the *_donut_id columns appearing only in later chunks).

Run on the RSP (needs lsst.daf.butler + the configured repo):

    python code/inspect_visit_provenance.py --param-set fam_danish_1_0_wep17_3_0_bin2x
    python code/inspect_visit_provenance.py --param-set all
"""
import argparse
import sys
from pathlib import Path

import yaml

# Packages highlighted in the printout; the full diff still compares them all.
DEFAULT_KEY_PKGS = ['lsst_distrib', 'ts_wep', 'ts_donut_viz', 'donut_viz', 'ts_ofc']


def first_visit_from_parquet(donut_parquet):
    """First (day_obs, seq_num) in a chunk donut parquet (row group 0), or None."""
    import pyarrow.parquet as pq
    p = Path(donut_parquet)
    if not p.exists():
        return None
    t = pq.ParquetFile(str(p)).read_row_group(0, columns=['day_obs', 'seq_num'])
    return int(t['day_obs'][0].as_py()), int(t['seq_num'][0].as_py())


def first_visit_from_butler(butler, dataset_type, dmin, dmax):
    """First (day_obs, seq_num) of `dataset_type` in [dmin, dmax] via the Butler."""
    dids = butler.registry.queryDataIds(
        ['visit'], datasets=dataset_type,
        where='instrument=:inst AND visit.day_obs >= :dmin AND visit.day_obs <= :dmax',
        bind={'inst': 'LSSTCam', 'dmin': int(dmin), 'dmax': int(dmax)})
    recs = [d.records['visit'] for d in dids.expanded()]
    if not recs:
        return None
    r = min(recs, key=lambda x: (x.day_obs, x.seq_num))
    return int(r.day_obs), int(r.seq_num)


def find_ref(butler, dataset_type, day_obs, seq_num):
    """DatasetRef for one visit (modern find_dataset, registry fallback)."""
    try:
        return butler.find_dataset(dataset_type, day_obs=day_obs, seq_num=seq_num)
    except AttributeError:
        return butler.registry.findDataset(
            dataset_type, instrument='LSSTCam', day_obs=day_obs, seq_num=seq_num)


def get_packages(butler, run):
    try:
        return butler.get('packages', collections=[run])
    except Exception:
        return None


def inspect_param_set(param_set, params, chunks, dataset_type, output_root,
                      key_pkgs, use_butler_first=False):
    from lsst.daf.butler import Butler
    repo = params['butler_repo']
    collections = params['fam_collections']
    print(f'\n{"=" * 78}\nparam_set: {param_set}')
    print(f'  repo={repo}\n  collections={collections}\n  {len(chunks)} chunk(s)')
    butler = Butler(repo, instrument='LSSTCam', collections=collections)

    records = []   # one dict per chunk
    for dmin, dmax in chunks:
        chunk = f'{dmin}_{dmax}'
        donut_parquet = f'{output_root}/{param_set}/chunks/{chunk}/donuts.parquet'
        visit = None if use_butler_first else first_visit_from_parquet(donut_parquet)
        if visit is None:
            try:
                visit = first_visit_from_butler(butler, dataset_type, dmin, dmax)
            except Exception as e:
                print(f'  [{chunk}] could not find first visit: '
                      f'{type(e).__name__}: {e}')
                continue
        if visit is None:
            print(f'  [{chunk}] no visits found')
            continue
        day_obs, seq_num = visit
        try:
            ref = find_ref(butler, dataset_type, day_obs, seq_num)
            tbl = butler.get(ref)
            cols = set(tbl.colnames)
            pkgs = get_packages(butler, ref.run)
        except Exception as e:
            print(f'  [{chunk}] visit {day_obs}/{seq_num}: '
                  f'{type(e).__name__}: {e}')
            continue
        rec = dict(chunk=chunk, day_obs=day_obs, seq_num=seq_num,
                   run=ref.run, cols=cols, pkgs=pkgs)
        records.append(rec)
        print(f'\n  [{chunk}] first visit day_obs={day_obs} seq_num={seq_num}')
        print(f'    RUN: {ref.run}')
        print(f'    {len(cols)} columns')
        if pkgs is not None:
            pdict = dict(pkgs)
            for k in key_pkgs:
                if k in pdict:
                    print(f'      {k:14s} = {pdict[k]}')

    _consistency_report(records, key_pkgs)
    return records


def _consistency_report(records, key_pkgs):
    print(f'\n  --- consistency across {len(records)} chunk(s) ---')
    if len(records) < 2:
        print('    (need >= 2 chunks to compare)')
        return
    ok = True

    runs = {r['run'] for r in records}
    if len(runs) == 1:
        print(f'    RUN:      CONSISTENT  ({runs.pop()})')
    else:
        ok = False
        print('    RUN:      INCONSISTENT — chunks span multiple RUNs:')
        for r in records:
            print(f'              [{r["chunk"]}] {r["run"]}')

    # Column sets relative to the first chunk.
    base = records[0]
    col_ok = True
    for r in records[1:]:
        only_base = base['cols'] - r['cols']
        only_r = r['cols'] - base['cols']
        if only_base or only_r:
            col_ok = False
            print(f'    COLUMNS:  differ {base["chunk"]} vs {r["chunk"]}:')
            if only_base:
                print(f'              only in {base["chunk"]}: {sorted(only_base)}')
            if only_r:
                print(f'              only in {r["chunk"]}: {sorted(only_r)}')
    if col_ok:
        print(f'    COLUMNS:  CONSISTENT  ({len(base["cols"])} columns)')
    else:
        ok = False

    # Package versions relative to the first chunk that has packages.
    base_pkgs = next((r['pkgs'] for r in records if r['pkgs'] is not None), None)
    if base_pkgs is None:
        print('    PACKAGES: unavailable (no packages dataset in any run)')
    else:
        pkg_ok = True
        for r in records:
            if r['pkgs'] is None or r['pkgs'] is base_pkgs:
                continue
            try:
                diff = base_pkgs.difference(r['pkgs'])
            except Exception:
                diff = {k: (dict(base_pkgs).get(k), dict(r['pkgs']).get(k))
                        for k in set(dict(base_pkgs)) | set(dict(r['pkgs']))
                        if dict(base_pkgs).get(k) != dict(r['pkgs']).get(k)}
            if diff:
                pkg_ok = False
                print(f'    PACKAGES: differ vs {r["chunk"]}:')
                for name, (va, vb) in sorted(diff.items()):
                    mark = '  <<' if name in key_pkgs else ''
                    print(f'              {name:20s} {va}  ->  {vb}{mark}')
        if pkg_ok:
            print('    PACKAGES: CONSISTENT')
        else:
            ok = False

    print(f'\n    VERDICT: {"PASS — consistent" if ok else "INCONSISTENT (see above)"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True,
                    help="param_set name, or 'all' to loop over snake_config.yaml")
    ap.add_argument('--dataset-type', default='aggregateAOSVisitTableRaw',
                    help='Butler dataset type to trace (default: %(default)s)')
    ap.add_argument('--output-root', default='output',
                    help='Root of the output/<param_set>/chunks tree (default: %(default)s)')
    ap.add_argument('--config', default=None,
                    help='snake_config.yaml path (default: ../snake_config.yaml '
                         'next to this script)')
    ap.add_argument('--from-butler', action='store_true',
                    help='Pick the first visit via the Butler date-range query '
                         'instead of reading the chunk parquet')
    ap.add_argument('--key-pkgs', nargs='+', default=DEFAULT_KEY_PKGS,
                    help='Packages to highlight (default: %(default)s)')
    args = ap.parse_args()

    aos_dir = Path(__file__).resolve().parent.parent
    cfg_path = Path(args.config) if args.config else aos_dir / 'snake_config.yaml'
    cfg = yaml.safe_load(open(cfg_path))['param_sets']
    param_sets = yaml.safe_load(open(aos_dir / 'param_sets.yaml'))

    names = list(cfg) if args.param_set == 'all' else [args.param_set]
    for name in names:
        if name not in cfg:
            print(f'param_set {name!r} not in {cfg_path} — skipping')
            continue
        if name not in param_sets:
            print(f'param_set {name!r} not in param_sets.yaml — skipping')
            continue
        chunks = [(int(a), int(b)) for a, b in cfg[name]['chunks']]
        inspect_param_set(name, param_sets[name], chunks, args.dataset_type,
                          args.output_root, args.key_pkgs,
                          use_butler_first=args.from_butler)


if __name__ == '__main__':
    main()
