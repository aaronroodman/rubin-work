#!/usr/bin/env python3
"""Pre-flight check for a pipeline chunk.

Surveys the data before mktable runs so you can spot two problems early:

  1. Visits listed in ConsDB but missing from the Butler collection
     (those would emit DatasetNotFoundError during mktable).
  2. Heterogeneous `nollIndices` across visits in the collection
     (mktable currently locks the first visit's nollIndices and skips
     visits with a different set, so a flavored chunk silently loses
     data).

Usage:
    # Survey one chunk by name (default — uses runs.yaml + param_sets.yaml):
    python code/check_chunk.py chunk1

    # Survey one chunk by param_set + explicit dates:
    python code/check_chunk.py --param-set fam_danish_v1_triplets_bin_2x \\
        --day-obs-min 20260315 --day-obs-max 20260327

    # Override the ConsDB URL (e.g. when running from outside the RSP):
    python code/check_chunk.py chunk1 \\
        --consdb-url https://usdf-rsp.slac.stanford.edu/consdb
"""

import argparse
import asyncio
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_runs(runs_yaml=None):
    p = Path(runs_yaml) if runs_yaml else \
        Path(__file__).resolve().parent.parent / 'runs.yaml'
    with open(p) as f:
        return yaml.safe_load(f)


async def check_chunk_data(butler_repo, fam_collections, day_obs_min,
                           day_obs_max, fam_programs,
                           consdb_url=None,
                           dataset_type='aggregateAOSVisitTableRaw',
                           detectors=(98, 99, 100),
                           verbose=True):
    """Run the pre-flight survey for a single chunk.

    Strategy:
      1. ConsDB → list of expected (day_obs, seq_num) visit pairs.
      2. For each pair, butler.get(dataset_type, day_obs=…, seq_num=…,
         detector=…). The first detector in `detectors` that returns
         data wins, and we read its `meta['nollIndices']` while we're
         at it — one round-trip per visit, on a small per-detector
         table (much faster than the full 188-CCD aggregate).
      3. Visits with no detector returning data are reported as
         missing.

    Returns a dict with:
        'consdb_pairs'   : list of (day_obs, seq_num) from ConsDB
        'butler_present' : list with Butler data
        'butler_missing' : list in ConsDB but not in Butler
        'noll_groups'    : dict tuple(nollIndices) -> list of (d,s)
        'other_errors'   : visits that errored when reading
    """
    # Imports kept inside so help/CLI parsing don't pay the LSST stack tax.
    from intrinsics_lib import (
        DEFAULT_CONSDB_URL, get_visit_pairs_from_consdb,
        print_band_counts_by_day,
    )
    from lsst.daf.butler import Butler
    from lsst.summit.utils import ConsDbClient
    from tqdm import tqdm

    if consdb_url is None:
        consdb_url = DEFAULT_CONSDB_URL

    # Setup ConsDB no_proxy + token (mirrors run_mktable behavior)
    os.environ.setdefault('no_proxy', '')
    if '.consdb' not in os.environ['no_proxy']:
        os.environ['no_proxy'] += ',.consdb'
    if '@' not in consdb_url and 'consdb-pq.consdb' not in consdb_url:
        token_file = Path.home() / '.lsst' / 'consdb_token'
        if token_file.exists():
            consdb_url = consdb_url.replace(
                '://', f'://user:{token_file.read_text().strip()}@', 1)
    consdb_client = ConsDbClient(consdb_url)

    if verbose:
        print(f"\n{'=' * 64}")
        print(f"Pre-flight check")
        print(f"  butler_repo:     {butler_repo}")
        print(f"  collection(s):   {fam_collections}")
        print(f"  date range:      {day_obs_min} – {day_obs_max}")
        print(f"  programs:        {fam_programs}")
        print(f"  dataset_type:    {dataset_type}")
        if detectors:
            print(f"  detector probe:  {list(detectors)} "
                  f"(first match wins per visit)")
        print(f"{'=' * 64}\n")

    # ---- 1. ConsDB query ---------------------------------------------------
    instrument = 'lsstcam'
    q = f"""
        SELECT v1.*, ql.physical_rotator_angle
        FROM cdb_{instrument}.visit1 v1
        LEFT JOIN cdb_{instrument}.visit1_quicklook ql
        ON v1.visit_id = ql.visit_id
        WHERE v1.day_obs >= {day_obs_min} AND v1.day_obs <= {day_obs_max}
    """
    visits_df = consdb_client.query(q).to_pandas()
    if verbose:
        print(f"ConsDB returned {len(visits_df)} visit rows in date range")
        print_band_counts_by_day(visits_df, fam_programs, 'cwfs')

    consdb_pairs = get_visit_pairs_from_consdb(
        visits_df, fam_programs, img_type='cwfs')
    if verbose:
        print(f"\nConsDB visit pairs (programs + cwfs + paired): "
              f"{len(consdb_pairs)}")

    # ---- 2. Butler — per-visit existence check + nollIndices peek -------
    # `aggregateAOSVisitTableRaw` is per (visit, detector) here, so each
    # butler.get loads just ~one CCD's slice (small) when we pass a
    # detector. Try `detectors` in order — the first one that exists for
    # a given visit is enough to count it as present. If detector= is
    # not a valid dim for this dataset, fall back to a no-detector get.
    from lsst.daf.butler import DatasetNotFoundError

    butler = Butler(butler_repo, instrument='LSSTCam',
                    collections=fam_collections)

    butler_present = []
    butler_missing = []
    noll_groups = defaultdict(list)
    other_errors = []

    detector_list = list(detectors) if detectors else [None]

    if verbose:
        det_msg = (f"first matching detector from {detector_list}"
                   if detectors else "no detector filter")
        print(f"\nProbing Butler for `{dataset_type}` on "
              f"{len(consdb_pairs)} visits ({det_msg})…")

    iterator = tqdm(consdb_pairs) if verbose else consdb_pairs
    for d, s in iterator:
        tbl = None
        last_err = None
        for det in detector_list:
            try:
                kwargs = dict(day_obs=d, seq_num=s)
                if det is not None:
                    kwargs['detector'] = det
                tbl = butler.get(dataset_type, **kwargs)
                break
            except DatasetNotFoundError:
                continue   # try next detector
            except Exception as e:
                last_err = e
                # Detector kwarg may be invalid if dataset is per-visit
                # only; try without it once.
                if det is not None:
                    try:
                        tbl = butler.get(
                            dataset_type, day_obs=d, seq_num=s)
                        break
                    except DatasetNotFoundError:
                        continue
                    except Exception as e2:
                        last_err = e2
                continue

        if tbl is None:
            if last_err is None:
                butler_missing.append((d, s))
            else:
                other_errors.append(
                    (d, s, type(last_err).__name__, str(last_err)))
            continue

        butler_present.append((d, s))
        noll = tbl.meta.get('nollIndices', None) \
            if hasattr(tbl, 'meta') else None
        if noll is None:
            noll_groups[None].append((d, s))
        else:
            key = tuple(int(n) for n in noll)
            noll_groups[key].append((d, s))

    # ---- 3. Summary --------------------------------------------------------
    if verbose:
        print(f"\n{'=' * 64}")
        print("Summary")
        print(f"{'=' * 64}")
        print(f"  ConsDB visit pairs:        {len(consdb_pairs)}")
        print(f"  Butler dataset present:    {len(butler_present)}")
        print(f"  Butler dataset missing:    {len(butler_missing)}")
        if other_errors:
            print(f"  Other errors (load fail):  {len(other_errors)}")
        print()
        if butler_missing:
            n_show = min(20, len(butler_missing))
            print(f"  First {n_show} missing (would emit DatasetNotFoundError):")
            # Group by day_obs for readability
            by_day = Counter(d for d, _ in butler_missing)
            for day in sorted(by_day):
                seqs = sorted(s for d, s in butler_missing if d == day)
                shown = seqs if len(seqs) <= 8 \
                    else seqs[:5] + ['…'] + seqs[-2:]
                print(f"    day_obs={day}: {by_day[day]} missing  {shown}")
            print()

        # nollIndices distribution (across every present visit)
        if noll_groups:
            print(f"  nollIndices distribution across "
                  f"{len(butler_present)} present visits:\n")
            sorted_groups = sorted(noll_groups.items(),
                                   key=lambda kv: -len(kv[1]))
            for i, (key, pairs) in enumerate(sorted_groups):
                tag = f"flavor #{i+1}"
                if key is None:
                    print(f"  {tag} (nollIndices = None): {len(pairs)} visits")
                else:
                    print(f"  {tag} ({len(key)} terms): {len(pairs)} visits")
                    print(f"     {list(key)}")
                # Show a few example (day_obs, seq_num) pairs
                head = pairs[:5]
                tail = pairs[-3:] if len(pairs) > 8 else []
                sample = (', '.join(f"{d}/{s}" for d, s in head)
                          + ('  …  ' + ', '.join(f"{d}/{s}" for d, s in tail)
                             if tail else ''))
                print(f"     sample: {sample}")
                print()
            if len(sorted_groups) > 1:
                most = sorted_groups[0]
                lost = sum(len(p) for k, p in sorted_groups[1:])
                print(f"  WARNING: {len(sorted_groups)} distinct "
                      f"nollIndices flavors present.")
                print(f"  mktable would lock onto the FIRST visit's flavor "
                      f"(by sort order) and skip the rest.")
                print(f"  If the {len(most[1])}-visit majority flavor is "
                      f"chosen as reference, "
                      f"{lost} other-flavor visits would be skipped.\n")

    return dict(
        consdb_pairs=consdb_pairs,
        butler_present=butler_present,
        butler_missing=butler_missing,
        noll_groups=dict(noll_groups),
        other_errors=other_errors,
    )


def main():
    parser = argparse.ArgumentParser(
        description='Pre-flight survey for a pipeline chunk.')
    parser.add_argument('chunk', nargs='?', default=None,
                        help='Chunk name from runs.yaml (e.g. chunk1). '
                             'If omitted, --param-set + --day-obs-min/max '
                             '+ --programs are required.')
    parser.add_argument('--runs-yaml', default=None,
                        help='Override runs.yaml path')

    parser.add_argument('--param-set', default=None,
                        help='Direct param_set name (skips runs.yaml)')
    parser.add_argument('--butler-repo', default=None)
    parser.add_argument('--collections', nargs='+', default=None)
    parser.add_argument('--programs', nargs='+', default=None)
    parser.add_argument('--day-obs-min', type=int, default=None)
    parser.add_argument('--day-obs-max', type=int, default=None)

    parser.add_argument('--consdb-url',
                        default='https://usdf-rsp.slac.stanford.edu/consdb',
                        help='ConsDB URL (default: external USDF URL; '
                             'token from ~/.lsst/consdb_token)')
    parser.add_argument('--dataset-type',
                        default='aggregateAOSVisitTableRaw',
                        help='Butler dataset type to probe (default: '
                             'aggregateAOSVisitTableRaw)')
    parser.add_argument('--detectors', nargs='+', type=int,
                        default=[98, 99, 100],
                        help='CCD IDs to try in order for the per-visit '
                             'existence check (default: 98 99 100). The '
                             'first detector with data wins; reading just '
                             'one CCD slice is much faster than the full '
                             '188-CCD aggregate. Pass `--detectors=` to '
                             'fall back to a no-detector get (per-visit '
                             'dataset types).')

    args = parser.parse_args()

    # Resolve params either from runs.yaml chunk or from explicit flags
    from intrinsics_lib import load_param_sets
    from run_pipeline import resolve_run

    if args.chunk:
        runs = _load_runs(args.runs_yaml)
        if args.chunk not in runs.get('runs', {}):
            parser.error(f"Chunk '{args.chunk}' not in runs.yaml. "
                         f"Available: {sorted(runs.get('runs', {}).keys())}")
        cfg = runs['runs'][args.chunk]
        resolved = resolve_run(cfg, load_param_sets())
    elif args.param_set:
        ps = load_param_sets()
        if args.param_set not in ps:
            parser.error(f"Unknown param_set '{args.param_set}'. "
                         f"Available: {sorted(ps.keys())}")
        resolved = dict(ps[args.param_set])
        if args.day_obs_min is not None:
            resolved['day_obs_min'] = args.day_obs_min
        if args.day_obs_max is not None:
            resolved['day_obs_max'] = args.day_obs_max
        if args.programs is not None:
            resolved['fam_programs'] = args.programs
        if args.butler_repo is not None:
            resolved['butler_repo'] = args.butler_repo
        if args.collections is not None:
            resolved['fam_collections'] = args.collections
    else:
        if not all([args.butler_repo, args.collections, args.programs,
                    args.day_obs_min, args.day_obs_max]):
            parser.error("Need either a chunk name OR a param_set OR all of: "
                         "--butler-repo, --collections, --programs, "
                         "--day-obs-min, --day-obs-max")
        resolved = dict(
            butler_repo=args.butler_repo,
            fam_collections=args.collections,
            fam_programs=args.programs,
            day_obs_min=args.day_obs_min,
            day_obs_max=args.day_obs_max,
        )

    asyncio.run(check_chunk_data(
        butler_repo=resolved['butler_repo'],
        fam_collections=resolved['fam_collections'],
        day_obs_min=resolved['day_obs_min'],
        day_obs_max=resolved['day_obs_max'],
        fam_programs=resolved['fam_programs'],
        consdb_url=args.consdb_url,
        dataset_type=args.dataset_type,
        detectors=tuple(args.detectors) if args.detectors else (),
    ))


if __name__ == '__main__':
    main()
