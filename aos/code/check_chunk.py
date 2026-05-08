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
                           noll_sample_n=20,
                           verbose=True):
    """Run the pre-flight survey for a single chunk.

    Strategy:
      1. ConsDB → list of expected (day_obs, seq_num) visit pairs.
      2. Butler.registry.queryDatasets — single round-trip query that
         tells us which (day_obs, seq_num) pairs have the dataset
         present, without actually loading any data. If `detectors` is
         given, restrict to those CCD IDs (lots faster on per-detector
         dataset types).
      3. Load up to `noll_sample_n` of the present datasets just to
         peek at meta['nollIndices'] and report the flavor distribution.

    Returns a dict with:
        'consdb_pairs'   : list of (day_obs, seq_num) from ConsDB
        'butler_present' : list with Butler data
        'butler_missing' : list in ConsDB but not in Butler
        'noll_groups'    : dict tuple(nollIndices) -> list of (d,s) sampled
        'other_errors'   : load failures during the nollIndices peek
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
            print(f"  detector subset: {list(detectors)}")
        print(f"  noll sample N:   {noll_sample_n}")
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

    # ---- 2. Butler — single registry query for existence ------------------
    butler = Butler(butler_repo, instrument='LSSTCam',
                    collections=fam_collections)

    where_parts = [
        "instrument='LSSTCam'",
        f"exposure.day_obs >= {day_obs_min} AND exposure.day_obs <= {day_obs_max}",
    ]
    if detectors:
        det_list = ','.join(str(d) for d in detectors)
        where_parts.append(f"detector IN ({det_list})")
    where = ' AND '.join(where_parts)

    if verbose:
        print(f"\nQuerying Butler registry for `{dataset_type}` "
              f"existence (single round-trip)…")

    refs_by_visit = defaultdict(list)
    other_errors = []
    try:
        ref_iter = butler.registry.queryDatasets(
            dataset_type, collections=fam_collections, where=where)
    except Exception as e:
        # `detector IN (…)` may be invalid if the dataset is per-visit
        # (no detector dim). Retry without the detector filter.
        if detectors:
            if verbose:
                print(f"  (detector filter not accepted: {e}; "
                      f"retrying without it)")
            where_no_det = ' AND '.join(
                p for p in where_parts if not p.startswith('detector'))
            ref_iter = butler.registry.queryDatasets(
                dataset_type, collections=fam_collections, where=where_no_det)
        else:
            raise

    for ref in ref_iter:
        d = ref.dataId.get('day_obs')
        s = ref.dataId.get('seq_num')
        if d is None or s is None:
            continue
        refs_by_visit[(int(d), int(s))].append(ref)

    butler_visits = set(refs_by_visit)
    consdb_set = set(consdb_pairs)
    butler_present = sorted(consdb_set & butler_visits)
    butler_missing = sorted(consdb_set - butler_visits)

    if verbose:
        n_refs = sum(len(v) for v in refs_by_visit.values())
        print(f"  Registry returned {n_refs} refs across "
              f"{len(refs_by_visit)} unique visits")

    # ---- 3. nollIndices peek on a sample subset ---------------------------
    noll_groups = defaultdict(list)
    if noll_sample_n and butler_present:
        sample = butler_present[:noll_sample_n]
        if verbose:
            print(f"\nLoading meta['nollIndices'] from "
                  f"{len(sample)} sample visits…")
        iterator = tqdm(sample) if verbose else sample
        for vk in iterator:
            ref = refs_by_visit[vk][0]    # any ref for this visit
            try:
                tbl = butler.get(ref)
            except Exception as e:
                other_errors.append((vk[0], vk[1],
                                     type(e).__name__, str(e)))
                continue
            noll = tbl.meta.get('nollIndices', None) \
                if hasattr(tbl, 'meta') else None
            if noll is None:
                noll_groups[None].append(vk)
            else:
                key = tuple(int(n) for n in noll)
                noll_groups[key].append(vk)

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

        # nollIndices distribution (computed on the sample, not the full set)
        n_sampled = sum(len(v) for v in noll_groups.values())
        if noll_groups:
            print(f"  nollIndices distribution among "
                  f"{n_sampled} sampled visits "
                  f"(of {len(butler_present)} present):\n")
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
                        help='CCD ID subset to query for existence '
                             '(default: 98 99 100). Pass --no-detectors '
                             '(an empty list, e.g. --detectors=) to query '
                             'every detector / use a per-visit dataset.')
    parser.add_argument('--noll-sample-n', type=int, default=20,
                        help='How many present visits to actually load '
                             'for the nollIndices peek (default 20). '
                             'Set to 0 to skip.')

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
        noll_sample_n=args.noll_sample_n,
    ))


if __name__ == '__main__':
    main()
