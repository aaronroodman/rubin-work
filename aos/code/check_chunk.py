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


def _strip_ecsv_comment(line):
    """Strip the ECSV comment prefix ('# ' or '#') from one line.
    Returns None if the line is a data row (no '#' prefix)."""
    if line.startswith('# '):
        return line[2:]
    if line.startswith('#'):
        return line[1:]
    return None


def _extract_noll_indices_from_text(text):
    """Pull `nollIndices` out of an ECSV partial-read string.

    The full ECSV header is megabytes (per-donut diagnostic arrays are
    serialized into the meta), but `nollIndices` lives near the top
    (~byte 2,400). We grep for its line, capture only its sub-tree,
    rebuild a small YAML doc, and let astropy's YAML loader handle
    the `!numpy.ndarray` block.

    Returns a tuple of ints, or None if extraction failed.
    """
    import re
    start_re = re.compile(r'^#\s*-\s*nollIndices:', re.M)
    m = start_re.search(text)
    if not m:
        return None

    lines = text[m.start():].splitlines()
    first_body = _strip_ecsv_comment(lines[0])
    if first_body is None:
        return None

    m_dash = re.match(r'(\s*)-', first_body)
    base_indent = len(m_dash.group(1)) if m_dash else 0

    block = [first_body]
    for line in lines[1:]:
        body = _strip_ecsv_comment(line)
        if body is None:
            break  # data row reached
        # Stop on next sibling list item or a dedent to the parent level
        if (body.startswith(' ' * base_indent + '- ')
                or (body.strip() and not body.startswith(' ' * (base_indent + 2)))):
            break
        block.append(body)

    first = block[0].lstrip()
    if first.startswith('- '):
        first = first[2:]
    wrapped = '\n'.join([first] + ['  ' + b for b in block[1:]])

    try:
        from astropy.io.misc.yaml import load as astropy_yaml_load
        parsed = astropy_yaml_load(wrapped)
    except Exception:
        return None

    if isinstance(parsed, dict) and 'nollIndices' in parsed:
        try:
            return tuple(int(n) for n in parsed['nollIndices'])
        except Exception:
            return None
    return None


def _fast_noll_indices(butler, dataset_type, day_obs, seq_num,
                      partial_bytes=32_768):
    """Cheaply read `nollIndices` for one visit by partial-reading the
    ECSV header at its S3 URI. Returns (status, value) where:
       status: 'present', 'missing', 'parse-fail', 'fallback-ok',
               'fallback-fail'
       value : tuple of ints (when status starts with 'present' or
               'fallback-ok'), or an error message string otherwise.
    """
    from lsst.daf.butler import DatasetNotFoundError
    try:
        ref = butler.find_dataset(
            dataset_type,
            {'instrument': 'LSSTCam', 'day_obs': day_obs,
             'seq_num': seq_num})
    except Exception as e:
        return 'parse-fail', f'find_dataset: {type(e).__name__}: {e}'
    if ref is None:
        return 'missing', None

    # Try partial read of the ECSV header
    try:
        uri = butler.getURI(ref)
        raw = uri.read(size=partial_bytes).decode('utf-8', errors='replace')
        noll = _extract_noll_indices_from_text(raw)
        if noll is not None:
            return 'present', noll
    except Exception as e:
        partial_err = f'partial-read: {type(e).__name__}: {e}'
    else:
        partial_err = None

    # Fall back: full butler.get and read the meta dict
    try:
        tbl = butler.get(ref)
    except DatasetNotFoundError:
        return 'missing', None
    except Exception as e:
        return ('fallback-fail',
                f'{partial_err or "partial: no nollIndices"}; '
                f'fallback: {type(e).__name__}: {e}')
    noll = tbl.meta.get('nollIndices', None) if hasattr(tbl, 'meta') \
        else None
    if noll is None:
        return 'present', None
    try:
        return 'fallback-ok', tuple(int(n) for n in noll)
    except Exception as e:
        return 'fallback-fail', f'cast nollIndices: {e}'


async def check_chunk_data(butler_repo, fam_collections, day_obs_min,
                           day_obs_max, fam_programs,
                           consdb_url=None,
                           dataset_type='aggregateAOSVisitTableRaw',
                           partial_bytes=32_768,
                           verbose=True):
    """Run the pre-flight survey for a single chunk.

    Strategy:
      1. ConsDB → list of expected (day_obs, seq_num) visit pairs.
      2. For each pair, find the dataset's S3 URI and partial-read
         ~32 KB of the ECSV header. Extract `meta['nollIndices']`
         from the small `!numpy.ndarray` block near the top of the
         file. Falls back to a full butler.get on parse failure.
      3. Visits with no dataset are reported as missing; visits
         where the partial read failed but the full load succeeded
         are flagged separately.

    Returns a dict with:
        'consdb_pairs'   : list of (day_obs, seq_num) from ConsDB
        'butler_present' : list with Butler data
        'butler_missing' : list in ConsDB but not in Butler
        'noll_groups'    : dict tuple(nollIndices) -> list of (d,s)
        'other_errors'   : visits where reading failed entirely
        'fallback_used'  : visits where partial-read failed but
                           the full load succeeded
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

    # ---- 2. Per-visit fast nollIndices read ------------------------------
    # ECSV files are ~18 MB each (per-donut diagnostic arrays embedded in
    # the meta), but `nollIndices` lives near byte 2,400. A 32 KB partial
    # read pulls it without loading the rest. _fast_noll_indices falls
    # back to a full butler.get if the partial-read parse fails.
    butler = Butler(butler_repo, instrument='LSSTCam',
                    collections=fam_collections)

    butler_present = []
    butler_missing = []
    noll_groups = defaultdict(list)
    other_errors = []
    fallback_used = []

    if verbose:
        print(f"\nProbing Butler for `{dataset_type}` on "
              f"{len(consdb_pairs)} visits "
              f"(partial-read {partial_bytes // 1024} KB per visit)…")

    iterator = tqdm(consdb_pairs) if verbose else consdb_pairs
    for d, s in iterator:
        status, value = _fast_noll_indices(
            butler, dataset_type, d, s, partial_bytes=partial_bytes)
        if status == 'missing':
            butler_missing.append((d, s))
        elif status in ('present', 'fallback-ok'):
            butler_present.append((d, s))
            if status == 'fallback-ok':
                fallback_used.append((d, s))
            if value is None:
                noll_groups[None].append((d, s))
            else:
                noll_groups[value].append((d, s))
        else:  # parse-fail or fallback-fail
            other_errors.append((d, s, status, str(value)))

    # ---- 3. Summary --------------------------------------------------------
    if verbose:
        print(f"\n{'=' * 64}")
        print("Summary")
        print(f"{'=' * 64}")
        print(f"  ConsDB visit pairs:        {len(consdb_pairs)}")
        print(f"  Butler dataset present:    {len(butler_present)}")
        print(f"  Butler dataset missing:    {len(butler_missing)}")
        if fallback_used:
            print(f"  Used full-load fallback:   {len(fallback_used)}")
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
        fallback_used=fallback_used,
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
    parser.add_argument('--partial-bytes', type=int, default=32_768,
                        help='Bytes to fetch per visit when probing the '
                             'ECSV header (default 32768 = 32 KB; the '
                             'nollIndices block is at ~byte 2,400).')

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
        partial_bytes=args.partial_bytes,
    ))


if __name__ == '__main__':
    main()
