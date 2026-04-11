#!/usr/bin/env python3
"""Build intrinsic Zernike tables from Rubin AOS FAM observations.

Usage:
    python run_mktable.py --param-set 4
    python run_mktable.py --param-set 4 --no-thermal
    python run_mktable.py --butler-repo /repo/embargo \
        --collections aos_fam_danish_triplets \
        --day-obs-min 20260315 --day-obs-max 20260317 \
        --programs T278 T381 T492 T539 T614
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from intrinsics_lib import run_mktable, PARAM_SETS


def main():
    parser = argparse.ArgumentParser(
        description='Build intrinsic Zernike tables from FAM observations.')

    # Option A: use a named parameter set
    parser.add_argument('--param-set', type=int, default=None,
                        help=f'Use a predefined parameter set '
                             f'({", ".join(str(k) for k in sorted(PARAM_SETS))})')

    # Option B: specify everything directly
    parser.add_argument('--butler-repo', default=None,
                        help='Butler repository path (e.g. /repo/embargo)')
    parser.add_argument('--collections', nargs='+', default=None,
                        help='Butler collection(s)')
    parser.add_argument('--day-obs-min', type=int, default=None,
                        help='Minimum day_obs (e.g. 20260315)')
    parser.add_argument('--day-obs-max', type=int, default=None,
                        help='Maximum day_obs (e.g. 20260317)')
    parser.add_argument('--programs', nargs='+', default=None,
                        help='Science program names (e.g. T278 T381)')

    # Optional overrides
    parser.add_argument('--prefix', default=None,
                        help='Output filename prefix (default: from param set)')
    parser.add_argument('--coord-sys', default='OCS', choices=['OCS', 'CCS'],
                        help='Coordinate system (default: OCS)')
    parser.add_argument('--output-dir', default='output',
                        help='Output directory (default: output)')
    parser.add_argument('--rotator-threshold', type=float, default=90.0,
                        help='Rotator angle flagging threshold in degrees')
    parser.add_argument('--fp-radius', type=float, default=1.8,
                        help='Focal plane radius in degrees')
    parser.add_argument('--fp-nsteps', type=int, default=73,
                        help='Grid points per axis for intrinsic model')
    parser.add_argument('--min-visits-per-day', type=int, default=5,
                        help='Minimum visit pairs per day_obs to include')
    parser.add_argument('--no-thermal', action='store_true',
                        help='Skip thermal data retrieval')
    parser.add_argument('--temp-time-window', type=float, default=0.2,
                        help='EFD temperature query post-padding (seconds)')
    parser.add_argument('--consdb-url', default=None,
                        help='ConsDB URL (default: Kubernetes-internal URL; '
                             'use http://usdf-rsp.slac.stanford.edu/consdb '
                             'from outside RSP pods)')

    args = parser.parse_args()

    # Resolve parameters
    if args.param_set is not None:
        if args.param_set not in PARAM_SETS:
            parser.error(f"Unknown param-set: {args.param_set}. "
                         f"Available: {sorted(PARAM_SETS.keys())}")
        params = dict(PARAM_SETS[args.param_set])
        # Allow CLI overrides
        if args.butler_repo is not None:
            params['butler_repo'] = args.butler_repo
        if args.collections is not None:
            params['fam_collections'] = args.collections
        if args.day_obs_min is not None:
            params['day_obs_min'] = args.day_obs_min
        if args.day_obs_max is not None:
            params['day_obs_max'] = args.day_obs_max
        if args.programs is not None:
            params['fam_programs'] = args.programs
        if args.prefix is not None:
            params['prefix'] = args.prefix
    else:
        if not all([args.butler_repo, args.collections,
                    args.day_obs_min, args.day_obs_max, args.programs]):
            parser.error("Must specify --param-set OR all of: "
                         "--butler-repo, --collections, --day-obs-min, "
                         "--day-obs-max, --programs")
        params = dict(
            butler_repo=args.butler_repo,
            fam_collections=args.collections,
            day_obs_min=args.day_obs_min,
            day_obs_max=args.day_obs_max,
            fam_programs=args.programs,
            prefix=args.prefix or 'fam_danish',
        )

    asyncio.run(run_mktable(
        butler_repo=params['butler_repo'],
        fam_collections=params['fam_collections'],
        day_obs_min=params['day_obs_min'],
        day_obs_max=params['day_obs_max'],
        fam_programs=params['fam_programs'],
        prefix=params['prefix'],
        coord_sys=args.coord_sys,
        output_dir=args.output_dir,
        rotator_threshold=args.rotator_threshold,
        fp_radius=args.fp_radius,
        fp_nsteps=args.fp_nsteps,
        min_visits_per_day=args.min_visits_per_day,
        include_thermal=not args.no_thermal,
        temp_time_window_sec=args.temp_time_window,
        consdb_url=args.consdb_url,
    ))


if __name__ == '__main__':
    main()
