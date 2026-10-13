#!/usr/bin/env python3
"""Check that ``/repo/embargo`` is readable from a batch node, for the phase 3c chunk.

One ``danish_1_2`` chunk, ``day_obs`` 20260713, lives only in ``/repo/embargo``: the 3c
pre-flight found 24 ``aggregateAOSVisitTableRaw`` datasets there and 0 in ``/repo/main``
(`notes/status/phase3_products_handoff.md`). The pre-flight ran interactively, where
embargo credentials are present; a Slurm job runs with a different environment, so
whether the repository resolves there has to be checked before the rebuild depends on it.

Reads only: it opens the Butler, queries the dataset count, and writes nothing.

Run it on a batch node (via ``sbatch --wrap``) or interactively for comparison::

    python -m rubinwork.products.fam_tables.refbuild.check_embargo_from_batch
"""

import argparse
import socket
import sys

CHUNK_DAY_OBS = 20260713
"""The one ``danish_1_2`` night that exists only in ``/repo/embargo``."""

COLLECTION = ('LSSTCam/runs/aos/fam/danish_1_2_0/wep_17_7_0/dv_4_7_0/bin_x2/'
              'paired/refitWcs')

EXPECTED_DATASETS = 24
"""``aggregateAOSVisitTableRaw`` datasets the 3c pre-flight found for this night."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='/repo/embargo',
                        help='Butler repository to check (default: /repo/embargo)')
    parser.add_argument('--day-obs', type=int, default=CHUNK_DAY_OBS,
                        help=f'Night to count datasets for (default: {CHUNK_DAY_OBS})')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print(f'host        {socket.gethostname()}')
    print(f'repo        {args.repo}')
    print(f'collection  {COLLECTION}')
    print(f'day_obs     {args.day_obs} (dimensionless date key)')
    print()

    from lsst.daf.butler import Butler

    try:
        butler = Butler(args.repo, instrument='LSSTCam', collections=[COLLECTION])
    except Exception as exc:
        print(f'FAIL: cannot open the Butler at {args.repo}: {type(exc).__name__}: {exc}')
        return 1
    print(f'Butler opened OK at {args.repo}')

    try:
        refs = list(butler.registry.queryDatasets(
            'aggregateAOSVisitTableRaw',
            collections=[COLLECTION],
            where=f'exposure.day_obs = {args.day_obs}',
        ))
    except Exception as exc:
        print(f'FAIL: query raised {type(exc).__name__}: {exc}')
        return 1

    n = len(refs)
    print(f'aggregateAOSVisitTableRaw datasets: {n} (dimensionless count), '
          f'pre-flight found {EXPECTED_DATASETS}')
    if n == 0:
        print('FAIL: the repository resolved but holds no datasets for this night')
        return 1

    # Reading one dataset's metadata is the part a credentialed-but-unreadable repository
    # fails on, so the check is not complete without it.  The file is large, so take the
    # URI rather than the data.
    try:
        uri = butler.getURI(refs[0])
        print(f'first dataset URI resolves: {uri}')
    except Exception as exc:
        print(f'FAIL: dataset listed but its URI does not resolve: '
              f'{type(exc).__name__}: {exc}')
        return 1

    print()
    print(f'PASS: /repo/embargo is readable from {socket.gethostname()} '
          f'for day_obs {args.day_obs}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
