"""Add ``intra_visit`` and ``extra_visit`` to an existing blitz ``visits.parquet``.

`run_blitz_mktable` writes both columns as of 2026-10-04, but the Danish 1.3 blitz tables
at ``output/fam_processing/danish_1_3_test/`` were built before that and carry only
``visit`` -- the **extra-focal** exposure of each Full Array Mode (FAM) triplet. Without
the intra-focal identifier there is no key to join per-side telemetry on, and rebuilding
the 13.3 GB ``donuts.parquet`` to recover two integers per visit is not worth it.

This script reads only the ``.meta`` of each visit's blitz table, takes
``intra_visit_id`` / ``extra_visit_id``, and rewrites ``visits.parquet`` in place with the
two columns appended. ``donuts.parquet`` is untouched.

The intra-focal identifier is **not** inferable from ``visit``: on ``day_obs`` 20260315
``seq_num`` 73 the pair is intra 2026031500072 / extra 2026031500073, while the visits in
the table are spaced three sequence numbers apart, so neither ``visit - 1`` nor
``visit - 2`` is right as a rule. Hence the per-visit metadata read.

Run from the repository root::

    python aos/code/fam_processing/backfill_visit_sides.py \
        --param-set danish_1_3_test

Notes
-----
One Butler read per visit, about 4 s each, so roughly an hour for 966 visits. The read
pulls the whole donut table because that is the dataset the metadata rides on; only
``.meta`` is used.
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml
from astropy.table import QTable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))   # repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))       # blitz_reader

import blitz_reader as br                                             # noqa: E402


def parse_args(argv=None):
    """Command-line interface.

    Returns
    -------
    args : `argparse.Namespace`
        Parsed arguments.
    """
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--param-set', required=True,
                   help='Entry in aos/param_sets.yaml giving butler_repo, '
                        'fam_collections and dir_name')
    p.add_argument('--dataset-type', default=br.BLITZ_FAM_DATASET,
                   help=f'Blitz dataset type (default {br.BLITZ_FAM_DATASET})')
    p.add_argument('--dry-run', action='store_true',
                   help='read and report, but do not rewrite visits.parquet')
    return p.parse_args(argv)


def main(argv=None):
    """Command-line entry point. See the module docstring."""
    args = parse_args(argv)
    topic = pathlib.Path(__file__).resolve().parents[2]                # aos/
    entry = yaml.safe_load(open(topic / 'param_sets.yaml'))[args.param_set]
    visits_file = (topic / 'output' / 'fam_processing' / entry['dir_name']
                   / 'visits.parquet')

    table = QTable.read(str(visits_file))
    if 'intra_visit' in table.colnames and 'extra_visit' in table.colnames:
        print(f'{visits_file.name} already carries intra_visit and extra_visit; '
              'nothing to do')
        return 0
    print(f'{visits_file}: {len(table)} rows, {len(table.colnames)} columns')

    from lsst.daf.butler import Butler
    butler = Butler(entry['butler_repo'])
    collections = entry['fam_collections']

    visit_ids = np.asarray(table['visit'], dtype=np.int64)
    intra = np.zeros(len(visit_ids), dtype=np.int64)
    extra = np.zeros(len(visit_ids), dtype=np.int64)
    n_failed = 0
    for i, visit in enumerate(visit_ids, start=0):
        try:
            meta = dict(br.read_blitz_visit(butler, int(visit), collections,
                                            dataset_type=args.dataset_type).meta)
            intra[i] = int(meta['intra_visit_id'])
            extra[i] = int(meta['extra_visit_id'])
        except Exception as exc:
            # A visit whose dataset has gone from the collection must not lose the other
            # 965 rows; -1 marks it so the gap is visible rather than silently plausible.
            print(f'  visit {visit}: FAILED -- {type(exc).__name__}: {exc}')
            intra[i] = -1
            extra[i] = -1
            n_failed += 1
        if (i + 1) % 50 == 0:
            print(f'  [{i + 1}/{len(visit_ids)}]')

    ok = extra > 0
    # extra_visit_id is the key the blitz dataset is registered under, so it must come
    # back equal to the `visit` column; a mismatch means the metadata is not this visit's.
    n_mismatch = int((extra[ok] != visit_ids[ok]).sum())
    if n_mismatch:
        raise RuntimeError(f'{n_mismatch} visits have extra_visit_id != visit; '
                           'the metadata does not match the row it was read for')
    n_same = int((intra[ok] == extra[ok]).sum())
    if n_same:
        raise RuntimeError(f'{n_same} visits have intra_visit_id == extra_visit_id')

    offsets, counts = np.unique(visit_ids[ok] - intra[ok], return_counts=True)
    print(f'\nvisit - intra_visit offsets (dimensionless): '
          + ', '.join(f'{o}: {c} visits' for o, c in zip(offsets, counts)))
    print(f'{int(ok.sum())} of {len(visit_ids)} visits resolved, {n_failed} failed')

    if args.dry_run:
        print('--dry-run: visits.parquet not rewritten')
        return 0

    table['intra_visit'] = intra
    table['extra_visit'] = extra
    table.write(str(visits_file), format='parquet', overwrite=True)
    print(f'\nWrote {len(table)} rows, {len(table.colnames)} columns to {visits_file}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
