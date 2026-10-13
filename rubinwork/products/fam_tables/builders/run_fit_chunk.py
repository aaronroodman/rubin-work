"""Run the Double-Zernike fit for one chunk, tolerating a chunk with no passing visits.

The external fitter, ``ts_intrinsic_wavefront/bin/run_dz_fit.py``, raises
``IndexError`` when the per-visit quality cut leaves no visits: it reads
``visit_info['nollIndices'][0]`` after the cut without testing for an empty
table (``dz_fitting.py:651``).  That never happened in the old chunked builds
because a chunk spanned many nights, so some visit always survived.  Phase 3c
splits the rebuild by night, and 8 of the 106 ``danish_1_2`` nights have no
visit that passes the cut.

Those 8 nights contribute **0 fit rows to the old tree as well** (measured
2026-10-13: ``aos/output/fam_processing/danish_1_2/fits.parquet`` has 2,528 rows
over 98 nights, against 106 nights in ``visits.parquet``), so skipping them loses
nothing.  This wrapper checks the cut first and writes a 0-row sentinel
``fits.parquet`` instead of calling the fitter, which is the form
``combine_parquets.py`` already drops on input.

Same mechanism as the ``< 5 visit_pairs`` filter one stage earlier, which is
defect A of ``notes/status/phase3_products_handoff.md``: a per-night split makes
an existing per-night cut visible as a job failure.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

#: Columns of the 0-row sentinel.  Only the join keys are needed: every other
#: column is recovered from the non-empty inputs by ``combine_parquets.py``,
#: which unions the schemas of the files it keeps.
SENTINEL_SCHEMA = pa.schema([
    pa.field('day_obs', pa.int64()),
    pa.field('seq_num', pa.int64()),
])


def n_visits_passing(visits_path: pathlib.Path) -> tuple[int, int]:
    """Count visits surviving the fitter's per-visit quality cut.

    Mirrors the default branch of `run_double_zernike_fits`: the precomputed
    ``visit_quality_pass`` flag if present, else no cut.

    Parameters
    ----------
    visits_path : `pathlib.Path`
        Path to a chunk's ``visits.parquet``.

    Returns
    -------
    n_pass : `int`
        Number of visits passing the cut (dimensionless count).
    n_total : `int`
        Number of visits in the table (dimensionless count).
    """
    names = pq.ParquetFile(str(visits_path)).schema_arrow.names
    if 'visit_quality_pass' not in names:
        n = pq.read_metadata(str(visits_path)).num_rows
        return n, n
    col = pq.read_table(str(visits_path), columns=['visit_quality_pass'])
    keep = np.asarray(col.column('visit_quality_pass').to_numpy(zero_copy_only=False),
                      dtype=bool)
    return int(keep.sum()), len(keep)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('donuts', type=pathlib.Path)
    p.add_argument('--visits', type=pathlib.Path, required=True)
    p.add_argument('--output', type=pathlib.Path, required=True)
    p.add_argument('--coord-sys', default='OCS')
    p.add_argument('--fitter', type=pathlib.Path, required=True,
                   help='path to ts_intrinsic_wavefront/bin/run_dz_fit.py')
    args = p.parse_args()

    n_pass, n_total = n_visits_passing(args.visits)
    print(f'quality cut: {n_pass}/{n_total} visits pass (dimensionless counts)')

    if n_pass == 0:
        print(f'no visit passes the cut -- writing a 0-row sentinel to {args.output} '
              f'instead of running the fit (see this module\'s docstring)')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(SENTINEL_SCHEMA.empty_table(), str(args.output))
        return 0

    cmd = [sys.executable, str(args.fitter), str(args.donuts),
           '--coord-sys', args.coord_sys,
           '--visits', str(args.visits), '--output', str(args.output)]
    print('running: ' + ' '.join(cmd), flush=True)
    return subprocess.call(cmd)


if __name__ == '__main__':
    sys.exit(main())
