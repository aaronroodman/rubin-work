"""Stage one night of a ``fam_tables`` build as the ``cwfs_tables`` reference input.

Phase 3a of ``notes/status/organization_plan.md`` checks the ``cwfs_tables`` move by
building a small reference slice with the unmoved code and reproducing it with the moved
code at zero tolerance.

``run_wfs_mktable.py`` has no ``--day-obs`` filter: it walks every FAM visit in the
``--tables-dir`` ``visits.parquet``. So a one-night reference needs a tables directory
holding only that night. This writes one, carrying all three tables because the
validation plot reads ``fits.parquet`` (the FAM k=1 overlay) and ``donuts.parquet``
(the FAM median in the corner-WFS radial shell).

Pick a night that has ``fits.parquet`` rows. On the phase-2 reference chunk, 20251116 has
6 FAM visits but **0** fits rows -- the DZ fit's ``median_blur_arcsec <= 1.2 arcsec``
quality cut drops all of them -- which makes the validation plot's overlay all-NaN and the
reference degenerate. 20251126 is the smallest night that does not: 6 FAM visits,
5 fits rows, 17,019 FAM donut rows.

Usage::

    python -m rubinwork.products.cwfs_tables.refbuild.stage_fam_night \\
        --reference /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/danish_1_2 \\
        --day-obs 20251126 \\
        --out .../\\_phase2_refbuild/cwfs_tables/_fam_20251126
"""

import argparse
import pathlib
import sys

import pandas as pd
import pyarrow.parquet as pq

__all__ = ["stage", "main"]

DONUT_BATCH_ROWS = 300_000
"""Row-batch size for streaming the FAM donut table, which is GB-scale for a full build."""


def stage(reference, day_obs, out):
    """Write the one-night slice of a FAM build's three tables.

    Parameters
    ----------
    reference : `pathlib.Path`
        A ``fam_tables`` build directory holding ``donuts``, ``visits`` and
        ``fits`` ``.parquet``.
    day_obs : `int`
        Observation night to keep, as ``YYYYMMDD``.
    out : `pathlib.Path`
        Directory to write the slice into. Created if absent.

    Returns
    -------
    counts : `dict` [`str`, `int`]
        Rows written per table name (count, dimensionless).
    """
    reference = pathlib.Path(reference)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}

    for name in ("visits", "fits"):
        df = pd.read_parquet(reference / f"{name}.parquet")
        sub = df[df.day_obs == day_obs]
        sub.to_parquet(out / f"{name}.parquet")
        counts[name] = len(sub)
        print(f"{name}.parquet: {len(sub)} rows of {len(df)}, "
              f"{len(sub.columns)} columns")

    # Streamed by row batch: a full build's donuts.parquet is 12.4 GB for danish_1_2.
    pf = pq.ParquetFile(str(reference / "donuts.parquet"))
    parts = [bd[bd.day_obs == day_obs]
             for batch in pf.iter_batches(batch_size=DONUT_BATCH_ROWS)
             for bd in (batch.to_pandas(),)]
    donuts = pd.concat(parts, ignore_index=True)
    donuts.to_parquet(out / "donuts.parquet")
    counts["donuts"] = len(donuts)
    print(f"donuts.parquet: {len(donuts)} rows, {len(donuts.columns)} columns")

    if counts["fits"] == 0:
        print(f"WARNING: no fits.parquet rows for day_obs={day_obs}; the validation "
              "plot's FAM k=1 overlay will be all-NaN. Pick another night.")
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", required=True,
                    help="fam_tables build directory to slice")
    ap.add_argument("--day-obs", required=True, type=int,
                    help="night to keep, YYYYMMDD")
    ap.add_argument("--out", required=True, help="directory to write the slice into")
    args = ap.parse_args(argv)
    stage(args.reference, args.day_obs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
