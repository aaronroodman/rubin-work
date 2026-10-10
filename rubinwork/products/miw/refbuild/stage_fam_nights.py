"""Stage a few nights of a ``fam_tables`` build as the ``miw`` reference input.

Phase 3a of ``notes/status/organization_plan.md`` checks the ``miw`` move by building a
small reference with the unmoved code and reproducing it with the moved code at zero
tolerance.

Why a slice is needed. ``build_intrinsic`` selects its own visits (band, program,
elevation window, rotator bin) and so is already small, but ``intrinsic_sidecar`` and
``refit_mi`` read the **whole** ``donuts.parquet`` of the FAM build -- 9,083,526 rows and
12.36 GB for ``danish_1_2`` -- which would dominate the reference. Restricting the FAM
tables to the nights the chosen rotator bins actually draw from keeps the whole chain
small without changing what the build sees in those bins.

The nights must cover every visit of the chosen rotator bins, or the reference's build
differs from a full-tree build of the same bins. For ``danish_1_2`` with the default cuts
(band ``i``, ``BLOCK-T614_triplets``, elevation 65 to 75 deg, ``day_obs <= 20260513``),
bins ``[-3, 3]`` and ``[55, 65]`` draw entirely from 20260315, 20260316 and 20260317: 37
and 36 visits respectively (count, dimensionless).

``visits.parquet`` carries ``alt`` in **radians**, while ``mi_config.yaml`` states the
elevation window in degrees; `bin_visit_counts` converts, so a bin count here matches what
the build selects.

The donut slice is copied row group by row group, because
``run_build_intrinsic.load_kept_donuts`` selects donuts by row group from the ``day_obs``
and ``seq_num`` column statistics -- a FAM ``donuts.parquet`` holds one row group per
visit, and a slice rewritten as one large group matches nothing.

Usage::

    python -m rubinwork.products.miw.refbuild.stage_fam_nights \\
        --reference aos/output/fam_processing/danish_1_2 \\
        --day-obs 20260315 20260316 20260317 \\
        --out .../_phase2_refbuild/miw/_fam_20260315_20260317
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

__all__ = ["bin_visit_counts", "stage", "main"]


def bin_visit_counts(visits, rotator_bins, band="i",
                     science_program="BLOCK-T614_triplets",
                     alt_min_deg=65.0, alt_max_deg=75.0, day_obs_max=None):
    """Visits per rotator bin under the measured-intrinsic build's own cuts.

    Parameters
    ----------
    visits : `pandas.DataFrame`
        A ``fam_tables`` ``visits.parquet``, with ``alt`` in radians.
    rotator_bins : `list` [`tuple` [`float`, `float`]]
        Camera rotator angle windows in degrees, as ``mi_config.yaml`` gives them.
    band : `str`, optional
        Band whitelist, a single band.
    science_program : `str`, optional
        ``science_program`` whitelist, a single program.
    alt_min_deg, alt_max_deg : `float`, optional
        Elevation window in degrees.
    day_obs_max : `int` or `None`, optional
        Latest night to include, as ``YYYYMMDD``. `None` keeps all.

    Returns
    -------
    counts : `dict` [`tuple`, `dict`]
        Per bin: ``n_visits`` (count, dimensionless) and ``nights`` (`list` of `int`).
    """
    sel = visits[(visits.band == band)
                 & (visits.science_program == science_program)
                 & (np.degrees(visits.alt) >= alt_min_deg)
                 & (np.degrees(visits.alt) <= alt_max_deg)]
    if day_obs_max is not None:
        sel = sel[sel.day_obs <= day_obs_max]
    out = {}
    for lo, hi in rotator_bins:
        b = sel[(sel.rotator_angle >= lo) & (sel.rotator_angle <= hi)]
        out[(lo, hi)] = dict(n_visits=len(b),
                             nights=sorted(int(d) for d in b.day_obs.unique()))
    return out


def stage(reference, day_obs, out):
    """Write the multi-night slice of a FAM build's three tables.

    Parameters
    ----------
    reference : `str` or `pathlib.Path`
        A ``fam_tables`` build directory holding ``donuts``, ``visits`` and
        ``fits`` ``.parquet``.
    day_obs : `iterable` [`int`]
        Observation nights to keep, as ``YYYYMMDD``.
    out : `str` or `pathlib.Path`
        Directory to write the slice into. Created if absent.

    Returns
    -------
    counts : `dict` [`str`, `int`]
        Rows written per table name (count, dimensionless).
    """
    reference = pathlib.Path(reference)
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    nights = sorted(int(d) for d in day_obs)
    counts = {}

    for name in ("visits", "fits"):
        df = pd.read_parquet(reference / f"{name}.parquet")
        sub = df[df.day_obs.isin(nights)]
        sub.to_parquet(out / f"{name}.parquet")
        counts[name] = len(sub)
        print(f"{name}.parquet: {len(sub)} rows of {len(df)}, "
              f"{len(sub.columns)} columns")

    # Copied ROW GROUP BY ROW GROUP, not streamed by row batch and rewritten.
    # run_build_intrinsic.load_kept_donuts selects donuts by row group, keying each group
    # on the `day_obs`/`seq_num` column statistics, so a FAM donuts.parquet holds exactly
    # one row group per visit. Rewriting the slice as a single large group makes every
    # lookup miss and the build dies on "No donut row groups matched the kept visits".
    src = pq.ParquetFile(str(reference / "donuts.parquet"))
    rows = 0
    with pq.ParquetWriter(str(out / "donuts.parquet"), src.schema_arrow) as writer:
        for i in range(src.num_row_groups):
            meta = src.metadata.row_group(i)
            day = next((meta.column(c).statistics.min
                        for c in range(meta.num_columns)
                        if meta.column(c).path_in_schema == "day_obs"
                        and meta.column(c).statistics is not None), None)
            if day is None or int(day) not in nights:
                continue
            table = src.read_row_group(i)
            writer.write_table(table, row_group_size=table.num_rows)
            rows += table.num_rows
    counts["donuts"] = rows
    kept_groups = pq.ParquetFile(str(out / "donuts.parquet")).num_row_groups
    print(f"donuts.parquet: {rows} rows in {kept_groups} row groups "
          f"of {src.num_row_groups}, {len(src.schema_arrow)} columns")

    if counts["fits"] == 0:
        print(f"WARNING: no fits.parquet rows for day_obs in {nights}; the build's "
              "quality cut has nothing to select. Pick other nights.")
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference", required=True,
                    help="fam_tables build directory to slice")
    ap.add_argument("--day-obs", required=True, type=int, nargs="+",
                    help="nights to keep, YYYYMMDD")
    ap.add_argument("--out", required=True, help="directory to write the slice into")
    args = ap.parse_args(argv)
    stage(args.reference, args.day_obs, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
