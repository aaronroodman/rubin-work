"""Stage a reference build's ``mktable`` output so the rest of the build can rerun.

Phase 2 of ``notes/status/organization_plan.md`` checks a code change by rerunning
everything **downstream** of ``mktable`` against a reference build and comparing at
zero tolerance. ``mktable`` itself is deliberately not rerun: it is in the external
``ts_intrinsic_wavefront`` package, its EFD thermal loop is most of the reference's
wall-clock, and it is where the known ``rotator_angle`` non-reproducibility lives.

So the per-chunk ``donuts.parquet`` and ``visits.parquet`` are copied out of the
reference into a fresh build's ``_work/chunks/``, and Snakemake takes it from there.

The subtlety this script exists for: the reference's chunk ``visits.parquet`` was
**merged in place** by its own ``attach_telemetry``, so it is not the file the
reference's ``fit`` saw. The pre-merge file is reconstructed by dropping the sidecar
columns the merge added, while **keeping** the 13 that ``mktable`` itself writes --
identified as the sidecar columns that also appear in the pre-merge
``fits.parquet`` (``cam_air_temp``, ``cam_m1m3_delta_t``, ``dome_delta_t``,
``m1m3_air_temp``, ``m2_air_temp``, ``m2_delta_t``, ``outside_temp`` in deg C,
``x_gradient``, ``y_gradient``, ``z_gradient``, ``radial_gradient`` in deg C/m, and
``tma_truss_temp_mxmy``, ``tma_truss_temp_pxpy`` in deg C) plus the join keys.

Usage::

    python -m rubinwork.products.fam_tables.refbuild.stage_reference_chunk \\
        --reference /sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/danish_1_2 \\
        --chunk 20251116_20251130 \\
        --build-dir .../(_scratch)/products/fam_tables/danish_1_2/<build>
"""

import argparse
import pathlib
import shutil
import sys

import pyarrow.parquet as pq

__all__ = ["premerge_visit_columns", "stage", "main"]


def premerge_visit_columns(chunk_dir):
    """Column names of a chunk's ``visits.parquet`` as ``mktable`` wrote it.

    Parameters
    ----------
    chunk_dir : `pathlib.Path`
        A reference build's ``chunks/<dmin>_<dmax>/`` directory, holding
        ``visits.parquet``, ``fits.parquet`` and the ``telemetry.parquet`` sidecar.

    Returns
    -------
    columns : `list` [`str`]
        The visits columns to keep, in their stored order. Every sidecar column
        is dropped except the ones that also appear in ``fits.parquet``, which
        ``mktable`` wrote and the merge only overwrote with equal values.

    Notes
    -----
    With no ``telemetry.parquet`` beside it, every column is kept: nothing was
    merged in.
    """
    chunk_dir = pathlib.Path(chunk_dir)
    visits = pq.ParquetFile(chunk_dir / "visits.parquet").schema_arrow.names
    sidecar = chunk_dir / "telemetry.parquet"
    if not sidecar.is_file():
        return list(visits)
    added = set(pq.ParquetFile(sidecar).schema_arrow.names)
    mktable_written = set(pq.ParquetFile(chunk_dir / "fits.parquet").schema_arrow.names)
    return [c for c in visits if c not in added or c in mktable_written]


def stage(reference, chunk, build_dir):
    """Copy one reference chunk's ``mktable`` output into a fresh build.

    Parameters
    ----------
    reference : `str` or `pathlib.Path`
        The reference build directory, holding ``chunks/<chunk>/``.
    chunk : `str`
        Chunk name, ``"<dmin>_<dmax>"``.
    build_dir : `str` or `pathlib.Path`
        The new build directory. ``_work/chunks/<chunk>/`` is created under it.

    Returns
    -------
    staged : `pathlib.Path`
        The staged chunk directory.
    """
    reference = pathlib.Path(reference)
    src = reference / "chunks" / chunk
    staged = pathlib.Path(build_dir) / "_work" / "chunks" / chunk
    staged.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src / "donuts.parquet", staged / "donuts.parquet")
    columns = premerge_visit_columns(src)
    table = pq.read_table(src / "visits.parquet", columns=columns)
    pq.write_table(table, staged / "visits.parquet")
    print(f"staged {src} -> {staged}")
    print(f"  donuts.parquet: {pq.ParquetFile(staged / 'donuts.parquet').metadata.num_rows}"
          " rows (count, dimensionless)")
    print(f"  visits.parquet: {table.num_rows} rows x {table.num_columns} columns "
          f"(counts, dimensionless), down from "
          f"{len(pq.ParquetFile(src / 'visits.parquet').schema_arrow.names)} columns "
          "by dropping the merged sidecar")
    return staged


def main(argv=None):
    """Command line entry point.

    Returns
    -------
    status : `int`
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(
        prog="python -m rubinwork.products.fam_tables.refbuild.stage_reference_chunk",
        description=__doc__.split("\n\n")[0])
    parser.add_argument("--reference", required=True,
                        help="Reference build directory holding chunks/<chunk>/")
    parser.add_argument("--chunk", required=True, help="Chunk name, <dmin>_<dmax>")
    parser.add_argument("--build-dir", required=True,
                        help="New build directory to stage into")
    args = parser.parse_args(argv)
    stage(args.reference, args.chunk, args.build_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
