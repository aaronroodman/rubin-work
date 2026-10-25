"""Build one ``wfs_corner_compare`` variant: the matched table and the per-corner stats.

Reads the variant's ``cwfs_tables`` build and the ``fam_tables`` build that CWFS build
was made from (its manifest's ``inputs``), unless ``--fam-build`` names another.

Usage, from the repository root::

    python -m rubinwork.products.wfs_corner_compare.builders.build_wfs_corner_compare \\
        --variant d12-refitWcs --out-dir <build dir>
"""

import argparse
import pathlib
import sys

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from rubinwork import corner_compare as cc
from rubinwork.products import catalog

from .. import reader


def resolve_inputs(variant, cwfs_build=catalog.CURRENT, fam_build=None):
    """The CWFS and FAM build directories a build of `variant` reads.

    Returns
    -------
    cwfs_dir, fam_dir : `pathlib.Path`
    inputs : `dict`
        ``{product: "variant@build"}`` for the manifest.
    """
    cv, fv = reader.cwfs_variant(variant), reader.fam_variant(variant)
    cwfs_dir = catalog.path("cwfs_tables", cv, cwfs_build)
    if fam_build is None:
        ref = (catalog.manifest("cwfs_tables", cv, cwfs_dir.name).get("inputs") or {}
               ).get("fam_tables", "")
        fam_build = ref.partition("@")[2] or catalog.CURRENT
    fam_dir = catalog.path("fam_tables", fv, fam_build)
    return cwfs_dir, fam_dir, {"cwfs_tables": f"{cv}@{cwfs_dir.name}",
                               "fam_tables": f"{fv}@{fam_dir.name}"}


def build(variant, out_dir, cwfs_dir, fam_dir):
    """Write both tables of one build into `out_dir`."""
    s = reader.settings(variant)
    zc, txc, tyc = f"zk_{s['coord']}", f"thx_{s['coord']}", f"thy_{s['coord']}"
    visits = pq.read_table(fam_dir / "visits.parquet").to_pandas()
    noll = [int(x) for x in np.asarray(visits["nollIndices"].iloc[0])]
    cw_cols = ["detector", "day_obs", "seq_num", "fam_seq_num", zc, txc, tyc]
    if "wfs_offset" in pq.read_schema(cwfs_dir / "donuts.parquet").names:
        cw_cols.append("wfs_offset")
    cwfs = pq.read_table(cwfs_dir / "donuts.parquet", columns=cw_cols).to_pandas()
    nights = sorted(int(d) for d in cwfs["day_obs"].unique())
    fam = pads.dataset(fam_dir / "donuts.parquet").to_table(
        columns=["day_obs", "seq_num", zc, txc, tyc],
        filter=pads.field("day_obs").isin(nights)).to_pandas()
    print(f"{variant}: {len(cwfs)} CWFS donuts, {len(fam)} FAM donuts (counts) over "
          f"{len(nights)} nights")
    table = cc.match_corners(fam, cwfs, visits, noll, settings=s)
    stats = cc.corner_stats(table)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / "wfs_corner_compare.parquet", index=False)
    stats.to_parquet(out_dir / "corner_stats.parquet", index=False)
    print(f"wrote {len(table)} matched rows and {len(stats)} stats rows (counts) to {out_dir}")
    return table, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cwfs-build", default=catalog.CURRENT)
    ap.add_argument("--fam-build", default=None,
                    help="default: the fam_tables build the CWFS build was made from")
    a = ap.parse_args(argv)
    cwfs_dir, fam_dir, inputs = resolve_inputs(a.variant, a.cwfs_build, a.fam_build)
    print(f"inputs: {inputs}")
    build(a.variant, a.out_dir, cwfs_dir, fam_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
