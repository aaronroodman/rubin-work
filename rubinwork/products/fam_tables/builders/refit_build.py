"""A new ``fam_tables`` build that reuses another build's donut table and refits.

With ``--reflag``, the source build's fits are kept and only the ``bad_fit`` flags are
recomputed (`lsst.ts.intrinsic.wavefront.dz_fitting.flag_bad_fits`), for a change of the
flag definition alone.

The donut table and the attached telemetry do not depend on the per-visit quality cut,
so a change of cut (`rubinwork.products.fam_tables.quality`) needs only a new
``visits.parquet`` flag and a new Double Zernike (DZ) fit, not a Butler read. This
builder makes that build:

1. hard-links ``donuts.parquet`` from the source build, and copies ``visits.parquet``
   and ``provenance.yaml``;
2. rewrites ``visit_quality_pass`` with `quality.apply`;
3. runs the k=1..3 and k=1..6 DZ fits on the passing visits, with the default Batoid
   intrinsic, as the source build did. The fitter joins the whole visits table, so the
   attached telemetry columns come through with it;
4. writes the manifest, recording the source build as an input.

Run from the repo root::

    python -m rubinwork.products.fam_tables.builders.refit_build \\
        --variant danish_1_2 --source 20261014 --build 20261025
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

from rubinwork.products import catalog
from rubinwork.products.fam_tables import quality, reader
from rubinwork.products.fam_tables.write_manifest import write as write_manifest


def reflag(src_fits, out_fits):
    """Copy a ``fits.parquet``, recomputing the three ``bad_fit`` columns.

    Uses the fitter's defaults, so the flags equal those of a fresh fit. Every other
    column and the schema metadata are copied unchanged.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from astropy.table import Table
    from lsst.ts.intrinsic.wavefront.dz_fitting import flag_bad_fits

    table = pq.read_table(str(src_fits))
    flags = {}
    for prefix in ("z1toz3", "z1toz6"):
        cols = [c for c in table.column_names if c.startswith(f"{prefix}_z") and "_c" in c
                and not c.endswith("_err") and not c.endswith("_scale")]
        t = Table({c: table[c].to_numpy() for c in cols + ["day_obs", "seq_num"]})
        t["n_donuts"] = table["n_donuts_1" if prefix == "z1toz3" else "n_donuts_2"].to_numpy()
        flags[f"{prefix}_bad_fit"] = np.asarray(flag_bad_fits(t, prefix))
    flags["bad_fit"] = flags["z1toz3_bad_fit"] | flags["z1toz6_bad_fit"]
    for name, value in flags.items():
        i = table.schema.get_field_index(name)
        old = table[name].to_numpy()
        print(f"{name}: {int(old.sum())} -> {int(value.sum())} of {len(value)} visits")
        table = table.set_column(i, name, pa.array(value, pa.bool_()))
    pq.write_table(table, str(out_fits))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variant", required=True)
    p.add_argument("--source", required=True, help="build to reuse the donut table of")
    p.add_argument("--build", required=True, help="name of the new build")
    p.add_argument("--coord-sys", default="OCS")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--reflag", action="store_true",
                   help="keep the source fits and visit flag; recompute only bad_fit")
    args = p.parse_args(argv)

    src = reader.build_dir(args.variant, args.source)
    out = reader.build_dir(args.variant, args.build)
    if not catalog.is_complete(src):
        raise SystemExit(f"{src} is not a complete build")
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} exists; pass --overwrite")
    (out / "_work" / "logs").mkdir(parents=True, exist_ok=True)

    donuts = out / "donuts.parquet"
    if donuts.exists():
        donuts.unlink()
    os.link(src / "donuts.parquet", donuts)
    for name in ("visits.parquet", "provenance.yaml"):
        if (src / name).exists():
            shutil.copy2(src / name, out / name)

    fits = out / "fits.parquet"
    if args.reflag:
        reflag(src / "fits.parquet", fits)
    else:
        n_pass, n = quality.apply(out / "visits.parquet")
        print(f"visit_quality_pass ({quality.CUT_DESCRIPTION}): {n_pass} of {n} visits pass")
        wf_bin = pathlib.Path(os.environ["TS_INTRINSIC_WAVEFRONT_DIR"]) / "bin"
        cmd = [sys.executable, str(wf_bin / "run_dz_fit.py"), str(donuts),
               "--coord-sys", args.coord_sys, "--visits", str(out / "visits.parquet"),
               "--output", str(fits)]
        print("running: " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    src_manifest = catalog.manifest(reader.PRODUCT, args.variant, args.source)
    options = dict((src_manifest.get("config") or {}).get("build_options") or {})
    options.update({"builder": "refit_build", "refit_from": args.source,
                    "reflag_only": args.reflag,
                    "coord_sys": args.coord_sys,
                    "visit_quality_cut": quality.CUT_DESCRIPTION,
                    "bad_fit": "more than 5 coefficients with |DZ| > 2.0 um, "
                               "or n_donuts < 200"})
    path = write_manifest(args.variant, build=args.build, build_options=options,
                          inputs={"fam_tables": f"{args.variant}@{args.source}"})
    print(f"manifest: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
