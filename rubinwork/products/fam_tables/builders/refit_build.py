"""A new ``fam_tables`` build that reuses another build's donut table and refits.

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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variant", required=True)
    p.add_argument("--source", required=True, help="build to reuse the donut table of")
    p.add_argument("--build", required=True, help="name of the new build")
    p.add_argument("--coord-sys", default="OCS")
    p.add_argument("--overwrite", action="store_true")
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

    n_pass, n = quality.apply(out / "visits.parquet")
    print(f"visit_quality_pass ({quality.CUT_DESCRIPTION}): {n_pass} of {n} visits pass")

    fits = out / "fits.parquet"
    wf_bin = pathlib.Path(os.environ["TS_INTRINSIC_WAVEFRONT_DIR"]) / "bin"
    cmd = [sys.executable, str(wf_bin / "run_dz_fit.py"), str(donuts),
           "--coord-sys", args.coord_sys, "--visits", str(out / "visits.parquet"),
           "--output", str(fits)]
    print("running: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    src_manifest = catalog.manifest(reader.PRODUCT, args.variant, args.source)
    options = dict((src_manifest.get("config") or {}).get("build_options") or {})
    options.update({"builder": "refit_build", "refit_from": args.source,
                    "coord_sys": args.coord_sys,
                    "visit_quality_cut": quality.CUT_DESCRIPTION})
    path = write_manifest(args.variant, build=args.build, build_options=options,
                          inputs={"fam_tables": f"{args.variant}@{args.source}"})
    print(f"manifest: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
