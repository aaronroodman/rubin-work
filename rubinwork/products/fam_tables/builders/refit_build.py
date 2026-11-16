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

With ``--pair-cut-nmad``, step 1 instead rewrites ``donuts.parquet`` without the donuts
`quality.pair_outliers` flags, per visit, and adds ``n_donuts_pair_cut`` (donuts dropped,
count) to ``visits.parquet``. Rows are dropped, not flagged: the DZ fitter and the MIW
build read every row and do not look at ``used``.

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

PAIR_CUT_FRAME = "OCS"


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


def write_pair_cut(src_donuts, out_donuts, n_nmad):
    """Copy a donut table without the donuts whose intra and extra fits disagree.

    One row group per visit is kept, as the fitter's row-group lookup needs. Returns a
    `pandas.DataFrame` of day_obs, seq_num and n_donuts_pair_cut (count).
    """
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(src_donuts))
    writer = pq.ParquetWriter(str(out_donuts), pf.schema_arrow)
    intra, extra = f"zk_{PAIR_CUT_FRAME}_intra", f"zk_{PAIR_CUT_FRAME}_extra"
    rows, seen, n_in, n_out = [], set(), 0, 0
    try:
        for i in range(pf.num_row_groups):
            t = pf.read_row_group(i)
            keep = np.ones(t.num_rows, bool)
            day = t["day_obs"].to_numpy()
            seq = t["seq_num"].to_numpy()
            for d, q in sorted(set(zip(day.tolist(), seq.tolist()))):
                if (d, q) in seen:
                    raise ValueError(f"visit {d} {q} spans more than one row group")
                seen.add((d, q))
                m = (day == d) & (seq == q)
                sub = t.filter(pa_mask(m))
                zi, ze = (stack_lists(sub[c]) for c in (intra, extra))
                flagged = np.zeros(m.sum(), bool) if zi is None or ze is None \
                    else quality.pair_outliers(zi, ze, n_nmad)
                keep[np.flatnonzero(m)[flagged]] = False
                rows.append(dict(day_obs=d, seq_num=q, n_donuts_pair_cut=int(flagged.sum())))
            writer.write_table(t.filter(pa_mask(keep)), row_group_size=max(t.num_rows, 1))
            n_in += t.num_rows
            n_out += int(keep.sum())
            if i % 200 == 0:
                print(f"row group {i}/{pf.num_row_groups}: kept {n_out} of {n_in} donuts",
                      flush=True)
    finally:
        writer.close()
    print(f"pair cut at {n_nmad} nMAD: kept {n_out} of {n_in} donuts, dropped "
          f"{n_in - n_out} ({100 * (n_in - n_out) / max(n_in, 1):.2f}%)", flush=True)
    return pd.DataFrame(rows)


def pa_mask(mask):
    import pyarrow as pa
    return pa.array(mask, pa.bool_())


def stack_lists(column):
    """(n, n_zk) float array from a list column; None if any entry is null."""
    import numpy as np
    if column.null_count:
        return None
    return np.stack(column.to_numpy(zero_copy_only=False)).astype(float)


def add_pair_cut_counts(visits_path, counts):
    """Add n_donuts_pair_cut (count) to visits.parquet, 0 for visits with no donuts."""
    import pandas as pd
    v = pd.read_parquet(visits_path)
    v = v.drop(columns=["n_donuts_pair_cut"], errors="ignore").merge(
        counts, on=["day_obs", "seq_num"], how="left")
    v["n_donuts_pair_cut"] = v["n_donuts_pair_cut"].fillna(0).astype(int)
    v.to_parquet(visits_path, index=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--variant", required=True)
    p.add_argument("--source", required=True, help="build to reuse the donut table of")
    p.add_argument("--build", required=True, help="name of the new build")
    p.add_argument("--coord-sys", default="OCS")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--reflag", action="store_true",
                   help="keep the source fits and visit flag; recompute only bad_fit")
    p.add_argument("--pair-cut-nmad", type=float, default=None,
                   help="drop donuts whose intra and extra fits differ by more than this "
                        "many nMAD in any Noll, per visit (dimensionless)")
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
    for name in ("visits.parquet", "provenance.yaml"):
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    if args.pair_cut_nmad is None:
        os.link(src / "donuts.parquet", donuts)
    else:
        if args.reflag:
            raise SystemExit("--pair-cut-nmad changes the donuts; it needs a refit")
        counts = write_pair_cut(src / "donuts.parquet", donuts, args.pair_cut_nmad)
        add_pair_cut_counts(out / "visits.parquet", counts)

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
                    "donut_pair_cut": (None if args.pair_cut_nmad is None else
                                       f"drop donuts with |(intra - extra) - median| > "
                                       f"{args.pair_cut_nmad:g} nMAD in any Noll, per "
                                       f"visit, zk_{PAIR_CUT_FRAME}"),
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
