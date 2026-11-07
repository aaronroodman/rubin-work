#!/usr/bin/env python
"""Rebuild the 50-DOF double-Zernike (DZ) sensitivity matrix into a study run.

Thin driver over `rubinwork.smatrix.compute_smatrix.compute`. Writes
``smatrix_dz_31_29_50_<band>_ofc_zcs.npy`` (field_k, pupil_j, dof; µm of
wavefront per µm or degree of DOF) and ``compare_ofc_<band>.json``, the
per-DOF comparison with the ts_ofc matrix, for each band.

Usage::

    python smatrix/ofc_reproduction/code/build_dense.py --jobs 16
"""
import argparse
import json
import os

import numpy as np

from rubinwork.common import study_run
import rubinwork.smatrix.compute_smatrix as cs
from compare_ofc import DOF_LABELS, find_ofc_file, load_ofc


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bands", nargs="+", default=["ref", "r"],
                   help="ref (LSST_g_500, 0.5 um) is what the ts_ofc matrix was built at")
    p.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    p.add_argument("--label", default="dense_build")
    args = p.parse_args()

    run = study_run("smatrix/ofc_reproduction", args.label)
    ofc = load_ofc(find_ofc_file(None))
    for band in args.bands:
        sens, _ = cs.compute(band, cs.DEFAULT_DATA_DIR, bend_dir="bend", jobs=args.jobs)
        out = run.path / f"smatrix_dz_31_29_50_{band}_{cs.CONFIG_TAG}.npy"
        np.save(out, sens)
        rows = []
        for d in range(50):
            a, b = sens[:, 4:, d], ofc[:, 4:, d]          # pupil Noll 4..28
            rows.append(dict(
                idof=d, dof=DOF_LABELS[d],
                scale=float(np.sum(a * b) / np.sum(b * b)),
                rel_rms=float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2))),
                max_abs_um=float(np.max(np.abs(a - b)))))
        (run.path / f"compare_ofc_{band}.json").write_text(json.dumps(rows, indent=1))
        print(f"== band {band}")
        for r in rows:
            print(f"{r['idof']:3d} {r['dof']:<10} scale {r['scale']:.5f} "
                  f"rel_rms {r['rel_rms']:.2e} max_abs {r['max_abs_um']:.2e} um")
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
