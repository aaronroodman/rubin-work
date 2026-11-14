"""Write the Danish-fit Smatrix of each fit set in the ts_ofc formats.

For each fit set, the side-averaged FAM field fit of the brute-force slopes
(`validation_pdf.smatrix`, kmax 30) becomes a (31 field, 29 pupil, 50 DOF) matrix in µm
of wavefront per DOF unit (µm, or deg for the tilts), ts_ofc v13 DOF order and signs.
Pupil Zernikes a fit set does not fit are zero rows, so a solve must restrict
``zn_selected`` to the fitted Noll indices (``noll.txt``).

Per fit set, in the run directory ``<name>/``:

- ``ofc/``: a copy of the v13 OFC config directory with
  ``sensitivity_matrix/lsst_sensitivity_dz_31_29_50.yaml`` replaced, for
  ``aos_state.make_state_estimator(config_dir=...)``;
- ``smatrix_dz_31_29_50_r_<name>.npy`` and ``.fits``, the `compute_smatrix` format;
- ``noll.txt``, the fitted pupil Noll indices.

Fit-set names: prod is ``sparse_no22``, prod_z22 is ``sparse_with22``, dense21 keeps its
name.

    python code/write_smatrix.py
"""
import argparse
import shutil
import textwrap
from pathlib import Path

import numpy as np
from astropy.io import fits

from refit_images import OUT_ROOT
from rubinwork import aos_state
from rubinwork.common import study_run
from validation_pdf import KMAX, smatrix

NAMES = {"dense21": "dense21", "prod": "sparse_no22", "prod_z22": "sparse_with22"}
SENS_YAML = "lsst_sensitivity_dz_31_29_50.yaml"


def write_ofc_yaml(path, S, fs):
    """Write S in the layout of ts_ofc ``save_sensitivity_matrix``.

    Copied rather than imported: ``lsst.ts.ofc.scripts`` imports ``lsst.ts.imsim``,
    which is not installed here.
    """
    with open(path, "w") as f:
        f.write("--- \n"
                f"# Sensitivity matrix from the sparse_fit Danish fit set {fs}.\n"
                "# Dimensions: (31,29,50): field Zernike k, pupil Zernike j (Noll), DOF.\n"
                "# The DOF are M2 dz, dx, dy (um), M2 rx, ry (deg), Cam dz, dx, dy (um),\n"
                "# Cam rx, ry (deg), 20 M1M3 bending modes (um), 20 M2 bending modes (um).\n"
                "# Units are um (Zk) / um or deg (dof).\n\n")
        for sublist in S.tolist():
            f.write("- ")
            for idx, row in enumerate(sublist):
                if idx != 0:
                    f.write("  ")
                f.write(textwrap.fill("- [" + ", ".join(map(str, row)), width=100,
                                      subsequent_indent="  ") + "] \n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--responses", default="20261111_responses_ofcsign")
    p.add_argument("--label", default="smatrix_files")
    args = p.parse_args()

    resp = np.load(f"{OUT_ROOT}/{args.responses}/responses.npz")
    v13_dir = Path(aos_state.resolve_ofc_config_dir("v13")).resolve()
    v13 = aos_state.make_state_estimator(dof_set="all_50")
    s1_v13 = np.linalg.svd(v13.ofc_data.sensitivity_matrix.reshape(-1, 50)
                           @ v13.normalization_matrix, compute_uv=False)[0]
    run = study_run("smatrix/sparse_fit", args.label)

    for fs, name in NAMES.items():
        S = smatrix(resp, fs)
        noll = resp[f"{fs}__noll"]
        out = Path(run.path) / name
        cfg = out / "ofc"
        shutil.copytree(v13_dir, cfg)
        write_ofc_yaml(cfg / "sensitivity_matrix" / SENS_YAML, S, fs)

        stem = f"smatrix_dz_31_29_50_r_{name}"
        np.save(out / f"{stem}.npy", S)
        hdr = fits.Header()
        hdr["CONTENT"] = "DZ sensitivity (field_k, pupil_j, dof)"
        hdr["DIM"] = "(31,29,50)"
        hdr["FITSET"] = (fs, "Danish fit set in sparse_fit")
        hdr["KMAX"] = (KMAX, "field Zernikes fitted over the FAM grid")
        hdr["NOLL"] = ",".join(map(str, noll))
        hdr["DOFSIGN"] = "OFC"
        hdr["UNITS"] = "um_zk_per_um_or_degree_dof"
        hdr["RESPONSE"] = args.responses
        fits.PrimaryHDU(S, header=hdr).writeto(out / f"{stem}.fits")
        np.savetxt(out / "noll.txt", noll, fmt="%d")

        se = aos_state.make_state_estimator(config_dir=str(cfg), dof_set="all_50")
        back = se.ofc_data.sensitivity_matrix
        s1 = np.linalg.svd(back.reshape(-1, 50) @ se.normalization_matrix,
                           compute_uv=False)[0]
        print(f"{name} ({fs}): {len(noll)} pupil Zernikes; reload max |diff| "
              f"{np.max(np.abs(back - S)):.2e} µm per DOF unit; singular value 1 "
              f"{s1:.2f} (v13 {s1_v13:.2f}), µm per unit normalized DOF")
    print(f"wrote {run.path}")


if __name__ == "__main__":
    main()
