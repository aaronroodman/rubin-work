#!/usr/bin/env python
"""Sparse-mode DOF observability: if the donut fit measures only the PRIMARY
aberrations (drop the secondary/tertiary pupil-Zernike rows of the sensitivity),
are the controlled v-modes still observable?  Repeated for the 50/34 and 22/12
(nDOF / nVmode-kept) schemes.

Method (matches ts_ofc StateEstimator exactly): the OFC's v-modes come from
  U,S,Vh = svd( sensitivity_matrix.reshape(-1, ndofs)[:, dof_idx] @ norm )
i.e. the raw DoubleZernike (all 31 field x 29 pupil) as observations, DOF
normalized.  Sparse mode = reshape using ONLY the primary pupil-Noll rows.

Primary (first radial order of each azimuthal family, Z4-28):
  Z4, Z5/6, Z7/8, Z9/10, Z11, Z14/15, Z20/21, Z27/28
Dropped (2nd/3rd): Z12/13, Z16/17, Z18/19, Z22, Z23/24, Z25/26.

Outputs, per scheme:
  * singular-value spectrum, full vs primary-only (with the n_keep truncation)
  * observability of each KEPT v-mode in primary-only (||A_prim @ v_m|| / S_m)
  * per-DOF observability ratio primary/full
Written to output/sparse_observability.pdf.
"""
import argparse
import asyncio
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

_PKG = "/Users/roodman/Astrophysics/Claude/packages"
DEFAULT_CONFIG_DIR = os.environ.get(
    "TS_CONFIG_MTTCS_DIR", _PKG + "/ts_config_mttcs") + "/MTAOS/v13/ofc"
if _PKG + "/ts_ofc/python" not in sys.path:
    sys.path.insert(0, _PKG + "/ts_ofc/python")

# The OFC measures the 21 packed Zernikes (zn_selected): Z4-Z26 minus Z20,Z21.
# Raw sensitivity_matrix pupil axis = Noll (verified: M2 tilt -> Noll7 coma), so we
# select these Noll indices directly (this also excludes piston/tilt Noll0-3).
ZN = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
PRIMARY = [4, 5, 6, 7, 8, 9, 10, 11, 14, 15]              # 1st radial order of each family
WFS_SENSORS = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]   # 4 corner wavefront sensors
HEX = ["dZ", "dX", "dY", "rX", "rY"]


def dof_labels(n_m1m3, n_m2):
    return (["M2:" + h for h in HEX] + ["Cam:" + h for h in HEX]
            + [f"M13b{i+1}" for i in range(n_m1m3)] + [f"M2b{i+1}" for i in range(n_m2)])


def build_scheme(config_dir, instrument, n_m1m3, n_m2):
    from lsst.ts.ofc import OFCData, SensitivityMatrix
    from lsst.ts.ofc.state_estimator import StateEstimator
    ofc = OFCData(instrument, config_dir=config_dir)
    ofc.configure_controller()
    asyncio.run(ofc.configure_instrument(instrument))
    ofc.zn_selected = np.array(ZN)                        # the 21 packed measured Zernikes
    ofc.comp_dof_idx = dict(
        m2HexPos=np.ones(5, bool), camHexPos=np.ones(5, bool),
        M1M3Bend=(np.arange(20) < n_m1m3), M2Bend=(np.arange(20) < n_m2))
    se = StateEstimator(ofc)
    sens = SensitivityMatrix(ofc)
    return ofc, se, sens


def _obs_matrices(ofc, se, sens, sampling):
    """(A_full, A_prim) observation matrices, DOF-normalized.

    sampling='dz': field-complete (raw DZ, all 31 field terms; pupil=Noll).
    sampling='wfs': evaluate at the 4 corner WFS field points (pupil axis = Noll-4)."""
    ndofs = ofc.ndofs
    norm = se.normalization_matrix
    if sampling == "wfs":
        fa = [ofc.sample_points[s] for s in WFS_SENSORS]
        A3d = np.nan_to_num(np.asarray(sens.evaluate(fa, 0.0), float))   # (4, npup, 50), Noll-4
        full = A3d[:, [j - 4 for j in ZN], :]
        prim = A3d[:, [j - 4 for j in PRIMARY], :]
    else:
        dz = np.nan_to_num(np.array(ofc.sensitivity_matrix, float))       # (31,29,50), pupil=Noll
        full = dz[:, ZN, :]
        prim = dz[:, PRIMARY, :]
    A_full = full.reshape(-1, ndofs)[:, ofc.dof_idx] @ norm
    A_prim = prim.reshape(-1, ndofs)[:, ofc.dof_idx] @ norm
    return A_full, A_prim


def scheme_observability(ofc, se, sens, sampling):
    """full/primary singular values, kept-vmode obs ratio, per-DOF ratio, at `sampling`.
    v-modes recomputed on the measured set (not se.Vh, which includes piston/tilt)."""
    A_full, A_prim = _obs_matrices(ofc, se, sens, sampling)
    U, S_full, Vh = np.linalg.svd(A_full, full_matrices=False)
    S_prim = np.linalg.svd(A_prim, compute_uv=False)
    obs = np.array([np.linalg.norm(A_prim @ Vh[m]) for m in range(len(S_full))])
    ratio_vmode = obs / np.where(S_full > 0, S_full, np.nan)
    ratio_dof = np.linalg.norm(A_prim, axis=0) / np.where(
        np.linalg.norm(A_full, axis=0) > 0, np.linalg.norm(A_full, axis=0), np.nan)
    return dict(S_full=S_full, S_prim=S_prim, ratio_vmode=ratio_vmode, ratio_dof=ratio_dof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    ap.add_argument("--instrument", default="lsst")
    ap.add_argument("--out", default="output/sparse_observability.pdf")
    args = ap.parse_args()

    schemes = [("50/34", 20, 20, 34), ("22/12", 7, 5, 12)]
    samplings = [("wfs", "4 corner WFS"), ("dz", "field-complete")]
    clr = lambda r: "tab:red" if r < 0.5 else "tab:orange" if r < 0.8 else "tab:green"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with PdfPages(args.out) as pdf:
        for tag, n_m1m3, n_m2, n_keep in schemes:
            ofc, se, sens = build_scheme(args.config_dir, args.instrument, n_m1m3, n_m2)
            labels = dof_labels(n_m1m3, n_m2)
            for smp, smp_lab in samplings:
                R = scheme_observability(ofc, se, sens, smp)
                nd = len(R["S_full"])
                nk = min(n_keep, nd)
                print(f"\n=== {tag} scheme, {smp_lab} (nDOF={nd}, keep {n_keep}) ===")
                print(f"  kept v-modes: min primary/full obs = "
                      f"{np.nanmin(R['ratio_vmode'][:nk]):.2f}  "
                      f"(median {np.nanmedian(R['ratio_vmode'][:nk]):.2f})")
                worst = np.argsort(R["ratio_dof"])[:6]
                print("  DOF most degraded:",
                      {labels[i]: round(float(R['ratio_dof'][i]), 2) for i in worst})

                fig, ax = plt.subplots(1, 3, figsize=(18, 5))
                ax[0].semilogy(np.arange(1, len(R["S_full"]) + 1), R["S_full"],
                               "o-", ms=4, label="full aberrations")
                ax[0].semilogy(np.arange(1, len(R["S_prim"]) + 1), R["S_prim"],
                               "s-", ms=4, label="primary only")
                ax[0].axvline(nk + 0.5, color="k", ls="--", lw=1, label=f"keep {n_keep}")
                ax[0].set_xlabel("v-mode index"); ax[0].set_ylabel("singular value")
                ax[0].set_title(f"{tag}: v-mode SVD spectrum"); ax[0].legend(fontsize=8)
                ax[0].grid(alpha=0.3)

                mk = np.arange(1, nk + 1)
                ax[1].bar(mk, R["ratio_vmode"][:nk], color=[clr(r) for r in R["ratio_vmode"][:nk]])
                ax[1].axhline(1, color="k", lw=0.5); ax[1].set_ylim(0, 1.1)
                ax[1].set_xlabel("kept v-mode index"); ax[1].set_ylabel("primary/full observability")
                ax[1].set_title(f"{tag}: observability of the {nk} CONTROLLED v-modes")
                ax[1].grid(alpha=0.3, axis="y")

                xd = np.arange(len(labels))
                ax[2].bar(xd, R["ratio_dof"], color=[clr(r) for r in R["ratio_dof"]])
                ax[2].axhline(1, color="k", lw=0.5); ax[2].set_ylim(0, 1.1)
                ax[2].set_xticks(xd); ax[2].set_xticklabels(labels, rotation=90, fontsize=6)
                ax[2].set_ylabel("primary/full observability")
                ax[2].set_title(f"{tag}: per-DOF observability")
                ax[2].grid(alpha=0.3, axis="y")

                fig.suptitle(f"Sparse (primary-only) DOF observability -- {tag} scheme, "
                             f"{smp_lab} sampling  (green ok, orange weakened, red lost)",
                             fontsize=13)
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
