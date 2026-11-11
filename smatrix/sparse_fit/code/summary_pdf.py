"""Study summary PDF from the brute-force outputs: field-model error against kmax, CWFS
against a single Smatrix, and nonlinearity at the DOF range, per DOF.

Usage::

    python summary_pdf.py --label summary
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from kmax_test import KMAXES
from refit_images import OUT_ROOT
from rubinwork.common import study_run

FIT_SETS = ("prod", "prod_z22", "dense21")
SUMMARY = """sparse_fit: brute-force sensitivity of sparse Danish fits (ray-traced batoid donuts)

940 configurations (26 check, 770 FAM, 144 CWFS) x 50 DOF x 5 amplitudes; fit sets
prod (Noll 4-11, 14, 15, 20, 21, 27, 28), prod_z22 (+ Z22), dense21 (Noll 4-26 but 20, 21).
Slopes from a quadratic through k = 0, +-1, +-2 steps; points with a lost fit (fwhm more
than 0.2 arcsec off nominal) dropped: 4, 6, 9 scans of 47000 (prod, prod_z22, dense21).

Relative error = |difference| / RMS over FAM of |response| (dimensionless, norm over the
fitted terms), median over points; prod unless named.

- Field model (FAM fit, field Zernikes to kmax, R_outer 1.75 deg): cross-validated error
  0.016-0.023 at kmax 30, 0.011-0.017 at kmax 91; the in-sample residual matches, so the
  floor is slope noise. kmax 30 is enough except camera dx/dy and M1M3 B15/B16/B20.
- CWFS (r <= 1.725 deg, n = 104): a single side-averaged Smatrix 0.021-0.035, as good as
  one per side (0.021-0.037; camera dx/dy 0.22 per side against 0.044 averaged).
  Per-corner averages 0.015-0.032. Detector shift against camera shift: no visible effect.
- In-focus Smatrix rows against the fitted responses: rigid body 0.07, B1-B10 0.5-0.6,
  B11-B20 1.0-1.8. The sparse Smatrix must come from the fitted responses.
- Nonlinearity at the full DOF range (quadratic / linear change): bending B3 and up
  <= 0.005; M1M3 and M2 B1-B2 median 0.023-0.044, max 0.18; rigid body not measurable
  from these steps (range 4.5-400x the +2 step, noise dominated).
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kmax-run", default="20261111_kmax_test")
    p.add_argument("--cwfs-run", default="20261111_cwfs_compare")
    p.add_argument("--lin-run", default="20261111_linearity_range")
    p.add_argument("--label", default="summary")
    args = p.parse_args()
    k = np.load(f"{OUT_ROOT}/{args.kmax_run}/kmax_test.npz")
    c = np.load(f"{OUT_ROOT}/{args.cwfs_run}/cwfs_compare.npz")
    lin = np.load(f"{OUT_ROOT}/{args.lin_run}/linearity_range.npz")
    run = study_run("smatrix/sparse_fit", args.label)
    dof = np.arange(50)
    with PdfPages(f"{run.path}/sparse_fit_summary.pdf") as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.05, 0.96, SUMMARY, va="top", family="monospace", fontsize=7.5)
        pdf.savefig(fig)
        plt.close(fig)

        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        for ax, fs in zip(axs, FIT_SETS):
            for km in KMAXES:
                rel = k[f"{fs}__k{km}__rel"]
                ax.plot(dof, np.nanmedian(rel[k["tag"] == "fam_cv"], 0), ".-",
                        label=f"FAM cross-validation, kmax {km}")
            ax.plot(dof, np.nanmedian(c[f"{fs}__single"], 0), "k.--",
                    label="CWFS, single Smatrix, kmax 30")
            ax.plot(dof, np.nanmedian(c[f"{fs}__own"], 0), "r.:",
                    label="CWFS, per-side Smatrix, kmax 30")
            ax.set(yscale="log", ylabel="median relative error", title=fs)
            for x in (9.5, 29.5):
                ax.axvline(x, c="0.7", lw=0.8)
        axs[0].legend(fontsize=6, ncol=2)
        axs[-1].set_xlabel("DOF index (0-9 rigid body, 10-29 M1M3 B1-B20, 30-49 M2 B1-B20)")
        fig.tight_layout()
        fig.savefig(f"{run.path}/field_model_error.png", dpi=110)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        for fs in FIT_SETS:
            q = lin[fs]
            ax.plot(dof, np.nanmedian(q, 0), ".-", label=f"{fs} median")
            ax.plot(dof, np.nanmax(q, 0), "x", label=f"{fs} max")
        ax.set(yscale="log", xlabel="DOF index",
               ylabel="quadratic / linear change at full range",
               title="Nonlinearity at the ts_ofc DOF range (rigid body: noise dominated)")
        for x in (9.5, 29.5):
            ax.axvline(x, c="0.7", lw=0.8)
        ax.legend(fontsize=7, ncol=3)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    print(f"wrote {run.path}")


if __name__ == "__main__":
    main()
