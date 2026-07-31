#!/usr/bin/env python
"""Analyze the full-mode DZ sensitivity matrix (all M1M3 + M2 bending modes).

Inputs ../output/smatrix_dz_31_29_238_<band>_bend_full.npy  (31,29,238).
DOF layout: 0-9 rigid, 10..10+NM1M3-1 M1M3 modes, then NM2 M2 modes.

Produces (../output/):
  fullmode_controllability_<band>.png  - wavefront leverage vs mode number
  fullmode_pupil_heatmap_<band>.png    - field-constant pupil-Zernike x mode
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

N_RIGID = 10
N_M1M3 = 156
N_M2 = 72
AOS_M1M3 = 20      # modes used by the current AOS / OFC
AOS_M2 = 20


def mode_wf_rms(sens, jlo=4):
    """Per-DOF wavefront leverage: RMS over all field-k and pupil-Noll>=jlo,
    in um wavefront per um of mode (drops pupil piston/tip/tilt j<4)."""
    s = sens[:, jlo:, :]                     # (31, 29-jlo, ndof)
    return np.sqrt(np.mean(s**2, axis=(0, 1)))  # (ndof,)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--band", default="r")
    args = p.parse_args()
    outdir = Path(__file__).resolve().parent.parent / "output"
    sens = np.load(outdir / f"smatrix_dz_31_29_238_{args.band}_bend_full.npy")
    ndof = sens.shape[2]
    assert ndof == N_RIGID + N_M1M3 + N_M2, ndof

    wf = mode_wf_rms(sens)
    m1 = wf[N_RIGID:N_RIGID + N_M1M3]
    m2 = wf[N_RIGID + N_M1M3:]

    # ---- controllability spectrum ----
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.semilogy(np.arange(1, N_M1M3 + 1), m1, "o-", ms=3, label="M1M3 (156 modes)", color="steelblue")
    ax.semilogy(np.arange(1, N_M2 + 1), m2, "s-", ms=3, label="M2 (72 modes)", color="darkorange")
    ax.axvspan(0.5, AOS_M1M3 + 0.5, color="gray", alpha=0.15,
               label=f"AOS-used (first {AOS_M1M3})")
    ax.set_xlabel("bending mode number (1-indexed)")
    ax.set_ylabel("wavefront leverage  RMS(DZ, Noll$\\geq$4)  [µm WF / µm mode]")
    ax.set_title(f"Bending-mode controllability spectrum — band {args.band}\n"
                 "(how much low-order wavefront each mirror mode produces)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out1 = outdir / f"fullmode_controllability_{args.band}.png"
    fig.savefig(out1, dpi=130); print("wrote", out1)

    # cumulative fraction of total wavefront leverage (variance)
    for nm, arr, aos in [("M1M3", m1, AOS_M1M3), ("M2", m2, AOS_M2)]:
        cum = np.cumsum(arr**2) / np.sum(arr**2)
        n90 = int(np.searchsorted(cum, 0.90)) + 1
        n99 = int(np.searchsorted(cum, 0.99)) + 1
        print(f"{nm}: leverage(variance) reaches 90% by mode {n90}, 99% by mode {n99}; "
              f"AOS uses {aos}. mode1={arr[0]:.3e}, last={arr[-1]:.3e} "
              f"(ratio {arr[0]/arr[-1]:.0f}x)")

    # ---- field-constant pupil-Zernike x mode heatmap ----
    PUP = np.arange(4, 29)     # pupil Noll 4..28
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for ax, (nm, lo, n) in zip(axes, [("M1M3", N_RIGID, N_M1M3),
                                      ("M2", N_RIGID + N_M1M3, N_M2)]):
        block = sens[1, 4:29, lo:lo + n]     # field Noll 1 (constant), pupil 4-28, modes
        vmax = np.percentile(np.abs(block), 99.5) or 1e-12
        im = ax.imshow(block, aspect="auto", cmap="RdBu_r",
                       norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                       extent=[1, n, 28.5, 3.5])
        ax.set_title(f"{nm}: field-constant DZ  (pupil Noll vs mode)")
        ax.set_xlabel("bending mode number"); ax.set_ylabel("pupil Noll")
        fig.colorbar(im, ax=ax, shrink=0.8, label="µm WF / µm mode")
    fig.suptitle(f"Full-mode wavefront sensitivity structure — band {args.band}", fontsize=13)
    out2 = outdir / f"fullmode_pupil_heatmap_{args.band}.png"
    fig.savefig(out2, dpi=130); print("wrote", out2)


if __name__ == "__main__":
    main()
