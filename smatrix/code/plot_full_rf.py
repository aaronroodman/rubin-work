#!/usr/bin/env python
"""Full-DOF normalization components: range r (ZEMAX and IM) and FWHM impact f.

Reads normalization_all.npz (from make_normalization.py).
Output: ../output/full_rf.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "output"
N_RIGID, N_M1M3_Z, N_M2_Z = 10, 153, 69
N_M1M3_I, N_M2_I = 156, 72


def main():
    d = np.load(OUT / "normalization_all.npz")
    fZ, rZ = d["full_zemax_f"], d["full_zemax_r"]     # 232
    rI = d["full_im_r"]                                # 238 (IM basis)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # f (FWHM) — from full_zemax
    ax = axes[0]
    ax.semilogy(range(1, N_M1M3_Z + 1), fZ[N_RIGID:N_RIGID + N_M1M3_Z], ".-", ms=4,
                color="steelblue", label="M1M3")
    ax.semilogy(range(1, N_M2_Z + 1), fZ[N_RIGID + N_M1M3_Z:], ".-", ms=4,
                color="darkorange", label="M2")
    ax.axvspan(0.5, 20.5, color="gray", alpha=0.12, label="AOS-used 20")
    ax.set_ylabel("f  (FWHM, arcsec / DOF-unit)")
    ax.set_title("FWHM impact f  (field-averaged, Noll 4-22)  [full_zemax]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    # r (range) — ZEMAX vs IM
    ax = axes[1]
    ax.semilogy(range(1, N_M1M3_Z + 1), rZ[N_RIGID:N_RIGID + N_M1M3_Z], ".-", ms=4,
                color="steelblue", label="M1M3 ZEMAX")
    ax.semilogy(range(1, N_M2_Z + 1), rZ[N_RIGID + N_M1M3_Z:], ".-", ms=4,
                color="darkorange", label="M2 ZEMAX")
    ax.semilogy(range(1, N_M1M3_I + 1), rI[N_RIGID:N_RIGID + N_M1M3_I], "x", ms=3,
                color="navy", alpha=0.6, label="M1M3 IM")
    ax.semilogy(range(1, N_M2_I + 1), rI[N_RIGID + N_M1M3_I:], "x", ms=3,
                color="saddlebrown", alpha=0.6, label="M2 IM")
    ax.axvspan(0.5, 20.5, color="gray", alpha=0.12)
    ax.set_ylabel("r  (range, DOF-unit)")
    ax.set_xlabel("bending mode number")
    ax.set_title("Range r  (force-per-micron; ZEMAX vs IM force)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = OUT / "full_rf.png"
    fig.savefig(out, dpi=130); print("wrote", out)


if __name__ == "__main__":
    main()
