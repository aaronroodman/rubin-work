#!/usr/bin/env python
"""Single large-format view of the full DZ sensitivity matrix (all 232 DOF).

Rows = double-Zernike terms (field Noll k x pupil Noll j, j>=4), columns = the
232 full_zemax DOF (10 rigid + 153 M1M3 + 69 M2).  Color = sensitivity
(um wavefront per DOF unit), symmetric log-ish scale.

Output: ../output/full_sensitivity_matrix.png
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

OUT = Path(__file__).resolve().parent.parent / "output"
KMAX, JMAX = 30, 28
N_RIGID, N_M1M3, N_M2 = 10, 153, 69


def main():
    sens = np.load(OUT / "smatrix_dz_31_29_232_r_bend_zemax.npy")   # (31,29,232)
    # rows: field k = 1..KMAX (skip k=0 unused), pupil j = 4..JMAX
    ks = np.arange(1, KMAX + 1)
    js = np.arange(4, JMAX + 1)
    block = sens[np.ix_(ks, js)]                 # (nk, nj, ndof)
    nk, nj, ndof = block.shape
    mat = block.transpose(0, 1, 2).reshape(nk * nj, ndof)   # (nk*nj rows, ndof)

    vmax = np.percentile(np.abs(mat), 99.5)
    fig, ax = plt.subplots(figsize=(26, 16))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", interpolation="nearest",
                   norm=SymLogNorm(linthresh=vmax * 1e-2, vmin=-vmax, vmax=vmax))
    # row ticks: one label per field-k block (every nj rows)
    ax.set_yticks(np.arange(nk) * nj + nj / 2)
    ax.set_yticklabels([f"field Z{k}" for k in ks], fontsize=7)
    for kk in range(1, nk):
        ax.axhline(kk * nj - 0.5, color="k", lw=0.3, alpha=0.4)
    # column group dividers + labels
    ax.axvline(N_RIGID - 0.5, color="k", lw=1)
    ax.axvline(N_RIGID + N_M1M3 - 0.5, color="k", lw=1)
    ax.set_xticks([N_RIGID / 2, N_RIGID + N_M1M3 / 2, N_RIGID + N_M1M3 + N_M2 / 2])
    ax.set_xticklabels(["rigid\n(10)", f"M1M3 bending ({N_M1M3})",
                        f"M2 bending ({N_M2})"], fontsize=11)
    ax.set_xlabel("DOF", fontsize=12)
    ax.set_ylabel("double-Zernike term  (field Noll block; pupil Noll 4-28 within each)", fontsize=11)
    ax.set_title("Full DZ sensitivity matrix — 232 DOF x (field Z1-30 x pupil Z4-28)  "
                 "[full_zemax, µm WF / DOF unit, symlog]", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.6, label="µm WF / DOF unit")
    fig.tight_layout()
    out = OUT / "full_sensitivity_matrix.png"
    fig.savefig(out, dpi=110); print("wrote", out)


if __name__ == "__main__":
    main()
