#!/usr/bin/env python
"""SVD of the geom-normalized full_zemax sensitivity matrix (all 232 DOF).

Normalization matches the OFC v-mode convention: a physical DOF vector d enters
as d/w (w = geometric weight sqrt(r/f)), so the normalized sensitivity is
A_norm = A . diag(w).  SVD(A_norm) = U S V^T gives the v-modes (columns of V, in
DOF space) and the principal-value (singular-value) spectrum S.

A = DZ sensitivity, pupil Noll Z4-Z22, field Noll k=1..30.

Outputs (../output/):
  full_svd_spectrum.png     - v-mode singular-value spectrum
  full_svd_dz_vmode.png     - DZ (pupil-Noll) content of each v-mode
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / "output"
N_RIGID, N_M1M3, N_M2 = 10, 153, 69
PUP = np.arange(4, 23)     # pupil Noll 4-22
FLD = np.arange(1, 31)     # field Noll 1-30


def main():
    sens = np.load(OUT / "smatrix_dz_31_29_232_r_bend_zemax.npy")   # (31,29,232)
    w = np.load(OUT / "normalization_all.npz")["full_zemax_w"]      # (232,)

    A = sens[np.ix_(FLD, PUP)].reshape(len(FLD) * len(PUP), -1)     # (nrows, 232)
    A_norm = A * w[None, :]                                          # columns scaled by w
    U, S, Vt = np.linalg.svd(A_norm, full_matrices=False)

    # ---- Page 1: singular-value (principal-value) spectrum ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.semilogy(np.arange(1, len(S) + 1), S, ".-", ms=4)
    ax.axvline(20.5, color="gray", ls="--", label="AOS-used ~20 v-modes")
    ax.set_xlabel("v-mode index"); ax.set_ylabel("singular value (normalized units)")
    ax.set_title("Full-mode v-mode principal-value spectrum  "
                 f"[full_zemax, {A.shape[1]} DOF, geom-normalized]")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    # annotate dynamic range
    ax.text(0.98, 0.95, f"s1/s_last = {S[0]/S[-1]:.1e}\n"
            f"n(s>1e-3·s1) = {int(np.sum(S>1e-3*S[0]))}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    fig.tight_layout(); fig.savefig(OUT / "full_svd_spectrum.png", dpi=130)
    print("wrote", OUT / "full_svd_spectrum.png")

    # ---- Page 2: DZ (pupil-Noll) content of each v-mode ----
    nshow = 60
    # U[:,m] reshaped (field, pupil); RMS over field -> per pupil Noll amplitude
    Uref = U[:, :nshow].reshape(len(FLD), len(PUP), nshow)
    dz_content = np.sqrt(np.mean(Uref**2, axis=0))                 # (pupil, mode)
    dz_content /= dz_content.max(axis=0, keepdims=True) + 1e-30    # normalize per v-mode
    fig, ax = plt.subplots(figsize=(15, 7))
    im = ax.imshow(dz_content, aspect="auto", cmap="viridis", origin="lower",
                   extent=[0.5, nshow + 0.5, PUP[0] - 0.5, PUP[-1] + 0.5])
    ax.set_xlabel("v-mode index"); ax.set_ylabel("pupil Noll Z")
    ax.set_yticks(PUP)
    ax.set_title("DZ (pupil-Zernike) content of each v-mode  "
                 "[field-RMS of left singular vector, per-mode normalized]")
    fig.colorbar(im, ax=ax, shrink=0.8, label="relative pupil-Noll content")
    fig.tight_layout(); fig.savefig(OUT / "full_svd_dz_vmode.png", dpi=130)
    print("wrote", OUT / "full_svd_dz_vmode.png")

    np.savez(OUT / "full_svd.npz", U=U, S=S, Vt=Vt, w=w, pupil=PUP, field=FLD)
    print("saved full_svd.npz;  s1=%.3g  s20=%.3g  s50=%.3g" % (S[0], S[19], S[49]))


if __name__ == "__main__":
    main()
