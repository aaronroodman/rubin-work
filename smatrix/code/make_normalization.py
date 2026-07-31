#!/usr/bin/env python
"""Generate geometric normalization weights under both force sources, for the
nominal 50 DOF and the full mode set, so the OFC v13 baseline and the
self-consistent full-mode weights are all available side by side.

Force sources (r = (force_range/20)/max|force-per-N-um|):
  "im"    Bo's IM force (ts_config v13 files; M2 already lbf->N).  This is what
          the official OFC v13 range weights used.  Self-consistent with the
          IM-basis modes (bend_full).
  "zemax" ZEMAX match_forces force (bend_zemax; M2 lbf->N here).  Self-consistent
          with the ZEMAX/OFC-basis modes (bend_zemax) that reproduce the OFC
          *sensitivity* exactly.

Combinations produced (all saved to ../output/normalization_all.npz):
  50dof_im     : OFC 50-DOF sensitivity + IM force (first-20)   -> reproduces v13
  50dof_zemax  : OFC 50-DOF sensitivity + ZEMAX force (bend.yaml modes)
  full_im      : bend_full (238 DOF, IM basis) + IM force (raw order)
  full_zemax   : bend_zemax (232 DOF, ZEMAX basis) + ZEMAX force (raw order)

f is field-averaged quadrature FWHM over Noll Z4-Z22 (the v13 convention).
"""
import numpy as np
import yaml
from pathlib import Path
from astropy.io import fits

import normalization_weights as NW

LBF2N = 4.4482216
OUT = Path(__file__).resolve().parent.parent / "output"
DATA = "/Users/roodman/LSST/batoid_rubin_data"
V13 = "/Users/roodman/Astrophysics/Claude/packages/ts_config_mttcs/MTAOS/v13/ofc"

# bend.yaml AOS mode selections (raw indices), per basis
AOS_M1M3_IM = list(range(20))                    # IM first-20 == bend.yaml modes
AOS_M2_IM = list(range(20))
AOS_M1M3_ZEMAX = list(range(19)) + [26]          # ZEMAX bend.yaml
AOS_M2_ZEMAX = list(range(17)) + [25, 26, 27]


def load_force(source):
    """Return (F_m1m3, F_m2) force-per-um matrices in Newtons, columns = raw modes.
    F shape (n_act, n_mode) so max over axis 0 gives per-mode peak force."""
    if source == "im":
        F1 = np.array(yaml.safe_load(open(f"{V13}/M1M3/M1M3_1um_156_force.yaml")))[:, 3:]   # (156,156) N
        F2 = np.array(yaml.safe_load(open(f"{V13}/M2/M2_1um_72_force.yaml")))[:, 3:]         # (72,72) N
        return np.abs(F1), np.abs(F2)            # (act, mode) already
    elif source == "zemax":
        F1 = fits.getdata(f"{DATA}/bend_zemax/M1M3_bend_forces.fits.gz").T                   # (mode,act)->(act,mode)
        F2 = fits.getdata(f"{DATA}/bend_zemax/M2_bend_forces.fits.gz").T * LBF2N             # lbf -> N
        return np.abs(F1), np.abs(F2)
    raise ValueError(source)


def compute(sens, F1, F2, sel1, sel2):
    f = NW.compute_f_quadrature(sens, rings=5, spokes=6, znmin=4, znmax=22)
    with np.errstate(divide="ignore"):
        r = NW.compute_range_weights(F1, F2, sel1, sel2)   # rb_stroke + bending
    # null-space modes (zero actuator force -> infinite range) are not physically
    # controllable; mark them NaN so they don't pollute the weights.
    r[~np.isfinite(r)] = np.nan
    w = NW.geometric_weights(r, f)
    return f, r, w


def main():
    res = {}

    # --- 50 DOF (OFC sensitivity; force source differs) ---
    sens50 = np.load(OUT / "smatrix_dz_31_29_50_r_ofc_zcs.npy")
    F1_im, F2_im = load_force("im")
    F1_zx, F2_zx = load_force("zemax")
    res["50dof_im"] = compute(sens50, F1_im, F2_im, AOS_M1M3_IM, AOS_M2_IM)
    res["50dof_zemax"] = compute(sens50, F1_zx, F2_zx, AOS_M1M3_ZEMAX, AOS_M2_ZEMAX)

    # --- full sets (each in its own self-consistent basis) ---
    sfull_im = np.load(OUT / "smatrix_dz_31_29_238_r_bend_full.npy")     # IM basis, 156+72
    sfull_zx = np.load(OUT / "smatrix_dz_31_29_232_r_bend_zemax.npy")    # ZEMAX basis, 153+69
    res["full_im"] = compute(sfull_im, F1_im, F2_im, list(range(156)), list(range(72)))
    res["full_zemax"] = compute(sfull_zx, F1_zx, F2_zx, list(range(153)), list(range(69)))

    # --- reproduce-v13 check ---
    w_off = np.array(yaml.safe_load(
        open(f"{V13}/normalization_weights/range0.5_fwhm-0.15.yaml")))
    ratio = res["50dof_im"][2] / w_off
    print(f"50dof_im vs official v13: max dev {np.max(np.abs(ratio-1))*100:.3f}%  "
          f"(reproduces v13)")
    # zemax-vs-im on the 50 DOF
    ri, rz = res["50dof_im"][1], res["50dof_zemax"][1]
    print(f"50dof range zemax/im: M1M3 median {np.median(rz[10:30]/ri[10:30]):.3f} "
          f"[{(rz[10:30]/ri[10:30]).min():.2f},{(rz[10:30]/ri[10:30]).max():.2f}];  "
          f"M2 median {np.median(rz[30:50]/ri[30:50]):.3f}")

    np.savez(OUT / "normalization_all.npz",
             **{f"{k}_{q}": v for k, (fv, rv, wv) in res.items()
                for q, v in [("f", fv), ("r", rv), ("w", wv)]},
             w_official_v13=w_off)
    print("saved", OUT / "normalization_all.npz")
    for k, (fv, rv, wv) in res.items():
        nnull = int(np.sum(~np.isfinite(rv)))
        print(f"  {k:<12} ndof={len(wv)}" + (f"  ({nnull} null-space modes -> NaN)" if nnull else ""))

    # comparison plot: 50-DOF im vs zemax range; full-set w
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ri, rz = res["50dof_im"][1], res["50dof_zemax"][1]
    x = np.arange(10, 50)
    axes[0].plot(x, (rz / ri)[10:50], "o-", ms=4)
    axes[0].axhline(1, color="k", lw=0.6)
    for xb in (29.5,):
        axes[0].axvline(xb, color="gray", lw=0.6)
    axes[0].set_title("50-DOF range ratio  r_ZEMAX / r_IM  (bending DOF)")
    axes[0].set_xlabel("DOF (10-29 M1M3 B1-20 · 30-49 M2 B1-20)")
    axes[0].set_ylabel("r_ZEMAX / r_IM"); axes[0].grid(alpha=0.3)
    for tag, sl, c in [("full_im (IM)", "full_im", "steelblue"),
                       ("full_zemax (ZEMAX)", "full_zemax", "darkorange")]:
        w = res[sl][2]
        axes[1].semilogy(range(len(w) - 10), w[10:], ".", ms=3, label=tag, color=c)
    axes[1].set_title("full-set geom weight w = sqrt(r/f)  (bending DOF)")
    axes[1].set_xlabel("bending DOF index"); axes[1].set_ylabel("w"); axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "normalization_compare.png", dpi=130)
    print("wrote", OUT / "normalization_compare.png")


if __name__ == "__main__":
    main()
