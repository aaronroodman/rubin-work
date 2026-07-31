#!/usr/bin/env python
"""Full-mode geometric normalization weights (all 153 M1M3 + 69 M2 modes).

Uses the ZEMAX/OFC-basis full sensitivity matrix (bend_zemax, 232 DOF) and the
matching ZEMAX force-per-micron (bend_zemax/*_bend_forces), so r and f/sensitivity
are self-consistent for every mode (unlike the official v13 low-20, which pairs
ZEMAX sensitivity with IM-basis force).

  f_i = field-averaged quadrature FWHM, Noll Z4-Z22            [arcsec/DOF-unit]
  r_i = (force_range/20)/max|force-per-um| ; rb_stroke         [DOF-unit]
  w_i = sqrt(r_i/f_i)                                          (v13 geometric)

Outputs ../output/:
  normalization_full.npz              (f, r, w, labels)
  fullmode_normalization.png          (f, r, w vs mode number)
"""
import numpy as np
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt

import normalization_weights as NW

N_RIGID = 10
N_M1M3 = 153
N_M2 = 69
DATA = "/Users/roodman/LSST/batoid_rubin_data/bend_zemax"
V13 = ("/Users/roodman/Astrophysics/Claude/packages/ts_config_mttcs/MTAOS/v13/ofc/"
       "normalization_weights/range0.5_fwhm-0.15.yaml")


def main():
    outdir = Path(__file__).resolve().parent.parent / "output"
    sens = np.load(outdir / "smatrix_dz_31_29_232_r_bend_zemax.npy")   # (31,29,232)
    assert sens.shape[2] == N_RIGID + N_M1M3 + N_M2

    # f: field-averaged FWHM, Noll 4-22
    f = NW.compute_f_quadrature(sens, rings=5, spokes=6, znmin=4, znmax=22)

    # r: ZEMAX force-per-um (self-consistent basis), all modes
    F1 = fits.getdata(f"{DATA}/M1M3_bend_forces.fits.gz")   # (153, n_act) mode x act
    F2 = fits.getdata(f"{DATA}/M2_bend_forces.fits.gz")     # (69, n_act)
    fpu1 = np.max(np.abs(F1), axis=1)                       # per-mode peak |force/um|
    fpu2 = np.max(np.abs(F2), axis=1)
    r_m1m3 = (NW.M1M3_FORCE_RANGE / 20.0) / fpu1
    r_m2 = (NW.M2_FORCE_RANGE / 20.0) / fpu2
    r = np.concatenate([NW.RB_STROKE, r_m1m3, r_m2])

    w = NW.geometric_weights(r, f)                          # sqrt(r/f)

    labels = (["M2dz", "M2dx", "M2dy", "M2Rx", "M2Ry",
               "Cdz", "Cdx", "Cdy", "CRx", "CRy"]
              + [f"M1M3_B{i}" for i in range(1, N_M1M3 + 1)]
              + [f"M2_B{i}" for i in range(1, N_M2 + 1)])
    np.savez(outdir / "normalization_full.npz", f=f, r=r, w=w, labels=labels)
    print("saved", outdir / "normalization_full.npz", " shape", w.shape)

    # low-20 sanity vs official v13
    import yaml
    w_off = np.array(yaml.safe_load(open(V13)))
    # map: rigid 0-9, M1M3 B1-20 (full 10-29), M2 B1-20 (full 163-182)
    idx_full = list(range(10)) + list(range(10, 30)) + list(range(163, 183))
    ratio = w[idx_full] / w_off
    print(f"\nlow-50-DOF geom weights vs official v13 (ZEMAX force):")
    print(f"  rigid (0-9) max dev: {np.max(np.abs(ratio[:10]-1))*100:.2f}%")
    print(f"  M1M3 B1-20 median ratio {np.median(ratio[10:30]):.3f}, "
          f"M2 B1-20 median ratio {np.median(ratio[30:50]):.3f}")
    print("  (deviations reflect ZEMAX-vs-IM force basis for the rotated modes.)")

    # plot f, r, w vs mode number
    m1 = slice(N_RIGID, N_RIGID + N_M1M3)
    m2 = slice(N_RIGID + N_M1M3, None)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, arr, name in zip(axes, [f, r, w],
                             ["f  (FWHM, arcsec/DOF)", "r  (range, DOF-unit)",
                              "w = sqrt(r/f)  (geom normalization)"]):
        ax.semilogy(range(1, N_M1M3 + 1), arr[m1], ".-", ms=4, label="M1M3", color="steelblue")
        ax.semilogy(range(1, N_M2 + 1), arr[m2], ".-", ms=4, label="M2", color="darkorange")
        ax.axvspan(0.5, 20.5, color="gray", alpha=0.12)
        ax.set_ylabel(name); ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(); axes[0].set_title("Full-mode normalization components "
                                        "(153 M1M3 + 69 M2, ZEMAX/OFC basis; gray = AOS-used 20)")
    axes[-1].set_xlabel("bending mode number")
    fig.tight_layout()
    out = outdir / "fullmode_normalization.png"
    fig.savefig(out, dpi=130); print("wrote", out)


if __name__ == "__main__":
    main()
