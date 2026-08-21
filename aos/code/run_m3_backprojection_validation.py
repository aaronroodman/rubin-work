#!/usr/bin/env python
"""Validation + stress tests for the M3 back-projection pipeline (simulation only).

Runs three checks with batoid (no real data):
  1. D1 injection/recovery — a known M3 surface figure is recovered.
  2. Source discrimination — back-project M1/M2/M3 injections ASSUMING M3.
  3. Two-surface stress — M3 + a fraction of M2; how the consistency degrades.

Produces a summary PDF and prints the numbers.

Usage:
  python code/run_m3_backprojection_validation.py [--out FILE.pdf] [--band r]
"""
import argparse
import warnings

import numpy as np
import batoid
import galsim.zernike as gz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import m3_backprojection as bp

warnings.filterwarnings("ignore")


def truth_surface_opd(coef_m, X, Y, cos_theta, R_outer, R_inner, wavelength):
    """Wavefront OPD [waves] imprinted by a surface figure: 2 cos(theta) h / lambda."""
    h = gz.Zernike(coef_m, R_inner=R_inner, R_outer=R_outer).evalCartesian(X, Y)
    return 2.0 * cos_theta * h / wavelength


def mean_cos_incidence(tel, surface, radius_deg=1.0):
    _, _, _, _, cz = bp.pupil_to_surface(tel, surface, radius_deg, 0.0, nrad=8, naz=32)
    return float(np.mean(cz))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output/m3_backprojection_d1_validation.pdf")
    ap.add_argument("--band", default="r")
    ap.add_argument("--grid-n", type=int, default=120)
    ap.add_argument("--nrad", type=int, default=28)
    ap.add_argument("--naz", type=int, default=110)
    ap.add_argument("--jmax", type=int, default=28)
    ap.add_argument("--radius-deg", type=float, default=1.7)
    args = ap.parse_args()

    tel = bp.load_telescope(args.band)
    lam = bp.WAVELENGTH
    pts = np.array(bp.field_grid(radius_deg=args.radius_deg, n_ring=6, per_ring0=6))
    R_out, R_in = bp.surface_aperture(tel, "M3")
    cz3 = mean_cos_incidence(tel, "M3")

    def bproj(tel_p, surface="M3"):
        zk = (bp.field_zernikes(tel_p, pts, jmax=args.jmax)
              - bp.field_zernikes(tel, pts, jmax=args.jmax))
        return bp.run_backprojection(tel, pts, zk, surface=surface,
                                     grid_n=args.grid_n, nrad=args.nrad, naz=args.naz)

    # ---- 1. D1 injection/recovery on M3 ----
    coef = np.zeros(args.jmax + 1)
    coef[5] = 200e-9; coef[7] = 100e-9; coef[11] = 150e-9   # astig + coma + spherical
    r = bproj(bp.perturb_surface(tel, "M3", coef))
    X, Y, glob, con, cnt, mask = r["X"], r["Y"], r["glob"], r["con"], r["count"], r["mask"]
    well = (cnt >= 10) & mask & ~np.isnan(glob)
    truth = truth_surface_opd(coef, X, Y, cz3, R_out, R_in, lam)
    G = bp.remove_plane_masked(glob, X, Y, well)
    T = bp.remove_plane_masked(truth, X, Y, well)
    a, b = G[well], T[well]
    rec_corr = np.corrcoef(a, b)[0, 1]
    rec_slope = np.polyfit(b, a, 1)[0]
    print("=== 1. D1 M3 injection/recovery ===")
    print(f"  recover corr = {rec_corr:+.4f}   amplitude slope = {rec_slope:+.3f}")
    print(f"  self-consistency: median VE = {np.nanmedian(con['var_explained']):.3f}"
          f"   RMS = {np.nanmedian(con['rms']):.4f} waves")

    # ---- 2. source discrimination (back-project ASSUMING M3) ----
    print("\n=== 2. source discrimination (back-project as M3) ===")
    print(f"  {'injected':16s} {'VE':>7s} {'RMS(w)':>8s}")
    disc = {}
    for name in ("M1", "M2", "M3"):
        Ro, Ri = bp.surface_aperture(tel, name)
        c = np.zeros(args.jmax + 1); c[5] = 200e-9; c[7] = 100e-9; c[11] = 150e-9
        rr = bproj(bp.perturb_surface(tel, name, c, R_outer=Ro, R_inner=Ri))
        ve = np.nanmedian(rr["con"]["var_explained"])
        rms = np.nanmedian(rr["con"]["rms"])
        disc[name] = (ve, rms)
        print(f"  {name+' source':16s} {ve:7.3f} {rms:8.4f}")

    # ---- 3. two-surface stress: M3 + fraction f of M2 astig ----
    print("\n=== 3. two-surface stress (M3 signal + f*M2 astig), back-project as M3 ===")
    print(f"  {'M2/M3 frac':>10s} {'VE':>7s} {'RMS(w)':>8s} {'M3-recover corr':>16s}")
    Ro2, Ri2 = bp.surface_aperture(tel, "M2")
    fracs = [0.0, 0.25, 0.5, 1.0]
    two_ve, two_corr = [], []
    for f in fracs:
        m3c = np.zeros(args.jmax + 1); m3c[5] = 200e-9; m3c[7] = 100e-9; m3c[11] = 150e-9
        m2c = np.zeros(args.jmax + 1); m2c[5] = f * 200e-9; m2c[6] = f * 150e-9
        telp = bp.perturb_surface(tel, "M3", m3c)
        telp = bp.perturb_surface(telp, "M2", m2c, R_outer=Ro2, R_inner=Ri2)
        rr = bproj(telp)
        wl = (rr["count"] >= 10) & rr["mask"] & ~np.isnan(rr["glob"])
        Gf = bp.remove_plane_masked(rr["glob"], X, Y, wl)
        Tf = bp.remove_plane_masked(truth_surface_opd(m3c, X, Y, cz3, R_out, R_in, lam),
                                    X, Y, wl)
        cc = np.corrcoef(Gf[wl], Tf[wl])[0, 1]
        ve = np.nanmedian(rr["con"]["var_explained"])
        rms = np.nanmedian(rr["con"]["rms"])
        two_ve.append(ve); two_corr.append(cc)
        print(f"  {f:10.2f} {ve:7.3f} {rms:8.4f} {cc:16.4f}")

    # ---- summary figure ----
    ext = [-R_out, R_out, -R_out, R_out]
    vmax = np.nanmax(np.abs(T))
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5))

    def show(a_, M, ttl, vm=vmax, cmap="RdBu_r", lab="waves"):
        im = a_.imshow(M, origin="lower", extent=ext, cmap=cmap, vmin=-vm, vmax=vm)
        a_.set_title(ttl, fontsize=11)
        plt.colorbar(im, ax=a_, shrink=.8, label=lab)
        a_.set_xlabel("M3 x [m]"); a_.set_ylabel("M3 y [m]")

    show(ax[0, 0], T, "Injected M3 OPD (truth, ptt-removed)")
    show(ax[0, 1], G, "Recovered M3 OPD (stitched)")
    show(ax[0, 2], G - T, f"Difference (RMS={np.nanstd((G - T)[well]):.3f} w)")

    imc = ax[1, 0].imshow(np.where(cnt > 0, cnt, np.nan), origin="lower",
                          extent=ext, cmap="viridis")
    ax[1, 0].set_title("FoV-point overlap count")
    plt.colorbar(imc, ax=ax[1, 0], shrink=.8)
    ax[1, 0].set_xlabel("M3 x [m]"); ax[1, 0].set_ylabel("M3 y [m]")

    sc = ax[1, 1].scatter(pts[:, 0], pts[:, 1], c=con["var_explained"], s=90,
                          cmap="viridis", vmin=0.9, vmax=1.0, edgecolor="k", lw=.3)
    ax[1, 1].set_title("Self-consistency VE per FoV (M3 source)")
    ax[1, 1].set_aspect("equal")
    ax[1, 1].set_xlabel("field x [deg]"); ax[1, 1].set_ylabel("field y [deg]")
    plt.colorbar(sc, ax=ax[1, 1], shrink=.8, label="var explained")

    axd = ax[1, 2]
    axd.plot(fracs, two_ve, "o-", label="self-consistency VE")
    axd.plot(fracs, two_corr, "s--", label="M3-recovery corr")
    axd.axhline(disc["M2"][0], color="gray", ls=":", lw=1,
                label=f"pure-M2 VE ({disc['M2'][0]:.2f})")
    axd.set_xlabel("M2 astig amplitude / M3 amplitude")
    axd.set_ylabel("metric")
    axd.set_title("Two-surface stress (M3 + M2)")
    axd.set_ylim(0.4, 1.02); axd.legend(fontsize=8); axd.grid(alpha=.3)

    fig.suptitle("M3 back-projection — D1 injection/recovery + stress tests (batoid, sim only)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
