#!/usr/bin/env python
"""Joint static-figure fit: can a simultaneous static surface-Zernike figure on
M3 + L1 + L2 + Filter reproduce the MIW?  The last static-optics loophole.

Free parameters (72): the amplitude of each Noll surface-Zernike Z5..Z22 (18
modes, NO Z4 -- power is cross-surface degenerate) on each of four SINGLE surfaces
  M3, L1_entrance, L2_entrance, Filter_entrance
(M1/M2 omitted: near-pupil, field-common, removed by the 50-DoF/34-mode optics
adjustment; L3 omitted: too near focus; each lens's 2nd surface omitted:
degenerate with the 1st).  Everything else -- geometry, positions, indices, and
per-surface piston/tip/tilt/power -- is fixed (the nominal batoid LSST model).

Method: each free parameter's batoid response (perturb the surface by a unit
Zernike figure -> change in the field Zernikes Z4..Z26 at every field point) is a
column of a response matrix; solve all coefficients at once by least squares to
the MIW.  Reports total + nested-subset VE, each surface's fitted figure RMS
(µm surface height -- for physical plausibility), and residual Z5..Z8 maps.

Frame MIW-OCS -> batoid is IDENTITY (validated).  i-band to match the MIW.

  python code/run_miw_joint_fit.py                       # M3 L1 L2 Filter, Z5..Z22
  python code/run_miw_joint_fit.py --optics M3 L1        # a subset
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import galsim.zernike as gz

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m3_backprojection as bp

warnings.filterwarnings("ignore")

JS = [j for j in range(4, 27) if j not in (20, 21)]          # MIW target Noll set
DEFAULT_MIW = ("output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/"
               "pathA_50_34_i_5rot/intrinsic_split_maps.parquet")
# optic -> the single surface carrying its figure
SURF_MAP = {"M1": "M1", "M2": "M2", "M3": "M3", "L1": "L1_entrance",
            "L2": "L2_entrance", "Filter": "Filter_entrance", "L3": "L3_entrance"}
UNIT = 1e-7   # response-column unit surface-height amplitude [m] (=100 nm)


def load_miw(path, stride):
    df = pd.read_parquet(path)
    cols = [f"Z{j}_OCS" for j in JS]
    ok = np.all(np.isfinite(df[cols].values), axis=1)
    df = df[ok].iloc[::stride].reset_index(drop=True)
    pts = np.column_stack([df.thx_deg.values, df.thy_deg.values])
    zk = np.zeros((len(df), 27))
    for j in JS:
        zk[:, j] = df[f"Z{j}_OCS"].values
    return pts, zk


def build_response(tel, pts, optics, modes, nx):
    """Response matrix A [N*len(JS), n_free] and ids=[(optic, surface, mode)]."""
    ptl = list(map(tuple, pts))
    z0 = bp.field_zernikes(tel, ptl, jmax=26, nx=nx)
    cols, ids = [], []
    for o in optics:
        s = SURF_MAP[o]
        Ro, Ri = bp.surface_aperture(tel, s)
        for k in modes:
            c = np.zeros(27); c[k] = UNIT
            telp = bp.perturb_surface(tel, s, c, R_outer=Ro, R_inner=Ri)
            dz = (bp.field_zernikes(telp, ptl, jmax=26, nx=nx) - z0) \
                * bp.WAVELENGTH * 1e6                 # µm field-Z per UNIT(100nm) figure
            cols.append(dz[:, JS].ravel()); ids.append((o, s, k))
        print(f"  built response for {o} ({s}): {len(modes)} modes", flush=True)
    return np.column_stack(cols), ids


def fit(A, dvec, sel):
    c, *_ = np.linalg.lstsq(A[:, sel], dvec, rcond=None)
    ve = 1 - np.var(dvec - A[:, sel] @ c) / np.var(dvec)
    return ve, c


def fig_residual(pts, zk, model_full, out, zs=(5, 6, 7, 8)):
    rows = [("MIW data", zk), ("joint model", model_full), ("residual", zk - model_full)]
    fig, ax = plt.subplots(3, len(zs), figsize=(3.5 * len(zs), 10), squeeze=False)
    ss = max(5, int(1800 / np.sqrt(len(pts))))
    vlim = {j: np.nanpercentile(np.abs(zk[:, j]), 98) for j in zs}
    for row, (name, arr) in enumerate(rows):
        for col, j in enumerate(zs):
            a = ax[row, col]
            sc = a.scatter(pts[:, 0], pts[:, 1], c=arr[:, j], s=ss, marker="s",
                           cmap="RdBu_r", vmin=-vlim[j], vmax=vlim[j])
            a.set_aspect("equal")
            a.set_title(f"{name}  Z{j}  (rms={np.sqrt(np.nanmean(arr[:, j]**2)):.3f})",
                        fontsize=9)
            plt.colorbar(sc, ax=a, shrink=.8)
    fig.suptitle("Joint static fit (M3+L1+L2+Filter, Z5-Z22): MIW vs model vs "
                 "residual, Z5-Z8 [µm]", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miw", default=DEFAULT_MIW)
    ap.add_argument("--band", default="i")
    ap.add_argument("--optics", nargs="+", default=["M3", "L1", "L2", "Filter"])
    ap.add_argument("--mode-min", type=int, default=5)
    ap.add_argument("--mode-max", type=int, default=22)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--nx", type=int, default=44)
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()

    modes = list(range(args.mode_min, args.mode_max + 1))
    tel = bp.load_telescope(args.band)
    pts, zk = load_miw(args.miw, args.stride)
    dvec = zk[:, JS].ravel()
    print(f"MIW: {args.miw}\n  {len(pts)} field pts (stride {args.stride}), band {args.band}"
          f"\n  optics {args.optics}, modes Z{args.mode_min}..Z{args.mode_max} "
          f"({len(args.optics) * len(modes)} free params)\n")

    A, ids = build_response(tel, pts, args.optics, modes, args.nx)
    opt_of = np.array([i[0] for i in ids])

    print("\n--- nested VE (add one optic at a time) ---")
    for k in range(1, len(args.optics) + 1):
        subset = args.optics[:k]
        sel = np.where(np.isin(opt_of, subset))[0]
        ve, _ = fit(A, dvec, sel)
        print(f"  {'+'.join(subset):24s} ({len(sel):2d} params): VE={ve:+.3f}")
    print("  --- each optic alone ---")
    for o in args.optics:
        sel = np.where(opt_of == o)[0]
        ve, _ = fit(A, dvec, sel)
        print(f"  {o:24s} ({len(sel):2d} params): VE={ve:+.3f}")

    ve_full, c = fit(A, dvec, np.arange(A.shape[1]))
    cond = float(np.linalg.cond(A))
    print(f"\n=== JOINT FIT (all {A.shape[1]} params): VE={ve_full:+.3f} ===")
    print(f"  response-matrix condition number = {cond:.3g}  "
          f"({'well' if cond < 1e3 else 'ILL' if cond > 1e5 else 'moderately'}-conditioned)")
    print("  fitted figure amplitude per surface (SURFACE HEIGHT; physical if <~0.1µm):")
    for o in args.optics:
        cc = c[opt_of == o]
        rms_um = float(np.sqrt(np.sum(cc ** 2)) * UNIT * 1e6)   # orthonormal Zernikes
        pk_um = float(np.max(np.abs(cc)) * UNIT * 1e6)
        kpk = int([k for (oo, s, k), b in zip(ids, opt_of == o) if b]
                  [int(np.argmax(np.abs(cc)))])
        print(f"    {o:8s}: RMS={rms_um:8.2f} µm   peak={pk_um:8.2f} µm (Z{kpk})")

    # VE vs PHYSICAL figure amplitude: ridge-regularized L-curve.  The unconstrained
    # fit uses ~20µm figures; this shows how little VE survives at plausible (<~0.1µm)
    # surface figures -- the honest "how much can real static optics explain".
    AtA = A.T @ A; Atd = A.T @ dvec
    print("\n--- VE vs allowed figure amplitude (ridge) ---")
    for lam in np.logspace(-6, 4, 11):
        cr = np.linalg.solve(AtA + lam * np.eye(A.shape[1]), Atd)
        ver = 1 - np.var(dvec - A @ cr) / np.var(dvec)
        amp = float(np.sqrt(np.sum(cr ** 2)) * UNIT * 1e6)   # total RMS figure µm
        print(f"    total figure RMS = {amp:9.3f} µm  ->  VE = {ver:+.3f}")

    # per-column response leverage (field-Z rms produced by a unit=100nm figure)
    col_resp = np.sqrt(np.mean(A ** 2, axis=0))
    print("  per-optic response leverage (median µm field-Z per 100nm figure):")
    for o in args.optics:
        print(f"    {o:8s}: {np.median(col_resp[opt_of == o]):.4g}")
    print(f"  max fitted surface amplitude = {np.abs(c).max()*UNIT*1e6:.2f} µm "
          f"(peak-to-plausible: a real figure error is <~0.1µm)")
    np.savez(os.path.join(args.outdir, "joint_fit_solution.npz"),
             c=c, ids=np.array(ids, dtype=object), col_resp=col_resp,
             optics=np.array(args.optics), modes=np.array(modes), ve=ve_full, cond=cond)

    # residual Z5-Z8 maps
    model_js = (A @ c).reshape(len(pts), len(JS))
    model_full = np.zeros((len(pts), 27))
    for col, j in enumerate(JS):
        model_full[:, j] = model_js[:, col]
    os.makedirs(args.outdir, exist_ok=True)
    out = os.path.join(args.outdir, "miw_joint_fit_residual.pdf")
    fig_residual(pts, zk, model_full, out)
    for j in (5, 6, 7, 8):
        dr = np.sqrt(np.mean(zk[:, j] ** 2)); rr = np.sqrt(np.mean((zk - model_full)[:, j] ** 2))
        print(f"  Z{j}: data rms={dr:.3f} -> residual rms={rr:.3f} µm  "
              f"({100*(1-rr/dr):+.0f}% reduced)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
