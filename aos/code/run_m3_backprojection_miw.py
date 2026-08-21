#!/usr/bin/env python
"""Apply the mirror back-projection pipeline to the real measured intrinsic
wavefront (MIW), test whether a static single-mirror (or joint multi-mirror) OPD
can explain it, and round-trip forward-model the result back to the MIW.

Inputs: the OCS columns of an `intrinsic_split_maps` parquet (use the `_5rot`
variant — proper multi-rotator OCS/CCS split; CCS is only Z4). Frame convention
MIW-OCS -> batoid is IDENTITY (validated against ts_ofc's DZ sensitivity matrix:
no axis swap / pupil rotation / parity), so thx_deg->theta_x, thy_deg->theta_y and
the annular-Noll (eps=0.612) OCS coefficients feed straight into batoid.

Tests (all validated by an injected-M3 control that must return VE~1):
  1. Back-project the MIW to M1/M2/M3 separately; stitch; per-FoV consistency.
  2. Round-trip: fit each stitched OPD to surface Zernikes, re-inject in batoid,
     forward-model the field Zernikes, compare to the data MIW (forward VE).
  3. Joint fit: least-squares fit of M1+M2+M3 surface-Zernike modes simultaneously
     to the MIW field Zernikes (the full static-mirror-optics model).

Result on 17_6_1 pathA_50_34_i_5rot: a static single-mirror OPD reproduces only
~5% of the MIW (M3 best, M2/M1 worse); rotator- and elevation-invariant per Aaron
(cal MIW at 70 deg predicts 30-40 deg FAM) -> points to a FAM retrieval/model bias,
not static optics. The joint-fit VE quantifies the last static-optics loophole.

Usage (fast local / full RSP):
  python code/run_m3_backprojection_miw.py --miw <path> --stride 8 --nx 48
  python code/run_m3_backprojection_miw.py            # defaults; --help for knobs
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

JS = [j for j in range(4, 27) if j not in (20, 21)]   # FAM Noll set
DEFAULT_MIW = ("output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/"
               "pathA_50_34_i_5rot/intrinsic_split_maps.parquet")


def load_miw(path, stride):
    """Return (pts[N,2] deg, zk[N,27] µm OCS Noll-indexed, df) from an MIW map."""
    df = pd.read_parquet(path)
    cols = [f"Z{j}_OCS" for j in JS]
    ok = np.all(np.isfinite(df[cols].values), axis=1)
    df = df[ok].iloc[::stride].reset_index(drop=True)
    pts = np.column_stack([df.thx_deg.values, df.thy_deg.values])
    zk = np.zeros((len(df), 27))
    for j in JS:
        zk[:, j] = df[f"Z{j}_OCS"].values
    return pts, zk, df


def mean_cos(tel, s):
    _, _, _, _, cz = bp.pupil_to_surface(tel, s, 1.0, 0.0, nrad=8, naz=32)
    return float(np.mean(cz))


def roundtrip(tel, pts, zk, surface, r=None, grid_n=110, nrad=22, naz=88,
              nx=48, fitmax=28):
    """Back-project -> stitch -> fit surface Zernikes -> forward-model -> VE.

    Returns (result_dict, model_field_zernikes[N,27] µm, forward_VE, surf_coef µm).
    """
    if r is None:
        r = bp.run_backprojection(tel, pts, zk, surface=surface,
                                  grid_n=grid_n, nrad=nrad, naz=naz)
    X, Y, glob, mask = r["X"], r["Y"], r["glob"], r["mask"]
    Ro, Ri = bp.surface_aperture(tel, surface)
    ct = mean_cos(tel, surface)
    m = np.isfinite(glob) & mask
    B = gz.zernikeBasis(fitmax, X[m], Y[m], R_inner=Ri, R_outer=Ro)
    coef, *_ = np.linalg.lstsq(B.T, glob[m] / (2 * ct), rcond=None)     # surf height µm
    telp = bp.perturb_surface(tel, surface, coef * 1e-6, R_outer=Ro, R_inner=Ri)
    ptl = list(map(tuple, pts))
    model = (bp.field_zernikes(telp, ptl, jmax=26, nx=nx)
             - bp.field_zernikes(tel, ptl, jmax=26, nx=nx)) * bp.WAVELENGTH * 1e6
    d = zk[:, JS].ravel(); mo = model[:, JS].ravel()
    ve = 1 - np.var(d - mo) / np.var(d)
    return r, model, ve, coef


def joint_fit(tel, pts, zk, surfaces, modes, nx=48):
    """Least-squares fit of surface-Zernike modes on `surfaces` to the MIW.

    Returns (VE, n_modes, coef, response_matrix A, dvec).
    """
    ptl = list(map(tuple, pts))
    z0 = bp.field_zernikes(tel, ptl, jmax=26, nx=nx)
    dvec = zk[:, JS].ravel()
    cols = []
    for s in surfaces:
        Ro, Ri = bp.surface_aperture(tel, s)
        for k in modes:
            coef = np.zeros(27); coef[k] = 1e-7
            telp = bp.perturb_surface(tel, s, coef, R_outer=Ro, R_inner=Ri)
            dz = (bp.field_zernikes(telp, ptl, jmax=26, nx=nx) - z0) * bp.WAVELENGTH * 1e6 / 1e-7
            cols.append(dz[:, JS].ravel())
    A = np.column_stack(cols)
    c, *_ = np.linalg.lstsq(A, dvec, rcond=None)
    ve = 1 - np.var(dvec - A @ c) / np.var(dvec)
    return ve, A.shape[1], c, A, dvec


def make_control(tel, pts, nx=48):
    """Injected-M3 'MIW' on the real field grid (µm), for the round-trip control."""
    coef = np.zeros(27); coef[5] = 0.2e-6; coef[7] = 0.12e-6; coef[11] = 0.15e-6
    ptl = list(map(tuple, pts))
    return (bp.field_zernikes(bp.perturb_surface(tel, "M3", coef), ptl, jmax=26, nx=nx)
            - bp.field_zernikes(tel, ptl, jmax=26, nx=nx)) * bp.WAVELENGTH * 1e6


def fig_by_surface(tel, pts, res, out):
    surfs = list(res)
    fig, ax = plt.subplots(len(surfs), 3, figsize=(15, 4.7 * len(surfs)))
    if len(surfs) == 1:
        ax = ax[None, :]
    for row, s in enumerate(surfs):
        r = res[s]; X = r["X"]; Ro, _ = bp.surface_aperture(tel, s); ext = [-Ro, Ro, -Ro, Ro]
        g = r["glob"]; ve = r["con"]["var_explained"]; rms = r["con"]["rms"]
        vm = np.nanpercentile(np.abs(g[np.isfinite(g)]), 98)
        im = ax[row, 0].imshow(g, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vm, vmax=vm)
        ax[row, 0].set_title(f"{s}: stitched OPD [µm] (medVE={np.nanmedian(ve):.2f})")
        ax[row, 0].set_xlabel(f"{s} x [m]"); plt.colorbar(im, ax=ax[row, 0], shrink=.8)
        sc = ax[row, 1].scatter(pts[:, 0], pts[:, 1], c=ve, s=24, cmap="viridis", vmin=0, vmax=1)
        ax[row, 1].set_title(f"{s}: consistency VE vs FP position"); ax[row, 1].set_aspect("equal")
        ax[row, 1].set_xlabel("field x [deg]"); plt.colorbar(sc, ax=ax[row, 1], shrink=.8, label="VE")
        vr = np.nanpercentile(rms[np.isfinite(rms)], 98)
        sc2 = ax[row, 2].scatter(pts[:, 0], pts[:, 1], c=rms, s=24, cmap="magma_r", vmin=0, vmax=vr)
        ax[row, 2].set_title(f"{s}: residual RMS vs FP [µm]"); ax[row, 2].set_aspect("equal")
        ax[row, 2].set_xlabel("field x [deg]"); plt.colorbar(sc2, ax=ax[row, 2], shrink=.8, label="µm")
    fig.suptitle("Real MIW back-projected to each mirror: recovered OPD + per-FP consistency", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def fig_roundtrip(pts, zk, model, ve, out):
    fig, ax = plt.subplots(3, 4, figsize=(18, 11))
    for col, j in enumerate((5, 6, 7, 8)):
        d = zk[:, j]; mo = model[:, j]; res = d - mo; vm = np.nanpercentile(np.abs(d), 98)
        for row, (val, ttl) in enumerate([(d, "data MIW"), (mo, "M3 model (round-trip)"), (res, "residual")]):
            a = ax[row, col]
            sc = a.scatter(pts[:, 0], pts[:, 1], c=val, s=22, cmap="RdBu_r", vmin=-vm, vmax=vm)
            a.set_aspect("equal"); a.set_title(f"Z{j} {ttl} (rms={np.std(val):.3f})", fontsize=10)
            a.set_xlabel("field x [deg]"); plt.colorbar(sc, ax=a, shrink=.8)
    fig.suptitle(f"Round-trip: data MIW vs MIW forward-modeled from stitched M3 OPD (forward VE={ve:.2f}) [µm]", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miw", default=DEFAULT_MIW)
    ap.add_argument("--band", default="r")
    ap.add_argument("--stride", type=int, default=8, help="field-point subsample stride")
    ap.add_argument("--grid-n", type=int, default=110)
    ap.add_argument("--nrad", type=int, default=22)
    ap.add_argument("--naz", type=int, default=88)
    ap.add_argument("--nx", type=int, default=48, help="batoid.zernike grid for forward models")
    ap.add_argument("--joint-modes", type=int, default=22, help="max surface Noll mode in joint fit")
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()

    tel = bp.load_telescope(args.band)
    pts, zk, _ = load_miw(args.miw, args.stride)
    print(f"MIW: {args.miw}\n  field points: {len(pts)} (stride {args.stride}), units µm, frame OCS==batoid (identity)")

    # 1. per-surface back-projection
    res = {s: bp.run_backprojection(tel, pts, zk, surface=s, grid_n=args.grid_n,
                                    nrad=args.nrad, naz=args.naz) for s in ("M3", "M2", "M1")}
    print("\n1. back-projection (per-FoV consistency of a single-surface OPD):")
    for s in ("M3", "M2", "M1"):
        c = res[s]["con"]
        print(f"   {s}: median VE={np.nanmedian(c['var_explained']):+.3f}  RMS={np.nanmedian(c['rms']):.4f} µm")

    # 2. round-trip (forward VE) + control
    print("\n2. round-trip forward VE (fraction of MIW a static single-mirror OPD reproduces):")
    rt = {}
    for s in ("M3", "M2", "M1"):
        _, model, ve, _ = roundtrip(tel, pts, zk, s, r=res[s], grid_n=args.grid_n,
                                    nrad=args.nrad, naz=args.naz, nx=args.nx)
        rt[s] = (model, ve)
        print(f"   {s}: forward VE={ve:+.3f}")
    zk_ctrl = make_control(tel, pts, nx=args.nx)
    _, _, ve_ctrl, _ = roundtrip(tel, pts, zk_ctrl, "M3", grid_n=args.grid_n,
                                 nrad=args.nrad, naz=args.naz, nx=args.nx)
    print(f"   control (injected M3): forward VE={ve_ctrl:+.3f}  (must be ~1.0)")

    # 3. joint multi-surface static fit
    modes = list(range(4, args.joint_modes + 1))
    print(f"\n3. joint static-mirror fit (surface Noll Z4..Z{args.joint_modes}, {len(modes)} modes/surface):")
    for surfs in [("M3",), ("M2", "M3"), ("M1", "M2", "M3")]:
        ve, ncol, *_ = joint_fit(tel, pts, zk, surfs, modes, nx=args.nx)
        print(f"   {'+'.join(surfs):12s} ({ncol:3d} modes): VE={ve:+.3f}")

    # figures
    os.makedirs(args.outdir, exist_ok=True)
    fig_by_surface(tel, pts, res, os.path.join(args.outdir, "miw_recovered_opd_by_surface.pdf"))
    fig_roundtrip(pts, zk, rt["M3"][0], rt["M3"][1], os.path.join(args.outdir, "miw_roundtrip_m3.pdf"))
    print(f"\nwrote figures to {args.outdir}/miw_recovered_opd_by_surface.pdf, miw_roundtrip_m3.pdf")


if __name__ == "__main__":
    main()
