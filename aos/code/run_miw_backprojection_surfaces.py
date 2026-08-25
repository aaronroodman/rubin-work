#!/usr/bin/env python
"""Which optical surface sources the MIW?  Back-project + fit the measured
intrinsic wavefront onto EVERY optic: M1, M2, M3, L1, L2, Filter, L3.

Extends run_m3_backprojection_miw.py from the three mirrors to the full column,
with the refractive elements handled as per-LENS UNITS (a lens's entrance+exit
surfaces are fit jointly).  Everything is i-band to match the MIW data.

Why more than the mirrors: a surface imprints FIELD-DEPENDENT aberrations only
where the beam footprint walks across it with field angle.  The footprint
walk/size grows monotonically down the column (M1~0.1 pupil-like -> M3~0.8 ->
L1~1.9 -> ... -> L3~20 focal-plane-like).  Two consequences:
  * the sub-aperture STITCH needs footprint overlap, so its consistency VE is
    trustworthy for M3/L1 (and marginally L2) but DEGENERATE for Filter/L3, where
    near-disjoint footprints let a static figure mimic almost any field pattern;
  * so the honest metric is the batoid least-squares FIT VE swept vs the number
    of surface Zernike modes -- a physical source reaches high VE at LOW order,
    an over-flexible near-focal-plane fit only as modes blow up.

Outputs (--outdir, default output/):
  miw_surfaces_footprints.pdf     beam footprints at several field points on each
                                  optic, over its clear aperture (geometry check)
  miw_surfaces_stitched_opd.pdf   per-surface stitched OPD + per-FoV consistency
  miw_surfaces_fit_ve.pdf         fit VE vs #modes, all units; + injection control
  miw_surfaces_azimuthal.pdf      recovered OPD split into m=0 vs m!=0

Injection control per surface (inject a known low-order figure, forward-model,
back-project+stitch, correlate with truth) doubles as the stitch-fidelity gauge:
corr~1 where the stitch is well-posed, dropping toward the focal plane.

Usage:
  python code/run_miw_backprojection_surfaces.py --stride 8 --nx 48       # full (RSP)
  python code/run_miw_backprojection_surfaces.py --stride 24 --nx 28 \
         --max-mode 15                                                     # fast (laptop)
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

JS = [j for j in range(4, 27) if j not in (20, 21)]     # FAM Noll set
DEFAULT_MIW = ("output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/"
               "pathA_50_34_i_5rot/intrinsic_split_maps.parquet")

# per-optic UNITS: a lens is its two surfaces fit jointly; the first surface is
# the representative geometry used for the stitch / footprint / azimuthal maps.
UNITS = {
    "M1": ["M1"], "M2": ["M2"], "M3": ["M3"],
    "L1": ["L1_entrance", "L1_exit"], "L2": ["L2_entrance", "L2_exit"],
    "Filter": ["Filter_entrance", "Filter_exit"], "L3": ["L3_entrance", "L3_exit"],
}
ORDER = ["M1", "M2", "M3", "L1", "L2", "Filter", "L3"]


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


# ---------------- batoid response matrix + mode-swept fit VE ----------------
def response_matrix(tel, pts, surfaces, modes, nx):
    """Columns = field-Zernike response [µm, JS-raveled] to a unit (1e-7 m)
    surface-Zernike on each (surface, mode).  Built once per unit; the mode sweep
    then fits nested column subsets.  Returns (A, ids=[(surface, mode)])."""
    ptl = list(map(tuple, pts))
    z0 = bp.field_zernikes(tel, ptl, jmax=26, nx=nx)
    cols, ids = [], []
    for s in surfaces:
        Ro, Ri = bp.surface_aperture(tel, s)
        for k in modes:
            c = np.zeros(27); c[k] = 1e-7
            telp = bp.perturb_surface(tel, s, c, R_outer=Ro, R_inner=Ri)
            dz = (bp.field_zernikes(telp, ptl, jmax=26, nx=nx) - z0) \
                * bp.WAVELENGTH * 1e6 / 1e-7
            cols.append(dz[:, JS].ravel()); ids.append((s, k))
    return np.column_stack(cols), ids


def ve_vs_modes(A, ids, dvec, caps):
    """Fit VE for nested surface-Zernike mode caps (fit all cols with mode<=cap)."""
    out = []
    for M in caps:
        sel = [i for i, (s, k) in enumerate(ids) if k <= M]
        if not sel:
            continue
        c, *_ = np.linalg.lstsq(A[:, sel], dvec, rcond=None)
        ve = 1 - np.var(dvec - A[:, sel] @ c) / np.var(dvec)
        out.append((M, len(sel), ve))
    return out


def injection_stitch_fidelity(tel, pts, surface, grid_n, nrad, naz, nx):
    """Inject a known low-order figure on `surface`, forward-model the field
    Zernikes, back-project+stitch, and correlate the stitched map with the truth
    figure over the aperture.  corr~1 => the stitch geometry is well-posed here."""
    Ro, Ri = bp.surface_aperture(tel, surface)
    coef = np.zeros(27); coef[5] = 0.2e-6; coef[7] = 0.12e-6; coef[11] = 0.15e-6
    telp = bp.perturb_surface(tel, surface, coef, R_outer=Ro, R_inner=Ri)
    ptl = list(map(tuple, pts))
    zk_inj = (bp.field_zernikes(telp, ptl, jmax=26, nx=nx)
              - bp.field_zernikes(tel, ptl, jmax=26, nx=nx)) * bp.WAVELENGTH * 1e6
    r = bp.run_backprojection(tel, pts, zk_inj, surface=surface,
                              grid_n=grid_n, nrad=nrad, naz=naz)
    X, Y, glob, mask = r["X"], r["Y"], r["glob"], r["mask"]
    m = np.isfinite(glob) & mask
    truth = gz.Zernike(coef * 1e6, R_inner=Ri, R_outer=Ro).evalCartesian(X, Y)
    a = bp.remove_plane_masked(glob, X, Y, m)[m]
    b = bp.remove_plane_masked(truth, X, Y, m)[m]
    corr = np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
    return float(corr)


def azimuthal_m0_fraction(r, nrb=24):
    """Fraction of the stitched-OPD variance that is azimuthally symmetric (m=0)."""
    X, Y, glob, mask = r["X"], r["Y"], r["glob"], r["mask"]
    m = np.isfinite(glob) & mask
    g = bp.remove_plane_masked(glob, X, Y, m)
    rr = np.hypot(X, Y)
    rb = np.linspace(rr[m].min(), rr[m].max(), nrb + 1)
    idx = np.digitize(rr, rb)
    m0 = np.full_like(g, np.nan)
    for b in range(1, len(rb) + 1):
        sel = m & (idx == b)
        if sel.sum() > 3:
            m0[sel] = np.nanmean(g[sel])
    tot = np.nanvar(g[m])
    return float(np.nanvar(m0[m]) / tot) if tot > 0 else np.nan, (X, Y, g, m0, m)


# ---------------------------------- figures ----------------------------------
def fig_footprints(tel, field_pts, out):
    surfs = [(n, UNITS[n][0]) for n in ORDER]
    nc = 4; nr = int(np.ceil(len(surfs) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 4.2 * nr))
    axes = np.atleast_1d(axes).ravel()
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(field_pts)))
    th = np.linspace(0, 2 * np.pi, 200)
    for ax, (name, s) in zip(axes, surfs):
        Ro, Ri = bp.surface_aperture(tel, s)
        ax.plot(Ro * np.cos(th), Ro * np.sin(th), "k-", lw=1.2)
        if Ri > 0:
            ax.plot(Ri * np.cos(th), Ri * np.sin(th), "k-", lw=1.2)
        for (fx, fy), c in zip(field_pts, colors):
            _, _, mx, my, _ = bp.pupil_to_surface(tel, s, fx, fy, nrad=7, naz=36)
            ax.scatter(mx, my, s=4, color=c, alpha=0.5,
                       label=f"({fx:+.1f},{fy:+.1f})°")
        ax.set_aspect("equal"); ax.set_title(f"{name}  ({s})", fontsize=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    for ax in axes[len(surfs):]:
        ax.axis("off")
    axes[0].legend(fontsize=7, loc="upper right", markerscale=2, title="field")
    fig.suptitle("Beam footprints per optic at several field points "
                 "(overlap = well-posed stitch; disjoint = focal-plane-like)",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def fig_stitched(tel, pts, res, out, order=ORDER):
    fig, ax = plt.subplots(len(order), 3, figsize=(15, 4.5 * len(order)))
    ax = np.atleast_2d(ax)
    for row, name in enumerate(order):
        r = res[name]["r"]; s = UNITS[name][0]
        X = r["X"]; Ro, _ = bp.surface_aperture(tel, s); ext = [-Ro, Ro, -Ro, Ro]
        g = r["glob"]; ve = r["con"]["var_explained"]; rms = r["con"]["rms"]
        fin = np.isfinite(g)
        vm = np.nanpercentile(np.abs(g[fin]), 98) if fin.any() else 1.0
        im = ax[row, 0].imshow(g, origin="lower", extent=ext, cmap="RdBu_r",
                               vmin=-vm, vmax=vm)
        ax[row, 0].set_title(f"{name}: stitched OPD [µm]  medVE={np.nanmedian(ve):+.2f}"
                             f"  m0={res[name]['m0']:.2f}", fontsize=10)
        ax[row, 0].set_xlabel(f"{s} x [m]"); plt.colorbar(im, ax=ax[row, 0], shrink=.8)
        sc = ax[row, 1].scatter(pts[:, 0], pts[:, 1], c=ve, s=22, cmap="viridis",
                                vmin=0, vmax=1)
        ax[row, 1].set_title(f"{name}: consistency VE vs field"); ax[row, 1].set_aspect("equal")
        ax[row, 1].set_xlabel("field x [deg]"); plt.colorbar(sc, ax=ax[row, 1], shrink=.8)
        vr = np.nanpercentile(rms[np.isfinite(rms)], 98) if np.isfinite(rms).any() else 1
        sc2 = ax[row, 2].scatter(pts[:, 0], pts[:, 1], c=rms, s=22, cmap="magma_r",
                                 vmin=0, vmax=vr)
        ax[row, 2].set_title(f"{name}: residual RMS [µm]"); ax[row, 2].set_aspect("equal")
        ax[row, 2].set_xlabel("field x [deg]"); plt.colorbar(sc2, ax=ax[row, 2], shrink=.8)
    fig.suptitle("MIW back-projected + stitched on each optic "
                 "(stitch trustworthy only where footprints overlap)", fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def fig_fit_ve(res, out, order=ORDER):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))
    cmap = plt.cm.turbo(np.linspace(0.05, 0.95, len(order)))
    for name, c in zip(order, cmap):
        sw = res[name]["sweep"]
        a1.plot([m for m, n, v in sw], [v for m, n, v in sw], "o-", color=c, label=name)
    a1.set_xlabel("max surface Zernike mode in fit"); a1.set_ylabel("fit VE (MIW variance explained)")
    a1.set_title("static-figure fit VE vs #modes, per optic"); a1.grid(alpha=0.3)
    a1.legend(fontsize=9); a1.axhline(0, color="k", lw=0.5)
    xi = np.arange(len(order))
    a2.bar(xi, [abs(res[n]["icorr"]) for n in order], color=cmap)  # |corr|: shape fidelity
    a2.set_xticks(xi); a2.set_xticklabels(order); a2.set_ylim(0, 1.05)
    a2.set_ylabel("injection→stitch |corr|"); a2.grid(alpha=0.3, axis="y")
    a2.set_title("stitch fidelity (injected known figure recovered)")
    fig.suptitle("Which optic can source the MIW: fit VE (left) is meaningful at "
                 "LOW mode order; right shows where the stitch is trustworthy",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def fig_azimuthal(res, out, order=ORDER):
    fig, ax = plt.subplots(len(order), 3, figsize=(13, 3.9 * len(order)))
    ax = np.atleast_2d(ax)
    for row, name in enumerate(order):
        X, Y, g, m0, m = res[name]["azpack"]
        ext = [X.min(), X.max(), Y.min(), Y.max()]
        gg = np.where(m, g, np.nan); mm0 = np.where(m, m0, np.nan)
        res_map = np.where(m, g - m0, np.nan)
        fin = np.isfinite(gg)
        vm = np.nanpercentile(np.abs(gg[fin]), 98) if fin.any() else 1.0
        for col, (M, ttl) in enumerate([(gg, "recovered OPD"),
                                        (mm0, f"m=0 part ({res[name]['m0']:.0%})"),
                                        (res_map, "m≠0 residual")]):
            im = ax[row, col].imshow(M, origin="lower", extent=ext, cmap="RdBu_r",
                                     vmin=-vm, vmax=vm)
            ax[row, col].set_title(f"{name}: {ttl}", fontsize=9)
            plt.colorbar(im, ax=ax[row, col], shrink=.75)
    fig.suptitle("Recovered OPD split into azimuthally-symmetric (m=0) vs m≠0",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--miw", default=DEFAULT_MIW)
    ap.add_argument("--band", default="i")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--grid-n", type=int, default=110)
    ap.add_argument("--nrad", type=int, default=22)
    ap.add_argument("--naz", type=int, default=88)
    ap.add_argument("--nx", type=int, default=48, help="batoid.zernike grid for forward models")
    ap.add_argument("--max-mode", type=int, default=22, help="max surface Noll mode in the fit")
    ap.add_argument("--mode-caps", type=int, nargs="+", default=[6, 11, 15, 22, 28])
    ap.add_argument("--only", nargs="+", default=None, help="subset of optics (e.g. M3 L1)")
    ap.add_argument("--no-fit", action="store_true", help="skip the (slow) response-matrix fit")
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()

    order = args.only or ORDER
    tel = bp.load_telescope(args.band)
    pts, zk = load_miw(args.miw, args.stride)
    dvec = zk[:, JS].ravel()
    print(f"MIW: {args.miw}\n  {len(pts)} field points (stride {args.stride}), "
          f"band {args.band}, frame OCS==batoid (identity)")

    field_pts = [(0.0, 0.0), (1.7, 0.0), (-1.7, 0.0), (0.0, 1.7), (0.0, -1.7),
                 (1.2, 1.2), (-1.2, -1.2)]
    os.makedirs(args.outdir, exist_ok=True)
    fig_footprints(tel, field_pts, os.path.join(args.outdir, "miw_surfaces_footprints.pdf"))
    print("  wrote miw_surfaces_footprints.pdf")

    caps = [c for c in args.mode_caps if c <= args.max_mode]
    modes = list(range(4, args.max_mode + 1))
    res = {}
    print(f"\n{'optic':7s} {'stitch_medVE':>12s} {'inj_corr':>9s} {'m0_frac':>8s} "
          f"{'fitVE@lowest':>12s} {'fitVE@max':>10s}")
    for name in order:
        surfs = UNITS[name]; rep = surfs[0]
        r = bp.run_backprojection(tel, pts, zk, surface=rep, grid_n=args.grid_n,
                                  nrad=args.nrad, naz=args.naz)
        medve = float(np.nanmedian(r["con"]["var_explained"]))
        icorr = injection_stitch_fidelity(tel, pts, rep, args.grid_n, args.nrad,
                                          args.naz, args.nx)
        m0, azpack = azimuthal_m0_fraction(r)
        sweep = []
        if not args.no_fit:
            A, ids = response_matrix(tel, pts, surfs, modes, args.nx)
            sweep = ve_vs_modes(A, ids, dvec, caps)
        res[name] = dict(r=r, con_medve=medve, icorr=icorr, m0=m0, azpack=azpack,
                         sweep=sweep)
        lo = f"{sweep[0][2]:+.2f}@Z{sweep[0][0]}" if sweep else "--"
        hi = f"{sweep[-1][2]:+.2f}@Z{sweep[-1][0]}" if sweep else "--"
        print(f"{name:7s} {medve:12.3f} {icorr:9.3f} {m0:8.2f} {lo:>12s} {hi:>10s}")

    fig_stitched(tel, pts, res, os.path.join(args.outdir, "miw_surfaces_stitched_opd.pdf"), order)
    fig_azimuthal(res, os.path.join(args.outdir, "miw_surfaces_azimuthal.pdf"), order)
    if not args.no_fit:
        fig_fit_ve(res, os.path.join(args.outdir, "miw_surfaces_fit_ve.pdf"), order)
    print(f"\nwrote figures to {args.outdir}/miw_surfaces_*.pdf")


if __name__ == "__main__":
    main()
