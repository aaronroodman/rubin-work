#!/usr/bin/env python
"""Estimate the retrieval-bias operator L from the coadd-vs-MIW residual maps.

Hypothesis (aos/MIW_COADD_EQUATIONS.md §4.2): a small bias B in the Danish donut
fit redistributes power among primary/secondary/tertiary radial orders.  The 50/34
software correction, computed from that biased wavefront, then subtracts a pattern
corresponding to no real optical state -- injecting structure rather than removing
it -- and leaves

    I_b - I_B  ⊃  R[ L . Δa ],
    L = (1-P) B_+ S_hat (U_eff^T B_+ S_hat)^+ ,

where Δa is the block's u-mode displacement from the MIW build state.  The key
property is that **L == 0 identically for an unbiased retrieval**, so a nonzero L
is direct evidence of retrieval bias, with a null hypothesis of exactly zero.

What this script does: regress the per-block residual MAPS on Δa,

    resid_b(theta, j)  =  sum_m  Δa_{b,m} * Phi_m(theta, j)  +  eps_b

solving for the field maps Phi_m (= the map-space columns of L) by ridge, and
scoring the fit with leave-night-out GroupKFold so the R^2 is honest.  Because the
unbiased model predicts Phi == 0, any cross-validated skill here is the signal.

Controls, all on the same folds and rows:
  * permutation null -- shuffle Δa across blocks (breaks the block pairing)
  * thermal-only baseline (z_gradient etc.), and thermal+Δa, to show whether Δa
    adds beyond the known thermal driver rather than proxying it
  * build blocks excluded from train AND test (they define Δa's origin)

Reads only block_grids.npz, so it runs on the laptop.

Caveat: the npz stores maps for Z5-Z8 only -- all PRIMARY aberrations.  Testing
B's radial-order structure directly needs the secondary maps (Z12/13, Z16/17)
saved too, which is an RSP-side change to run_coadd_blocks_miw.py.

  python code/analyze_miw_bias_regression.py
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recompute_coadd_metrics import (          # noqa: E402
    coarsen, coarsen_centers, umode_reference, umode_features, _block_programs)

Z_TERMS = [5, 6, 7, 8]
THERMAL = ["z_gradient", "y_gradient", "x_gradient", "radial_gradient",
           "m2_delta_t", "cam_m1m3_delta_t", "dome_delta_t", "outside_temp"]


# ---------------------------------------------------------------- regression
def ridge_fit(X, Y, lam):
    """Multi-target ridge: Phi = (X'X + lam I)^-1 X'Y.  X is centred/scaled."""
    n, p = X.shape
    G = X.T @ X + lam * np.eye(p)
    return np.linalg.solve(G, X.T @ Y)


def cv_r2(X, Y, groups, lam, n_splits=5):
    """Leave-night-out grouped CV; returns (pooled R^2, per-cell R^2, OOF pred).

    Pooled R^2 is computed over ALL cells and blocks at once -- the honest
    headline for 'do the residual maps respond linearly to Δa'."""
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    pred = np.zeros_like(Y)
    for tr, te in gkf.split(X, Y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd > 0, sd, 1.0)
        ym = Y[tr].mean(0)
        Phi = ridge_fit((X[tr] - mu) / sd, Y[tr] - ym, lam)
        pred[te] = ((X[te] - mu) / sd) @ Phi + ym
    ss_res = np.sum((Y - pred) ** 2)
    ss_tot = np.sum((Y - Y.mean(0)) ** 2)
    denom = np.sum((Y - Y.mean(0)) ** 2, axis=0)
    per_cell = 1.0 - np.sum((Y - pred) ** 2, axis=0) / np.where(denom > 0, denom, np.nan)
    return 1.0 - ss_res / ss_tot, per_cell, pred


def scan_lambda(X, Y, groups, lams):
    out = [(lam, cv_r2(X, Y, groups, lam)[0]) for lam in lams]
    best = max(out, key=lambda t: t[1])
    return best[0], out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param-set", default="fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x")
    ap.add_argument("--out-name", default="coadd_50_34")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--rebin", type=int, default=3,
                    help="coarsen the maps before regressing (cuts per-cell noise)")
    ap.add_argument("--n-modes", type=int, default=34,
                    help="how many leading u-modes to use as predictors")
    ap.add_argument("--umode-ref", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()

    cdir = os.path.join(args.output_root, args.param_set, args.out_name)
    d = np.load(os.path.join(cdir, "block_grids.npz"), allow_pickle=True)
    n = d["grids"].shape[0]
    f = args.rebin

    # ---- residual maps, coarsened, flattened over (Zernike, cell) ----
    resid = np.stack([
        np.stack([coarsen(d["grids"][i, k] - d["miw"][i, k], f)
                  for k in range(len(Z_TERMS))]) for i in range(n)])   # (n,4,ny,nx)
    flat = resid.reshape(n, -1)
    good = np.isfinite(flat).all(0) & (np.nanstd(flat, 0) > 0)
    Y = flat[:, good]
    print(f"residual maps: rebin {f}x{f} -> {resid.shape[2]}x{resid.shape[3]} per "
          f"Zernike, {int(good.sum())} usable cells x {len(Z_TERMS)} terms")

    # ---- Δa ----
    Um = np.asarray(d["umodes"], float)
    day = np.asarray(d["day_obs"]).astype(int)
    alt = np.asarray(d["alt"], float)
    band = np.asarray(d["band"]).astype(str)
    infam = np.asarray(d["in_family"]).astype(bool)
    prog = _block_programs(cdir, np.asarray(d["block"]), n)
    build = (np.isin(band, ["i"]) & np.array([p in {"T614_triplets"} for p in prog])
             & (alt >= 65) & (alt <= 75) & infam & (day <= 20260513))
    side = np.load(args.umode_ref) if args.umode_ref else None
    u_ref, sig_b, _, src = umode_reference(
        Um, build, np.asarray(d["n_visits"], float), side)
    dU, _ = umode_features(Um, u_ref, sig_b)
    nm = min(args.n_modes, dU.shape[1])
    print(f"Δa from {src}; using {nm} modes, {int(build.sum())} build blocks excluded")

    keep = ~build
    Y, X, grp = Y[keep], dU[keep, :nm], day[keep]
    tele = {k: np.asarray(d[k], float)[keep] for k in THERMAL if k in d.files}
    T = np.column_stack([v for v in tele.values()])
    ok = np.isfinite(X).all(1) & np.isfinite(T).all(1)
    Y, X, T, grp = Y[ok], X[ok], T[ok], grp[ok]
    print(f"  n={len(Y)} non-build blocks, {len(np.unique(grp))} nights, "
          f"{X.shape[1]} Δa + {T.shape[1]} thermal predictors")

    # ---- lambda scan, then the model comparison ----
    lams = np.logspace(-2, 4, 13)
    lam, scan = scan_lambda(X, Y, grp, lams)
    print(f"\nridge lambda scan (Δa only): best lam={lam:.3g}")
    for l_, r_ in scan:
        print(f"    lam={l_:9.3g}  CV R2={r_:+.4f}")

    models = {"Δa only": X, "thermal only": T, "thermal + Δa": np.column_stack([T, X])}
    res = {}
    for lab, M in models.items():
        lm, _ = scan_lambda(M, Y, grp, lams)
        r2, per_cell, _ = cv_r2(M, Y, grp, lm)
        res[lab] = (r2, per_cell, lm)
        print(f"  {lab:14s}: CV R2={r2:+.4f}  (lam={lm:.3g})")

    # ---- permutation null for Δa ----
    rng = np.random.default_rng(0)
    null = np.array([cv_r2(X[rng.permutation(len(X))], Y, grp, lam)[0]
                     for _ in range(args.n_perm)])
    r2_obs = res["Δa only"][0]
    p = float((null >= r2_obs).mean())
    print(f"\npermutation null ({args.n_perm} shuffles of Δa): "
          f"mean {null.mean():+.4f}, 95th {np.percentile(null,95):+.4f}; "
          f"observed {r2_obs:+.4f}  ->  p={p:.3f}")

    # ---- full-sample Phi (the map-space columns of L) ----
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Phi = ridge_fit((X - mu) / sd, Y - Y.mean(0), lam)          # (nm, ncell)
    power = np.linalg.norm(Phi, axis=1)
    order = np.argsort(-power)
    print("\nper-mode |Phi_m| (map response per 1 sigma of Δa_m), top 10:")
    for i in order[:10]:
        print(f"    mode u{i+1:<3d} |Phi|={power[i]:.4f}")

    # ---- drop-one-mode CV contribution ----
    contrib = []
    for m in range(nm):
        cols = [c for c in range(nm) if c != m]
        contrib.append(r2_obs - cv_r2(X[:, cols], Y, grp, lam)[0])
    contrib = np.array(contrib)
    top = np.argsort(-contrib)
    print("\ndrop-one-mode CV R2 loss (positive = mode carries unique signal):")
    for i in top[:8]:
        print(f"    u{i+1:<3d} ΔR2={contrib[i]:+.4f}")

    # ---------------------------------------------------------------- figures
    ny, nx = resid.shape[2], resid.shape[3]
    cx = coarsen_centers(0.5 * (d["xbins"][:-1] + d["xbins"][1:]), f)
    cy = coarsen_centers(0.5 * (d["ybins"][:-1] + d["ybins"][1:]), f)
    out = args.out or os.path.join(cdir, "miw_bias_regression.pdf")

    def unflat(vec):
        """cell vector -> (4, ny, nx) with NaN where masked"""
        full = np.full(len(Z_TERMS) * ny * nx, np.nan)
        full[good] = vec
        return full.reshape(len(Z_TERMS), ny, nx)

    with PdfPages(out) as pdf:
        # page 1: model comparison + null
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
        labs = list(res)
        a1.bar(np.arange(len(labs)), [res[k][0] for k in labs],
               color=["tab:purple", "0.5", "tab:blue"])
        a1.axhline(0, color="k", lw=0.6)
        a1.set_xticks(np.arange(len(labs))); a1.set_xticklabels(labs, fontsize=9)
        a1.set_ylabel("leave-night-out CV R$^2$ of the residual MAPS")
        a1.set_title("Does the residual map respond linearly to $\\Delta a$?")
        a1.grid(alpha=0.3, axis="y")
        a2.hist(null, 30, color="0.7", label=f"permutation null (n={args.n_perm})")
        a2.axvline(r2_obs, color="tab:purple", lw=2, label=f"observed {r2_obs:+.3f}")
        a2.axvline(np.percentile(null, 95), color="k", ls="--", lw=1, label="null 95th")
        a2.set_xlabel("CV R$^2$"); a2.legend(fontsize=8)
        a2.set_title(f"$\\Delta a$ permutation test, p={p:.3f}")
        fig.suptitle("Retrieval-bias test: the unbiased model predicts $L\\equiv0$, "
                     "so any CV skill here is bias signal\n"
                     f"{len(Y)} non-build blocks, {len(np.unique(grp))} nights, "
                     f"rebin {f}x{f}", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # page 2: per-mode response power + drop-one contribution
        fig, (b1, b2) = plt.subplots(2, 1, figsize=(max(11, 0.3 * nm + 5), 8))
        mi = np.arange(1, nm + 1)
        b1.bar(mi, power, color="tab:purple")
        b1.set_ylabel("$|\\Phi_m|$  (map response per 1$\\sigma$ $\\Delta a_m$)")
        b1.set_title("Estimated columns of $L$ in map space -- magnitude per u-mode")
        b1.grid(alpha=0.3, axis="y"); b1.set_xticks(mi)
        b1.set_xticklabels(mi, fontsize=7)
        b2.bar(mi, contrib, color=["tab:red" if c > 0 else "0.6" for c in contrib])
        b2.axhline(0, color="k", lw=0.6)
        b2.set_ylabel("drop-one CV $\\Delta R^2$"); b2.set_xlabel("u-mode index")
        b2.set_title("Unique cross-validated contribution of each mode "
                     "(guards against the collinearity of $\\Delta a$)")
        b2.grid(alpha=0.3, axis="y"); b2.set_xticks(mi)
        b2.set_xticklabels(mi, fontsize=7)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # pages 3+: Phi_m maps for the leading modes
        for i in order[:6]:
            M = unflat(Phi[i])
            fig, ax = plt.subplots(1, len(Z_TERMS), figsize=(4.1 * len(Z_TERMS), 4.3))
            vmax = np.nanpercentile(np.abs(M), 99) or 1.0
            for c, z in enumerate(Z_TERMS):
                s = ax[c].pcolormesh(cx, cy, M[c].T, cmap="RdBu_r",
                                     vmin=-vmax, vmax=vmax)
                ax[c].set_aspect("equal"); ax[c].set_title(f"Z{z}", fontsize=10)
                plt.colorbar(s, ax=ax[c], fraction=0.046)
            fig.suptitle(f"$\\Phi_{{{i+1}}}$: residual-map response to u-mode "
                         f"{i+1} displacement  ($|\\Phi|$={power[i]:.4f}, "
                         f"drop-one $\\Delta R^2$={contrib[i]:+.4f})\n"
                         "under an unbiased retrieval this map would be zero",
                         fontsize=11)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # last page: per-cell CV R2 map for the Δa model
        pc = unflat(res["Δa only"][1])
        fig, ax = plt.subplots(1, len(Z_TERMS), figsize=(4.1 * len(Z_TERMS), 4.3))
        for c, z in enumerate(Z_TERMS):
            s = ax[c].pcolormesh(cx, cy, pc[c].T, cmap="viridis", vmin=0,
                                 vmax=max(0.05, np.nanpercentile(pc, 98)))
            ax[c].set_aspect("equal"); ax[c].set_title(f"Z{z}", fontsize=10)
            plt.colorbar(s, ax=ax[c], fraction=0.046)
        fig.suptitle("Per-cell leave-night-out CV R$^2$ from $\\Delta a$ alone -- "
                     "where in the field the residual is predicted by the u-modes",
                     fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
