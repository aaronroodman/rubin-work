#!/usr/bin/env python
"""Per-mode coupling of the u-mode amplitudes into the UNREACHABLE (null) subspace.

The sharp form of the retrieval-bias test (aos/docs/miw_coadd_equations.md §4.2).  Split
each visit's 126-coefficient DZ vector w into three orthogonal pieces:

    a       = U_eff^T w        (34 amps, µm)  the CONTROLLED subspace, removed
    d       = U_disc^T w       (16 amps, µm)  reachable but discarded
    n_null  = U_null^T w       (76 amps, µm)  NOT reachable by any of the 50 DOF

A real optical state lives entirely in span(U_all) = span(U_eff) + span(U_disc), so
under an UNBIASED retrieval n_null is pure noise and carries NO information about a.
A retrieval bias B_+ sends the state out of col(S_hat), giving n_null = L_null a with
L_null a fixed 76x34 operator -- i.e. the null-space content becomes PREDICTABLE from
the u-mode amplitudes.  We estimate L_null by ridge regression of n_null on the
SIGNED a, scored leave-night-out.

CRITICAL CONTROL.  a and n_null are orthogonal projections of the same w, so they
would be independent only if the noise were isotropic.  The empirical covariance
Sigma_emp is strongly non-isotropic, so noise ALONE induces some apparent coupling.
The null distribution here is therefore built by simulation: keep each visit's
reachable part U_all U_all^T w (the real optical state) and replace the rest with
noise drawn from Sigma_emp, then rerun the identical regression.  Observed skill is
only meaningful relative to THAT null, not relative to zero.

All amplitudes are in µm of wavefront.  Correlations and R^2 are dimensionless.

  python code/analyze_umode_null_coupling.py
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_dz_goodness_of_fit import (        # noqa: E402
    load_bases, read_dz, assign_blocks, within_block_covariance)


def cv_r2_multi(X, Y, groups, lam, n_splits=5):
    """Leave-night-out ridge CV; pooled R^2 (dimensionless) over all targets."""
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    pred = np.zeros_like(Y)
    for tr, te in gkf.split(X, Y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd > 0, sd, 1.0)
        ym = Y[tr].mean(0)
        Xs = (X[tr] - mu) / sd
        B = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ (Y[tr] - ym))
        pred[te] = ((X[te] - mu) / sd) @ B + ym
    return 1.0 - np.sum((Y - pred) ** 2) / np.sum((Y - Y.mean(0)) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default="output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/"
                    "pathA_50_34_i_5rot/fits.parquet")
    ap.add_argument("--blocks-summary",
                    default="output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/"
                    "coadd_50_34/blocks_summary.parquet")
    ap.add_argument("--stable-programs", nargs="+", default=["T614_triplets"])
    ap.add_argument("--n-sim", type=int, default=100)
    ap.add_argument("--out", default="output/umode_null_coupling.pdf")
    args = ap.parse_args()

    U_eff, U_all, kj = load_bases()
    Q, _ = np.linalg.qr(U_all, mode="complete")
    U_null = Q[:, U_all.shape[1]:]                       # (126, 76) orthonormal
    U_disc = U_all[:, U_eff.shape[1]:]                   # (126, 16)
    print(f"subspaces: U_eff {U_eff.shape}, U_disc {U_disc.shape}, "
          f"U_null {U_null.shape}  (126 DZ coefficients total)")

    W, E, meta = read_dz(args.fits, kj)
    blk, prog = assign_blocks(meta, args.blocks_summary)
    blk_cov = np.where(np.isin(prog, args.stable_programs), blk, -1)
    Sigma, cdof, nb, _ = within_block_covariance(W, blk_cov, min_visits=3)
    print(f"Sigma_emp from {cdof} within-block residual dof, {nb} stable blocks")

    ok = np.isfinite(W).all(1)
    Wo = W[ok]
    day = meta["day_obs"].to_numpy()[ok]
    A = Wo @ U_eff                     # (n, 34) u-mode amplitudes, µm
    N = Wo @ U_null                    # (n, 76) null-space amplitudes, µm
    D = Wo @ U_disc                    # (n, 16) reachable-but-discarded, µm
    n = len(Wo)
    print(f"{n} visits, {len(np.unique(day))} nights")
    print(f"RMS amplitudes [µm]: ||a||={np.sqrt((A**2).sum(1)).mean():.4f}, "
          f"||d||={np.sqrt((D**2).sum(1)).mean():.4f}, "
          f"||n_null||={np.sqrt((N**2).sum(1)).mean():.4f}")

    lams = np.logspace(-1, 4, 11)
    r2 = {float(l): cv_r2_multi(A, N, day, l) for l in lams}
    lam = max(r2, key=r2.get)
    r2_obs = r2[lam]
    print(f"\nOBSERVED: CV R^2 (dimensionless) of signed a [µm] -> n_null [µm] "
          f"= {r2_obs:+.4f}  (ridge lambda={lam:g}, leave-night-out)")

    # ---- simulated null: real reachable state + noise from Sigma_emp ----
    rng = np.random.default_rng(0)
    L_chol = np.linalg.cholesky(Sigma + 1e-18 * np.eye(Sigma.shape[0]))
    W_reach = Wo @ U_all @ U_all.T                       # keep the real state
    null_r2 = []
    for _ in range(args.n_sim):
        Ws = W_reach + rng.standard_normal((n, Sigma.shape[0])) @ L_chol.T
        null_r2.append(cv_r2_multi(Ws @ U_eff, Ws @ U_null, day, lam))
    null_r2 = np.array(null_r2)
    p = float((null_r2 >= r2_obs).mean())
    print(f"NOISE NULL ({args.n_sim} sims): CV R^2 mean {null_r2.mean():+.4f}, "
          f"95th pct {np.percentile(null_r2,95):+.4f}  ->  p = {p:.3f}")
    print("  (null keeps each visit's REAL reachable state and redraws the rest from"
          " Sigma_emp, so it absorbs any coupling created by non-isotropic noise)")

    # ---- per-mode: column norm of L_null, and the simple correlation ----
    mu, sd = A.mean(0), A.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    As = (A - mu) / sd
    L = np.linalg.solve(As.T @ As + lam * np.eye(34), As.T @ (N - N.mean(0)))
    col = np.linalg.norm(L, axis=1)                      # µm of null per 1 sigma of a_m
    drop = np.array([r2_obs - cv_r2_multi(np.delete(A, m, axis=1), N, day, lam)
                     for m in range(34)])
    rn = np.sqrt((N ** 2).sum(1))
    rho = np.array([pearsonr(np.abs(A[:, m]), rn)[0] for m in range(34)])
    print(f"\n{'mode':>5} {'|L col| [µm/sigma_a]':>22} {'drop-one CV dR2':>17} "
          f"{'r(|a_m|,||n_null||)':>20}")
    for m in np.argsort(-drop)[:10]:
        print(f"u{m+1:<4d} {col[m]:22.4f} {drop[m]:+17.4f} {rho[m]:+20.3f}")

    # ---------------------------------------------------------------- figure
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with PdfPages(args.out) as pdf:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        ax[0].hist(null_r2, 25, color="0.7",
                   label=f"noise null ({args.n_sim} sims)")
        ax[0].axvline(r2_obs, color="tab:red", lw=2,
                      label=f"observed {r2_obs:+.3f}")
        ax[0].set_xlabel("leave-night-out CV $R^2$  (dimensionless)")
        ax[0].set_ylabel("simulations")
        ax[0].legend(fontsize=8)
        ax[0].set_title(f"$a$ [µm] $\\to$ $n_{{null}}$ [µm],  p = {p:.3f}")
        mi = np.arange(1, 35)
        ax[1].bar(mi, drop, color=["tab:red" if d > 0 else "0.6" for d in drop])
        ax[1].axhline(0, color="k", lw=0.6)
        ax[1].set_xlabel("u-mode index $m$")
        ax[1].set_ylabel("drop-one CV $\\Delta R^2$ (dimensionless)")
        ax[1].set_title("unique contribution of each $a_m$ to $n_{null}$")
        fig.suptitle("Does the u-mode amplitude predict the UNREACHABLE subspace?\n"
                     "unbiased retrieval $\\Rightarrow$ no skill above the noise null",
                     fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
