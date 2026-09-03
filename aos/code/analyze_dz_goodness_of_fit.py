#!/usr/bin/env python
"""Per-visit goodness-of-fit of the Double-Zernike fit against the sensitivity matrix.

The build fits n_j x n_k = 21 x 6 = **126** DZ coefficients per visit but only ever
uses them to constrain 34 v-modes, so the measurement is massively overdetermined.
That redundancy is testable: is each visit's 126-vector w actually consistent with
being the response of SOME physical DOF state?

Two nested null hypotheses (aos/MIW_COADD_EQUATIONS.md §2):

  chi2_50 : w = S_hat q  for some q in R^50  -- consistent with ANY reachable state.
            dof = 126 - 50 = 76.  The residual lives in null(S_hat^T), which NO
            combination of the 50 DOF can produce.
  chi2_34 : w = U_eff c   for some c in R^34 -- consistent with the CONTROLLED
            subspace only.  dof = 126 - 34 = 92.

and their difference (dof 16) is the power in the reachable-but-discarded v-modes
u_35..50 -- real optical states the truncation declines to correct.  chi2_50 is the
honest "is this consistent with the sensitivity matrix"; chi2_34 additionally
penalises physically real states, so it is the secondary number.

**THE KEY TEST.** Under an unbiased retrieval, chi2_50 is pure noise and therefore
INDEPENDENT of the state amplitude ||a|| = ||U_eff^T w||.  Under a retrieval bias
(§4.2) the measured vector is B_+ W, which leaves col(S_hat), so the null-space
residual grows in proportion to the state itself: chi2_50 should rise with ||a||^2.
That is a per-VISIT version of the block-level discriminator, with ~1126 points
instead of 221 -- and it needs no rerun.  It is also robust to a constant
mis-calibration of the fit errors, which shifts chi2/dof but not the SLOPE.

Per-pupil-j and per-radial-order-family breakdowns of the excess say WHICH Zernikes
are inconsistent, i.e. the structure of B.

Reference intrinsic: fits.parquet from a normal build is the fit against the
converged MIW (the final iteration).  To get the BATOID-intrinsic (BIW) version for
comparison, rerun the build with `--n-iter 1` -- iteration 1 fits against the batoid
tabulated intrinsic by construction -- and pass it as --fits-biw.  NOTE the MIW case
is partly self-fulfilling for the BUILD visits (the MIW absorbed exactly their mean
non-DOF structure), so the honest comparison is on the NON-build visits; the script
splits them.

  python code/analyze_dz_goodness_of_fit.py
  python code/analyze_dz_goodness_of_fit.py --fits-biw output/<ps>/<biw_build>/fits.parquet
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

_PKG = "/Users/roodman/Astrophysics/Claude/packages"
K_MIN, K_MAX, N_KEEP = 1, 6, 34
ZK = [z for z in range(4, 27) if z not in (20, 21)]
# azimuthal families -> radial orders, for the B-structure breakdown
FAMILIES = {
    "astig m=2":  {"1st": (6, 5), "2nd": (12, 13), "3rd": (24, 23)},
    "coma m=1":   {"1st": (8, 7), "2nd": (16, 17)},
    "trefoil m=3": {"1st": (10, 9), "2nd": (18, 19)},
    "tetra m=4":  {"1st": (14, 15), "2nd": (26, 25)},
    "spher m=0":  {"1st": (11, None), "2nd": (22, None)},
    "focus":      {"1st": (4, None)},
}


def load_bases(instrument="lsst", pkg=_PKG):
    """(U_eff, U_all, kj_grid) for the k<=6 x 21-pupil x 50-DOF slice."""
    if pkg + "/ts_ofc/python" not in sys.path:
        sys.path.insert(0, pkg + "/ts_ofc/python")
    from lsst.ts.ofc import OFCData
    import importlib.util as iu
    spec = iu.spec_from_file_location(
        "ofc_svd", pkg + "/ts_intrinsic_wavefront/python/lsst/ts/intrinsic/"
        "wavefront/ofc_svd.py")
    osv = iu.module_from_spec(spec)
    sys.modules["ofc_svd"] = osv
    spec.loader.exec_module(osv)
    svd = osv.build_ofc_svd(ZK, K_MIN, K_MAX, N_KEEP, instrument=instrument)
    S_full = np.asarray(OFCData(instrument).sensitivity_matrix)
    S_slab = S_full[K_MIN:K_MAX + 1, np.asarray(ZK, int), :]
    Sh = (S_slab.reshape(-1, S_slab.shape[-1])[:, svd.dof_idx]
          @ np.diag(svd.normalization_weights))
    U_all = np.linalg.svd(np.nan_to_num(np.asarray(Sh, float)),
                          full_matrices=False)[0]
    return np.asarray(svd.U_eff, float), U_all, list(svd.kj_grid)


def read_dz(path, kj_grid, prefix="z1toz6"):
    """(W, E, meta) -- per-visit DZ coefficients and their errors in kj_grid order."""
    df = pd.read_parquet(path)
    n = len(df)
    W = np.full((n, len(kj_grid)), np.nan)
    E = np.full((n, len(kj_grid)), np.nan)
    for c, (k, j) in enumerate(kj_grid):
        cv, ce = f"{prefix}_z{j}_c{k}", f"{prefix}_z{j}_c{k}_err"
        if cv in df.columns:
            W[:, c] = df[cv].to_numpy(float)
        if ce in df.columns:
            E[:, c] = df[ce].to_numpy(float)
    keep = [c for c in ("day_obs", "seq_num", "visit", "rotator_angle",
                        "n_donuts_1", "visit_quality_pass", "rotator_flagged",
                        "z_gradient") if c in df.columns]
    return W, E, df[keep].copy()


def assign_blocks(meta, blocks_summary):
    """(block id, program) per visit, by (day_obs, seq_min<=seq_num<=seq_max).

    Block id -1 / program "" = unmatched.  The program matters: the S-matrix test
    programs (T377/T378/T379/T380/T381) deliberately STEP the optical state inside
    a block, so their within-block scatter is real DOF motion, not noise, and they
    must be excluded from the error estimate.  T614_triplets are the stable blocks."""
    bs = pd.read_parquet(blocks_summary)
    blk = np.full(len(meta), -1, int)
    prog = np.full(len(meta), "", dtype=object)
    dob = meta["day_obs"].to_numpy(int)
    snm = meta["seq_num"].to_numpy(int)
    for _, r in bs.iterrows():
        m = ((dob == int(r["day_obs"])) & (snm >= int(r["seq_min"]))
             & (snm <= int(r["seq_max"])))
        blk[m] = int(r["block"])
        prog[m] = str(r["program"]).replace("BLOCK-", "")
    return blk, prog


def _shrink_cov(D, label=""):
    """Covariance of centred residuals D, shrinking ONLY the correlation matrix.

    Ledoit-Wolf shrinks toward a SCALED IDENTITY, which is wrong here: the 126 DZ
    coefficients span ~30x in sigma (Z5 ~1.7e-3, Z16 ~3e-4), so identity shrinkage
    leaks the large variances into the small ones and corrupts both the quoted
    errors and the chi2.  Instead: keep the empirical variances exactly, standardize,
    shrink the CORRELATION matrix toward the identity, then rescale."""
    n = D.shape[0]
    v = np.maximum((D ** 2).mean(0), 1e-300)
    d = np.sqrt(v)
    Z = D / d[None, :]
    R = (Z.T @ Z) / n
    from sklearn.covariance import LedoitWolf
    lam = float(LedoitWolf(assume_centered=True).fit(Z).shrinkage_)
    R = (1.0 - lam) * R + lam * np.eye(R.shape[0])
    print(f"  correlation shrinkage{(' ' + label) if label else ''}: "
          f"lambda={lam:.3f} ({n} residuals, {D.shape[1]} dims)")
    return (R * d[None, :]) * d[:, None]


def adjacent_pair_covariance(W, blk, seq, shrink=True):
    """Noise covariance from CONSECUTIVE-visit differences within a block.

    Cross-check on the block-mean estimator: differencing consecutive visits
    (minutes apart) suppresses any slow real drift while keeping the full
    turbulence + retrieval noise.  Divided by sqrt(2) because a difference of two
    noisy measurements carries twice the variance.  If this agrees with the
    block-mean estimator, the block-mean scatter really is stationary noise."""
    rows = []
    for b in np.unique(blk[blk >= 0]):
        idx = np.where(blk == b)[0]
        idx = idx[np.argsort(np.asarray(seq)[idx])]
        idx = [i for i in idx if np.isfinite(W[i]).all()]
        if len(idx) >= 2:
            rows.append(np.diff(W[idx], axis=0) / np.sqrt(2.0))
    if not rows:
        raise SystemExit("no block had two consecutive usable visits")
    D = np.vstack(rows)
    Sig = _shrink_cov(D, "(adjacent-pair)") if shrink else (D.T @ D) / D.shape[0]
    return Sig, D.shape[0], len(rows), D


def within_block_covariance(W, blk, min_visits=3, shrink=True):
    """EMPIRICAL covariance of the 126 DZ coefficients from within-block scatter.

    Inside one T614 block (~6-12 triplets at a fixed pointing/rotator, minutes
    apart) the optical state is essentially constant, so the visit-to-visit
    scatter of w is a direct measurement of the coefficient error -- atmosphere,
    dome seeing, retrieval noise and their SPATIAL CORRELATIONS all included, at
    the right scale and with no modelling.  This is the quantity the RLM formal
    errors get wrong: those divide the per-donut scale by sqrt(N_donuts), i.e.
    assume the ~2800 donuts are independent samples, which turbulence violates.

    Pooled estimator with the correct divisor sum_b (m_b - 1); Ledoit-Wolf
    shrinkage keeps the 126x126 inverse well behaved at ~10 samples per dimension.

    CAVEAT: a block-CONSTANT non-DOF component of w (e.g. an imperfect intrinsic)
    is invisible to this estimator by construction, so chi2 will still flag it --
    correctly, but that failure mode is not turbulence.

    Returns (Sigma, n_resid, n_blocks, D) with D the stacked residuals."""
    rows, nb = [], 0
    for b in np.unique(blk):
        if b < 0:
            continue
        m = blk == b
        Wb = W[m]
        good = np.isfinite(Wb).all(1)
        if int(good.sum()) < min_visits:
            continue
        Wb = Wb[good]
        rows.append(Wb - Wb.mean(0))          # m_b residuals, spanning m_b - 1 dims
        nb += 1
    if not rows:
        raise SystemExit("no block had enough visits for a within-block covariance")
    D = np.vstack(rows)
    dof = D.shape[0] - nb                      # sum_b (m_b - 1)
    # rescale from the naive 1/n to the correct 1/sum_b (m_b - 1)
    fac = D.shape[0] / max(dof, 1)
    Sigma = (_shrink_cov(D, "(block-mean)") * fac if shrink
             else (D.T @ D) / max(dof, 1))
    return Sigma, dof, nb, D


def gls_chi2(W, Sigma, U_eff, U_all):
    """Per-visit GLS chi2 against the 50-DOF and 34-v-mode subspaces, using the
    empirical covariance.  Whitens with the Cholesky factor, then projects."""
    from scipy.linalg import solve_triangular
    L = np.linalg.cholesky(Sigma)
    out = {}
    Aw = {lab: solve_triangular(L, B, lower=True)
          for lab, B in (("50", U_all), ("34", U_eff))}
    n, m = W.shape
    for lab in ("50", "34"):
        A = Aw[lab]
        chi = np.full(n, np.nan)
        for i in range(n):
            if not np.isfinite(W[i]).all():
                continue
            y = solve_triangular(L, W[i], lower=True)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = y - A @ coef
            chi[i] = float(r @ r)
        out[f"chi2_{lab}"] = chi
        out[f"dof_{lab}"] = np.full(n, m - A.shape[1], float)
    return out


def gof(W, E, U_eff, U_all, floor_frac=0.05):
    """Per-visit weighted chi2 against the two nested subspace models.

    Weighted (GLS) residual against a model subspace spanned by ``B``:
        chi2 = || (1 - Pi_w) D w ||^2,  D = diag(1/sigma),  Pi_w = proj onto col(D B)
    Rows with NaN coefficient or error are dropped for that visit (dof adjusted).
    Error floor: sigma is floored at floor_frac of the per-visit median sigma, so a
    single spuriously tiny RLM error cannot dominate the chi2."""
    n, m = W.shape
    out = {k: np.full(n, np.nan) for k in
           ("chi2_50", "dof_50", "chi2_34", "dof_34", "a_norm",
            "r_null", "r_disc", "r_tot")}
    per_j = np.full((n, m), np.nan)          # weighted sq residual per coefficient
    for i in range(n):
        w, e = W[i], E[i]
        ok = np.isfinite(w) & np.isfinite(e) & (e > 0)
        if ok.sum() < 60:
            continue
        s = e[ok]
        s = np.maximum(s, floor_frac * np.median(s))
        y = w[ok] / s
        res = {}
        for lab, B, nb in (("50", U_all[ok], U_all.shape[1]),
                           ("34", U_eff[ok], U_eff.shape[1])):
            A = B / s[:, None]
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = y - A @ coef
            res[lab] = r
            out[f"chi2_{lab}"][i] = float(r @ r)
            out[f"dof_{lab}"][i] = int(ok.sum() - nb)
        per_j[i, ok] = res["50"] ** 2
        # unweighted decomposition -- what actually survives into the coadd
        wf = np.where(np.isfinite(w), w, 0.0)
        a = U_eff.T @ wf
        rr = wf - U_eff @ a
        rn = rr - U_all @ (U_all.T @ rr)
        out["a_norm"][i] = np.linalg.norm(a)
        out["r_tot"][i] = np.linalg.norm(rr)
        out["r_null"][i] = np.linalg.norm(rn)
        out["r_disc"][i] = np.sqrt(max(out["r_tot"][i] ** 2 - out["r_null"][i] ** 2, 0))
    return out, per_j


def slope_test(a, chi2, dof):
    """Regress chi2/dof on ||a||^2 -- the bias signature.  Returns (slope, r, p)."""
    from scipy.stats import pearsonr
    x, y = np.asarray(a) ** 2, np.asarray(chi2) / np.asarray(dof)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 20:
        return np.nan, np.nan, np.nan
    r, p = pearsonr(x[ok], y[ok])
    sl = np.polyfit(x[ok], y[ok], 1)[0]
    return float(sl), float(r), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", default="output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/"
                    "pathA_50_34_i_5rot/fits.parquet",
                    help="build fits.parquet -- the MIW-referenced per-visit DZ fit")
    ap.add_argument("--fits-biw", default=None,
                    help="second fits.parquet fit against the BATOID intrinsic "
                    "(produce it by rerunning the build with --n-iter 1)")
    ap.add_argument("--instrument", default="lsst")
    ap.add_argument("--blocks-summary",
                    default="output/fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x/"
                    "coadd_50_34/blocks_summary.parquet",
                    help="block table used to build the EMPIRICAL within-block "
                    "coefficient covariance (the turbulence-aware error model)")
    ap.add_argument("--min-visits", type=int, default=3,
                    help="minimum visits in a block to contribute within-block residuals")
    ap.add_argument("--stable-programs", nargs="+", default=["T614_triplets"],
                    help="ONLY these programs contribute to the empirical covariance. "
                    "The S-matrix test programs (T377/T378/T379/T380/T381) step the "
                    "optical state WITHIN a block, so their scatter is real DOF motion "
                    "and would inflate the error estimate. T614_triplets are stable.")
    ap.add_argument("--out", default="output/dz_goodness_of_fit.pdf")
    args = ap.parse_args()

    U_eff, U_all, kj = load_bases(args.instrument)
    print(f"bases: U_eff {U_eff.shape}, U_all {U_all.shape}, {len(kj)} DZ coefficients"
          f"  ->  dof: chi2_50 = {len(kj)}-50 = {len(kj)-50}, "
          f"chi2_34 = {len(kj)}-34 = {len(kj)-34}")

    cases = [("MIW", args.fits)] + ([("BIW", args.fits_biw)] if args.fits_biw else [])
    R = {}
    for lab, path in cases:
        W, E, meta = read_dz(path, kj)
        g, per_j = gof(W, E, U_eff, U_all)
        R[lab] = (g, per_j, meta, W, None, None)
        ok = np.isfinite(g["chi2_50"])
        red50 = g["chi2_50"][ok] / g["dof_50"][ok]
        red34 = g["chi2_34"][ok] / g["dof_34"][ok]
        d16 = ((g["chi2_34"] - g["chi2_50"])[ok]
               / np.maximum(g["dof_34"][ok] - g["dof_50"][ok], 1))
        print(f"\n=== {lab}-referenced fit: {path}")
        print(f"  {int(ok.sum())} of {len(W)} visits usable")
        print(f"  chi2_50/dof (vs ANY reachable state): median {np.median(red50):.2f}, "
              f"16-84th {np.percentile(red50,16):.2f}-{np.percentile(red50,84):.2f}")
        print(f"  chi2_34/dof (vs the 34 controlled):   median {np.median(red34):.2f}")
        print(f"  reachable-but-discarded excess/dof16: median {np.median(d16):.2f}")
        print(f"  unweighted residual norms (um): (1-P)w median "
              f"{np.nanmedian(g['r_tot']):.4f}, null part {np.nanmedian(g['r_null']):.4f}"
              f", discarded part {np.nanmedian(g['r_disc']):.4f}, "
              f"||a|| {np.nanmedian(g['a_norm']):.4f}")
        # ---- EMPIRICAL within-block covariance, and the calibrated chi2 ----
        if args.blocks_summary and os.path.exists(args.blocks_summary):
            blk, prog = assign_blocks(meta, args.blocks_summary)
            print(f"  block assignment: {int((blk >= 0).sum())}/{len(blk)} visits "
                  f"matched, {len(set(blk[blk >= 0]))} blocks")
            # Only STABLE-program blocks may define the error: the S-matrix test
            # programs step the optical state inside a block, so their scatter is
            # real DOF motion.  Blocks outside the list are masked to -1 here, which
            # excludes them from BOTH covariance estimators (they are still scored).
            stable = np.isin(prog, args.stable_programs)
            dropped = sorted(set(prog[(blk >= 0) & ~stable]))
            blk_cov = np.where(stable, blk, -1)
            print(f"  covariance restricted to {args.stable_programs}: "
                  f"{int(((blk_cov >= 0)).sum())} visits, "
                  f"{len(set(blk_cov[blk_cov >= 0]))} blocks"
                  f"  (excluded programs: {dropped})")
            Sig, cdof, nb, Dres = within_block_covariance(W, blk_cov,
                                                          min_visits=args.min_visits)
            sd_emp = np.sqrt(np.diag(Sig))
            sd_rlm = np.nanmedian(E, axis=0)
            infl = sd_emp / sd_rlm
            print(f"  within-block covariance: {cdof} residual dof from {nb} blocks")
            print(f"  ERROR INFLATION sigma_emp/sigma_RLM: median {np.median(infl):.1f}, "
                  f"16-84th {np.percentile(infl,16):.1f}-{np.percentile(infl,84):.1f}"
                  f"  -> effective independent donuts ~ "
                  f"{np.nanmedian(meta['n_donuts_1'])/np.median(infl)**2:.0f} "
                  f"of {np.nanmedian(meta['n_donuts_1']):.0f}")
            Sig_adj, nadj, npair_b, _ = adjacent_pair_covariance(
                W, blk_cov, meta["seq_num"].to_numpy())
            print(f"  adjacent-pair covariance: {nadj} pairs from {npair_b} blocks; "
                  f"inflation vs RLM median "
                  f"{np.median(np.sqrt(np.diag(Sig_adj))/sd_rlm):.1f}")
            for cl, Sg in (("block-mean  (conservative: includes drift)", Sig),
                           ("adjacent-pair (drift-suppressed NOISE)", Sig_adj)):
                gg = gls_chi2(W, Sg, U_eff, U_all)
                ok2 = np.isfinite(gg["chi2_50"])
                c50 = gg["chi2_50"][ok2] / gg["dof_50"][ok2]
                c34 = gg["chi2_34"][ok2] / gg["dof_34"][ok2]
                sl2, r2_, p2_ = slope_test(g["a_norm"][ok2], gg["chi2_50"][ok2],
                                           gg["dof_50"][ok2])
                print(f"  [{cl}]")
                print(f"     chi2_50/dof median {np.median(c50):.2f} "
                      f"(16-84th {np.percentile(c50,16):.2f}-{np.percentile(c50,84):.2f})"
                      f";  chi2_34/dof median {np.median(c34):.2f}")
                print(f"     BIAS TEST vs ||a||^2: slope {sl2:+.3g}, r={r2_:+.3f}, "
                      f"p={p2_:.2e}")
            # drift control: adjacent-visit pairs vs the full block spread
            adj = []
            for b in np.unique(blk_cov[blk_cov >= 0]):
                idx = np.where(blk_cov == b)[0]
                idx = idx[np.argsort(meta["seq_num"].to_numpy()[idx])]
                W_ok = [i for i in idx if np.isfinite(W[i]).all()]
                if len(W_ok) >= 2:
                    adj.append(np.diff(W[W_ok], axis=0) / np.sqrt(2))
            if adj:
                sd_adj = np.sqrt(np.mean(np.vstack(adj) ** 2, axis=0))
                print(f"  DRIFT CONTROL adjacent-pair sigma / block-spread sigma: "
                      f"median {np.median(sd_adj/sd_emp):.2f} "
                      f"(<1 => real optical drift within blocks inflates Sigma_emp, "
                      f"making the chi2 conservative)")
            R[lab] = (g, per_j, meta, W, Sig, gg)
        sl, rr, pp = slope_test(g["a_norm"][ok], g["chi2_50"][ok], g["dof_50"][ok])
        print(f"  *** BIAS TEST: chi2_50/dof vs ||a||^2 -> slope {sl:+.3g}, "
              f"r={rr:+.3f}, p={pp:.2e}")
        print("      (unbiased retrieval predicts r = 0; positive r means the "
              "non-DOF-representable residual GROWS with the state amplitude)")
        # null-space FRACTION vs ||a|| -- scale-free version of the same test
        fn = g["r_null"] / np.maximum(g["r_tot"], 1e-30)
        _, rf, pf = slope_test(g["a_norm"][ok], fn[ok] ** 2 * g["dof_50"][ok],
                               g["dof_50"][ok])
        print(f"      null-space share of (1-P)w: median {np.nanmedian(fn**2):.3f}"
              f"  (vs ||a||^2: r={rf:+.3f}, p={pf:.2e})")

    # ---------------------------------------------------------------- figures
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with PdfPages(args.out) as pdf:
        # page 1: chi2/dof distributions
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
        for lab, (g, *_rest) in R.items():
            ok = np.isfinite(g["chi2_50"])
            ax[0].hist(g["chi2_50"][ok] / g["dof_50"][ok], 60, histtype="step",
                       lw=1.6, label=f"{lab}: vs 50 DOF (dof {int(g['dof_50'][ok][0])})")
            ax[0].hist(g["chi2_34"][ok] / g["dof_34"][ok], 60, histtype="step",
                       lw=1.0, ls="--", label=f"{lab}: vs 34 v-modes")
        ax[0].axvline(1, color="k", lw=1, ls=":")
        ax[0].set_xlabel("$\\chi^2$/dof"); ax[0].set_ylabel("visits")
        ax[0].legend(fontsize=7); ax[0].set_title("per-visit goodness of fit")
        for lab, (g, *_rest) in R.items():
            ok = np.isfinite(g["chi2_50"])
            ax[1].scatter(g["a_norm"][ok] ** 2, g["chi2_50"][ok] / g["dof_50"][ok],
                          s=6, alpha=0.35, label=lab)
            x = g["a_norm"][ok] ** 2
            y = g["chi2_50"][ok] / g["dof_50"][ok]
            sl, ic = np.polyfit(x, y, 1)
            xx = np.array([x.min(), x.max()])
            ax[1].plot(xx, sl * xx + ic, lw=1.5)
        ax[1].set_xlabel("$\\|a\\|^2$  (state amplitude$^2$, µm$^2$)")
        ax[1].set_ylabel("$\\chi^2_{50}$/dof")
        ax[1].legend(fontsize=8)
        ax[1].set_title("BIAS TEST: unbiased retrieval $\\Rightarrow$ flat")
        for lab, (g, *_rest) in R.items():
            ok = np.isfinite(g["r_tot"])
            ax[2].scatter(g["a_norm"][ok], g["r_null"][ok], s=6, alpha=0.35, label=lab)
        ax[2].set_xlabel("$\\|a\\|$ (µm)")
        ax[2].set_ylabel("$\\|$null-space part of $(1-P)w\\|$ (µm)")
        ax[2].legend(fontsize=8)
        ax[2].set_title("un-removable, non-DOF residual vs state")
        fig.suptitle(f"DZ fit consistency with the sensitivity matrix: "
                     f"{len(kj)} measured coefficients, 50 reachable DOF, "
                     f"34 controlled v-modes", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # page 2: per-pupil-j excess -- the structure of B
        fig, ax = plt.subplots(len(R), 1, figsize=(13, 4.6 * len(R)), squeeze=False)
        for r, (lab, (g, per_j, *_rest)) in enumerate(R.items()):
            a_ = ax[r][0]
            byj = {j: np.nanmean(per_j[:, [c for c, (k, jj) in enumerate(kj) if jj == j]])
                   for j in ZK}
            xs = np.arange(len(ZK))
            a_.bar(xs, [byj[j] for j in ZK], color="tab:blue")
            a_.axhline(1, color="k", lw=1, ls=":")
            a_.set_xticks(xs); a_.set_xticklabels([f"Z{j}" for j in ZK], fontsize=8)
            a_.set_ylabel("mean weighted sq residual per coefficient")
            a_.set_title(f"{lab}: which pupil Zernikes are inconsistent with the "
                         "sensitivity matrix (1.0 = consistent with the quoted errors)")
            a_.grid(alpha=0.3, axis="y")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # page 3: family / radial-order breakdown
        fig, ax = plt.subplots(len(R), 1, figsize=(12, 4.4 * len(R)), squeeze=False)
        for r, (lab, (g, per_j, *_rest)) in enumerate(R.items()):
            a_ = ax[r][0]
            labs, vals = [], []
            for fam, orders in FAMILIES.items():
                for o, (jc, js_) in orders.items():
                    cols = [c for c, (k, jj) in enumerate(kj)
                            if jj == jc or (js_ is not None and jj == js_)]
                    labs.append(f"{fam}\n{o}")
                    vals.append(float(np.nanmean(per_j[:, cols])))
            a_.bar(np.arange(len(labs)), vals, color="tab:purple")
            a_.axhline(1, color="k", lw=1, ls=":")
            a_.set_xticks(np.arange(len(labs)))
            a_.set_xticklabels(labs, fontsize=7)
            a_.set_ylabel("mean weighted sq residual")
            a_.set_title(f"{lab}: excess by azimuthal family and RADIAL ORDER -- "
                         "the structure a retrieval bias B would imprint")
            a_.grid(alpha=0.3, axis="y")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
