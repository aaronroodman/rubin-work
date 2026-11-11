"""What changes the PSF ellipticity from visit to visit.

Targets: |e| p50 and p90 over CCDs, the coherent |mean e| and its ratio to mean |e|, the
raw two-point sums xi_e1e1 + xi_e2e2 at two separations, and signed mean e1, e2 (OCS).
Inputs: thermal, airflow, intra - extra blur, louvers, time after twilight, pointing, and
the corner-mean wavefront deviation Z4, Z5, Z6 (OCS) from cwfs_science.

Pages: variance budget between and within nights; Spearman rho over all visits and within
night; Huber forward selection with night-grouped cross-validation; boosted trees with
permutation importance; signed ellipticity against corner astigmatism.

Usage::

    python nightlyiq/iq_environment/code/ellipticity_drivers.py
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import iq_correlations as iqc  # noqa: E402

from rubinwork.common import study_run  # noqa: E402
from rubinwork.common.utils import nmad  # noqa: E402
from rubinwork.products import cwfs_science  # noqa: E402
from rubinwork.products.value_added import reader  # noqa: E402

TARGETS = {
    "psf_e_azel_p50": "|e| p50 over CCDs",
    "psf_e_azel_p90": "|e| p90 over CCDs",
    "e_coh": "|mean e|",
    "coh_frac": "|mean e| / mean |e|",
    "xi_sum_r2": "xi_e1e1 + xi_e2e2, 0.35-0.59 deg",
    "xi_sum_r6": "xi_e1e1 + xi_e2e2, 1.29-1.52 deg",
    "e1_ocs": "mean e1, OCS",
    "e2_ocs": "mean e2, OCS",
}  # all dimensionless
SIGNED = {"e1_azel": "mean e1, Az/El", "e2_azel": "mean e2, Az/El"}
# Residuals after the per-band pointing model; e_coh_res is hypot of the e1, e2 residuals.
FIT_TARGETS = ["e1_azel_res", "e2_azel_res", "e_coh_res", "psf_e_azel_p50_res",
               "psf_e_azel_p90_res", "xi_sum_r6_res"]
MODELLED = ["e1_azel", "e2_azel", "psf_e_azel_p50", "psf_e_azel_p90", "xi_sum_r6"]
BANDS = ("u", "g", "r", "i", "z", "y")
RES = {"e1_azel_res": "mean e1 Az/El, pointing-model residual",
       "e2_azel_res": "mean e2 Az/El, pointing-model residual",
       "e_coh_res": "hypot(e1, e2 residuals)",
       "psf_e_azel_p50_res": "|e| p50, pointing-model residual",
       "psf_e_azel_p90_res": "|e| p90, pointing-model residual",
       "xi_sum_r6_res": "xi sum 1.29-1.52 deg, pointing-model residual"}
LABELS = {**TARGETS, **SIGNED, **RES}
OPTICS = {
    "dev_z4": "corner Z4 deviation, OCS (um)",
    "dev_z5": "corner Z5 deviation, OCS (um)",
    "dev_z6": "corner Z6 deviation, OCS (um)",
    "astig_um": "corner hypot(Z5, Z6) deviation (um)",
    "aos_fwhm": "AOS FWHM (arcsec)",
}
EXTRA = {
    "louvers_open_sum": "louvers, equivalent fully open (count)",
    "min_after_twilight": "time after evening twilight (min)",
    "abs_into_wind_deg": "|pointing - wind direction| (deg, 0 into wind)",
}
INPUTS = {**iqc.THERMAL, **iqc.AIRFLOW, "blur_diff": iqc.OTHER["blur_diff"], **OPTICS,
          **EXTRA}
N_STEPS = 5
N_SELECT = 12000
N_FOLDS = 5
SEED = 20261111


def corner_zernikes():
    """Corner-mean wavefront deviation (measured - intrinsic, OCS) per visit [um]."""
    ccd = cwfs_science.load_table("ccd_visits", iqc.CWFS_VARIANT)
    ccd = ccd[ccd["n_quality"] >= iqc.MIN_DONUTS].copy()
    for j in (4, 5, 6):
        ccd[f"dev_z{j}"] = ccd[f"zk_OCS_median_z{j}"] - ccd[f"zk_intrinsic_OCS_median_z{j}"]
    g = ccd.groupby("visit")
    z = g[["dev_z4", "dev_z5", "dev_z6"]].mean()
    z = z[g.size() >= 2 * iqc.MIN_RAFTS]
    z["astig_um"] = np.hypot(z.dev_z5, z.dev_z6)
    return z.reset_index().rename(columns={"visit": "visit_id"})


def load():
    d = iqc.load()
    con = reader.open_db(reader.catalog_db_path("v1", iqc.VA_BUILD))
    try:
        x = con.execute("SELECT visit_id, louvers_open_sum, min_after_twilight "
                        "FROM visit_telemetry").df()
    finally:
        con.close()
    d = d.merge(x, on="visit_id", how="left").merge(corner_zernikes(), on="visit_id",
                                                     how="left")
    d["abs_into_wind_deg"] = d.into_wind_deg.abs()
    for r in ("r2", "r6"):
        d[f"xi_sum_{r}"] = d[f"psf_xi_e1e1_{r}"] + d[f"psf_xi_e2e2_{r}"]
    d["e1_ocs"] = d.psf_e1_azel_mean
    d["e2_ocs"] = -d.psf_e2_azel_mean
    d["e1_azel"] = d.psf_e1_azel_mean
    d["e2_azel"] = d.psf_e2_azel_mean
    bad = ~(d.psf_n_ccd >= 150)
    for c in list(TARGETS) + list(SIGNED):
        d.loc[bad, c] = np.nan
    d = d[d.band.isin(BANDS)].copy()
    d["tanz"] = np.tan(np.radians(90.0 - d.altitude_deg))
    return d


def pointing_model(d):
    """Per band, Huber target ~ 1 + tan z + tan^2 z; adds ``<target>_res`` columns."""
    rows = []
    for t in MODELLED:
        d[f"{t}_res"] = np.nan
        for b in BANDS:
            m = (d.band == b) & d[t].notna() & d.tanz.notna()
            X = np.column_stack([np.ones(m.sum()), d.tanz[m], d.tanz[m] ** 2])
            fit = sm.RLM(d.loc[m, t].to_numpy(), X, M=sm.robust.norms.HuberT()).fit()
            d.loc[m, f"{t}_res"] = d.loc[m, t] - X @ fit.params
            rows.append(dict(target=t, band=b, n=int(m.sum()), c0=fit.params[0],
                             c_tanz=fit.params[1], c_tanz2=fit.params[2],
                             nmad_before=nmad(d.loc[m, t]),
                             nmad_after=nmad(d.loc[m, f"{t}_res"])))
    d["e_coh_res"] = np.hypot(d.e1_azel_res, d.e2_azel_res)
    return pd.DataFrame(rows)


def within(d, cols):
    d = d[d.groupby("day_obs")["visit_id"].transform("size") >= iqc.MIN_NIGHT].copy()
    d[cols] = d[cols] - d.groupby("day_obs")[cols].transform("median")
    return d


def variance_budget(d):
    """Fraction of robust variance between nights, per target."""
    rows = []
    for t in list(SIGNED) + list(RES):
        s = d[["day_obs", t]].dropna()
        s = s[s.groupby("day_obs")[t].transform("size") >= iqc.MIN_NIGHT]
        tot = nmad(s[t]) ** 2
        win = nmad(s[t] - s.groupby("day_obs")[t].transform("median")) ** 2
        rows.append(dict(target=t, total_nmad=np.sqrt(tot), within_nmad=np.sqrt(win),
                         between_frac=1 - win / tot, n=len(s), nights=s.day_obs.nunique()))
    return pd.DataFrame(rows)


def rho_matrix(d):
    rho = pd.DataFrame(index=list(RES), columns=list(INPUTS), dtype=float)
    for t in RES:
        for x in INPUTS:
            s = d[[t, x]].dropna()
            if len(s) > 300:
                rho.loc[t, x] = stats.spearmanr(s[t], s[x])[0]
    return rho


def huber_cv(s, t, xs, folds):
    """Held-out nMAD of a Huber fit t ~ xs, folds grouped by night."""
    res = np.empty(len(s))
    y = s[t].to_numpy()
    X = sm.add_constant(s[xs].to_numpy(), has_constant="add") if xs else np.ones((len(s), 1))
    for tr, te in folds:
        fit = sm.RLM(y[tr], X[tr], M=sm.robust.norms.HuberT()).fit(maxiter=30)
        res[te] = y[te] - X[te] @ fit.params
    return nmad(res)


def forward_select(s, t, candidates):
    """Greedy forward selection on held-out nMAD."""
    rng = np.random.default_rng(SEED)
    sub = s.iloc[rng.choice(len(s), min(N_SELECT, len(s)), replace=False)].reset_index(drop=True)
    folds = list(GroupKFold(N_FOLDS).split(sub, groups=sub.day_obs))
    chosen, path = [], [("(constant)", huber_cv(sub, t, [], folds))]
    for _ in range(N_STEPS):
        scores = {x: huber_cv(sub, t, chosen + [x], folds) for x in candidates if x not in chosen}
        best = min(scores, key=scores.get)
        chosen.append(best)
        path.append((best, scores[best]))
    return path


def trees(s, t, inputs):
    """Boosted trees: held-out nMAD and permutation importance (night-grouped)."""
    X, y = s[inputs], s[t].to_numpy()
    folds = list(GroupKFold(N_FOLDS).split(X, groups=s.day_obs))
    res = np.empty(len(s))
    imp = np.zeros(len(inputs))
    for tr, te in folds:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          loss="absolute_error", random_state=SEED)
        m.fit(X.iloc[tr], y[tr])
        res[te] = y[te] - m.predict(X.iloc[te])
        k = te[:4000]
        p = permutation_importance(m, X.iloc[k], y[k], n_repeats=3, random_state=SEED,
                                   scoring="neg_mean_absolute_error")
        imp += p.importances_mean / len(folds)
    return nmad(res), pd.Series(imp, index=inputs).sort_values(ascending=False)


def page_text(pdf, title, lines, size=8.5):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.95, title, fontsize=15, weight="bold", va="top")
    fig.text(0.05, 0.89, "\n".join(lines), fontsize=size, va="top", family="monospace")
    pdf.savefig(fig)
    plt.close(fig)


def page_heatmap(pdf, rho, title):
    fig, ax = plt.subplots(figsize=(18, 6))
    im = ax.imshow(rho.to_numpy(float), cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_yticks(range(len(rho.index)), [LABELS[t] for t in rho.index], fontsize=8)
    ax.set_xticks(range(len(INPUTS)), list(INPUTS.values()), rotation=65, ha="right",
                  fontsize=6)
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            v = rho.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(v) > 0.35 else "black")
    plt.colorbar(im, ax=ax, label="Spearman rho (dimensionless)", fraction=0.02)
    ax.set_title(title)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_selection(pdf, paths, base):
    fig, axes = plt.subplots(2, len(FIT_TARGETS), figsize=(20, 8.5), sharey=False)
    for c, t in enumerate(FIT_TARGETS):
        for r, mode in enumerate(("all visits", "within night")):
            ax = axes[r, c]
            p = paths[(t, mode)]
            ax.plot(range(len(p)), [v for _, v in p], "o-", color="#0072B2")
            ax.axhline(base[(t, mode)], color="#D55E00", ls="--", lw=1,
                       label="boosted trees, all inputs")
            ax.set_xticks(range(len(p)), [x for x, _ in p], rotation=60, ha="right",
                          fontsize=6)
            ax.set_ylabel("held-out nMAD (dimensionless)", fontsize=7)
            ax.set_title(f"{LABELS[t]}, {mode}", fontsize=7)
            ax.legend(fontsize=6)
    fig.suptitle("Huber forward selection, 5 night-grouped folds "
                 f"({N_SELECT} visit subsample)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_importance(pdf, imps):
    fig, axes = plt.subplots(2, len(FIT_TARGETS), figsize=(20, 8.5))
    for c, t in enumerate(FIT_TARGETS):
        for r, mode in enumerate(("all visits", "within night")):
            ax = axes[r, c]
            s = imps[(t, mode)][:12][::-1]
            ax.barh(range(len(s)), s.values, color="#0072B2", height=0.7)
            ax.set_yticks(range(len(s)), [INPUTS[x] for x in s.index], fontsize=6)
            ax.set_xlabel("increase in MAE when permuted (dimensionless)", fontsize=7)
            ax.set_title(f"{LABELS[t]}, {mode}", fontsize=7)
    fig.suptitle("Boosted-tree permutation importance, top 12, night-grouped folds")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_pointing(pdf, d, pm):
    """Each modelled target against tan z by band, before and after the model."""
    fig, axes = plt.subplots(2, len(MODELLED), figsize=(20, 7.5))
    colors = dict(zip(BANDS, ["#56B4E9", "#009E73", "#E69F00", "#D55E00", "#CC79A7", "#000000"]))
    bins = np.linspace(0, 1.5, 13)
    c = 0.5 * (bins[1:] + bins[:-1])
    for j, t in enumerate(MODELLED):
        for r, col in enumerate((t, f"{t}_res")):
            ax = axes[r, j]
            for b in BANDS:
                s = d[d.band == b]
                idx = np.digitize(s.tanz, bins)
                med = [s[col][idx == k].median() if (idx == k).sum() > 50 else np.nan
                       for k in range(1, len(bins))]
                ax.plot(c, med, "o-", ms=3, color=colors[b], label=b)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_xlabel("tan(zenith angle) (dimensionless)", fontsize=7)
            ax.set_title(("" if r == 0 else "residual: ") + LABELS[t], fontsize=7)
            ax.set_ylabel("median (dimensionless)", fontsize=7)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Pointing model: per band, Huber target ~ 1 + tan z + tan^2 z "
                 "(top: before, bottom: residual)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def page_signed(pdf, d):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for r, e in enumerate(("e1_ocs", "e2_ocs")):
        for c, z in enumerate(("dev_z5", "dev_z6")):
            ax = axes[r, c]
            s = d[[z, e]].dropna()
            lo, hi = s.quantile(0.005), s.quantile(0.995)
            s = s[(s[z] >= lo[z]) & (s[z] <= hi[z]) & (s[e] >= lo[e]) & (s[e] <= hi[e])]
            hb = ax.hexbin(s[z], s[e], gridsize=60, norm=LogNorm(), cmap="viridis", mincnt=1)
            iqc.binned_median(ax, s[z].to_numpy(), s[e].to_numpy(), "#D55E00")
            ax.axhline(0, color="0.6", lw=0.8)
            ax.axvline(0, color="0.6", lw=0.8)
            ax.set_xlabel(OPTICS[z])
            ax.set_ylabel(f"{TARGETS[e]} (dimensionless)")
            ax.set_title(f"Pearson r {stats.pearsonr(s[z], s[e])[0]:+.3f}, Spearman rho "
                         f"{stats.spearmanr(s[z], s[e])[0]:+.3f}, n {len(s)}", fontsize=8)
            plt.colorbar(hb, ax=ax, label="visits per cell")
    fig.suptitle("Signed coherent ellipticity against corner astigmatism, all visits "
                 "(central 99%)")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main():
    run = study_run("nightlyiq/iq_environment", "ellipticity_drivers")
    d = load()
    pm = pointing_model(d)
    pm.to_csv(run.path / "pointing_model.csv", index=False)
    cols = list(TARGETS) + list(SIGNED) + list(RES) + list(INPUTS)
    dw = within(d, cols)
    budget = variance_budget(d)
    budget.to_csv(run.path / "variance_budget.csv", index=False)
    rho_a, rho_w = rho_matrix(d), rho_matrix(dw)
    rho_a.to_csv(run.path / "spearman_all.csv")
    rho_w.to_csv(run.path / "spearman_within_night.csv")

    paths, base, imps, sel_rows = {}, {}, {}, []
    inputs = list(INPUTS)
    for mode, frame in (("all visits", d), ("within night", dw)):
        for t in FIT_TARGETS:
            s = frame[["day_obs", t] + inputs].dropna().reset_index(drop=True)
            p = forward_select(s, t, inputs)
            paths[(t, mode)] = p
            st = s  # same sample as the selection, so the nMADs compare
            base[(t, mode)], imps[(t, mode)] = trees(st, t, inputs)
            for k, (x, v) in enumerate(p):
                sel_rows.append(dict(mode=mode, target=t, step=k, input=x, cv_nmad=v,
                                     n_complete=len(s), n_trees=len(st),
                                     trees_cv_nmad=base[(t, mode)]))
            print(f"{mode} {t}: n {len(s)}, " + "; ".join(f"{x} {v:.4f}" for x, v in p)
                  + f"; trees {base[(t, mode)]:.4f} (n {len(st)})", flush=True)
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(run.path / "forward_selection.csv", index=False)
    pd.DataFrame({f"{t} | {m}": s for (t, m), s in imps.items()}).to_csv(
        run.path / "tree_importance.csv")

    lines = [f"{len(d)} science visits, {d.day_obs.nunique()} nights (sample as in "
             f"iq_correlations); ellipticity needs psf_n_ccd >= 150",
             "Corner Z4-Z6: mean over corner CCDs of (measured - intrinsic), OCS, um of "
             "wavefront; cwfs_science covers 20251021 to 20260714.",
             "Pointing model: per band, Huber target ~ 1 + tan z + tan^2 z; all fits below "
             "use its residuals. Elevation is not an input.",
             "", "Variance budget (robust; between = 1 - within nMAD^2 / total nMAD^2):"]
    for r in budget.itertuples():
        lines.append(f"  {LABELS[r.target]:46s} total nMAD {r.total_nmad:.4f}  within "
                     f"{r.within_nmad:.4f}  between fraction {r.between_frac:.2f} "
                     f"(of variance; n {r.n}, {r.nights} nights)")
    lines += ["", "Forward selection (held-out nMAD, dimensionless) and trees:"]
    for (t, mode), p in paths.items():
        lines.append(f"  {t}, {mode}: const {p[0][1]:.4f} -> "
                     + " -> ".join(f"{x} {v:.4f}" for x, v in p[1:4])
                     + f"; {N_STEPS} inputs {p[-1][1]:.4f}; trees {base[(t, mode)]:.4f}")
    out = run.path / "ellipticity_drivers.pdf"
    with PdfPages(out) as pdf:
        page_text(pdf, "What changes the ellipticity visit to visit", lines, size=7)
        page_pointing(pdf, d, pm)
        page_heatmap(pdf, rho_w, "Spearman rho, within night (night medians subtracted)")
        page_heatmap(pdf, rho_a, "Spearman rho, all visits")
        page_selection(pdf, paths, base)
        page_importance(pdf, imps)
        page_signed(pdf, d)
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
