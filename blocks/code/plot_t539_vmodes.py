#!/usr/bin/env python
"""V-mode diagnostics for the T539 closed-loop table -> one multi-page PDF.

Pages:
  1. v-mode corner plot (v1..vN) -- each off-diagonal panel annotated with the
     robust Spearman rho and Theil-Sen slope, colored by significance.  Diagonal
     1-D histograms share each column's x-range with the scatter panels.
  2. pairwise summary: Spearman-rho and Theil-Sen-slope heatmaps, with
     significance stars, to see which v_i-v_j couplings are real.
  3. v-modes vs M1M3 z_gradient       (vertical thermal gradient)
  4. v-modes vs M1M3 radial_gradient  (in-plane thermal gradient)
  5. v1 vs mean TMA truss temperature  ((pxpy + mxmy) / 2)

Correlations use ROBUST Spearman rank rho + Theil-Sen slope.  Significance is the
Spearman p-value; with N=n*(n-1)/2 pairs we also flag the Bonferroni threshold
(alpha/N).  Points are colored by band (legend on every page).

Cuts: donut_blur < 1.1, elevation >= 59, |cam rotator| <= 10, drop u-band, keep
only the 22/12 closed loop (closed_loop_22dof_trunc12), exclude Apr 24 / Apr 28
(fixed 50-DOF/21-v-mode single-value LUT; see memory apr-2026-50dof-lut).
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, theilslopes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

BAND_COLORS = {"u": "#6a3d9a", "g": "#1f78b4", "r": "#33a02c",
               "i": "#e31a1c", "z": "#ff7f00", "y": "#b15928"}
ALPHA = 0.05


def apply_cuts(d, exclude_days, annotation="closed_loop_22dof_trunc12"):
    m = (
        (d["donut_blur_fwhm"] < 1.1)
        & (d["altitude"] >= 59.0)
        & (d["physical_rotator_angle"].abs() <= 10.0)
        & (d["band"] != "u")
        & (~d["day_obs"].isin(exclude_days))
    )
    if annotation:
        m = m & (d["scheduler_note"] == annotation)
    return d[m].copy()


def robust_stats(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return None
    rho, p = spearmanr(x[ok], y[ok])
    slope, intercept, lo, hi = theilslopes(y[ok], x[ok])
    return dict(rho=rho, p=p, slope=slope, intercept=intercept,
                n=int(ok.sum()), xok=x[ok])


def _band_handles(d):
    bands = [b for b in BAND_COLORS if b in set(d["band"])]
    return bands, [plt.Line2D([], [], marker="o", ls="", color=BAND_COLORS[b],
                              label=b) for b in bands]


def data_ranges(d, vcols, pad=0.06):
    r = {}
    for c in vcols:
        v = d[c].to_numpy(); v = v[np.isfinite(v)]
        lo, hi = float(np.min(v)), float(np.max(v))
        m = pad * (hi - lo) if hi > lo else (abs(hi) or 1.0) * 0.1
        r[c] = (lo - m, hi + m)
    return r


def pairwise_robust(d, vcols):
    n = len(vcols)
    rho = np.full((n, n), np.nan); pval = np.full((n, n), np.nan)
    slope = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            st = robust_stats(d[vcols[j]].to_numpy(), d[vcols[i]].to_numpy())
            if st:                       # panel (i,j): x=v_j, y=v_i
                rho[i, j], pval[i, j], slope[i, j] = st["rho"], st["p"], st["slope"]
    return rho, pval, slope


def _sig_color(p, bonf):
    if not np.isfinite(p):
        return "0.6", "normal"
    if p < bonf:
        return "#c81e1e", "bold"        # survives Bonferroni
    if p < ALPHA:
        return "black", "normal"        # nominally significant
    return "0.6", "normal"              # not significant


def corner_plot(d, vcols, title, rho, pval, slope, bonf, ranges):
    n = len(vcols)
    fig, axes = plt.subplots(n, n, figsize=(1.95 * n, 1.95 * n))
    bands, handles = _band_handles(d)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off"); continue
            if i == j:                   # diagonal hist: SAME x-range as column
                bins = np.linspace(*ranges[vcols[i]], 16)
                for b in bands:
                    ax.hist(d.loc[d["band"] == b, vcols[i]].to_numpy(), bins=bins,
                            histtype="step", color=BAND_COLORS[b], lw=1.1)
                ax.set_xlim(ranges[vcols[i]])
            else:
                for b in bands:
                    sub = d[d["band"] == b]
                    ax.scatter(sub[vcols[j]], sub[vcols[i]], s=8, alpha=0.7,
                               color=BAND_COLORS[b], edgecolors="none")
                ax.set_xlim(ranges[vcols[j]]); ax.set_ylim(ranges[vcols[i]])
                col, wt = _sig_color(pval[i, j], bonf)
                ax.text(0.04, 0.96,
                        f"ρ {rho[i, j]:+.2f}\nm {slope[i, j]:+.2f}",
                        transform=ax.transAxes, va="top", ha="left",
                        fontsize=6.3, color=col, weight=wt)
            ax.set_xlabel(vcols[j], fontsize=8) if i == n - 1 else ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(vcols[i], fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=6)
    note = (r"$\rho$/slope color:  red bold = Bonferroni (p<%.1g)   "
            r"black = p<%.2g   grey = n.s." % (bonf, ALPHA))
    fig.legend(handles=handles, loc="upper right", title="band", fontsize=9)
    fig.suptitle(title + "\n" + note, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def pair_summary(vcols, rho, pval, slope, bonf, title):
    n = len(vcols)
    tri = np.tril(np.ones((n, n), bool), -1)            # lower triangle only
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.2 * 2, 8.5))
    for ax, mat, fmt, sub in [(axL, rho, "{:+.2f}", "Spearman ρ"),
                              (axR, slope, "{:+.2g}", "Theil-Sen slope")]:
        disp = np.where(tri, rho, np.nan)               # color BOTH by rho
        im = ax.imshow(disp, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(n):
            for j in range(n):
                if not tri[i, j] or not np.isfinite(rho[i, j]):
                    continue
                star = "**" if pval[i, j] < bonf else ("*" if pval[i, j] < ALPHA else "")
                wt = "bold" if star else "normal"
                tc = "white" if abs(rho[i, j]) > 0.55 else "black"
                ax.text(j, i, fmt.format(mat[i, j]) + star, ha="center",
                        va="center", fontsize=7.5, color=tc, weight=wt)
        ax.set_xticks(range(n)); ax.set_xticklabels(vcols, fontsize=8, rotation=45)
        ax.set_yticks(range(n)); ax.set_yticklabels(vcols, fontsize=8)
        ax.set_title(sub, fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"Spearman $\rho$")
    fig.suptitle(title + "   (** p<%.1g Bonferroni,  * p<%.2g)" % (bonf, ALPHA),
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def vmode_vs_var(d, vcols, xcol, title, ranges, stats_rows=None):
    n = len(vcols); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow))
    axes = np.atleast_1d(axes).ravel()
    bands, handles = _band_handles(d)
    x = d[xcol].to_numpy()
    for k, vc in enumerate(vcols):
        ax = axes[k]
        for b in bands:
            sub = d[d["band"] == b]
            ax.scatter(sub[xcol], sub[vc], s=14, alpha=0.75,
                       color=BAND_COLORS[b], edgecolors="none")
        st = robust_stats(x, d[vc].to_numpy())
        if st:
            xs = np.array([st["xok"].min(), st["xok"].max()])
            ax.plot(xs, st["intercept"] + st["slope"] * xs, "k--", lw=1.3)
            sig = "**" if st["p"] < (ALPHA / (n * (n - 1) / 2)) else ("*" if st["p"] < ALPHA else "")
            ax.set_title(f"{vc}   ρ={st['rho']:+.2f}{sig}  (TS {st['slope']:+.2f})",
                         fontsize=9)
            ax.set_ylim(ranges[vc])
            if stats_rows is not None:
                stats_rows.append(dict(var=xcol, vmode=vc, spearman_rho=st["rho"],
                                       p=st["p"], theilsen_slope=st["slope"], n=st["n"]))
        ax.set_xlabel(xcol, fontsize=8); ax.tick_params(labelsize=7)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.legend(handles=handles, loc="upper right", title="band", fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def v1_vs_truss(d, title, ranges, stats_rows=None):
    x = 0.5 * (d["tma_truss_temp_pxpy"].to_numpy() + d["tma_truss_temp_mxmy"].to_numpy())
    y = d["v1"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    bands, handles = _band_handles(d)
    for b in bands:
        m = (d["band"] == b).to_numpy()
        ax.scatter(x[m], y[m], s=28, alpha=0.8, color=BAND_COLORS[b], edgecolors="none")
    st = robust_stats(x, y)
    if st:
        xs = np.array([st["xok"].min(), st["xok"].max()])
        ax.plot(xs, st["intercept"] + st["slope"] * xs, "k--", lw=1.5)
        ax.set_title(f"{title}\nρ={st['rho']:+.2f} (p={st['p']:.1e})   "
                     f"Theil-Sen slope={st['slope']:+.3f} v1/degC   n={st['n']}",
                     fontsize=11)
        ax.set_ylim(ranges["v1"])
        if stats_rows is not None:
            stats_rows.append(dict(var="truss_mean_temp", vmode="v1",
                                   spearman_rho=st["rho"], p=st["p"],
                                   theilsen_slope=st["slope"], n=st["n"]))
    ax.set_xlabel("mean TMA truss temperature  (pxpy+mxmy)/2  [degC]", fontsize=11)
    ax.set_ylabel("v1", fontsize=11)
    ax.legend(handles=handles, title="band", fontsize=10)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", default="output/t539_vmode_analysis_noLUT.pdf")
    ap.add_argument("--n-vmode", type=int, default=12)
    ap.add_argument("--exclude-days", type=int, nargs="*", default=[20260424, 20260428])
    ap.add_argument("--annotation", default="closed_loop_22dof_trunc12")
    args = ap.parse_args()

    d = pd.read_parquet(args.infile)
    vcols = [f"v{i}" for i in range(1, args.n_vmode + 1)]
    cut = apply_cuts(d, set(args.exclude_days), annotation=args.annotation or None)
    npairs = args.n_vmode * (args.n_vmode - 1) / 2
    bonf = ALPHA / npairs
    print(f"total {len(d)} -> after cuts {len(cut)} rows; Bonferroni p<{bonf:.2g} "
          f"({int(npairs)} pairs)")

    ranges = data_ranges(cut, vcols)
    rho, pval, slope = pairwise_robust(cut, vcols)
    tag = "22/12 only, cuts + no Apr24/28 (50/21 LUT)"
    rows = []
    with PdfPages(args.out) as pdf:
        pdf.savefig(corner_plot(cut, vcols, f"T539 v-mode corner  [{tag}]  n={len(cut)}",
                                rho, pval, slope, bonf, ranges))
        pdf.savefig(pair_summary(vcols, rho, pval, slope, bonf,
                                 f"T539 pairwise v-mode correlations  [{tag}]  n={len(cut)}"))
        pdf.savefig(vmode_vs_var(cut, vcols, "z_gradient",
                    f"v-modes vs M1M3 z_gradient (vertical)  [{tag}]  n={len(cut)}", ranges, rows))
        pdf.savefig(vmode_vs_var(cut, vcols, "radial_gradient",
                    f"v-modes vs M1M3 radial_gradient (in-plane)  [{tag}]  n={len(cut)}", ranges, rows))
        pdf.savefig(v1_vs_truss(cut, f"v1 vs mean TMA truss temp  [{tag}]", ranges, rows))
    plt.close("all")
    print(f"wrote {args.out}")

    # significant pairwise couplings
    sig = [(vcols[i], vcols[j], rho[i, j], slope[i, j], pval[i, j])
           for i in range(len(vcols)) for j in range(i)
           if np.isfinite(pval[i, j]) and pval[i, j] < ALPHA]
    sig.sort(key=lambda t: -abs(t[2]))
    print(f"\n=== significant v_i-v_j pairs (p<{ALPHA}, ** = Bonferroni) ===")
    for a, b, r, s, p in sig:
        print(f"  {a:>3}-{b:<3}  rho={r:+.2f}  TS slope={s:+.3f}  p={p:.1e}"
              f"  {'**' if p < bonf else '*'}")


if __name__ == "__main__":
    main()
