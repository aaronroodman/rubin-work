#!/usr/bin/env python3
"""v-mode <-> temperature correlations + v-mode autocorrelation (FAM, per ps/mi).

Projects the MI-subtracted per-visit Double-Zernikes in output/<ps>/<mi>/fits.parquet
onto the OFC v-modes (the build's n_dof/n_keep, e.g. 50/34), then correlates each
v-mode against (a) the temperature telemetry suite and (b) the other v-modes
(autocorrelation).  Unlike thermal_correlations (Pearson) this uses the ROBUST
Spearman rho + Theil-Sen slope, and gives one v-mode-centric summary: discrete
-1..1 rho heatmaps + slope heatmaps with statistically-significant cells flagged.

Pages: (1) v-mode x temperature rho + slope, (2) v-mode x v-mode rho + slope,
(3..) the full v_i-v_j lower triangle tiled into `corner_block` x `corner_block`
panels per page (scatter + robust Theil-Sen fit + Spearman rho), (last) the
top-|rho| individual v-mode pair scatters at readable size.  Also
vmode_correlations_summary.parquet  (kind, mode_i, term, rho, slope, p, n).

Needs lsst.ts.ofc (RSP).  Knobs: analysis_config.yaml `vmode_correlations`.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, theilslopes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.backends.backend_pdf import PdfPages

from lsst.ts.intrinsic.wavefront import mi_config as mc
from lsst.ts.intrinsic.wavefront import ofc_svd as osv
from astropy.table import QTable

DEFAULT_THERMAL_VARS = [
    "cam_air_temp", "m2_air_temp", "m1m3_air_temp", "outside_temp",
    "m2_delta_t", "cam_m1m3_delta_t", "dome_delta_t",
    "x_gradient", "y_gradient", "z_gradient", "radial_gradient",
    "tma_truss_temp_pxpy", "tma_truss_temp_mxmy",
]
DEFAULT = dict(dz_prefix="z1toz6", max_coeff_um=2.0,
               thermal_vars=list(DEFAULT_THERMAL_VARS), alpha=0.05,
               corner_block=6,    # v_i-v_j corner tiled into corner_block^2 panels/page
               top_pairs=16)      # strongest |rho| pairs shown as large scatters


def dz_coeff_columns(df, prefix):
    pat = re.compile(rf"^{re.escape(prefix)}_z\d+_c\d+$")
    return [c for c in df.columns if pat.match(c)]


def quality_cut(df, prefix, maxc):
    cols = dz_coeff_columns(df, prefix)
    if not cols or not maxc:
        return df
    bad = (df[cols].abs() > maxc).any(axis=1)
    return df[~bad]


def project_vmodes(df, prefix, svd):
    W = np.full((len(df), len(svd.kj_grid)), np.nan)
    for ci, (k, j) in enumerate(svd.kj_grid):
        col = f"{prefix}_z{j}_c{k}"
        if col in df.columns:
            W[:, ci] = df[col].to_numpy(dtype=float)
    return svd.vmodes(svd.project_amplitudes(W))


def robust(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 5:
        return None
    rho, p = spearmanr(x[m], y[m])
    slope, intercept, lo, hi = theilslopes(y[m], x[m])
    return dict(rho=float(rho), p=float(p), slope=float(slope), n=int(m.sum()))


def _discrete_heatmap(ax, M, P, row_labels, col_labels, bonf, title, fmt,
                      color_by_rho=None):
    """M = values to annotate; color by rho (=M unless color_by_rho given), on a
    discrete -1..1 RdBu_r scale; bold + star cells with p<alpha, ** if p<bonf."""
    C = color_by_rho if color_by_rho is not None else M
    levels = np.linspace(-1, 1, 11)
    cmap = plt.get_cmap("RdBu_r", len(levels) - 1)
    norm = BoundaryNorm(levels, cmap.N)
    im = ax.imshow(C, cmap=cmap, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]):
                continue
            p = P[i, j]
            star = "**" if (np.isfinite(p) and p < bonf) else (
                   "*" if (np.isfinite(p) and p < 0.05) else "")
            wt = "bold" if star else "normal"
            tc = "white" if (np.isfinite(C[i, j]) and abs(C[i, j]) > 0.6) else "black"
            ax.text(j, i, fmt.format(M[i, j]) + star, ha="center", va="center",
                    fontsize=5.5, color=tc, weight=wt)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=6, rotation=90)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=6)
    ax.set_title(title, fontsize=10)
    return im


def _corr_grid(V, cols, xs):
    """rho/slope/p matrices of each v-mode (rows) vs each column series in xs."""
    nR, nC = V.shape[1], len(cols)
    rho = np.full((nR, nC), np.nan); slope = np.full((nR, nC), np.nan)
    pval = np.full((nR, nC), np.nan)
    rows = []
    for i in range(nR):
        for c, name in enumerate(cols):
            st = robust(xs[c], V[:, i])
            if st:
                rho[i, c], slope[i, c], pval[i, c] = st["rho"], st["slope"], st["p"]
                rows.append(dict(mode_i=i + 1, term=name, rho=st["rho"],
                                 slope=st["slope"], p=st["p"], n=st["n"]))
    return rho, slope, pval, rows


def _fit(x, y):
    """Robust Spearman rho + Theil-Sen (slope, intercept) on finite pairs."""
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 5:
        return None
    rho, p = spearmanr(x[m], y[m])
    slope, intercept, _, _ = theilslopes(y[m], x[m])
    return dict(rho=float(rho), p=float(p), slope=float(slope),
                intercept=float(intercept), n=int(m.sum()), mask=m)


def _star(p, bonf):
    return "**" if (np.isfinite(p) and p < bonf) else (
           "*" if (np.isfinite(p) and p < 0.05) else "")


def _scatter_panel(ax, x, y, bonf):
    """One v_col(x)-vs-v_row(y) scatter with the robust Theil-Sen line and the
    Spearman rho annotation (bold red for |rho|>=0.3, significance stars)."""
    mk = np.isfinite(x) & np.isfinite(y)
    xx, yy = x[mk], y[mk]
    ax.scatter(xx, yy, s=4, alpha=0.35, color="C0", edgecolors="none",
               rasterized=True)
    f = _fit(x, y)
    if f is not None:
        xr = np.array([xx.min(), xx.max()])
        ax.plot(xr, f["intercept"] + f["slope"] * xr, color="C3", lw=0.9)
        strong = abs(f["rho"]) >= 0.3
        ax.text(0.05, 0.93, f"{f['rho']:+.2f}{_star(f['p'], bonf)}",
                transform=ax.transAxes, fontsize=7, va="top",
                color=("C3" if strong else "0.4"),
                weight=("bold" if strong else "normal"))


def vmode_corner_pages(V, vlabels, block, bonf, title_prefix):
    """The full v_i-v_j lower triangle, tiled into pages of `block` x `block`
    panels.  Split the modes into consecutive blocks; emit one page per
    (row-block >= col-block) pair.  On a diagonal block the page is itself a
    lower-triangle corner (histograms on the diagonal, upper corner blank);
    off-diagonal block-pages are a full grid of scatters.  Yields figures one at
    a time (reading order: row-block outer, col-block inner) so the caller can
    save and close each before the next is built."""
    n = V.shape[1]
    nb = int(np.ceil(n / block))
    for rb in range(nb):
        rows = list(range(rb * block, min((rb + 1) * block, n)))
        for cb in range(rb + 1):
            cols = list(range(cb * block, min((cb + 1) * block, n)))
            diag = (rb == cb)
            fig, axes = plt.subplots(
                len(rows), len(cols),
                figsize=(2.2 * len(cols) + 0.6, 2.2 * len(rows) + 0.9),
                squeeze=False)
            for ri, i in enumerate(rows):
                for cj, j in enumerate(cols):
                    ax = axes[ri][cj]
                    if diag and j > i:                       # upper corner blank
                        ax.axis("off"); continue
                    if diag and i == j:                      # diagonal histogram
                        v = V[:, i][np.isfinite(V[:, i])]
                        ax.hist(v, bins=25, color="0.7"); ax.set_yticks([])
                    else:
                        _scatter_panel(ax, V[:, j], V[:, i], bonf)
                    if ri == len(rows) - 1:
                        ax.set_xlabel(vlabels[j], fontsize=8)
                    if cj == 0:
                        ax.set_ylabel(vlabels[i], fontsize=8)
                    ax.tick_params(labelsize=6, length=2)
            rr = f"{vlabels[rows[0]]}-{vlabels[rows[-1]]}"
            cc = f"{vlabels[cols[0]]}-{vlabels[cols[-1]]}"
            fig.suptitle(f"{title_prefix}: rows {rr} x cols {cc}", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            yield fig


def ranked_pairs(V):
    """All i<j v-mode pairs with a robust fit, sorted by |rho| descending."""
    out = []
    for i in range(V.shape[1]):
        for j in range(i + 1, V.shape[1]):
            f = _fit(V[:, i], V[:, j])
            if f is not None:
                out.append({"i": i, "j": j, **f})
    out.sort(key=lambda d: -abs(d["rho"]))
    return out


def top_pairs_page(V, vlabels, pairs, bonf, ncol, title):
    """The strongest v-mode pairs as large individual scatters (x=v_i, y=v_j)
    with the robust fit, Spearman rho, Theil-Sen slope and n."""
    k = len(pairs)
    if k == 0:
        return None
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                             squeeze=False)
    for idx in range(nrow * ncol):
        ax = axes[idx // ncol][idx % ncol]
        if idx >= k:
            ax.axis("off"); continue
        pr = pairs[idx]; i, j = pr["i"], pr["j"]
        x, y = V[:, i], V[:, j]
        mk = np.isfinite(x) & np.isfinite(y)
        xx, yy = x[mk], y[mk]
        ax.scatter(xx, yy, s=9, alpha=0.4, color="C0", edgecolors="none",
                   rasterized=True)
        xr = np.array([xx.min(), xx.max()])
        ax.plot(xr, pr["intercept"] + pr["slope"] * xr, color="C3", lw=1.2)
        ax.set_xlabel(vlabels[i], fontsize=8)
        ax.set_ylabel(vlabels[j], fontsize=8)
        ax.set_title(f"rho={pr['rho']:+.2f}{_star(pr['p'], bonf)}  "
                     f"slope={pr['slope']:+.2g}  n={pr['n']}", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--param-set", required=True)
    ap.add_argument("--mi-name", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--analysis-config", default=None, dest="analysis_config")
    ap.add_argument("--output-root", default="output")
    args = ap.parse_args()

    sec = {**DEFAULT, **mc.analysis_section(
        "vmode_correlations", args.param_set, args.mi_name,
        config_path=(Path(args.analysis_config) if args.analysis_config else None))}
    prefix = sec["dz_prefix"]
    thermal_vars = list(sec["thermal_vars"])
    cfg = mc.load_mi_config(args.param_set, args.mi_name,
                            config_path=(Path(args.config) if args.config else None))

    base = Path(args.output_root) / args.param_set / args.mi_name
    out_dir = base / "plots"; out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(base / "fits.parquet")
    if "visit_quality_pass" in df.columns:
        df = df[df["visit_quality_pass"].astype(bool)]
    df = quality_cut(df, prefix, sec["max_coeff_um"])
    present_tv = [tv for tv in thermal_vars if tv in df.columns]
    print(f"[vmode_correlations] {base}  {len(df)} visits, {len(present_tv)} temps",
          flush=True)

    # OFC SVD from the build's n_dof/n_keep (e.g. 50/34)
    b = cfg["build"]
    visits = QTable.read(str(base.parent / "visits.parquet"))
    iZs = [int(j) for j in np.asarray(visits["nollIndices"][0]).tolist()]
    svd = osv.build_ofc_svd(iZs, int(b["k_min"]), int(b["k_max"]), cfg["n_keep"],
                            n_dof=cfg.get("n_dof"),
                            ofc_normalization_yaml=b.get("ofc_normalization_yaml"))
    V = project_vmodes(df, prefix, svd)
    nkeep = V.shape[1]
    vlabels = [f"v{i + 1}" for i in range(nkeep)]
    print(f"  SVD n_keep_eff={nkeep}; projecting {len(df)} visits", flush=True)

    # v-mode x temperature and v-mode x v-mode (autocorr)
    trho, tslope, tp, trows = _corr_grid(
        V, present_tv, [df[tv].to_numpy(float) for tv in present_tv])
    vrho, vslope, vp, vrows = _corr_grid(V, vlabels, [V[:, i] for i in range(nkeep)])
    np.fill_diagonal(vrho, np.nan); np.fill_diagonal(vslope, np.nan)  # drop self-corr

    npairs = len(present_tv) * nkeep + nkeep * (nkeep - 1) // 2
    bonf = sec["alpha"] / max(1, npairs)

    with PdfPages(str(out_dir / "vmode_correlations.pdf")) as pdf:
        # page 1: v-mode x temperature
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(max(8, 1.0 * len(present_tv) + 6), 0.32 * nkeep + 3))
        im = _discrete_heatmap(a1, trho, tp, vlabels, present_tv, bonf,
                               "v-mode x temperature  (Spearman rho)", "{:+.2f}")
        _discrete_heatmap(a2, tslope, tp, vlabels, present_tv, bonf,
                          "Theil-Sen slope (color=rho)", "{:+.2g}", color_by_rho=trho)
        fig.colorbar(im, ax=[a1, a2], fraction=0.03, pad=0.02, label="Spearman rho")
        fig.suptitle(f"{args.param_set} / {args.mi_name}: v-modes vs temperature  "
                     f"(n={len(df)}, {nkeep} modes; ** p<{bonf:.1g}, * p<0.05)", fontsize=11)
        pdf.savefig(fig); plt.close(fig)
        # page 2: v-mode autocorrelation
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(0.4 * nkeep + 6, 0.32 * nkeep + 3))
        im = _discrete_heatmap(a1, vrho, vp, vlabels, vlabels, bonf,
                               "v-mode x v-mode  (Spearman rho)", "{:+.2f}")
        _discrete_heatmap(a2, vslope, vp, vlabels, vlabels, bonf,
                          "Theil-Sen slope (color=rho)", "{:+.2g}", color_by_rho=vrho)
        fig.colorbar(im, ax=[a1, a2], fraction=0.03, pad=0.02, label="Spearman rho")
        fig.suptitle(f"{args.param_set} / {args.mi_name}: v-mode autocorrelation  "
                     f"(n={len(df)}, {nkeep} modes)", fontsize=11)
        pdf.savefig(fig); plt.close(fig)
        # pages 3+: full v_i-v_j lower triangle, tiled block x block per page
        block = int(sec["corner_block"])
        n_corner = 0
        for fig in vmode_corner_pages(
                V, vlabels, block, bonf,
                f"{args.mi_name}: v-mode corner ({nkeep} modes, {block}x{block}/page;"
                f" Theil-Sen line, Spearman rho; ** p<{bonf:.1g}, * p<0.05)"):
            pdf.savefig(fig); plt.close(fig); n_corner += 1
        # final page: the strongest individual v_i-v_j pair scatters
        pairs = ranked_pairs(V)[: int(sec["top_pairs"])]
        fig = top_pairs_page(
            V, vlabels, pairs, bonf, 4,
            f"{args.param_set} / {args.mi_name}: top {len(pairs)} v-mode pairs by "
            f"|rho| (robust fit; ** p<{bonf:.1g}, * p<0.05)")
        if fig is not None:
            pdf.savefig(fig); plt.close(fig)

    summ = ([dict(kind="vmode_temp", **r) for r in trows]
            + [dict(kind="vmode_vmode", **r) for r in vrows])
    pd.DataFrame(summ).to_parquet(out_dir / "vmode_correlations_summary.parquet")
    print(f"  wrote vmode_correlations.pdf ({2 + n_corner + 1} pages: "
          f"temp heatmap, autocorr heatmap, {n_corner} corner blocks "
          f"({block}x{block}), top-{len(pairs)} pairs) "
          f"+ _summary.parquet ({len(summ)} rows)", flush=True)


if __name__ == "__main__":
    main()
