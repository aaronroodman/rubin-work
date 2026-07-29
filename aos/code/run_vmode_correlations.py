#!/usr/bin/env python3
"""v-mode <-> temperature correlations + v-mode autocorrelation (FAM, per ps/mi).

Projects the MI-subtracted per-visit Double-Zernikes in output/<ps>/<mi>/fits.parquet
onto the OFC v-modes, then correlates each v-mode against (a) the temperature
telemetry suite and (b) the other v-modes (autocorrelation).  Unlike
thermal_correlations (Pearson) this uses the ROBUST Spearman rho + Theil-Sen
slope.

Everything is produced for TWO OFC subspaces and written to scheme-tagged files:
  50_34 -> all 50 DOF, keep 34 modes   (vmode_correlations_50_34.pdf)
  22_12 -> reduced 22 DOF, keep 12      (vmode_correlations_22_12.pdf)

Pages per scheme (each summary panel one-to-a-page, equal aspect, own colorbar):
  1  v-mode x temperature  Spearman rho heatmap
  2  v-mode x temperature  Theil-Sen slope heatmap (colored by rho)
  3  v-mode autocorrelation Spearman rho heatmap (lower triangle only)
  4  v-mode autocorrelation Theil-Sen slope heatmap (lower triangle, color=rho)
  5+ v_i-v_j lower-triangle corner tiled `corner_block` x `corner_block` per page
  .  top-|rho| individual v-mode pair scatters
  .  top-|rho| v-mode x thermal scatters
  .  v1 vs every thermal-telemetry item
Also  vmode_correlations_summary_<tag>.parquet  (kind, mode_i, term, rho, slope, p, n).

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
               corner_block=6,       # v_i-v_j corner: corner_block^2 panels/page
               top_pairs=16,         # strongest |rho| v-mode pairs as scatters
               top_thermal_pairs=16) # strongest |rho| v-mode x thermal scatters

# 22-DOF reduced set: M2 hex(0-4) + Cam hex(5-9) + M1M3 B1-7(10-16) + M2 B1-5(30-34)
DOF22 = list(range(0, 10)) + list(range(10, 17)) + list(range(30, 35))
SCHEMES = [("50_34", None, 34), ("22_12", DOF22, 12)]   # (tag, n_dof, n_keep)


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


def _fit(x, y):
    """Robust Spearman rho + Theil-Sen (slope, intercept) on finite pairs."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 5:
        return None
    rho, p = spearmanr(x[m], y[m])
    slope, intercept, _, _ = theilslopes(y[m], x[m])
    return dict(rho=float(rho), p=float(p), slope=float(slope),
                intercept=float(intercept), n=int(m.sum()))


def _star(p, bonf):
    return "**" if (np.isfinite(p) and p < bonf) else (
           "*" if (np.isfinite(p) and p < 0.05) else "")


def _corr_grid(V, cols, xs):
    """rho/slope/p matrices of each v-mode (rows) vs each column series in xs."""
    nR, nC = V.shape[1], len(cols)
    rho = np.full((nR, nC), np.nan); slope = np.full((nR, nC), np.nan)
    pval = np.full((nR, nC), np.nan)
    rows = []
    for i in range(nR):
        for c, name in enumerate(cols):
            f = _fit(xs[c], V[:, i])
            if f:
                rho[i, c], slope[i, c], pval[i, c] = f["rho"], f["slope"], f["p"]
                rows.append(dict(mode_i=i + 1, term=name, rho=f["rho"],
                                 slope=f["slope"], p=f["p"], n=f["n"]))
    return rho, slope, pval, rows


# ---------------------------------------------------------------- heatmap pages
def _discrete_heatmap(ax, M, P, row_labels, col_labels, bonf, title, fmt,
                      color_by_rho=None, aspect="equal"):
    """Annotate cells with M on a discrete -1..1 RdBu_r scale (colored by rho =
    M unless color_by_rho given); bold + star cells with p<0.05, ** if p<bonf.
    NaN cells (e.g. the blanked upper triangle) render as light grey."""
    C = color_by_rho if color_by_rho is not None else M
    levels = np.linspace(-1, 1, 11)
    cmap = plt.get_cmap("RdBu_r", len(levels) - 1).copy()
    cmap.set_bad("0.9")
    norm = BoundaryNorm(levels, cmap.N)
    im = ax.imshow(C, cmap=cmap, norm=norm, aspect=aspect)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]):
                continue
            p = P[i, j]
            star = _star(p, bonf)
            tc = "white" if (np.isfinite(C[i, j]) and abs(C[i, j]) > 0.6) else "black"
            ax.text(j, i, fmt.format(M[i, j]) + star, ha="center", va="center",
                    fontsize=6, color=tc, weight=("bold" if star else "normal"))
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=7, rotation=90)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_title(title, fontsize=11)
    return im


def heatmap_page(M, P, row_labels, col_labels, bonf, title, fmt, cbar_label,
                 color_by_rho=None, lower_only=False):
    """One heatmap to a page, equal aspect, with its own colorbar.  lower_only
    blanks the strictly-upper triangle (symmetric v-mode autocorrelation)."""
    M = np.array(M, float)
    C = np.array(color_by_rho, float) if color_by_rho is not None else M.copy()
    if lower_only:
        r = np.arange(M.shape[0])[:, None]; c = np.arange(M.shape[1])[None, :]
        up = c > r
        M[up] = np.nan; C[up] = np.nan
    nrow, ncol = M.shape
    fig, ax = plt.subplots(figsize=(max(6.0, 0.42 * ncol + 4.0),
                                    max(5.0, 0.42 * nrow + 3.0)))
    im = _discrete_heatmap(ax, M, P, row_labels, col_labels, bonf, title, fmt,
                           color_by_rho=C, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label,
                 ticks=np.linspace(-1, 1, 11))
    fig.tight_layout()
    return fig


# ------------------------------------------------------------- scatter helpers
def _scatter_panel(ax, x, y, bonf):
    """One v_col(x)-vs-v_row(y) scatter with the robust Theil-Sen line and the
    Spearman rho annotation (bold red for |rho|>=0.3, significance stars)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
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
    panels.  Diagonal block-pages are corners (histograms on the diagonal, upper
    corner blank); off-diagonal block-pages are full scatter grids.  Yields one
    figure at a time so the caller can save+close before the next is built."""
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


def ranked_vmode_thermal(V, present_tv, xs):
    """All (v-mode, thermal) pairs with a robust fit, sorted by |rho| desc."""
    out = []
    for c, name in enumerate(present_tv):
        for i in range(V.shape[1]):
            f = _fit(xs[c], V[:, i])
            if f is not None:
                out.append({"mode": i, "term": name, **f})
    out.sort(key=lambda d: -abs(d["rho"]))
    return out


def scatter_list_page(specs, ncol, bonf, title):
    """Draw a list of scatters (each spec: x, y, xlabel, ylabel) as a grid, with
    the robust Theil-Sen line and a rho/slope/n title per panel."""
    k = len(specs)
    if k == 0:
        return None
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.9 * nrow),
                             squeeze=False)
    for idx in range(nrow * ncol):
        ax = axes[idx // ncol][idx % ncol]
        if idx >= k:
            ax.axis("off"); continue
        sp = specs[idx]
        x = np.asarray(sp["x"], float); y = np.asarray(sp["y"], float)
        mk = np.isfinite(x) & np.isfinite(y)
        xx, yy = x[mk], y[mk]
        ax.scatter(xx, yy, s=9, alpha=0.4, color="C0", edgecolors="none",
                   rasterized=True)
        f = _fit(x, y)
        if f is not None:
            xr = np.array([xx.min(), xx.max()])
            ax.plot(xr, f["intercept"] + f["slope"] * xr, color="C3", lw=1.2)
            ax.set_title(f"rho={f['rho']:+.2f}{_star(f['p'], bonf)}  "
                         f"slope={f['slope']:+.2g}  n={f['n']}", fontsize=8)
        else:
            ax.set_title(f"{sp['ylabel']} vs {sp['xlabel']}", fontsize=8)
        ax.set_xlabel(sp["xlabel"], fontsize=8)
        ax.set_ylabel(sp["ylabel"], fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


# ----------------------------------------------------------------- per scheme
def run_scheme(tag, n_dof, n_keep, df, prefix, present_tv, iZs, b, sec,
               out_dir, ps, mi):
    svd = osv.build_ofc_svd(iZs, int(b["k_min"]), int(b["k_max"]), n_keep,
                            n_dof=n_dof,
                            ofc_normalization_yaml=b.get("ofc_normalization_yaml"))
    V = project_vmodes(df, prefix, svd)
    nkeep = V.shape[1]
    vlabels = [f"v{i + 1}" for i in range(nkeep)]
    xs = [df[tv].to_numpy(float) for tv in present_tv]
    xs_by = dict(zip(present_tv, xs))

    trho, tslope, tp, trows = _corr_grid(V, present_tv, xs)
    vrho, vslope, vp, vrows = _corr_grid(V, vlabels, [V[:, i] for i in range(nkeep)])
    np.fill_diagonal(vrho, np.nan); np.fill_diagonal(vslope, np.nan)

    npairs = len(present_tv) * nkeep + nkeep * (nkeep - 1) // 2
    bonf = sec["alpha"] / max(1, npairs)
    pref = f"{ps}/{mi} [{tag}]"
    sig = f"n={len(df)}, {nkeep} modes; ** p<{bonf:.1g}, * p<0.05"

    pdf_path = out_dir / f"vmode_correlations_{tag}.pdf"
    npages = 0
    with PdfPages(str(pdf_path)) as pdf:
        # -- one summary heatmap to a page, equal aspect, own colorbar --
        pdf.savefig(heatmap_page(
            trho, tp, vlabels, present_tv, bonf,
            f"{pref}: v-mode x temperature  (Spearman rho; {sig})",
            "{:+.2f}", "Spearman rho")); plt.close(); npages += 1
        pdf.savefig(heatmap_page(
            tslope, tp, vlabels, present_tv, bonf,
            f"{pref}: v-mode x temperature  (Theil-Sen slope, color=rho; {sig})",
            "{:+.2g}", "Spearman rho", color_by_rho=trho)); plt.close(); npages += 1
        pdf.savefig(heatmap_page(
            vrho, vp, vlabels, vlabels, bonf,
            f"{pref}: v-mode autocorrelation  (Spearman rho, lower triangle; {sig})",
            "{:+.2f}", "Spearman rho", lower_only=True)); plt.close(); npages += 1
        pdf.savefig(heatmap_page(
            vslope, vp, vlabels, vlabels, bonf,
            f"{pref}: v-mode autocorrelation  (Theil-Sen slope, color=rho, "
            f"lower triangle; {sig})",
            "{:+.2g}", "Spearman rho", color_by_rho=vrho, lower_only=True))
        plt.close(); npages += 1

        # -- v_i-v_j corner, tiled block x block per page --
        block = int(sec["corner_block"])
        for fig in vmode_corner_pages(
                V, vlabels, block, bonf,
                f"{pref}: v-mode corner ({nkeep} modes, {block}x{block}/page; {sig})"):
            pdf.savefig(fig); plt.close(fig); npages += 1

        # -- strongest individual v-mode pair scatters --
        pairs = ranked_pairs(V)[: int(sec["top_pairs"])]
        fig = scatter_list_page(
            [dict(x=V[:, p["i"]], y=V[:, p["j"]],
                  xlabel=vlabels[p["i"]], ylabel=vlabels[p["j"]]) for p in pairs],
            4, bonf, f"{pref}: top {len(pairs)} v-mode pairs by |rho|")
        if fig is not None:
            pdf.savefig(fig); plt.close(fig); npages += 1

        # -- strongest v-mode x thermal scatters --
        vt = ranked_vmode_thermal(V, present_tv, xs)[: int(sec["top_thermal_pairs"])]
        fig = scatter_list_page(
            [dict(x=xs_by[r["term"]], y=V[:, r["mode"]],
                  xlabel=r["term"], ylabel=vlabels[r["mode"]]) for r in vt],
            4, bonf, f"{pref}: top {len(vt)} v-mode x thermal by |rho|")
        if fig is not None:
            pdf.savefig(fig); plt.close(fig); npages += 1

        # -- v1 vs every thermal telemetry item --
        fig = scatter_list_page(
            [dict(x=xs_by[tv], y=V[:, 0], xlabel=tv, ylabel="v1")
             for tv in present_tv],
            4, bonf, f"{pref}: v1 vs all thermal telemetry ({len(present_tv)} items)")
        if fig is not None:
            pdf.savefig(fig); plt.close(fig); npages += 1

    summ = ([dict(kind="vmode_temp", **r) for r in trows]
            + [dict(kind="vmode_vmode", **r) for r in vrows])
    pd.DataFrame(summ).to_parquet(out_dir / f"vmode_correlations_summary_{tag}.parquet")
    print(f"  [{tag}] n_dof={svd.n_dof} n_keep_eff={nkeep}: "
          f"vmode_correlations_{tag}.pdf ({npages} pages) + "
          f"_summary_{tag}.parquet ({len(summ)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--param-set", required=True)
    ap.add_argument("--mi-name", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--analysis-config", default=None, dest="analysis_config")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--schemes", nargs="+", default=None,
                    help="subset of scheme tags to run (default: all: "
                         + ", ".join(t for t, _, _ in SCHEMES) + ")")
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

    b = cfg["build"]
    visits = QTable.read(str(base.parent / "visits.parquet"))
    iZs = [int(j) for j in np.asarray(visits["nollIndices"][0]).tolist()]

    want = set(args.schemes) if args.schemes else None
    for tag, n_dof, n_keep in SCHEMES:
        if want and tag not in want:
            continue
        run_scheme(tag, n_dof, n_keep, df, prefix, present_tv, iZs, b, sec,
                   out_dir, args.param_set, args.mi_name)


if __name__ == "__main__":
    main()
