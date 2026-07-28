#!/usr/bin/env python3
"""v-mode <-> temperature correlations + v-mode autocorrelation (FAM, per ps/mi).

Projects the MI-subtracted per-visit Double-Zernikes in output/<ps>/<mi>/fits.parquet
onto the OFC v-modes (the build's n_dof/n_keep, e.g. 50/34), then correlates each
v-mode against (a) the temperature telemetry suite and (b) the other v-modes
(autocorrelation).  Unlike thermal_correlations (Pearson) this uses the ROBUST
Spearman rho + Theil-Sen slope, and gives one v-mode-centric summary: discrete
-1..1 rho heatmaps + slope heatmaps with statistically-significant cells flagged.

Pages: (1) v-mode x temperature rho + slope, (2) v-mode x v-mode rho + slope.
Also  vmode_correlations_summary.parquet  (kind, mode_i, term, rho, slope, p, n).

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
               thermal_vars=list(DEFAULT_THERMAL_VARS), alpha=0.05)


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

    summ = ([dict(kind="vmode_temp", **r) for r in trows]
            + [dict(kind="vmode_vmode", **r) for r in vrows])
    pd.DataFrame(summ).to_parquet(out_dir / "vmode_correlations_summary.parquet")
    print(f"  wrote vmode_correlations.pdf + _summary.parquet ({len(summ)} rows)",
          flush=True)


if __name__ == "__main__":
    main()
