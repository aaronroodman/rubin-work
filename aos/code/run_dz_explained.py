#!/usr/bin/env python3
"""Per-visit fraction of the measured Double-Zernike explained by the OFC
sensitivity subspace, for the 22/12 and 50/34 schemes (FAM, per ps/mi).

For output/<ps>/<mi>/fits.parquet, pack the MI-subtracted per-visit DZ into W and,
for each scheme, project onto the OFC u-mode subspace (U_eff) and report the
fraction of DZ power captured:  frac = ||U_eff U_effT W||^2 / ||W||^2  (per visit),
i.e. how much of the measured wavefront is reachable by that DOF/mode truncation.

Schemes: 22/12 (DOF = M2hex+CamHex+M1M3 B1-7+M2 B1-5, keep 12 modes) and
50/34 (all 50 DOF, keep 34).  Emits dz_explained.parquet (per visit: frac_22_12,
frac_50_34, gain) + dz_explained.pdf (fraction vs elevation and vs visit).

This did not exist before -- run_dz_correlations only printed a single
dataset-wide "removed %" for one scheme.  Needs lsst.ts.ofc (RSP).
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from lsst.ts.intrinsic.wavefront import mi_config as mc
from lsst.ts.intrinsic.wavefront import ofc_svd as osv
from astropy.table import QTable

# 22-DOF reduced set: M2 hex(0-4) + Cam hex(5-9) + M1M3 B1-7(10-16) + M2 B1-5(30-34)
DOF22 = list(range(0, 10)) + list(range(10, 17)) + list(range(30, 35))
SCHEMES = {"22_12": (DOF22, 12), "50_34": (None, 34)}   # (n_dof, n_keep)
DEFAULT = dict(dz_prefix="z1toz6", max_coeff_um=2.0)


def dz_coeff_columns(df, prefix):
    pat = re.compile(rf"^{re.escape(prefix)}_z\d+_c\d+$")
    return [c for c in df.columns if pat.match(c)]


def quality_cut(df, prefix, maxc):
    cols = dz_coeff_columns(df, prefix)
    if not cols or not maxc:
        return df
    return df[~(df[cols].abs() > maxc).any(axis=1)]


def _pack_W(df, prefix, kj_grid):
    W = np.full((len(df), len(kj_grid)), np.nan)
    for ci, (k, j) in enumerate(kj_grid):
        col = f"{prefix}_z{j}_c{k}"
        if col in df.columns:
            W[:, ci] = df[col].to_numpy(dtype=float)
    return W


def _fraction_explained(W, svd):
    """Per-row ||proj||^2 / ||W||^2 with NaN zero-filled (matches the aggregate
    'removed %' convention in run_dz_correlations)."""
    W0 = np.nan_to_num(W, nan=0.0)
    A = svd.project_amplitudes(W0)          # (n, n_keep)
    proj = A @ svd.U_eff.T                   # (n, n_kj) reconstruction
    num = np.nansum(proj ** 2, axis=1)
    den = np.nansum(W0 ** 2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


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
        "dz_explained", args.param_set, args.mi_name,
        config_path=(Path(args.analysis_config) if args.analysis_config else None))}
    prefix = sec["dz_prefix"]
    cfg = mc.load_mi_config(args.param_set, args.mi_name,
                            config_path=(Path(args.config) if args.config else None))

    base = Path(args.output_root) / args.param_set / args.mi_name
    out_dir = base / "plots"; out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(base / "fits.parquet").reset_index(drop=True)
    if "visit_quality_pass" in df.columns:
        df = df[df["visit_quality_pass"].astype(bool)].reset_index(drop=True)
    df = quality_cut(df, prefix, sec["max_coeff_um"]).reset_index(drop=True)
    print(f"[dz_explained] {base}  {len(df)} visits", flush=True)

    b = cfg["build"]
    visits = QTable.read(str(base.parent / "visits.parquet"))
    iZs = [int(j) for j in np.asarray(visits["nollIndices"][0]).tolist()]

    out = df[[c for c in ["day_obs", "seq_num", "visit", "alt", "rotator_angle",
                          "band", "science_program"] if c in df.columns]].copy()
    for name, (n_dof, n_keep) in SCHEMES.items():
        svd = osv.build_ofc_svd(iZs, int(b["k_min"]), int(b["k_max"]), n_keep,
                                n_dof=n_dof,
                                ofc_normalization_yaml=b.get("ofc_normalization_yaml"))
        W = _pack_W(df, prefix, svd.kj_grid)
        out[f"frac_{name}"] = _fraction_explained(W, svd)
        print(f"  {name}: n_dof={svd.n_dof} n_keep_eff={svd.n_keep_eff}  "
              f"median frac = {np.nanmedian(out[f'frac_{name}']):.3f}", flush=True)
    out["gain_50_34_over_22_12"] = out["frac_50_34"] - out["frac_22_12"]
    out.to_parquet(out_dir / "dz_explained.parquet", index=False)

    # summary plot: fraction vs elevation, and the marginal gain
    el = np.rad2deg(out["alt"].to_numpy(float)) if "alt" in out else np.arange(len(out))
    with PdfPages(str(out_dir / "dz_explained.pdf")) as pdf:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        for name, c in [("22_12", "C0"), ("50_34", "C3")]:
            ax[0].scatter(el, out[f"frac_{name}"], s=8, alpha=0.6, color=c,
                          label=f"{name} (med {np.nanmedian(out[f'frac_{name}']):.3f})")
        ax[0].set_xlabel("elevation (deg)"); ax[0].set_ylabel("fraction of DZ explained")
        ax[0].set_ylim(0, 1.02); ax[0].legend(fontsize=8); ax[0].set_title("vs elevation")
        ax[1].scatter(el, out["gain_50_34_over_22_12"], s=8, alpha=0.6, color="C2")
        ax[1].set_xlabel("elevation (deg)"); ax[1].set_ylabel("frac(50/34) - frac(22/12)")
        ax[1].set_title("marginal gain 22/12 -> 50/34")
        for name, c in [("22_12", "C0"), ("50_34", "C3")]:
            ax[2].hist(out[f"frac_{name}"].dropna(), bins=30, histtype="step",
                       color=c, label=name)
        ax[2].set_xlabel("fraction of DZ explained"); ax[2].legend(fontsize=8)
        ax[2].set_title("distribution")
        fig.suptitle(f"{args.param_set} / {args.mi_name}: DZ fraction explained "
                     f"(22/12 vs 50/34), n={len(out)}", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95]); pdf.savefig(fig); plt.close(fig)
    print(f"  wrote dz_explained.parquet + dz_explained.pdf", flush=True)


if __name__ == "__main__":
    main()
