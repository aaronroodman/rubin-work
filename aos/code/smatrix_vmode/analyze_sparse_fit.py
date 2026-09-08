#!/usr/bin/env python3
"""analyze_sparse_fit — can a sparse donut fit still constrain the optical state?

A *sparse* donut fit measures only the PRIMARY aberration of each azimuthal family and
fixes the secondary and tertiary terms at nominal. That is attractive -- fewer free
parameters per donut -- but it raises two questions, and this script answers both.

**Part 1, `sensitivity`.** Which degrees of freedom (DOF) drive the secondary and
tertiary aberrations of a family, and do those same DOF also drive a field-CORRELATED
primary of that family? Those are the DOF whose mis-separation would push real
secondary/tertiary content into the primary and so manufacture exactly the radial-order
correlations the Measured Intrinsic Wavefront (MIW) shows -- astig Z5/6 <-> Z12/13 <->
Z23/24, coma Z7/8 <-> Z16/17, and so on. A revised sparse sensitivity matrix has to
handle them.

**Part 2, `observability`.** If the fit reports only the primaries, are the controlled
v-modes still observable at all? The sensitivity matrix is re-formed using only the
primary pupil-Noll rows, and the singular-value spectrum, per-v-mode observability and
per-DOF observability ratio are compared against the full matrix, for both the 50-DOF /
34-v-mode and 22-DOF / 12-v-mode schemes.

Neither part needs any FAM data: both work directly on the ts_ofc DoubleZernike
sensitivity matrix S[k, j, d] (31 field-Zernike k, 29 pupil-Noll j, 50 DOF d). Because
the field basis is orthonormal, the field-map correlation of two pupil Zernikes'
responses to a DOF is just the correlation of their field-coefficient vectors, so no
simulation is required.

Primary terms (first radial order per azimuthal family, Z4-Z28):
  Z4, Z5/6, Z7/8, Z9/10, Z11, Z14/15, Z20/21, Z27/28
Dropped as secondary/tertiary: Z12/13, Z16/17, Z18/19, Z22, Z23/24, Z25/26.

Both parts together are the study, written to
``output/smatrix_vmode/sparse_fit_study.pdf`` -- outside any param_set, since nothing
here depends on FAM data. A single ``--part`` is for a quick look and must be given its
own ``--out``, so that a partial run cannot overwrite the full PDF.

Usage:
  python code/smatrix_vmode/analyze_sparse_fit.py                    # the study
  python code/smatrix_vmode/analyze_sparse_fit.py --part sensitivity --out /tmp/s.pdf

Needs lsst.ts.ofc and $TS_CONFIG_MTTCS_DIR.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# The DM stack, ts_ofc and ts_intrinsic_wavefront all come from the environment on both
# the RSP and the s3df/sdfiana nodes, and that setup exports TS_CONFIG_MTTCS_DIR.
_OFC_CONFIG_FALLBACK = ("/sdf/group/rubin/u/roodman/LSST/packages/ts_config_mttcs"
                        "/MTAOS/v13/ofc")
_ENV_MTTCS = os.environ.get("TS_CONFIG_MTTCS_DIR")
DEFAULT_CONFIG_DIR = (_ENV_MTTCS + "/MTAOS/v13/ofc" if _ENV_MTTCS
                      else _OFC_CONFIG_FALLBACK)

DOF_NAMES = (["M2:dZ", "M2:dX", "M2:dY", "M2:rX", "M2:rY"]
             + ["Cam:dZ", "Cam:dX", "Cam:dY", "Cam:rX", "Cam:rY"]
             + [f"M13b{i+1}" for i in range(20)]
             + [f"M2b{i+1}" for i in range(20)])
DOF_GROUPS = [(0, 5, "M2 hex"), (5, 10, "Cam hex"), (10, 30, "M1M3 bend"),
              (30, 50, "M2 bend")]

# family -> {order: (cos_Noll, sin_Noll)}; sin=None for m=0
FAMILIES = [
    ("Astigmatism", "m=2", {"1st": (6, 5), "2nd": (12, 13), "3rd": (24, 23)}),
    ("Coma", "m=1", {"1st": (8, 7), "2nd": (16, 17)}),
    ("Trefoil", "m=3", {"1st": (10, 9), "2nd": (18, 19)}),
    ("Tetrafoil", "m=4", {"1st": (14, 15), "2nd": (26, 25)}),
    ("Spherical", "m=0", {"1st": (11, None), "2nd": (22, None)}),
]

ZN = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
PRIMARY = [4, 5, 6, 7, 8, 9, 10, 11, 14, 15]              # 1st radial order of each family
WFS_SENSORS = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]   # 4 corner wavefront sensors
HEX = ["dZ", "dX", "dY", "rX", "rY"]


def load_sensitivity(config_dir, instrument="lsst"):
    """Load the DZ sensitivity matrix the way the OFC does (OFCData), returning
    (S[31 field, 29 pupil, 50 dof], provenance str).  Falls back to reading the
    lsst_sensitivity yaml directly if ts_ofc cannot be imported."""
    try:
        from lsst.ts.ofc import OFCData
        ofc = OFCData(instrument, config_dir=config_dir)
        ofc.configure_controller()
        asyncio.run(ofc.configure_instrument(instrument))
        S = np.nan_to_num(np.array(ofc.sensitivity_matrix, dtype=float))
        return S, f"OFCData({instrument}, {config_dir}) -> ofc.sensitivity_matrix"
    except Exception as e:
        import glob as _g
        hit = sorted(_g.glob(f"{config_dir}/sensitivity_matrix/"
                             f"{instrument}_sensitivity*.yaml"))[0]
        print(f"  (OFCData load failed: {e}; falling back to {hit})")
        return np.nan_to_num(np.array(yaml.safe_load(open(hit)), dtype=float)), hit


def power(S, j, d):
    """Total field-response power (const + varying) of pupil Noll j to DOF d."""
    return float(np.sqrt(np.sum(S[1:, j, d] ** 2)))

def pair_power(S, oc, os_, d):
    p = power(S, oc, d) ** 2
    if os_ is not None:
        p += power(S, os_, d) ** 2
    return float(np.sqrt(p))

def field_corr(S, ja, jb, d):
    """Field-map correlation of two pupil Zernikes' responses to DOF d."""
    a, b = S[1:, ja, d], S[1:, jb, d]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


def dof_labels(n_m1m3, n_m2):
    return (["M2:" + h for h in HEX] + ["Cam:" + h for h in HEX]
            + [f"M13b{i+1}" for i in range(n_m1m3)] + [f"M2b{i+1}" for i in range(n_m2)])

def _obs_matrices(ofc, se, sens, sampling):
    """(A_full, A_prim) observation matrices, DOF-normalized.

    sampling='dz': field-complete (raw DZ, all 31 field terms; pupil=Noll).
    sampling='wfs': evaluate at the 4 corner WFS field points (pupil axis = Noll-4)."""
    ndofs = ofc.ndofs
    norm = se.normalization_matrix
    if sampling == "wfs":
        fa = [ofc.sample_points[s] for s in WFS_SENSORS]
        A3d = np.nan_to_num(np.asarray(sens.evaluate(fa, 0.0), float))   # (4, npup, 50), Noll-4
        full = A3d[:, [j - 4 for j in ZN], :]
        prim = A3d[:, [j - 4 for j in PRIMARY], :]
    else:
        dz = np.nan_to_num(np.array(ofc.sensitivity_matrix, float))       # (31,29,50), pupil=Noll
        full = dz[:, ZN, :]
        prim = dz[:, PRIMARY, :]
    A_full = full.reshape(-1, ndofs)[:, ofc.dof_idx] @ norm
    A_prim = prim.reshape(-1, ndofs)[:, ofc.dof_idx] @ norm
    return A_full, A_prim

def build_scheme(config_dir, instrument, n_m1m3, n_m2):
    from lsst.ts.ofc import OFCData, SensitivityMatrix
    from lsst.ts.ofc.state_estimator import StateEstimator
    ofc = OFCData(instrument, config_dir=config_dir)
    ofc.configure_controller()
    asyncio.run(ofc.configure_instrument(instrument))
    ofc.zn_selected = np.array(ZN)                        # the 21 packed measured Zernikes
    ofc.comp_dof_idx = dict(
        m2HexPos=np.ones(5, bool), camHexPos=np.ones(5, bool),
        M1M3Bend=(np.arange(20) < n_m1m3), M2Bend=(np.arange(20) < n_m2))
    se = StateEstimator(ofc)
    sens = SensitivityMatrix(ofc)
    return ofc, se, sens

def scheme_observability(ofc, se, sens, sampling):
    """full/primary singular values, kept-vmode obs ratio, per-DOF ratio, at `sampling`.
    v-modes recomputed on the measured set (not se.Vh, which includes piston/tilt)."""
    A_full, A_prim = _obs_matrices(ofc, se, sens, sampling)
    U, S_full, Vh = np.linalg.svd(A_full, full_matrices=False)
    S_prim = np.linalg.svd(A_prim, compute_uv=False)
    obs = np.array([np.linalg.norm(A_prim @ Vh[m]) for m in range(len(S_full))])
    ratio_vmode = obs / np.where(S_full > 0, S_full, np.nan)
    ratio_dof = np.linalg.norm(A_prim, axis=0) / np.where(
        np.linalg.norm(A_full, axis=0) > 0, np.linalg.norm(A_full, axis=0), np.nan)
    return dict(S_full=S_full, S_prim=S_prim, ratio_vmode=ratio_vmode, ratio_dof=ratio_dof)


def part_sensitivity(args, pdf):

    S, prov = load_sensitivity(args.config_dir, args.instrument)
    nk, nj, nd = S.shape
    print(f"sensitivity via {prov}\n  shape (field {nk}, pupil {nj}, DOF {nd}); "
          f"NaN->0; coma2 Z16 power={np.sqrt(np.sum(S[1:,16,:]**2)):.2f} (all orders present)")

    x = np.arange(nd)
    # page 0: which pupil-Noll terms are populated in the OFC's matrix
    fam_of = {5: "as1", 6: "as1", 7: "co1", 8: "co1", 9: "tr1", 10: "tr1", 11: "sp1",
              12: "as2", 13: "as2", 14: "te1", 15: "te1", 16: "co2", 17: "co2",
              18: "tr2", 19: "tr2", 22: "sp2", 23: "as3", 24: "as3", 25: "te2", 26: "te2"}
    jj = np.arange(4, 29)
    pw_j = np.array([np.sqrt(np.sum(S[1:, j, :] ** 2)) for j in jj])
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(jj, pw_j, 0.7, color="tab:blue")
    ax.set_yscale("log"); ax.set_xticks(jj)
    ax.set_xticklabels([f"Z{j}\n{fam_of.get(j,'')}" for j in jj], fontsize=7)
    ax.set_ylabel("total field-response power over all DOF")
    ax.set_title("Sensitivity population per pupil Noll (the matrix OFCData "
                 "loads: v13 lsst_sensitivity_dz_31_29_50.yaml) -- ALL orders present")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    for name, ml, orders in FAMILIES:
        keys = list(orders)
        # per-order per-DOF power
        pw = {o: np.array([pair_power(S, *orders[o], d) for d in range(nd)])
              for o in keys}
        # primary<->secondary (and tertiary) field correlation per DOF (cos, sin)
        corr = {}
        for hi in keys[1:]:
            cc = np.array([field_corr(S, orders["1st"][0], orders[hi][0], d)
                           for d in range(nd)])
            sc = (np.array([field_corr(S, orders["1st"][1], orders[hi][1], d)
                            for d in range(nd)])
                  if orders[hi][1] is not None else np.full(nd, np.nan))
            corr[hi] = (cc, sc)

        fig, (a1, a2) = plt.subplots(2, 1, figsize=(15, 9))
        # panel 1: field power per DOF for each radial order
        w = 0.8 / len(keys)
        for i, o in enumerate(keys):
            a1.bar(x + i * w, pw[o], w, label=f"{o} (Z{orders[o][0]}"
                   + (f"/{orders[o][1]}" if orders[o][1] else "") + ")")
        a1.set_yscale("log"); a1.set_ylabel("field-response power (native units)")
        a1.set_title(f"{name} ({ml}): per-DOF field response by radial order")
        a1.legend(fontsize=8, ncol=len(keys))
        for lo, hi, lab in DOF_GROUPS:
            a1.axvline(hi - 0.5, color="0.7", lw=0.6, ls=":")
            a1.text((lo + hi) / 2, a1.get_ylim()[1], lab, ha="center", va="top",
                    fontsize=8, color="0.4")

        # panel 2: primary<->secondary field correlation, only where secondary
        # is a real driver (power above threshold)
        thr = args.min_frac * pw[keys[1]].max()
        drv = pw[keys[1]] > thr
        for hi, (cc, sc) in corr.items():
            a2.scatter(x[drv], cc[drv], s=36, label=f"1st-{hi} cos", marker="o")
            ok = drv & np.isfinite(sc)
            a2.scatter(x[ok], sc[ok], s=36, label=f"1st-{hi} sin", marker="^")
        a2.axhline(0, color="k", lw=0.5); a2.set_ylim(-1.05, 1.05)
        a2.set_ylabel("primary<->higher field-map correlation")
        a2.set_xlabel("DOF index")
        a2.set_title(f"{name}: does a secondary-driving DOF also drive a "
                     f"field-correlated primary?  (only DOF with 2nd power > "
                     f"{args.min_frac:.0%} of max)")
        a2.legend(fontsize=8, ncol=4)
        for ax in (a1, a2):
            ax.set_xticks(x[::2]); ax.set_xticklabels([DOF_NAMES[i] for i in x[::2]],
                                                      rotation=90, fontsize=6)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # shortlist: DOF that drive secondary AND a field-correlated primary
        print(f"\n{name} ({ml}) -- DOF driving 2nd (>{args.min_frac:.0%} max) "
              f"and their primary coupling:")
        order2 = np.argsort(pw[keys[1]])[::-1]
        for d in order2:
            if pw[keys[1]][d] <= thr:
                break
            cc, sc = corr[keys[1]]
            ratio = pw[keys[1]][d] / (pw["1st"][d] + 1e-12)
            print(f"  {DOF_NAMES[d]:8s}: 2nd={pw[keys[1]][d]:7.2f}  1st={pw['1st'][d]:7.2f}"
                  f"  2nd/1st={ratio:5.2f}  corr(1st,2nd) cos={cc[d]:+.2f} "
                  f"sin={sc[d] if np.isfinite(sc[d]) else float('nan'):+.2f}")


def part_observability(args, pdf):

    schemes = [("50/34", 20, 20, 34), ("22/12", 7, 5, 12)]
    samplings = [("wfs", "4 corner WFS"), ("dz", "field-complete")]
    clr = lambda r: "tab:red" if r < 0.5 else "tab:orange" if r < 0.8 else "tab:green"
    for tag, n_m1m3, n_m2, n_keep in schemes:
        ofc, se, sens = build_scheme(args.config_dir, args.instrument, n_m1m3, n_m2)
        labels = dof_labels(n_m1m3, n_m2)
        for smp, smp_lab in samplings:
            R = scheme_observability(ofc, se, sens, smp)
            nd = len(R["S_full"])
            nk = min(n_keep, nd)
            print(f"\n=== {tag} scheme, {smp_lab} (nDOF={nd}, keep {n_keep}) ===")
            print(f"  kept v-modes: min primary/full obs = "
                  f"{np.nanmin(R['ratio_vmode'][:nk]):.2f}  "
                  f"(median {np.nanmedian(R['ratio_vmode'][:nk]):.2f})")
            worst = np.argsort(R["ratio_dof"])[:6]
            print("  DOF most degraded:",
                  {labels[i]: round(float(R['ratio_dof'][i]), 2) for i in worst})

            fig, ax = plt.subplots(1, 3, figsize=(18, 5))
            ax[0].semilogy(np.arange(1, len(R["S_full"]) + 1), R["S_full"],
                           "o-", ms=4, label="full aberrations")
            ax[0].semilogy(np.arange(1, len(R["S_prim"]) + 1), R["S_prim"],
                           "s-", ms=4, label="primary only")
            ax[0].axvline(nk + 0.5, color="k", ls="--", lw=1, label=f"keep {n_keep}")
            ax[0].set_xlabel("v-mode index"); ax[0].set_ylabel("singular value")
            ax[0].set_title(f"{tag}: v-mode SVD spectrum"); ax[0].legend(fontsize=8)
            ax[0].grid(alpha=0.3)

            mk = np.arange(1, nk + 1)
            ax[1].bar(mk, R["ratio_vmode"][:nk], color=[clr(r) for r in R["ratio_vmode"][:nk]])
            ax[1].axhline(1, color="k", lw=0.5); ax[1].set_ylim(0, 1.1)
            ax[1].set_xlabel("kept v-mode index"); ax[1].set_ylabel("primary/full observability")
            ax[1].set_title(f"{tag}: observability of the {nk} CONTROLLED v-modes")
            ax[1].grid(alpha=0.3, axis="y")

            xd = np.arange(len(labels))
            ax[2].bar(xd, R["ratio_dof"], color=[clr(r) for r in R["ratio_dof"]])
            ax[2].axhline(1, color="k", lw=0.5); ax[2].set_ylim(0, 1.1)
            ax[2].set_xticks(xd); ax[2].set_xticklabels(labels, rotation=90, fontsize=6)
            ax[2].set_ylabel("primary/full observability")
            ax[2].set_title(f"{tag}: per-DOF observability")
            ax[2].grid(alpha=0.3, axis="y")

            fig.suptitle(f"Sparse (primary-only) DOF observability -- {tag} scheme, "
                         f"{smp_lab} sampling  (green ok, orange weakened, red lost)",
                         fontsize=13)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="both",
                    choices=["sensitivity", "observability", "both"],
                    help="which half of the sparse-fit question to answer; the default "
                         "'both' is the full study, and a single part is for a quick "
                         "look and needs --out so it cannot overwrite the full PDF")
    ap.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR,
                    help="OFC config dir (v13); defaults to $TS_CONFIG_MTTCS_DIR")
    ap.add_argument("--instrument", default="lsst")
    ap.add_argument("--out", default=None,
                    help="output PDF; default output/smatrix_vmode/sparse_fit_study.pdf")
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--min-frac", type=float, default=0.05,
                    help="[sensitivity] report DOF with at least this fraction of the "
                         "family's field power (dimensionless)")
    args = ap.parse_args()

    if args.part != "both" and not args.out:
        ap.error(f"--part {args.part} writes only half the study, so it needs an "
                 f"explicit --out rather than overwriting sparse_fit_study.pdf")

    out = (Path(args.out) if args.out
           else Path(args.output_root) / "smatrix_vmode" / "sparse_fit_study.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(out)) as pdf:
        if args.part in ("sensitivity", "both"):
            print("=== part 1: which DOF drive secondary/tertiary aberrations ===")
            part_sensitivity(args, pdf)
        if args.part in ("observability", "both"):
            print("\n=== part 2: are the v-modes observable from primaries alone ===")
            part_observability(args, pdf)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
