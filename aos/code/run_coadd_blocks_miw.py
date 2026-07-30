#!/usr/bin/env python3
"""Per-FAM-block MIW-style Z5-Z8 focal-plane maps (Path A, 50/34, 3 iterations).

Purpose: the standard measured-intrinsic (MIW) build for pathA_50_34_i(_5rot)
pools ALL FAM visits in a camera-rotator bin (over many nights) and extracts the
remnant wavefront after the 3-iteration 50/34 optical-state correction.  That
mixes nights.  Here we run the SAME iterative analysis but aggregate differently:
one build PER CONTIGUOUS FAM BLOCK (a T614 block ~ 12 triplets at a single
pointing/rotator, one night).  This yields one MIW-style remnant map per block so
we can see whether the anomalous wavefront seen in 4 of the 9 rotator-bin MIW
maps (the out-of-family bins, ~20260409) is confined to that day_obs or recurs.

Reuse: the iterative optical-state removal is the package core
`measured_intrinsic.build_measured_intrinsic_uconstrained` (unchanged) with the
resolved mi_config for the chosen mi_name; only the visit SELECTION differs
(per block instead of per rotator window).  Z5-Z8 are read straight off the
final-iteration `measured_grid` (no CCD-height/Z4 step needed for j>=5).

Block definition (NO elevation or rotator cut -- take every block):
  * Standard programs (e.g. T614): a block is a set of visits with the same
    program, ~equal (alt, az, rotator), and a seq_num span within ~3*12=36 --
    so missing triplets inside the set are fine (grouped by pointing, not by
    strict seq spacing).
  * Bounce programs (T720/T724): the run alternates between two pointings over a
    wide seq span, so it is split by the bounced state instead -- all visits at
    one (alt, rotator) state form one block, the other state its own block
    (e.g. an elevation bounce 70<->40 -> a 70 block + a 40 block).

Each coadd's Z5-Z8 remnant is compared to the MIW reference (the per-donut MIW
sidecar binned on the same OCS grid): per-Zernike residual RMS (um) + spatial
Pearson r, plus a combined pooled-RMS and mean-r.  Blocks are ordered in time
(day_obs, seq) as a "coadd ordinal".

Outputs (into output/<ps>/<out-name>/, default out-name=coadd_50_34):
  blocks_summary.csv              every detected block: program, day_obs, seq
                                  range, n_visits, alt/az/rot means, median-blur
                                  FWHM, in-5rot-family flag  (also printed)
  coadd_metrics.parquet           one row per coadd: metadata + ordinal + per-Z
                                  and combined resid RMS / corr + vmode_1..34
  coadd_blocks_miw_perblock.pdf   one block per page: rows coadd / MIW / diff x
                                  Z5..Z8, metrics in titles, ordered by day/seq
  coadd_blocks_miw_timeseries.pdf metrics + telemetry (alt/az/rot/blur/
                                  z_gradient) + per-v-mode value pages (5/page),
                                  all vs coadd ordinal on a common x-scale
  block_grids.npz                 stacked coadd+MIW grids, v-modes, metadata

RSP-only (needs lsst.ts.intrinsic.wavefront + lsst.ts.ofc); requires the mi_name
MIW sidecar (zk_intrinsic.parquet) to exist.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.table import QTable

from lsst.ts.intrinsic.wavefront import ofc_svd as osv
from lsst.ts.intrinsic.wavefront import mi_config as mc
from lsst.ts.intrinsic.wavefront import intrinsic_build_plots as ibp
from lsst.ts.intrinsic.wavefront.dz_fitting import derive_noll_indices
from lsst.ts.intrinsic.wavefront.measured_intrinsic import (
    build_measured_intrinsic_uconstrained, bin_median_focal)

Z_TERMS = [5, 6, 7, 8]


def _fixed_list_to_2d(table, col):
    """A fixed-length list<double> arrow column -> (n, L) float array."""
    arr = table[col].combine_chunks()
    flat = arr.values.to_numpy(zero_copy_only=False).astype(float)
    return flat.reshape(len(table), -1)


# ------------------------------------------------------- donut loading (fast)
def _visit_row_groups(donuts_parquet):
    """(day_obs, seq_num) -> row_group index, from per-visit column stats.
    (Same technique as run_build_intrinsic.load_kept_donuts, built once.)"""
    pf = pq.ParquetFile(str(donuts_parquet))
    lut = {}
    for i in range(pf.num_row_groups):
        meta = pf.metadata.row_group(i)
        d = s = None
        for ci in range(meta.num_columns):
            cm = meta.column(ci)
            if cm.path_in_schema == "day_obs" and cm.statistics is not None:
                d = cm.statistics.min
            elif cm.path_in_schema == "seq_num" and cm.statistics is not None:
                s = cm.statistics.min
        if d is not None and s is not None:
            lut[(int(d), int(s))] = i
    return pf, lut


def load_block_donuts(pf, lut, keys):
    """Concatenate only the donut row groups for this block's (day_obs, seq_num)."""
    frames = [pf.read_row_group(lut[k]).to_pandas() for k in keys if k in lut]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------- block grouping
def _circ_mean_deg(a):
    """Circular mean of angles in degrees (for azimuth)."""
    a = np.deg2rad(np.asarray(a, float))
    return float(np.mod(np.rad2deg(np.angle(np.mean(np.exp(1j * a)))), 360.0))


def _wrapdiff(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def assign_blocks(visits_pd, allowed_bands, bounce_programs, pointing_tol,
                  max_seq_span, bounce_round, wide_programs=(), wide_tol=20.0,
                  max_blur=None):
    """Assign every FAM visit to a block (NO elevation / rotator cut).

    Standard program: greedy pointing-set -- a new block starts when the program
    changes, the (alt, az, rot) drifts beyond tolerance, or seq_num reaches
    `max_seq_span` past the block's first visit (>= , so a 12-triplet block
    spanning 33 stays whole and a back-to-back repeat splits cleanly; missing
    triplets inside the span are tolerated).  `wide_programs` (T710/T710_v2,
    whose image plan drifts in az/rotator) use `wide_tol` on az and rot instead
    of `pointing_tol`.  Bounce program (T720/T724): one block per rounded
    (alt, rot) state, combining all visits at that state regardless of seq span.
    Optional `max_blur` drops visits with median_blur_arcsec above it.
    Returns the DataFrame with a 'block' integer label + deg columns."""
    v = visits_pd.dropna(subset=["alt", "az", "rotator_angle"]).copy()
    if allowed_bands and "band" in v.columns:
        v = v[v["band"].astype(str).isin(set(allowed_bands))]
    if max_blur is not None and "median_blur_arcsec" in v.columns:
        v = v[v["median_blur_arcsec"].to_numpy(float) <= float(max_blur)]
    v["alt_deg"] = np.rad2deg(v["alt"].to_numpy(float))
    v["az_deg"] = np.mod(np.rad2deg(v["az"].to_numpy(float)), 360.0)
    v["rot_deg"] = v["rotator_angle"].to_numpy(float)
    v = v.sort_values(["science_program", "day_obs", "seq_num"]).reset_index(drop=True)

    block = np.full(len(v), -1, dtype=int)
    seq = v["seq_num"].to_numpy(float)
    alt = v["alt_deg"].to_numpy(); az = v["az_deg"].to_numpy(); rot = v["rot_deg"].to_numpy()
    wide = set(wide_programs)
    nb = 0
    for (pr, day), pos in v.groupby(["science_program", "day_obs"]).indices.items():
        pos = np.sort(np.asarray(pos))
        if str(pr) in bounce_programs:
            # split by bounced (alt, rot) state; combine all visits per state
            seen = {}
            for p in pos:
                key = (round(alt[p] / bounce_round) * bounce_round,
                       round(rot[p] / bounce_round) * bounce_round)
                if key not in seen:
                    seen[key] = nb; nb += 1
                block[p] = seen[key]
        else:
            # T710 family: image plan drifts in az/rot over a long session, so
            # use a wide az/rot tolerance AND drop the seq-span split (let the
            # az/rot tolerance bound the block instead) to keep each session one
            # block.  Standard programs keep tight tol + the ~36 seq-span split.
            atol = pointing_tol
            if str(pr) in wide:
                aztol = rtol = wide_tol; span = np.inf
            else:
                aztol = rtol = pointing_tol; span = max_seq_span
            cur = None; start_seq = None; ref = None
            for p in pos:
                new = (start_seq is None
                       or (seq[p] - start_seq) >= span
                       or abs(alt[p] - ref[0]) > atol
                       or _wrapdiff(az[p], ref[1]) > aztol
                       or abs(rot[p] - ref[2]) > rtol)
                if new:
                    cur = nb; nb += 1; start_seq = seq[p]; ref = (alt[p], az[p], rot[p])
                block[p] = cur
    v["block"] = block
    return v


def in_family(rot, windows):
    return bool(any(lo <= rot <= hi for lo, hi in (windows or [])))


def block_summary(v, rot_windows):
    """Per-block summary DataFrame (one row per block), from the visit table."""
    rows = []
    for blk, g in v.groupby("block"):
        rot = float(np.mean(g["rot_deg"]))
        rows.append(dict(
            block=int(blk),
            program=str(g["science_program"].iloc[0]).replace("BLOCK-", ""),
            day_obs=int(g["day_obs"].iloc[0]),
            seq_min=int(g["seq_num"].min()), seq_max=int(g["seq_num"].max()),
            n_visits=int(len(g)),
            alt=float(np.mean(g["alt_deg"])), az=_circ_mean_deg(g["az_deg"]),
            rot=rot,
            fwhm=float(np.nanmedian(g["median_blur_arcsec"]))
            if "median_blur_arcsec" in g else np.nan,
            z_gradient=float(np.nanmedian(g["z_gradient"]))
            if "z_gradient" in g else np.nan,
            in_family=in_family(rot, rot_windows)))
    return pd.DataFrame(rows).sort_values(["day_obs", "seq_min"]).reset_index(drop=True)


# ------------------------------------------------------------- plotting
def _term_scales(blocks, pct):
    """Common per-term |vmax| across all block grids (percentile of |median|)."""
    vmax = {}
    for z in Z_TERMS:
        vals = [np.abs(b["grid"][z][np.isfinite(b["grid"][z])]).ravel()
                for b in blocks if np.isfinite(b["grid"][z]).any()]
        v = np.concatenate(vals) if vals else np.array([0.0])
        vmax[z] = float(np.nanpercentile(v, pct)) if v.size else 1.0
        vmax[z] = vmax[z] or 1.0
    return vmax


def _pearson(a, b):
    if a.size < 5 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def map_metrics(coadd, miw):
    """Per-Zernike residual RMS (um) and spatial Pearson r over co-valid cells,
    plus a pooled-RMS and mean-r combined metric."""
    m = {}; diffs = []; rs = []
    for z in Z_TERMS:
        c, w = coadd[z], miw[z]
        ok = np.isfinite(c) & np.isfinite(w)
        if int(ok.sum()) < 5:
            m[f"resid_rms_z{z}"] = np.nan; m[f"corr_z{z}"] = np.nan; continue
        d = c[ok] - w[ok]
        m[f"resid_rms_z{z}"] = float(np.sqrt(np.mean(d ** 2)))
        m[f"corr_z{z}"] = _pearson(c[ok], w[ok])
        diffs.append(d); rs.append(m[f"corr_z{z}"])
    alld = np.concatenate(diffs) if diffs else np.array([np.nan])
    m["resid_rms_combined"] = float(np.sqrt(np.mean(alld ** 2))) if diffs else np.nan
    m["corr_combined"] = float(np.nanmean(rs)) if rs else np.nan
    return m


def miw_ref_grid(dd, mi_full, coord, iZidx, n_bins, fp_grid):
    """Bin the per-donut MIW sidecar (row-aligned to dd) on the SAME focal grid
    as the coadd remnant -> {z: 2D} reference map for Z5-Z8."""
    thx = np.rad2deg(np.asarray(dd[f"thx_{coord}"], float))
    thy = np.rad2deg(np.asarray(dd[f"thy_{coord}"], float))
    grid, *_ = bin_median_focal(thx, thy, mi_full, iZidx,
                                n_bins=n_bins, fp_radius=fp_grid)
    return {z: grid.get(z) for z in Z_TERMS}


def block_vmodes(final, svd):
    """Block-median v-mode amplitudes (a/sigma) from the per-visit raw DZ fits,
    projected onto the OFC subspace (same convention as vmode_correlations)."""
    W = ibp.stack_per_visit_coeffs(final["fit_rows_raw"], list(svd.kj_grid))
    V = svd.vmodes(svd.project_amplitudes(np.nan_to_num(W, nan=0.0)))
    return np.nanmedian(V, axis=0)


def render_perblock(blocks, xbins, ybins, out_pdf, pct):
    """One block per page: rows = coadd remnant / MIW reference / difference,
    cols = Z5..Z8.  Common per-term color scale (all three rows) so good
    agreement reads faint; the difference row's titles carry the residual RMS
    and spatial correlation.  Blocks arrive in day_obs/seq order."""
    vmax = _term_scales(blocks, pct)
    with PdfPages(out_pdf) as pdf:
        for b in blocks:
            fig, axes = plt.subplots(3, 4, figsize=(16, 11.5))
            rows = [("coadd", b["grid"]), ("MIW ref", b["miw"]),
                    ("coadd - MIW", {z: b["grid"][z] - b["miw"][z] for z in Z_TERMS})]
            for r, (rlab, grd) in enumerate(rows):
                for c, z in enumerate(Z_TERMS):
                    ax = axes[r][c]
                    pcm = ax.pcolormesh(xbins, ybins, grd[z].T, cmap="RdBu_r",
                                        vmin=-vmax[z], vmax=vmax[z], shading="flat")
                    ax.set_aspect("equal"); ax.tick_params(labelsize=5)
                    if r == 0:
                        ax.set_title(f"Z{z}", fontsize=11)
                    if r == 2:
                        ax.set_title(f"rms={b['metrics'][f'resid_rms_z{z}']:.3f}  "
                                     f"r={b['metrics'][f'corr_z{z}']:+.2f}", fontsize=8)
                    if c == 0:
                        ax.set_ylabel(rlab, fontsize=10)
                    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
            fam = "" if b["in_family"] else "   [out-of-5rot-family]"
            fig.suptitle(
                f"{b['program']}  {b['day_obs']}  seq {b['seq_min']}-{b['seq_max']}"
                f"   alt={b['alt']:.1f} az={b['az']:.1f} rot={b['rot']:+.1f}"
                f"   FWHM={b['fwhm']:.2f}\"  n_vis={b['n_visits']} n_don={b['n_donuts']}"
                f"   [ordinal {b['ordinal']}]{fam}\n"
                f"combined resid RMS={b['metrics']['resid_rms_combined']:.3f} um   "
                f"mean r={b['metrics']['corr_combined']:+.2f}   "
                f"(Path A 50/34, 3 iter, OCS)", fontsize=11,
                color=("black" if b["in_family"] else "tab:red"))
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig); plt.close(fig)
    print(f"wrote {out_pdf}  ({len(blocks)} blocks x [coadd/MIW/diff]; per-term "
          f"|vmax| Z5-8 = {', '.join(f'{vmax[z]:.3f}' for z in Z_TERMS)} um)",
          flush=True)


def render_timeseries(blocks, out_pdf):
    """Comparison metrics, block telemetry, and per-v-mode value plots, all vs
    coadd ordinal on a COMMON horizontal scale.  Single-column panels; day_obs
    boundaries drawn as dotted vlines on every panel (out-of-family blocks
    shaded), with day_obs labelled on the top panel of each page.  v-modes are
    5 panels/page (7 pages for 34)."""
    n = len(blocks); ordn = np.arange(n)
    days = np.array([b["day_obs"] for b in blocks])
    outfam = np.array([not b["in_family"] for b in blocks])
    Vm = np.array([b["vmodes"] for b in blocks])                 # (n, n_keep)
    nkeep = Vm.shape[1]
    W = max(12.0, 0.30 * n + 3.0)                                # common width
    xlim = (-0.6, n - 0.4)                                       # common x-range
    bnd = np.where(days[1:] != days[:-1])[0] + 0.5              # day boundaries
    day_start = [0] + list(np.where(days[1:] != days[:-1])[0] + 1)

    def decorate(ax, first=False):
        for x in bnd:
            ax.axvline(x, color="0.5", lw=0.6, ls=":")
        for i in np.where(outfam)[0]:
            ax.axvspan(i - 0.5, i + 0.5, color="tab:red", alpha=0.07)
        ax.set_xlim(*xlim); ax.grid(alpha=0.3)
        if first:                                    # day_obs labels, top panel only
            for i in day_start:
                ax.text(i, 0.99, str(days[i]), transform=ax.get_xaxis_transform(),
                        rotation=90, fontsize=5, va="top", ha="center", color="0.4")

    def series_page(pdf, items, suptitle, colors=None):
        fig, axes = plt.subplots(len(items), 1, figsize=(W, 2.0 * len(items) + 0.6),
                                 sharex=True, squeeze=False)
        axes = axes[:, 0]
        for k, (lab, vals) in enumerate(items):
            axes[k].plot(ordn, vals, "o-", ms=3,
                         color=(colors[k] if colors else "C0"))
            axes[k].set_ylabel(lab, fontsize=8)
            decorate(axes[k], first=(k == 0))
        axes[-1].set_xlabel("coadd ordinal (day_obs / seq order)")
        fig.suptitle(suptitle, fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    with PdfPages(out_pdf) as pdf:
        # comparison metrics (resid RMS + corr), multi-series per panel
        fig, ax = plt.subplots(2, 1, figsize=(W, 6.8), sharex=True)
        for z in Z_TERMS:
            ax[0].plot(ordn, [b["metrics"][f"resid_rms_z{z}"] for b in blocks],
                       "o-", ms=3, label=f"Z{z}")
        ax[0].plot(ordn, [b["metrics"]["resid_rms_combined"] for b in blocks],
                   "k-", lw=2, label="combined")
        ax[0].set_ylabel("resid RMS (um)"); ax[0].legend(fontsize=7, ncol=5)
        for z in Z_TERMS:
            ax[1].plot(ordn, [b["metrics"][f"corr_z{z}"] for b in blocks],
                       "o-", ms=3, label=f"Z{z}")
        ax[1].plot(ordn, [b["metrics"]["corr_combined"] for b in blocks],
                   "k-", lw=2, label="mean")
        ax[1].set_ylabel("spatial corr r"); ax[1].legend(fontsize=7, ncol=5)
        ax[1].set_xlabel("coadd ordinal (day_obs / seq order)")
        decorate(ax[0], first=True); decorate(ax[1])
        fig.suptitle("Coadd-vs-MIW comparison metrics vs ordinal  "
                     "(red bands = out-of-5rot-family)", fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # telemetry page: elevation / azimuth / rotator / blur FWHM / z_gradient
        series_page(pdf, [
            ("elevation (deg)", [b["alt"] for b in blocks]),
            ("azimuth (deg)", [b["az"] for b in blocks]),
            ("rotator (deg)", [b["rot"] for b in blocks]),
            ("donut blur FWHM (\")", [b["fwhm"] for b in blocks]),
            ("M1M3 z_gradient", [b["z_gradient"] for b in blocks]),
        ], "Block telemetry vs coadd ordinal")

        # per-v-mode value plots, 5 panels/page
        per = 5
        for pg in range((nkeep + per - 1) // per):
            lo, hi = pg * per, min((pg + 1) * per, nkeep)
            series_page(pdf, [(f"v{m + 1}", Vm[:, m]) for m in range(lo, hi)],
                        f"v-mode amplitude vs coadd ordinal  (v{lo + 1}-v{hi})")
    npg = 2 + (nkeep + 4) // 5
    print(f"wrote {out_pdf}  (metrics + telemetry + {(nkeep + 4) // 5} v-mode "
          f"pages = {npg} pages, n={n})", flush=True)


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--param-set", required=True)
    ap.add_argument("--mi-name", default="pathA_50_34_i_5rot",
                    help="mi_config entry whose build knobs (n_dof/n_keep/iter/"
                         "cuts) define the analysis")
    ap.add_argument("--config", default=None)
    ap.add_argument("--output-root", default="output")
    ap.add_argument("--out-name", default="coadd_50_34",
                    help="subdir under output/<ps>/ for this analysis")
    ap.add_argument("--min-visits", type=int, default=6,
                    help="build/plot only blocks with >= this many visits "
                         "(the summary table lists ALL detected blocks)")
    ap.add_argument("--bands", nargs="*", default=None,
                    help="band whitelist (default: mi_config filter, i.e. i)")
    ap.add_argument("--bounce-programs", nargs="*",
                    default=["BLOCK-T720", "BLOCK-T724"],
                    help="programs split by bounced (alt,rot) state")
    ap.add_argument("--wide-programs", nargs="*",
                    default=["BLOCK-T710", "BLOCK-T710_v2"],
                    help="programs with a drifting image plan -> wider az/rot tol")
    ap.add_argument("--pointing-tol", type=float, default=5.0,
                    help="deg tolerance on (alt,az,rot) within a standard block")
    ap.add_argument("--wide-tol", type=float, default=20.0,
                    help="deg az/rot tolerance for --wide-programs (T710 family)")
    ap.add_argument("--max-seq-span", type=float, default=36.0,
                    help="a block breaks when seq_num reaches this far past its "
                         "first visit (>=), so a 12-triplet run stays whole")
    ap.add_argument("--bounce-round", type=float, default=10.0,
                    help="deg rounding to separate bounce states")
    ap.add_argument("--max-blur", type=float, default=1.4,
                    help="drop visits with median_blur_arcsec above this "
                         "(relaxed seeing cut; None-> no cut)")
    ap.add_argument("--n-bins", type=int, default=None,
                    help="focal-plane bins/axis (default: build.n_bins)")
    ap.add_argument("--n-iter", type=int, default=None,
                    help="iterations (default: build.n_iter, i.e. 3)")
    ap.add_argument("--pct", type=float, default=98.0)
    args = ap.parse_args()

    cfg = mc.load_mi_config(args.param_set, args.mi_name,
                            config_path=(Path(args.config) if args.config else None))
    b = cfg["build"]
    coord = cfg.get("coord_sys", "OCS")
    n_keep = cfg["n_keep"]; n_dof = cfg.get("n_dof")
    k_min, k_max = int(b["k_min"]), int(b["k_max"])
    n_iter = int(args.n_iter if args.n_iter is not None else b["n_iter"])
    n_bins = int(args.n_bins if args.n_bins is not None else b["n_bins"])
    fp_basis = float(b["fp_radius_basis"]); fp_grid = float(b["fp_radius_grid"])
    min_donuts = int(b["min_donuts"]); bad_fit = float(b["bad_fit_threshold"])
    ofc_norm = b.get("ofc_normalization_yaml")
    allowed_bands = args.bands if args.bands is not None else mc.as_band_list(cfg.get("filter"))
    rot_windows = (cfg.get("split") or {}).get("rotator_select")
    bounce_progs = set(args.bounce_programs or [])
    wide_progs = set(args.wide_programs or [])

    base = Path(args.output_root) / args.param_set
    out_dir = base / args.out_name; out_dir.mkdir(parents=True, exist_ok=True)
    donuts_pq = base / "donuts.parquet"

    # ---- block assignment (NO alt/rot cut) + summary table over ALL blocks ----
    vpd = pd.read_parquet(base / "visits.parquet", columns=[
        "day_obs", "seq_num", "alt", "az", "rotator_angle",
        "science_program", "band", "median_blur_arcsec", "z_gradient"])
    bdf = assign_blocks(vpd, allowed_bands, bounce_progs, args.pointing_tol,
                        args.max_seq_span, args.bounce_round,
                        wide_programs=wide_progs, wide_tol=args.wide_tol,
                        max_blur=args.max_blur)
    summ = block_summary(bdf, rot_windows)
    summ["plotted"] = summ["n_visits"] >= args.min_visits
    summ.to_csv(out_dir / "blocks_summary.csv", index=False)
    print(f"[coadd_blocks_miw] {args.param_set}/{args.mi_name}: "
          f"{len(summ)} blocks (bands={allowed_bands}, no alt/rot cut, "
          f"max_blur={args.max_blur}; bounce={sorted(bounce_progs)}, "
          f"wide={sorted(wide_progs)})  -> {out_dir}", flush=True)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(summ.to_string(
            index=False,
            formatters={"alt": "{:.1f}".format, "az": "{:.1f}".format,
                        "rot": "{:+.1f}".format, "fwhm": "{:.2f}".format}),
              flush=True)
    print(f"  wrote blocks_summary.csv;  {int(summ['plotted'].sum())} blocks "
          f">= {args.min_visits} visits will be built/plotted", flush=True)

    vtab = QTable.read(str(base / "visits.parquet"))
    noll_arr = (np.array(vtab["nollIndices"][0])
                if "nollIndices" in vtab.colnames else None)
    pf, lut = _visit_row_groups(donuts_pq)

    # per-donut MIW sidecar (row-aligned to donuts.parquet), indexed per visit so
    # each block's reference map is binned on the same grid as its remnant
    sidecar_pq = base / args.mi_name / "zk_intrinsic.parquet"
    if not sidecar_pq.exists():
        raise SystemExit(f"MIW sidecar not found: {sidecar_pq}  -- build it first "
                         "(rule intrinsic_sidecar) so the coadd can compare to the MIW")
    st = pq.read_table(str(sidecar_pq),
                       columns=["day_obs", "seq_num", "zk_intrinsic_MI"])
    sc_mi = _fixed_list_to_2d(st, "zk_intrinsic_MI")          # (N, nZk), Noll order
    scdf = pd.DataFrame({"day_obs": np.asarray(st["day_obs"]),
                         "seq_num": np.asarray(st["seq_num"])})
    sc_groups = {(int(k[0]), int(k[1])): v for k, v in
                 scdf.groupby(["day_obs", "seq_num"], sort=False).indices.items()}
    print(f"  MIW sidecar: {len(sc_mi)} donuts, {len(sc_groups)} visits", flush=True)

    svd = None; iZs = iZidx = None; xbins = ybins = None
    blocks = []
    meta = summ.set_index("block")
    for blk, g in bdf.groupby("block"):
        if meta.loc[blk, "n_visits"] < args.min_visits:
            continue
        keys = list(zip(g["day_obs"].astype(int), g["seq_num"].astype(int)))
        dd = load_block_donuts(pf, lut, keys)
        if dd is None or len(dd) == 0:
            print(f"  block {blk}: no donuts, skipped"); continue
        if svd is None:                                    # build SVD once
            nZk = np.stack(dd[f"zk_{coord}"].values).shape[1]
            iZs, iZidx = derive_noll_indices(nZk, noll_arr)
            svd = osv.build_ofc_svd(iZs, k_min, k_max, n_keep, n_dof=n_dof,
                                    ofc_normalization_yaml=ofc_norm)
            print(f"  SVD: n_dof={svd.n_dof} n_keep_eff={svd.n_keep_eff}; "
                  f"pupil Noll={iZs}", flush=True)
        res = build_measured_intrinsic_uconstrained(
            dd, None, coord, iZs, kj_grid=svd.kj_grid, U_eff=svd.U_eff,
            n_iter=n_iter, n_bins=n_bins, fp_radius_basis=fp_basis,
            fp_radius_grid=fp_grid, min_donuts=min_donuts,
            bad_fit_threshold=bad_fit)
        final = res["iter_results"][-1]
        xbins, ybins = res["xbins"], res["ybins"]
        grid = {z: final["measured_grid"].get(z) for z in Z_TERMS}
        if any(grid[z] is None for z in Z_TERMS):
            print(f"  block {blk}: missing a Z5-8 grid, skipped"); continue
        # MIW reference (sidecar rows aligned per-visit to dd) on the same grid
        kept = [k for k in keys if k in lut]
        if any(k not in sc_groups for k in kept):
            print(f"  block {blk}: sidecar missing a visit, skipped"); continue
        mi_full = np.concatenate([sc_mi[sc_groups[k]] for k in kept])
        if len(mi_full) != len(dd):
            print(f"  block {blk}: sidecar/donut count mismatch "
                  f"({len(mi_full)} vs {len(dd)}), skipped"); continue
        miw = miw_ref_grid(dd, mi_full, coord, iZidx, n_bins, fp_grid)
        if any(miw[z] is None for z in Z_TERMS):
            print(f"  block {blk}: missing a MIW Z5-8 grid, skipped"); continue
        r = meta.loc[blk]
        blocks.append(dict(
            block=int(blk), program=str(r["program"]), day_obs=int(r["day_obs"]),
            seq_min=int(r["seq_min"]), seq_max=int(r["seq_max"]),
            rot=float(r["rot"]), alt=float(r["alt"]), az=float(r["az"]),
            fwhm=float(r["fwhm"]), z_gradient=float(r["z_gradient"]),
            n_visits=int(r["n_visits"]),
            n_donuts=int(len(dd)), in_family=bool(r["in_family"]),
            grid=grid, miw=miw, metrics=map_metrics(grid, miw),
            vmodes=block_vmodes(final, svd)))
        print(f"  block {blk}: {r['program']} {int(r['day_obs'])} "
              f"rot={r['rot']:+.1f} n_visits={int(r['n_visits'])} "
              f"n_donuts={len(dd)} in_family={bool(r['in_family'])}  "
              f"resid_rms={blocks[-1]['metrics']['resid_rms_combined']:.3f} "
              f"r={blocks[-1]['metrics']['corr_combined']:+.2f}", flush=True)

    if not blocks:
        raise SystemExit("no block produced a remnant grid "
                         "(is donuts.parquet the full combine + is the sidecar built?)")

    # time order (day_obs, seq) -> coadd ordinal
    blocks.sort(key=lambda bb: (bb["day_obs"], bb["seq_min"]))
    for i, bb in enumerate(blocks):
        bb["ordinal"] = i

    # metrics + v-modes parquet (one row per coadd)
    rows = []
    for bb in blocks:
        row = {k: bb[k] for k in ("ordinal", "block", "program", "day_obs",
               "seq_min", "seq_max", "alt", "az", "rot", "fwhm", "z_gradient",
               "n_visits", "n_donuts", "in_family")}
        row.update(bb["metrics"])
        for i, vv in enumerate(bb["vmodes"]):
            row[f"vmode_{i + 1}"] = float(vv)
        rows.append(row)
    pd.DataFrame(rows).to_parquet(out_dir / "coadd_metrics.parquet", index=False)
    print(f"  wrote coadd_metrics.parquet ({len(rows)} coadds, "
          f"{len(blocks[0]['vmodes'])} v-modes each)", flush=True)

    # grids + metadata for cheap re-plots
    np.savez_compressed(
        out_dir / "block_grids.npz",
        xbins=xbins, ybins=ybins, terms=np.array(Z_TERMS),
        grids=np.array([[bb["grid"][z] for z in Z_TERMS] for bb in blocks]),
        miw=np.array([[bb["miw"][z] for z in Z_TERMS] for bb in blocks]),
        vmodes=np.array([bb["vmodes"] for bb in blocks]),
        **{k: np.array([bb[k] for bb in blocks]) for k in
           ("ordinal", "block", "day_obs", "seq_min", "seq_max", "rot", "alt",
            "az", "fwhm", "z_gradient", "n_visits", "n_donuts", "in_family")})

    render_perblock(blocks, xbins, ybins,
                    str(out_dir / "coadd_blocks_miw_perblock.pdf"), args.pct)
    render_timeseries(blocks, str(out_dir / "coadd_blocks_miw_timeseries.pdf"))
    print("[coadd_blocks_miw] done.", flush=True)


if __name__ == "__main__":
    main()
