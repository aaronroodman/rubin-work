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
Pearson r, plus a combined pooled-RMS and mean-r.  The same metrics are ALSO
computed over just the corner-WFS radial annulus (cameraGeom SW0 inner corner
~1.5178 deg to 1.725 deg, --cwfs-inner/--cwfs-outer) as `cwfs_`-prefixed
columns, to highlight the high-radius region where the CWFS sit.  Blocks are
ordered in time (day_obs, seq) as a "coadd ordinal".

Outputs (into output/<ps>/<out-name>/, default out-name=coadd_50_34):
  blocks_summary.parquet          every detected block: program, day_obs, seq
                                  range, n_visits, alt/az/rot means, blur FWHM,
                                  band, in-5rot-family flag, 13 thermal means
                                  (a compact subset is also printed)
  coadd_metrics.parquet           one row per coadd: metadata + ordinal + band +
                                  13 thermal means + per-Z and combined resid RMS
                                  / corr (full FP + cwfs_ annulus) + umode_1..34
                                  (u-mode amplitudes, um wavefront)
  coadd_blocks_miw_perblock.pdf   one block per page: rows coadd / MIW / diff x
                                  Z5..Z8, metrics in titles, ordered by day/seq
  coadd_blocks_miw_timeseries.pdf comparison metrics (day_obs-labelled) + a
                                  telemetry<->metric correlation bar chart +
                                  telemetry-vs-corr scatters (z/y_gradient,
                                  dome_delta_t) + telemetry strips (alt/az/rot/
                                  blur/z_gradient/band, 2/page) + a u-mode
                                  amplitude heatmap (um), all vs coadd ordinal
  block_grids.npz                 stacked coadd+MIW Z5-Z8 grids, u-modes, metadata,
                                  plus the DZ PRIMITIVES: block-median raw w
                                  (`raw_dz`) and residual (1-P)w (`resid_dz`),
                                  and the subspace bases `U_eff` / `U_all` /
                                  `U_disc` (+ kj_grid, iZs, sigma, V,
                                  norm_weights).  With those saved, the u-mode
                                  amplitudes, the removed part Pw, and the
                                  reachable-but-discarded vs null(S^T) split of
                                  the residual are all derivable OFF-RSP -- the
                                  discriminating test for the retrieval-bias
                                  hypothesis (aos/MIW_COADD_EQUATIONS.md §4.2).
  block_grids_ext.npz             coadd+MIW grids for the EXTRA pupil Zernikes
                                  (`--ext-terms`, default the 2nd/3rd radial
                                  orders).  Z5-Z8 are all primary, so these are
                                  what reveal which radial orders a retrieval
                                  bias mixes.  Row-aligned by `block`.

RSP-only (needs lsst.ts.intrinsic.wavefront + lsst.ts.ofc); requires the mi_name
MIW sidecar (zk_intrinsic.parquet) to exist.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr, theilslopes
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

# 13 core thermal telemetry columns (ESS temps/deltas + M1M3 gradients + truss);
# block-mean of each is carried into the metrics parquet + the correlation bars.
THERMAL_VARS = [
    "cam_air_temp", "m2_air_temp", "m1m3_air_temp", "outside_temp",
    "m2_delta_t", "cam_m1m3_delta_t", "dome_delta_t",
    "x_gradient", "y_gradient", "z_gradient", "radial_gradient",
    "tma_truss_temp_pxpy", "tma_truss_temp_mxmy",
]
# numeric telemetry correlated against the metrics (az excluded: circular)
CORR_VARS = ["alt", "rot", "fwhm"] + THERMAL_VARS
# telemetry scatter-plotted directly against the coadd-vs-MIW spatial correlation
SCATTER_VARS = ["z_gradient", "y_gradient", "dome_delta_t"]

# Blocks used to BUILD the 5rot MIW -- the ACTUAL selection from mi_config.yaml
# (defaults + the pathA_50_34_i_5rot entry), verified against the build
# fits.parquet: i-band, program BLOCK-T614_triplets, elevation 65-75 deg, one of
# the 5 in-family rotator windows (== in_family), day_obs <= 20260513.  On the
# real data this is 16 blocks over 4 nights (2026-03-15..03-24) at elevation ~70.
# 2025 FAM (folded in later, different programs) is NOT in the build.  Shaded
# lightly on the time series to mark the self-comparison / reference set.
BUILD_BANDS = ("i",)
BUILD_PROGRAMS = ("T614_triplets",)   # blocks_summary program has the BLOCK- prefix stripped
BUILD_ALT_MIN = 65.0
BUILD_ALT_MAX = 75.0
BUILD_DAY_MAX = 20260513


def _build_used(blocks):
    """Boolean per-block: was this block part of the 5rot MIW build set?"""
    return np.array([
        (str(b.get("band")) in BUILD_BANDS)
        and (str(b.get("program", "")).replace("BLOCK-", "") in BUILD_PROGRAMS)
        and (BUILD_ALT_MIN <= float(b["alt"]) <= BUILD_ALT_MAX)
        and bool(b.get("in_family"))
        and (int(b["day_obs"]) <= BUILD_DAY_MAX)
        for b in blocks])


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
            band=(str(g["band"].mode().iloc[0])
                  if "band" in g and len(g["band"].mode()) else ""),
            in_family=in_family(rot, rot_windows),
            **{tv: (float(np.nanmean(g[tv])) if tv in g.columns else np.nan)
               for tv in THERMAL_VARS}))
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


def _metrics_over(coadd, miw, extra_mask, prefix=""):
    """Per-Zernike residual RMS (um) + spatial Pearson r over co-valid cells
    (optionally AND'd with extra_mask), plus a pooled-RMS and mean-r combined."""
    m = {}; diffs = []; rs = []
    for z in Z_TERMS:
        c, w = coadd[z], miw[z]
        ok = np.isfinite(c) & np.isfinite(w)
        if extra_mask is not None:
            ok = ok & extra_mask
        if int(ok.sum()) < 5:
            m[f"{prefix}resid_rms_z{z}"] = np.nan
            m[f"{prefix}corr_z{z}"] = np.nan
            continue
        d = c[ok] - w[ok]
        m[f"{prefix}resid_rms_z{z}"] = float(np.sqrt(np.mean(d ** 2)))
        m[f"{prefix}corr_z{z}"] = _pearson(c[ok], w[ok])
        diffs.append(d); rs.append(m[f"{prefix}corr_z{z}"])
    alld = np.concatenate(diffs) if diffs else np.array([np.nan])
    m[f"{prefix}resid_rms_combined"] = (
        float(np.sqrt(np.mean(alld ** 2))) if diffs else np.nan)
    m[f"{prefix}corr_combined"] = float(np.nanmean(rs)) if rs else np.nan
    return m


def map_metrics(coadd, miw, cwfs_mask=None):
    """Coadd-vs-MIW metrics over the whole focal plane, plus (if `cwfs_mask` is
    given) a second `cwfs_`-prefixed set restricted to the corner-WFS annulus."""
    out = _metrics_over(coadd, miw, None, "")
    if cwfs_mask is not None:
        out.update(_metrics_over(coadd, miw, cwfs_mask, "cwfs_"))
    return out


def miw_ref_grid(dd, mi_full, coord, iZidx, n_bins, fp_grid, terms=None):
    """Bin the per-donut MIW sidecar (row-aligned to dd) on the SAME focal grid
    as the coadd remnant -> {z: 2D} reference map (default Z5-Z8)."""
    thx = np.rad2deg(np.asarray(dd[f"thx_{coord}"], float))
    thy = np.rad2deg(np.asarray(dd[f"thy_{coord}"], float))
    grid, *_ = bin_median_focal(thx, thy, mi_full, iZidx,
                                n_bins=n_bins, fp_radius=fp_grid)
    return {z: grid.get(z) for z in (terms if terms is not None else Z_TERMS)}


def residual_subspace_basis(svd, instrument="lsst"):
    """Orthonormal bases for the DZ subspaces that SURVIVE the n_dof/n_keep removal.

    ``(1-P)w`` splits in two (see aos/MIW_COADD_EQUATIONS.md §2, §4.2):
      * ``u_35..50``  -- REACHABLE by the n_dof DOF but discarded by the n_keep
        truncation (the ill-conditioned v-modes); 16 dims for 50/34.
      * ``null(S^T)`` -- not reachable by ANY combination of the DOF; 76 dims.

    The retrieval-bias hypothesis predicts residual power in the NULL part
    proportional to Δa, which an unbiased retrieval cannot produce (there
    ``(1-P)W`` lives entirely in the 16 reachable-but-discarded dims).  So this
    split is the discriminating measurement, which is why the bases are saved.

    Returns (U_all, U_disc): the full reachable basis (n_kj, n_dof) and its
    discarded columns (n_kj, n_dof - n_keep)."""
    from lsst.ts.ofc import OFCData
    S_full = np.asarray(OFCData(instrument).sensitivity_matrix)
    S_slab = S_full[svd.k_min:svd.k_max + 1, np.asarray(svd.iZs, int), :]
    Sh = (S_slab.reshape(-1, S_slab.shape[-1])[:, svd.dof_idx]
          @ np.diag(svd.normalization_weights))
    U_all = np.linalg.svd(np.nan_to_num(np.asarray(Sh, float)),
                          full_matrices=False)[0]
    keep = set(svd.keep_idx if svd.keep_idx else range(svd.U_eff.shape[1]))
    disc = [i for i in range(U_all.shape[1]) if i not in keep]
    return U_all, U_all[:, disc]


def block_dz_vectors(final, svd):
    """Block-median RAW and RESIDUAL Double-Zernike coefficient vectors (µm).

    ``w`` is the unconstrained per-visit DZ fit (``fit_rows_raw``); the build
    subtracts only ``P w``, so ``(1-P) w`` is what survives into this block's
    coadd.  Saving both primitives (rather than just the u-mode amplitudes)
    lets the whole §4.2 analysis -- a, Pw, the 16/76 subspace split, the L
    regression -- be redone off-RSP from block_grids.npz alone."""
    W = ibp.stack_per_visit_coeffs(final["fit_rows_raw"], list(svd.kj_grid))
    W = np.where(np.isfinite(W), W, 0.0)
    R = W - (W @ svd.U_eff) @ svd.U_eff.T           # (1-P) w, per visit
    return np.nanmedian(W, axis=0), np.nanmedian(R, axis=0)


def block_umodes(final, svd):
    """Block-median u-mode amplitudes (MICRONS of wavefront) from the per-visit
    raw DZ fits: a = U_eff^T . W, the measured Double-Zernike projected onto the
    OFC left-singular vectors.  NOT divided by sigma -- this is physical
    wavefront (um), keeping all modes on a common scale, unlike the v-mode
    a/sigma."""
    W = ibp.stack_per_visit_coeffs(final["fit_rows_raw"], list(svd.kj_grid))
    A = svd.project_amplitudes(W)          # (n_visits, n_keep) u-mode amps, um
    return np.nanmedian(A, axis=0)


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
                f"resid RMS={b['metrics']['resid_rms_combined']:.3f} um "
                f"(CWFS {b['metrics'].get('cwfs_resid_rms_combined', float('nan')):.3f})   "
                f"mean r={b['metrics']['corr_combined']:+.2f} "
                f"(CWFS {b['metrics'].get('cwfs_corr_combined', float('nan')):+.2f})   "
                f"(Path A 50/34, 3 iter, OCS)", fontsize=11,
                color=("black" if b["in_family"] else "tab:red"))
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig); plt.close(fig)
    print(f"wrote {out_pdf}  ({len(blocks)} blocks x [coadd/MIW/diff]; per-term "
          f"|vmax| Z5-8 = {', '.join(f'{vmax[z]:.3f}' for z in Z_TERMS)} um)",
          flush=True)


def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 5 or np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0:
        return np.nan
    return float(spearmanr(x[m], y[m]).correlation)


def _corr_bar_page(pdf, blocks, n):
    """Bar chart of each numeric telemetry var's Spearman correlation with the
    coadd-vs-MIW deviation (residual RMS) and with the mean spatial agreement."""
    rms = np.array([b["metrics"]["resid_rms_combined"] for b in blocks], float)
    cor = np.array([b["metrics"]["corr_combined"] for b in blocks], float)
    rows = []
    for tv in CORR_VARS:
        x = np.array([b.get(tv, np.nan) for b in blocks], float)
        rows.append((tv, _spearman(x, rms), _spearman(x, cor)))
    rows = [r for r in rows if np.isfinite(r[1]) or np.isfinite(r[2])]
    rows.sort(key=lambda r: -(abs(r[1]) if np.isfinite(r[1]) else 0.0))
    labels = [r[0] for r in rows]
    rrms = np.array([r[1] for r in rows]); rcor = np.array([r[2] for r in rows])
    x = np.arange(len(labels))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(max(10, 0.45 * len(labels) + 3), 8.5))
    for ax, vals, ylab, ttl in [
            (a1, rrms, "rho vs resid_rms_combined",
             "telemetry vs deviation magnitude (residual RMS)"),
            (a2, rcor, "rho vs corr_combined",
             "telemetry vs spatial agreement (mean r)")]:
        ax.bar(x, np.nan_to_num(vals),
               color=["tab:red" if v > 0 else "tab:blue" for v in vals])
        ax.axhline(0, color="k", lw=0.5); ax.set_ylim(-1, 1); ax.grid(alpha=0.3)
        ax.set_ylabel(ylab, fontsize=9); ax.set_title(ttl, fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    fig.suptitle(f"Telemetry <-> coadd-vs-MIW metric  Spearman correlation "
                 f"(n={n} coadds; red rho>0, blue rho<0)", fontsize=11)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def _scatter_vs_metric_page(pdf, blocks, tvars, metric_key, metric_label):
    """Scatter each telemetry var against a per-coadd metric (default the
    combined Z5-Z8 spatial correlation), colored by 5rot-family, with a robust
    Theil-Sen line + Spearman rho."""
    bu = _build_used(blocks)
    y = np.array([b["metrics"].get(metric_key, np.nan) for b in blocks], float)
    fig, axes = plt.subplots(1, len(tvars), figsize=(4.6 * len(tvars), 4.6),
                             squeeze=False)
    axes = axes[0]
    for ax, tv in zip(axes, tvars):
        x = np.array([b.get(tv, np.nan) for b in blocks], float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok & ~bu], y[ok & ~bu], s=20, c="0.55",
                   alpha=0.6, label="other")
        ax.scatter(x[ok & bu], y[ok & bu], s=26, c="tab:green",
                   alpha=0.85, label="MIW build")
        rho = _spearman(x, y)
        if np.isfinite(rho) and int(ok.sum()) >= 5:
            sl, ic, _, _ = theilslopes(y[ok], x[ok])
            xr = np.array([np.nanmin(x[ok]), np.nanmax(x[ok])])
            ax.plot(xr, ic + sl * xr, "k--", lw=1)
        ax.set_xlabel(tv, fontsize=9); ax.grid(alpha=0.3)
        ax.set_title(f"{tv}   Spearman rho={rho:+.2f}", fontsize=9)
    axes[0].set_ylabel(metric_label, fontsize=9); axes[0].legend(fontsize=7)
    fig.suptitle(f"telemetry vs {metric_label} (one point per coadd)", fontsize=11)
    fig.tight_layout(); pdf.savefig(fig); plt.close(fig)


def render_timeseries(blocks, out_pdf):
    """Comparison metrics, a telemetry<->metric correlation bar chart, block
    telemetry (2 panels/page incl. filter band), and per-u-mode value plots, all
    vs coadd ordinal on a COMMON horizontal scale.  day_obs boundaries are dotted
    vlines on every panel (out-of-family blocks shaded); day_obs is labelled in
    small vertical text at each day's start.  u-modes are 5 panels/page."""
    n = len(blocks); ordn = np.arange(n)
    days = np.array([b["day_obs"] for b in blocks])
    build_used = _build_used(blocks)
    Um = np.array([b["umodes"] for b in blocks])                 # (n, n_keep), um
    nkeep = Um.shape[1]
    W = max(12.0, 0.30 * n + 3.0)                                # common width
    xlim = (-0.6, n - 0.4)                                       # common x-range
    bnd = np.where(days[1:] != days[:-1])[0] + 0.5              # day boundaries
    day_start = [0] + list(np.where(days[1:] != days[:-1])[0] + 1)

    def decorate(ax, first=False):
        for x in bnd:
            ax.axvline(x, color="0.5", lw=0.6, ls=":")
        for i in np.where(build_used)[0]:
            ax.axvspan(i - 0.5, i + 0.5, color="tab:blue", alpha=0.10)
        ax.set_xlim(*xlim); ax.grid(alpha=0.3)
        if first:                     # small day_obs labels at each day's start
            for i in day_start:
                ax.text(i, 0.985, str(days[i]), transform=ax.get_xaxis_transform(),
                        rotation=90, fontsize=6, va="top", ha="left", color="0.35")

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
                   "k-", lw=2, label="combined (full FP)")
        ax[0].plot(ordn, [b["metrics"].get("cwfs_resid_rms_combined", np.nan)
                          for b in blocks],
                   color="tab:red", ls="--", lw=2, label="combined (CWFS annulus)")
        ax[0].set_ylabel("resid RMS (um)"); ax[0].legend(fontsize=7, ncol=6)
        for z in Z_TERMS:
            ax[1].plot(ordn, [b["metrics"][f"corr_z{z}"] for b in blocks],
                       "o-", ms=3, label=f"Z{z}")
        ax[1].plot(ordn, [b["metrics"]["corr_combined"] for b in blocks],
                   "k-", lw=2, label="mean (full FP)")
        ax[1].plot(ordn, [b["metrics"].get("cwfs_corr_combined", np.nan)
                          for b in blocks],
                   color="tab:red", ls="--", lw=2, label="mean (CWFS annulus)")
        ax[1].set_ylabel("spatial corr r"); ax[1].legend(fontsize=7, ncol=6)
        ax[1].set_xlabel("coadd ordinal (day_obs / seq order)")
        decorate(ax[0], first=True); decorate(ax[1], first=True)
        fig.suptitle("Coadd-vs-MIW comparison metrics vs ordinal  "
                     "(vertical = day_obs; blue bands = 5rot MIW build blocks)",
                     fontsize=11)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # telemetry <-> metric correlation bar chart, then direct scatters
        _corr_bar_page(pdf, blocks, n)
        _scatter_vs_metric_page(pdf, blocks, SCATTER_VARS, "corr_combined",
                                "spatial corr (combined Z5-Z8)")

        # block telemetry, 2 panels/page (incl. filter band)
        tele = [("elevation (deg)", "line", [b["alt"] for b in blocks]),
                ("azimuth (deg)", "line", [b["az"] for b in blocks]),
                ("camera rotator (deg)", "line", [b["rot"] for b in blocks]),
                ("donut blur FWHM (\")", "line", [b["fwhm"] for b in blocks]),
                ("M1M3 z_gradient", "line", [b["z_gradient"] for b in blocks]),
                ("filter band", "band", [b["band"] for b in blocks])]
        for p0 in range(0, len(tele), 2):
            chunk = tele[p0:p0 + 2]
            fig, axes = plt.subplots(len(chunk), 1,
                                     figsize=(W, 4.2 * len(chunk) + 0.4),
                                     sharex=True, squeeze=False)
            axes = axes[:, 0]
            for k, (lab, kind, vals) in enumerate(chunk):
                if kind == "band":
                    cats = sorted({v for v in vals if v})
                    idx = {c: i for i, c in enumerate(cats)}
                    axes[k].plot(ordn, [idx.get(v, np.nan) for v in vals],
                                 "o", ms=5, color="C0")
                    axes[k].set_yticks(range(len(cats)))
                    axes[k].set_yticklabels(cats)
                    axes[k].set_ylim(-0.5, max(0.5, len(cats) - 0.5))
                    axes[k].set_ylabel("filter band", fontsize=9)
                else:
                    axes[k].plot(ordn, vals, "o-", ms=4, color="C0")
                    axes[k].set_ylabel(lab, fontsize=9)
                decorate(axes[k], first=(k == 0))
            axes[-1].set_xlabel("coadd ordinal (day_obs / seq order)")
            fig.suptitle("Block telemetry vs coadd ordinal", fontsize=11)
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # u-mode amplitude heatmap (um): rows = u-mode, cols = coadd ordinal
        fig, ax = plt.subplots(figsize=(W, max(6.0, 0.20 * nkeep + 2.5)))
        vlim = float(np.nanpercentile(np.abs(Um), 98)) or 1.0
        im = ax.imshow(Um.T, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                       origin="lower", extent=[-0.5, n - 0.5, 0.5, nkeep + 0.5])
        for x in bnd:
            ax.axvline(x, color="0.3", lw=0.6, ls=":")
        for i in np.where(build_used)[0]:
            ax.axvline(i, color="tab:blue", lw=0.5, alpha=0.3)
        for i in day_start:
            ax.text(i, 1.005, str(days[i]), transform=ax.get_xaxis_transform(),
                    rotation=90, fontsize=6, va="bottom", ha="left", color="0.35")
        ax.set_xlabel("coadd ordinal (day_obs / seq order)")
        ax.set_ylabel("u-mode index"); ax.set_yticks(list(range(1, nkeep + 1, 2)))
        ax.set_title("u-mode amplitude (um wavefront) vs coadd ordinal  "
                     "(blue lines = 5rot MIW build blocks; vertical dotted = day_obs)")
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                     label="u-mode amplitude (um)")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print(f"wrote {out_pdf}  (metrics + corr-bars + scatter + 3 telemetry + "
          f"u-mode heatmap, n={n})", flush=True)


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
    ap.add_argument("--instrument", default="lsst",
                    help="ts_ofc instrument name, for the residual subspace bases")
    ap.add_argument("--ext-terms", default="secondary",
                    help="extra pupil Noll maps written to block_grids_ext.npz, on top "
                    "of the Z5-Z8 in block_grids.npz.  'secondary' (default) = the "
                    "2nd/3rd radial orders that resolve which orders a retrieval bias "
                    "mixes; 'all' = every fitted pupil Noll; 'none'; or a comma list")
    ap.add_argument("--cwfs-inner", type=float, default=None,
                    help="corner-WFS annulus inner radius (deg); default = the "
                         "cameraGeom SW0 extra-focal inner corner (~1.5178)")
    ap.add_argument("--cwfs-outer", type=float, default=1.725,
                    help="corner-WFS annulus outer radius (deg); default 1.725 "
                         "(AOS-online radial limit)")
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
    # Read visits PER CHUNK and union columns: the combined visits.parquet keeps
    # only columns present in every chunk, so if a chunk (e.g. the newer
    # reprocessing) lacks thermal telemetry the combine drops z_gradient for all
    # blocks.  Per-chunk concat keeps z_gradient where it exists (NaN elsewhere).
    need = (["day_obs", "seq_num", "alt", "az", "rotator_angle",
             "science_program", "band", "median_blur_arcsec"] + THERMAL_VARS)
    chunk_vis = sorted((base / "chunks").glob("*/visits.parquet"))
    srcs = chunk_vis if chunk_vis else [base / "visits.parquet"]
    parts = []
    for f in srcs:
        avail = set(pq.read_schema(str(f)).names)
        parts.append(pd.read_parquet(f, columns=[c for c in need if c in avail]))
    vpd = pd.concat(parts, ignore_index=True)
    for c in need:
        if c not in vpd.columns:
            vpd[c] = np.nan
    if not vpd["z_gradient"].notna().any():
        print("  NOTE: z_gradient absent from all visits (no thermal attached) — "
              "the telemetry z_gradient panel will be empty", flush=True)
    bdf = assign_blocks(vpd, allowed_bands, bounce_progs, args.pointing_tol,
                        args.max_seq_span, args.bounce_round,
                        wide_programs=wide_progs, wide_tol=args.wide_tol,
                        max_blur=args.max_blur)
    summ = block_summary(bdf, rot_windows)
    summ["plotted"] = summ["n_visits"] >= args.min_visits
    summ.to_parquet(out_dir / "blocks_summary.parquet", index=False)
    print(f"[coadd_blocks_miw] {args.param_set}/{args.mi_name}: "
          f"{len(summ)} blocks (bands={allowed_bands}, no alt/rot cut, "
          f"max_blur={args.max_blur}; bounce={sorted(bounce_progs)}, "
          f"wide={sorted(wide_progs)})  -> {out_dir}", flush=True)
    show = ["block", "program", "day_obs", "seq_min", "seq_max", "n_visits",
            "alt", "az", "rot", "fwhm", "band", "in_family", "plotted"]
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(summ[[c for c in show if c in summ.columns]].to_string(
            index=False,
            formatters={"alt": "{:.1f}".format, "az": "{:.1f}".format,
                        "rot": "{:+.1f}".format, "fwhm": "{:.2f}".format}),
              flush=True)
    print(f"  wrote blocks_summary.parquet (+thermal means);  "
          f"{int(summ['plotted'].sum())} blocks >= {args.min_visits} visits "
          f"will be built/plotted", flush=True)

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

    # corner-WFS annulus (radial region of the CWFS), matched from cameraGeom
    cwfs_inner = (float(args.cwfs_inner) if args.cwfs_inner is not None
                  else float(ibp.resolve_wfs_inner_edge(None)))
    cwfs_outer = float(args.cwfs_outer)

    # extra pupil Zernikes to map (resolved against the fitted iZs once known)
    SECONDARY = [12, 13, 16, 17, 18, 19, 22, 23, 24, 25, 26]   # 2nd/3rd radial orders
    ext_spec = str(args.ext_terms).strip().lower()
    ext_terms = []

    svd = None; iZs = iZidx = None; xbins = ybins = None; cwfs_mask = None
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
            if ext_spec in ("none", ""):
                want = []
            elif ext_spec == "all":
                want = list(iZs)
            elif ext_spec == "secondary":
                want = SECONDARY
            else:
                want = [int(x) for x in ext_spec.replace(" ", "").split(",") if x]
            ext_terms = [z for z in want if z in iZidx and z not in Z_TERMS]
            print(f"  extra mapped pupil Noll ({ext_spec}): {ext_terms}", flush=True)
        res = build_measured_intrinsic_uconstrained(
            dd, None, coord, iZs, kj_grid=svd.kj_grid, U_eff=svd.U_eff,
            n_iter=n_iter, n_bins=n_bins, fp_radius_basis=fp_basis,
            fp_radius_grid=fp_grid, min_donuts=min_donuts,
            bad_fit_threshold=bad_fit)
        final = res["iter_results"][-1]
        xbins, ybins = res["xbins"], res["ybins"]
        if cwfs_mask is None:                   # radial CWFS mask on the grid
            cx = 0.5 * (xbins[:-1] + xbins[1:]); cy = 0.5 * (ybins[:-1] + ybins[1:])
            rr = np.hypot(cy[:, None], cx[None, :])
            cwfs_mask = (rr >= cwfs_inner) & (rr <= cwfs_outer)
            print(f"  CWFS annulus [{cwfs_inner:.4f}, {cwfs_outer:.3f}] deg: "
                  f"{int(cwfs_mask.sum())} of {cwfs_mask.size} focal cells", flush=True)
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
        # extra pupil Zernikes (secondary/tertiary radial orders) -> sidecar npz.
        # These are what resolve WHICH radial orders the retrieval bias mixes;
        # Z_TERMS (Z5-Z8) are all PRIMARY, so the main npz cannot show it.
        nanmap = np.full_like(np.asarray(grid[Z_TERMS[0]], float), np.nan)
        _eg = {z: final["measured_grid"].get(z) for z in ext_terms}
        _em = miw_ref_grid(dd, mi_full, coord, iZidx, n_bins, fp_grid,
                           terms=ext_terms)
        # a missing extra term must not abort the block -- fill it with NaN
        ext_grid = {z: (nanmap if _eg[z] is None else _eg[z]) for z in ext_terms}
        ext_miw = {z: (nanmap if _em[z] is None else _em[z]) for z in ext_terms}
        raw_dz, resid_dz = block_dz_vectors(final, svd)
        r = meta.loc[blk]
        blocks.append(dict(
            block=int(blk), program=str(r["program"]), day_obs=int(r["day_obs"]),
            seq_min=int(r["seq_min"]), seq_max=int(r["seq_max"]),
            rot=float(r["rot"]), alt=float(r["alt"]), az=float(r["az"]),
            fwhm=float(r["fwhm"]), band=str(r["band"]),
            n_visits=int(r["n_visits"]),
            n_donuts=int(len(dd)), in_family=bool(r["in_family"]),
            grid=grid, miw=miw, metrics=map_metrics(grid, miw, cwfs_mask),
            umodes=block_umodes(final, svd),
            raw_dz=raw_dz, resid_dz=resid_dz,
            ext_grid=ext_grid, ext_miw=ext_miw,
            **{tv: float(r[tv]) for tv in THERMAL_VARS}))
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

    # metrics + u-modes parquet (one row per coadd)
    rows = []
    for bb in blocks:
        row = {k: bb[k] for k in ("ordinal", "block", "program", "day_obs",
               "seq_min", "seq_max", "alt", "az", "rot", "fwhm", "band",
               "n_visits", "n_donuts", "in_family")}
        row.update({tv: bb[tv] for tv in THERMAL_VARS})
        row.update(bb["metrics"])
        for i, vv in enumerate(bb["umodes"]):
            row[f"umode_{i + 1}"] = float(vv)          # u-mode amplitude, um
        rows.append(row)
    pd.DataFrame(rows).to_parquet(out_dir / "coadd_metrics.parquet", index=False)
    print(f"  wrote coadd_metrics.parquet ({len(rows)} coadds, "
          f"{len(blocks[0]['umodes'])} u-modes each)", flush=True)

    # grids + metadata for cheap re-plots.  NOTE the legacy keys (terms/grids/miw)
    # keep their exact Z5-Z8 meaning so existing consumers (recompute_coadd_metrics,
    # analyze_miw_bias_regression) are unaffected; the new DZ primitives are added
    # alongside and the extra Zernike maps go to a sidecar file.
    U_all, U_disc = residual_subspace_basis(svd, args.instrument)
    np.savez_compressed(
        out_dir / "block_grids.npz",
        xbins=xbins, ybins=ybins, terms=np.array(Z_TERMS),
        grids=np.array([[bb["grid"][z] for z in Z_TERMS] for bb in blocks]),
        miw=np.array([[bb["miw"][z] for z in Z_TERMS] for bb in blocks]),
        umodes=np.array([bb["umodes"] for bb in blocks]),
        # --- DZ primitives: block-median raw w and residual (1-P)w, plus the
        # subspace bases, so a, Pw and the 16/76 split are all derivable off-RSP
        raw_dz=np.array([bb["raw_dz"] for bb in blocks]),
        resid_dz=np.array([bb["resid_dz"] for bb in blocks]),
        U_eff=np.asarray(svd.U_eff, float), U_all=U_all, U_disc=U_disc,
        kj_grid=np.array(svd.kj_grid, int), iZs=np.array(svd.iZs, int),
        sigma=np.asarray(svd.Sigma, float),
        V=np.asarray(svd.V, float),
        norm_weights=np.asarray(svd.normalization_weights, float),
        band=np.array([bb["band"] for bb in blocks]),
        **{k: np.array([bb[k] for bb in blocks]) for k in
           (("ordinal", "block", "day_obs", "seq_min", "seq_max", "rot", "alt",
             "az", "fwhm", "n_visits", "n_donuts", "in_family") + tuple(THERMAL_VARS))})
    print(f"  wrote block_grids.npz (+ raw_dz/resid_dz {len(svd.kj_grid)}-vectors, "
          f"U_all {U_all.shape}, U_disc {U_disc.shape})", flush=True)

    if ext_terms:
        np.savez_compressed(
            out_dir / "block_grids_ext.npz",
            xbins=xbins, ybins=ybins, terms=np.array(ext_terms),
            block=np.array([bb["block"] for bb in blocks]),
            grids=np.array([[bb["ext_grid"][z] for z in ext_terms]
                            for bb in blocks], dtype=float),
            miw=np.array([[bb["ext_miw"][z] for z in ext_terms]
                          for bb in blocks], dtype=float))
        print(f"  wrote block_grids_ext.npz ({len(ext_terms)} extra pupil Zernikes "
              f"{ext_terms}; row-aligned to block_grids.npz by `block`)", flush=True)

    render_perblock(blocks, xbins, ybins,
                    str(out_dir / "coadd_blocks_miw_perblock.pdf"), args.pct)
    render_timeseries(blocks, str(out_dir / "coadd_blocks_miw_timeseries.pdf"))
    print("[coadd_blocks_miw] done.", flush=True)


if __name__ == "__main__":
    main()
