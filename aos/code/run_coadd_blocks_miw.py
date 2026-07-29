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

Outputs (into output/<ps>/<out-name>/, default out-name=coadd_50_34):
  coadd_blocks_miw_perblock.pdf   rows = blocks (sorted by rotator, then day),
                                  cols = Z5..Z8, common per-term color scale
  coadd_blocks_miw_by_rotator.pdf donut-count-weighted mean of block grids per
                                  rounded rotator bin (the "combine after" view,
                                  comparable to the 9/5-bin MIW maps)
  block_grids.npz                 stacked grids + per-block metadata for re-plots

RSP-only (needs lsst.ts.intrinsic.wavefront + lsst.ts.ofc).
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
from lsst.ts.intrinsic.wavefront.dz_fitting import derive_noll_indices
from lsst.ts.intrinsic.wavefront.measured_intrinsic import (
    build_measured_intrinsic_uconstrained)

Z_TERMS = [5, 6, 7, 8]


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
def build_blocks(visits_pd, programs, allowed_bands, alt_min, alt_max,
                 min_triplets):
    """FAM contiguous blocks (seq_num spacing 3, day boundary) after the MIW
    band/elevation/program cuts.  One block per (day, contiguous run, pointing).
    Returns a DataFrame with a 'block' label and the pointing summary columns."""
    v = visits_pd.dropna(subset=["alt", "az", "rotator_angle"]).copy()
    if allowed_bands and "band" in v.columns:
        v = v[v["band"].astype(str).isin(set(allowed_bands))]
    if programs and "science_program" in v.columns:
        v = v[v["science_program"].astype(str).isin(set(programs))]
    v["alt_deg"] = np.rad2deg(v["alt"].to_numpy(float))
    v["az_deg"] = np.mod(np.rad2deg(v["az"].to_numpy(float)), 360.0)
    v["rot_deg"] = v["rotator_angle"].to_numpy(float)
    if alt_min is not None:
        v = v[v["alt_deg"] >= float(alt_min)]
    if alt_max is not None:
        v = v[v["alt_deg"] <= float(alt_max)]
    v = v.sort_values(["day_obs", "seq_num"]).reset_index(drop=True)
    newblk = ((v["day_obs"] != v["day_obs"].shift())
              | ((v["seq_num"] - v["seq_num"].shift()) != 3))
    v["block"] = newblk.cumsum()
    size = v.groupby("block")["seq_num"].transform("size")
    v = v[size >= min_triplets].copy()
    return v


def in_family(rot, windows):
    return bool(any(lo <= rot <= hi for lo, hi in (windows or [])))


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


def _row(fig, gs_row, axes_row, grids, xbins, ybins, vmax, label, in_fam):
    ec = "tab:red" if not in_fam else "0.3"
    for c, z in enumerate(Z_TERMS):
        ax = axes_row[c]
        g = grids[z]
        pcm = ax.pcolormesh(xbins, ybins, g.T, cmap="RdBu_r",
                            vmin=-vmax[z], vmax=vmax[z], shading="flat")
        ax.set_aspect("equal"); ax.tick_params(labelsize=5)
        if c == 0:
            ax.set_ylabel(label, fontsize=7, color=ec)
            for sp in ax.spines.values():
                sp.set_color(ec); sp.set_linewidth(1.6 if not in_fam else 0.8)
        fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
    return pcm


def render_perblock(blocks, xbins, ybins, out_pdf, pct, per_page=6):
    vmax = _term_scales(blocks, pct)
    order = sorted(range(len(blocks)),
                   key=lambda i: (blocks[i]["rot"], blocks[i]["day_obs"]))
    with PdfPages(out_pdf) as pdf:
        for p0 in range(0, len(order), per_page):
            idxs = order[p0:p0 + per_page]
            fig, axes = plt.subplots(len(idxs), 4,
                                     figsize=(15, 2.7 * len(idxs)),
                                     squeeze=False)
            for r, bi in enumerate(idxs):
                b = blocks[bi]
                lab = (f"{b['day_obs']}\nrot={b['rot']:+.1f}\n"
                       f"n={b['n_triplets']}t"
                       + ("" if b["in_family"] else "\n[out]"))
                _row(fig, None, axes[r], b["grid"], xbins, ybins, vmax,
                     lab, b["in_family"])
                if r == 0:
                    for c, z in enumerate(Z_TERMS):
                        axes[r][c].set_title(f"Z{z}", fontsize=10)
            fig.suptitle("Per-block MIW-style remnant (Path A 50/34, 3 iter, OCS)  "
                         "— red = out-of-5rot-family rotator", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig); plt.close(fig)
    print(f"wrote {out_pdf}  ({len(blocks)} blocks; per-term |vmax| Z5-8 = "
          f"{', '.join(f'{vmax[z]:.3f}' for z in Z_TERMS)} um)", flush=True)


def render_by_rotator(blocks, xbins, ybins, out_pdf, pct, rot_round):
    """Donut-count-weighted mean of block grids per rounded rotator bin."""
    key = {}
    for b in blocks:
        rb = round(b["rot"] / rot_round) * rot_round
        key.setdefault(rb, []).append(b)
    coadds = []
    for rb in sorted(key):
        bs = key[rb]
        grid = {}
        for z in Z_TERMS:
            stack = np.array([bs_i["grid"][z] for bs_i in bs])          # (nb,ny,nx)
            wts = np.array([bs_i["n_donuts"] for bs_i in bs], float)
            m = np.isfinite(stack)
            wsum = (np.where(m, wts[:, None, None], 0.0)).sum(0)
            gsum = np.nansum(np.where(m, stack * wts[:, None, None], 0.0), 0)
            with np.errstate(invalid="ignore"):
                grid[z] = np.where(wsum > 0, gsum / wsum, np.nan)
        days = sorted({bi["day_obs"] for bi in bs})
        coadds.append(dict(rot=rb, grid=grid, nblk=len(bs), days=days,
                           in_family=bs[0]["in_family"], n_triplets=sum(
                               bi["n_triplets"] for bi in bs)))
    vmax = _term_scales(coadds, pct)
    with PdfPages(out_pdf) as pdf:
        per_page = 6
        for p0 in range(0, len(coadds), per_page):
            chunk = coadds[p0:p0 + per_page]
            fig, axes = plt.subplots(len(chunk), 4,
                                     figsize=(15, 2.7 * len(chunk)), squeeze=False)
            for r, cd in enumerate(chunk):
                lab = (f"rot~{cd['rot']:+.0f}\n{cd['nblk']} blk\n"
                       + ",".join(str(d)[4:] for d in cd["days"])
                       + ("" if cd["in_family"] else "\n[out]"))
                _row(fig, None, axes[r], cd["grid"], xbins, ybins, vmax,
                     lab, cd["in_family"])
                if r == 0:
                    for c, z in enumerate(Z_TERMS):
                        axes[r][c].set_title(f"Z{z}", fontsize=10)
            fig.suptitle("Rotator-binned coadd of per-block MIW remnants "
                         "(count-weighted mean)  — red = out-of-5rot-family",
                         fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig); plt.close(fig)
    print(f"wrote {out_pdf}  ({len(coadds)} rotator bins)", flush=True)


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
    ap.add_argument("--min-triplets", type=int, default=6)
    ap.add_argument("--n-bins", type=int, default=None,
                    help="focal-plane bins/axis (default: build.n_bins)")
    ap.add_argument("--n-iter", type=int, default=None,
                    help="iterations (default: build.n_iter, i.e. 3)")
    ap.add_argument("--pct", type=float, default=98.0)
    ap.add_argument("--rot-round", type=float, default=5.0,
                    help="rotator rounding (deg) for the by-rotator coadd")
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
    allowed_bands = mc.as_band_list(cfg.get("filter"))
    programs = cfg.get("programs")
    alt_min, alt_max = cfg.get("alt_min_deg"), cfg.get("alt_max_deg")
    rot_windows = (cfg.get("split") or {}).get("rotator_select")

    base = Path(args.output_root) / args.param_set
    out_dir = base / args.out_name; out_dir.mkdir(parents=True, exist_ok=True)
    donuts_pq = base / "donuts.parquet"

    # blocks (pandas) + Noll indices (QTable)
    vpd = pd.read_parquet(base / "visits.parquet", columns=[
        "day_obs", "seq_num", "alt", "az", "rotator_angle",
        "science_program", "band"])
    blocks_df = build_blocks(vpd, programs, allowed_bands, alt_min, alt_max,
                             args.min_triplets)
    n_blk = blocks_df["block"].nunique()
    print(f"[coadd_blocks_miw] {args.param_set}/{args.mi_name}: {n_blk} FAM blocks "
          f">= {args.min_triplets} triplets (bands={allowed_bands}, "
          f"programs={programs}, alt=[{alt_min},{alt_max}])  -> {out_dir}",
          flush=True)
    if n_blk == 0:
        raise SystemExit("no FAM blocks pass the cuts")

    vtab = QTable.read(str(base / "visits.parquet"))
    noll_arr = (np.array(vtab["nollIndices"][0])
                if "nollIndices" in vtab.colnames else None)

    pf, lut = _visit_row_groups(donuts_pq)

    svd = None
    xbins = ybins = None
    blocks = []
    for blk, g in blocks_df.groupby("block"):
        keys = list(zip(g["day_obs"].astype(int), g["seq_num"].astype(int)))
        dd = load_block_donuts(pf, lut, keys)
        if dd is None or len(dd) == 0:
            print(f"  block {blk}: no donuts, skipped"); continue
        if svd is None:                                    # build SVD once
            nZk = np.stack(dd[f"zk_{coord}"].values).shape[1]
            iZs, _ = derive_noll_indices(nZk, noll_arr)
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
        rot = float(np.median(g["rot_deg"]))
        blocks.append(dict(
            block=int(blk), day_obs=int(g["day_obs"].iloc[0]),
            rot=rot, alt=float(np.median(g["alt_deg"])),
            az=float(np.median(g["az_deg"])),
            n_triplets=int(len(g)), n_donuts=int(len(dd)),
            in_family=in_family(rot, rot_windows), grid=grid))
        print(f"  block {blk}: day={blocks[-1]['day_obs']} rot={rot:+.1f} "
              f"n_triplets={len(g)} n_donuts={len(dd)} "
              f"in_family={blocks[-1]['in_family']}", flush=True)

    if not blocks:
        raise SystemExit("no block produced a remnant grid")

    # save grids + metadata for cheap re-plots
    np.savez_compressed(
        out_dir / "block_grids.npz",
        xbins=xbins, ybins=ybins, terms=np.array(Z_TERMS),
        grids=np.array([[b["grid"][z] for z in Z_TERMS] for b in blocks]),
        block=np.array([b["block"] for b in blocks]),
        day_obs=np.array([b["day_obs"] for b in blocks]),
        rot=np.array([b["rot"] for b in blocks]),
        alt=np.array([b["alt"] for b in blocks]),
        az=np.array([b["az"] for b in blocks]),
        n_triplets=np.array([b["n_triplets"] for b in blocks]),
        n_donuts=np.array([b["n_donuts"] for b in blocks]),
        in_family=np.array([b["in_family"] for b in blocks]))

    render_perblock(blocks, xbins, ybins,
                    str(out_dir / "coadd_blocks_miw_perblock.pdf"), args.pct)
    render_by_rotator(blocks, xbins, ybins,
                      str(out_dir / "coadd_blocks_miw_by_rotator.pdf"),
                      args.pct, args.rot_round)
    print("[coadd_blocks_miw] done.", flush=True)


if __name__ == "__main__":
    main()
