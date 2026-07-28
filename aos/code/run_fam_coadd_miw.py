#!/usr/bin/env python
"""Coadded FAM measured-intrinsic-subtracted focal-plane maps (Z5-Z8, OCS).

Motivation: single FAM triplets have too few donuts and carry turbulence, so we
coadd the ~12 triplets of a block taken at a common pointing.  For each coadd we
subtract the measured intrinsic wavefront (the pathA_50_34_i MIW sidecar) from
every donut and bin Z5,Z6,Z7,Z8 over the focal plane (OCS).  One page per coadd,
all coadds in a single PDF, a common color scale PER TERM across coadds -- so the
pages can be flipped through to see how many distinct MIW families are present
(the two-families question from the 5-rotator MIW decomposition).

Coadd = one contiguous FAM block (consecutive triplets, seq_num spacing 3) split
by pointing bins of `bin_deg` (default 1 deg) in (elevation, rotator, azimuth).
So a T614 block (12 triplets, one pointing) -> one coadd; a T720/T724 bounce
block -> one coadd per elevation/rotator value (the two values never combine).

Reuses: aos/code/dz_plotting (block detection helper), the MIW sidecar
(row-aligned to donuts.parquet), and the scipy binned_statistic_2d + imshow
RdBu_r map pattern from dz_plotting.plot_zernike_trio.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binned_statistic_2d

sys.path.insert(0, str(Path(__file__).resolve().parent))  # aos/code

# donuts/sidecar zk vectors are Noll [4,5,6,7,8,...]; Z5-Z8 = columns 1..4
Z_COLS = {5: 1, 6: 2, 7: 3, 8: 4}


def _fixed_list_to_2d(table, col):
    """A fixed-length list<double> arrow column -> (n, L) float array (fast)."""
    arr = table[col].combine_chunks()
    flat = arr.values.to_numpy(zero_copy_only=False).astype(float)
    return flat.reshape(len(table), -1)


def _wrap_deg(a):
    return np.mod(a, 360.0)


def build_coadd_groups(visits, bin_deg, programs, min_triplets):
    """Assign each triplet (visit) to a coadd group.

    Group key = (contiguous-block id within a day) x pointing bin in
    (alt, rot, az) rounded to bin_deg.  Returns a DataFrame with a 'group' label
    column, restricted to groups with >= min_triplets triplets.
    """
    v = visits.copy()
    if programs:
        v = v[v["science_program"].isin(programs)]
    v = v.dropna(subset=["alt", "az", "rotator_angle"]).copy()
    # alt/az are radians in visits.parquet; rotator_angle is degrees
    v["alt_deg"] = np.rad2deg(v["alt"].to_numpy(float))
    v["az_deg"] = _wrap_deg(np.rad2deg(v["az"].to_numpy(float)))
    v["rot_deg"] = v["rotator_angle"].to_numpy(float)
    v = v.sort_values(["day_obs", "seq_num"]).reset_index(drop=True)
    # contiguous FAM block: new block whenever seq_num gap != 3 (triplet spacing)
    # or the day changes
    newblk = ((v["day_obs"] != v["day_obs"].shift())
              | ((v["seq_num"] - v["seq_num"].shift()) != 3))
    v["block"] = newblk.cumsum()

    def _b(x):
        return np.round(x / bin_deg) * bin_deg
    v["group"] = (v["day_obs"].astype(str) + "_blk" + v["block"].astype(str)
                  + f"_el" + _b(v["alt_deg"]).round(0).astype(int).astype(str)
                  + "_rot" + _b(v["rot_deg"]).round(0).astype(int).astype(str)
                  + "_az" + _b(v["az_deg"]).round(0).astype(int).astype(str))
    size = v.groupby("group")["seq_num"].transform("size")
    v = v[size >= min_triplets].copy()
    return v


def coadd_maps(donuts_path, sidecar_path, groups, grid_n, fp_radius):
    """For each group, median-coadd MIW-subtracted Z5-Z8 over the focal plane.

    Returns dict group -> {5: 2D, 6: 2D, 7: 2D, 8: 2D} and the bin edges.
    Reads the whole donut + sidecar tables (row-aligned) once.
    """
    dt = pq.read_table(donuts_path,
                       columns=["day_obs", "seq_num", "thx_OCS_extra",
                                "thy_OCS_extra", "zk_OCS"])
    st = pq.read_table(sidecar_path, columns=["day_obs", "seq_num", "zk_intrinsic_MI"])
    if len(dt) != len(st):
        raise SystemExit(f"donuts ({len(dt)}) and sidecar ({len(st)}) row counts "
                         f"differ -- not row-aligned")
    # sanity: row alignment on (day_obs, seq_num)
    if not (np.array_equal(dt["day_obs"].to_numpy(), st["day_obs"].to_numpy())
            and np.array_equal(dt["seq_num"].to_numpy(), st["seq_num"].to_numpy())):
        raise SystemExit("donuts/sidecar not row-aligned on (day_obs, seq_num)")

    zk = _fixed_list_to_2d(dt, "zk_OCS")
    mi = _fixed_list_to_2d(st, "zk_intrinsic_MI")
    dev = zk - mi                                    # MIW-subtracted, Noll-aligned
    thx = np.rad2deg(dt["thx_OCS_extra"].to_numpy(zero_copy_only=False).astype(float))
    thy = np.rad2deg(dt["thy_OCS_extra"].to_numpy(zero_copy_only=False).astype(float))
    d_day = dt["day_obs"].to_numpy()
    d_seq = dt["seq_num"].to_numpy()

    # map each donut to its group via (day_obs, seq_num)
    key = pd.DataFrame({"day_obs": d_day, "seq_num": d_seq})
    g = groups[["day_obs", "seq_num", "group"]].drop_duplicates()
    key = key.merge(g, on=["day_obs", "seq_num"], how="left")
    grp = key["group"].to_numpy(dtype=object)

    edges = np.linspace(-fp_radius, fp_radius, grid_n + 1)
    out = {}
    for gname in pd.unique(groups["group"]):
        sel = grp == gname
        if not sel.any():
            continue
        maps = {}
        for z, col in Z_COLS.items():
            stat, _, _, _ = binned_statistic_2d(
                thy[sel], thx[sel], dev[sel, col], statistic="median", bins=edges)
            maps[z] = stat.T          # (thx rows, thy cols) for imshow
        out[gname] = maps
    return out, edges


def render_pdf(maps_by_group, edges, groups, out_pdf, pct):
    # per-term common color scale from the pooled binned medians across all coadds
    vmax = {}
    for z in Z_COLS:
        vals = np.concatenate([np.abs(m[z][np.isfinite(m[z])]).ravel()
                               for m in maps_by_group.values()
                               if np.isfinite(m[z]).any()]) if maps_by_group else np.array([0.0])
        vmax[z] = float(np.nanpercentile(vals, pct)) if vals.size else 1.0
        vmax[z] = vmax[z] or 1.0
    ext = [edges[0], edges[-1], edges[0], edges[-1]]

    # order pages by day_obs then group name
    ginfo = (groups.groupby("group")
             .agg(day_obs=("day_obs", "first"), n=("seq_num", "size"),
                  alt=("alt_deg", "mean"), rot=("rot_deg", "mean"),
                  az=("az_deg", "mean"), band=("band", "first"),
                  prog=("science_program", "first"))
             .reset_index().sort_values(["day_obs", "group"]))
    with PdfPages(out_pdf) as pdf:
        for _, r in ginfo.iterrows():
            if r["group"] not in maps_by_group:
                continue
            m = maps_by_group[r["group"]]
            fig, axes = plt.subplots(1, 4, figsize=(17, 4.6))
            for ax, z in zip(axes, Z_COLS):
                im = ax.imshow(m[z], origin="lower", extent=ext, cmap="RdBu_r",
                               vmin=-vmax[z], vmax=vmax[z], aspect="equal")
                ax.set_title(f"Z{z}", fontsize=10)
                ax.set_xlabel("thy (deg)", fontsize=8)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            axes[0].set_ylabel("thx (deg)", fontsize=8)
            fig.suptitle(
                f"{int(r['day_obs'])}  {r['prog']}  el={r['alt']:.1f} "
                f"rot={r['rot']:.1f} az={r['az']:.1f}  band={r['band']}  "
                f"n_triplets={int(r['n'])}   (MIW-subtracted, OCS)", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {out_pdf}  ({len(maps_by_group)} coadds; "
          f"per-term |vmax| Z5-8 = "
          f"{', '.join(f'{vmax[z]:.3f}' for z in Z_COLS)} um)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--param-set", required=True, dest="param_set")
    ap.add_argument("--mi-name", default="pathA_50_34_i", dest="mi_name")
    ap.add_argument("--output-dir", default=None, dest="output_dir")
    ap.add_argument("--bin-deg", type=float, default=1.0, dest="bin_deg")
    ap.add_argument("--min-triplets", type=int, default=6, dest="min_triplets")
    ap.add_argument("--grid-n", type=int, default=24, dest="grid_n")
    ap.add_argument("--fp-radius", type=float, default=1.8, dest="fp_radius")
    ap.add_argument("--pct", type=float, default=98.0, dest="pct",
                    help="percentile of |binned median| for the per-term color scale")
    ap.add_argument("--programs", nargs="*",
                    default=["BLOCK-T614_triplets", "BLOCK-T720", "BLOCK-T724",
                             "BLOCK-T710", "BLOCK-T710_v2"])
    args = ap.parse_args()

    base = args.output_dir or f"output/{args.param_set}"
    donuts = f"{base}/donuts.parquet"
    visits_p = f"{base}/visits.parquet"
    sidecar = f"{base}/{args.mi_name}/zk_intrinsic.parquet"
    out_pdf = f"{base}/{args.mi_name}/plots/fam_coadd_miw_maps.pdf"
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    visits = pd.read_parquet(visits_p, columns=[
        "day_obs", "seq_num", "alt", "az", "rotator_angle", "science_program", "band"])
    groups = build_coadd_groups(visits, args.bin_deg,
                                set(args.programs) if args.programs else None,
                                args.min_triplets)
    ng = groups["group"].nunique()
    print(f"{len(groups)} triplets in {ng} coadd groups "
          f"(>= {args.min_triplets} triplets each)", flush=True)
    if ng == 0:
        raise SystemExit("no coadd groups -- check programs / min_triplets")

    maps_by_group, edges = coadd_maps(donuts, sidecar, groups,
                                      args.grid_n, args.fp_radius)
    render_pdf(maps_by_group, edges, groups, out_pdf, args.pct)


if __name__ == "__main__":
    main()
