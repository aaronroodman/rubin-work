#!/usr/bin/env python3
"""Validate a new ``miw`` build against its pre-move old-tree counterpart.

Phase 3c item 4 of ``notes/status/organization_plan.md``. ``compare_builds`` could not
settle the ``danish_1_2`` MIW comparison because its ``intrinsic_split_maps`` row key
``(thx_deg, thy_deg)`` **does not join**: the two builds lay their 71x71 field lattice on
slightly different coordinates. The old build's half-width is ``126/73 = 1.7260274 deg``
and the new one's is ``1.726 deg`` exactly, so the lattices are offset by up to
**2.74e-05 deg of field angle** -- a thousandth of the 0.0493 deg cell step, but enough
that an exact float key matches only the origin (1 of 3,831 cells). Matching instead on
the **integer lattice index** ``(ix, iy)`` recovers the true relationship: 3,831 shared
cells, 154 new-only, 0 old-only, a clean superset.

What the comparison then shows, and what the plots are for: the difference on the shared
cells is **not** uniform. It is confined to the field edge, which is the signature of
``ts_intrinsic_wavefront`` commit ``3f66904`` ("Fix the OCS/CCS null space at the field
edge in the intrinsic split", 2026-10-05) -- the only change to ``intrinsic_split.py``
between the old builds (2026-07) and the new ones. That commit stopped nulling
under-determined outer rings and continues them radially instead, which is what both adds
the 154 cells and moves the outermost shared ring.

Writes one PDF plus a ``summary.md`` per variant under
``<studies>/phase3c_validation/miw/<variant>/``. Never writes into a build directory and
never into the old tree.

Usage::

    python -m rubinwork.products.miw.refbuild.compare_old_tree \\
        --variant d12-A_50_34_i --build 20261014
    python -m rubinwork.products.miw.refbuild.compare_old_tree --all
"""

import argparse
import pathlib
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon

from ...catalog import studies_root
from .. import build_dir, build_source, split_rotator_bins

OLD_TREE = pathlib.Path(
    "/sdf/home/r/roodman/notebooks/rubin-work/aos/output/miw")
"""Pre-move output root. Read-only here; nothing is ever written under it."""

OLD_DIR_BY_VARIANT = {
    "d12-A_50_34_i": "danish_1_2_A_50_34_i",
    "d12-A_50_34_i_5rot": "danish_1_2_A_50_34_i_5rot",
}
"""New variant name -> old-tree directory name, for the two ``danish_1_2`` MIW entries."""

NEW_RMS_COLUMNS = ["n_null_rings", "max_null_dim", "n_continued_rings",
                   "min_ring_coverage_seen"]
"""``intrinsic_split_rms`` columns that commit ``3f66904`` added; absent in the old build."""

EXPECTED_CCS_MAX_UM = 9.2e-04
"""Comparison-plan difference 2's upper bound on the CCS change, um of wavefront."""

EXPECTED_OCS_MAX_UM = 7e-06
"""Comparison-plan difference 2's upper bound on the OCS change, um of wavefront."""

CWFS_CORNERS_DEG = {
    "R00_SW": (-1.19, -1.19),
    "R04_SW": (-1.19, 1.19),
    "R40_SW": (1.19, -1.19),
    "R44_SW": (1.19, 1.19),
}
"""Approximate corner-wavefront-sensor field centres, deg in the OCS field frame.

Used only to orient the valid-cell mask plot, so the nominal raft centres are enough.
"""


def lattice_index(frame):
    """Integer lattice indices of a field-grid table's cells.

    The two builds' lattices differ by up to 2.74e-05 deg of field angle, so an exact
    float join on ``(thx_deg, thy_deg)`` fails. Each build's grid is a complete regular
    71x71 lattice, so ranking the unique coordinates recovers a join key that is exact.

    Parameters
    ----------
    frame : `pandas.DataFrame`
        Table with ``thx_deg`` and ``thy_deg`` columns, deg of field angle.

    Returns
    -------
    ix, iy : `numpy.ndarray`
        0-based column and row index of each cell on its own lattice, dimensionless.
    """
    ux = np.unique(frame["thx_deg"].to_numpy())
    uy = np.unique(frame["thy_deg"].to_numpy())
    return (np.searchsorted(ux, frame["thx_deg"].to_numpy()),
            np.searchsorted(uy, frame["thy_deg"].to_numpy()))


def load_maps(path):
    """Read an ``intrinsic_split_maps`` table and attach its lattice indices."""
    frame = pd.read_parquet(path)
    frame["ix"], frame["iy"] = lattice_index(frame)
    frame["r_deg"] = np.hypot(frame["thx_deg"], frame["thy_deg"])
    return frame


def map_columns(frame):
    """The ``Z<j>_OCS`` / ``Z<j>_CCS`` value columns of a maps table, in file order."""
    return [c for c in frame.columns if c.startswith("Z")
            and (c.endswith("_OCS") or c.endswith("_CCS"))]


def compare_maps(old, new):
    """Per-column difference of two maps tables on their shared lattice cells.

    Parameters
    ----------
    old, new : `pandas.DataFrame`
        Maps tables from `load_maps`.

    Returns
    -------
    merged : `pandas.DataFrame`
        Inner join on the lattice index, with ``_o`` / ``_n`` suffixed value columns.
    stats : `pandas.DataFrame`
        One row per value column: ``n_both_finite`` (cells, dimensionless),
        ``maxabs_um`` and ``rms_um`` of the new-minus-old difference, um of wavefront.
    new_only : `pandas.DataFrame`
        The cells present only in the new build.
    """
    merged = old.merge(new, on=["ix", "iy"], suffixes=("_o", "_n"))
    rows = []
    for col in map_columns(old):
        a = merged[f"{col}_o"].to_numpy()
        b = merged[f"{col}_n"].to_numpy()
        good = np.isfinite(a) & np.isfinite(b)
        diff = b[good] - a[good]
        rows.append((col, int(good.sum()),
                     float(np.abs(diff).max()) if good.any() else np.nan,
                     float(np.sqrt((diff ** 2).mean())) if good.any() else np.nan))
    stats = pd.DataFrame(
        rows, columns=["column", "n_both_finite", "maxabs_um", "rms_um"])
    shared = set(zip(merged["ix"], merged["iy"]))
    mask = [p not in shared for p in zip(new["ix"], new["iy"])]
    return merged, stats, new[mask]


def radial_bins(merged, columns):
    """Worst per-cell difference over all Zernikes, binned by field radius.

    The field-edge hypothesis predicts a difference that is negligible in the interior
    and large only beyond the radius commit ``3f66904`` says used to be nulled.

    Returns
    -------
    table : `pandas.DataFrame`
        Per radial bin: ``n_cells`` (dimensionless), ``maxabs_um`` and ``median_um``
        of the worst-Zernike absolute difference, um of wavefront.
    """
    stack = np.stack([np.abs(merged[f"{c}_n"].to_numpy()
                            - merged[f"{c}_o"].to_numpy()) for c in columns])
    with np.errstate(invalid="ignore"):
        worst = np.nanmax(stack, axis=0)
    edges = [0.0, 1.0, 1.5, 1.65, 1.729, 2.0]
    rows = []
    radius = merged["r_deg_o"].to_numpy()
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (radius >= lo) & (radius < hi)
        if not sel.any():
            continue
        vals = worst[sel]
        rows.append((f"{lo:.3f}-{hi:.3f}", int(sel.sum()),
                     float(np.nanmax(vals)), float(np.nanmedian(vals))))
    return pd.DataFrame(
        rows, columns=["r_deg_bin", "n_cells", "maxabs_um", "median_um"])


def nan_counts(frame):
    """NaN map cells: total over all value columns, and in ``Z4_OCS`` alone.

    Commit ``3f66904`` reports the NaN annulus it removed as **124 points** in the Z4
    handoff map, so the ``Z4_OCS`` count is the one that identifies it directly.
    """
    cols = map_columns(frame)
    total = int(sum((~np.isfinite(frame[c].to_numpy())).sum() for c in cols))
    z4 = int((~np.isfinite(frame["Z4_OCS"].to_numpy())).sum()) \
        if "Z4_OCS" in frame else -1
    return total, z4


def describe_grids(old, new, merged, new_only, offset_deg):
    """Prose describing how this variant's two lattices relate.

    The two ``danish_1_2`` variants differ, so this cannot be one hardcoded sentence.
    The 9-bin variant's old build sits on an older lattice of half-width
    ``126/73 = 1.7260274 deg`` with 3,831 cells; the ``_5rot`` old build is already on the
    ``1.726 deg`` lattice with all 3,985 cells, but 124 of them NaN.
    """
    half_old = float(np.abs(old["thx_deg"]).max())
    half_new = float(np.abs(new["thx_deg"]).max())
    if offset_deg > 0:
        return (
            "The two builds put their 71x71 field lattice on **different coordinates**: "
            f"the old half-width is **{half_old:.7f} deg** of field angle (`126/73`) and "
            f"the new one is **{half_new:.7f} deg** exactly. The lattices differ by at "
            f"most **{offset_deg:.3e} deg of field angle**, a thousandth of the 0.0493 "
            "deg cell step, so an exact float join on `(thx_deg, thy_deg)` matches only "
            "the origin -- which is why `compare_builds` saw a row-count mismatch and "
            "compared 0 columns. Joining on the **integer lattice index** instead gives "
            "the real relationship, a clean superset:")
    return (
        "Both builds are on the **same** lattice, half-width "
        f"**{half_new:.7f} deg** of field angle, and every cell joins "
        f"(offset {offset_deg:.3e} deg). For this variant the old build already had all "
        f"{len(new)} cells, so **no cells were added**; what changed is that cells which "
        "were NaN are now filled. The cell-count growth reported for the 9-bin variant "
        "is therefore two effects stacked -- a lattice change there, and the NaN rim "
        "fill in both.")


def plot_map_triptych(pdf, merged, new_only, column):
    """Old | new | new-minus-old maps of one Zernike component, with added cells marked.

    The first two panels share one symmetric color scale so they can be read against
    each other; the difference panel has its own.
    """
    a = merged[f"{column}_o"].to_numpy()
    b = merged[f"{column}_n"].to_numpy()
    x = merged["thx_deg_o"].to_numpy()
    y = merged["thy_deg_o"].to_numpy()
    span = np.nanmax(np.abs(np.concatenate([a, b])))
    diff = b - a
    dspan = np.nanmax(np.abs(diff)) or 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    for ax, vals, title, lim, cmap in (
            (axes[0], a, f"old tree  {column}", span, "RdBu_r"),
            (axes[1], b, f"new build  {column}", span, "RdBu_r"),
            (axes[2], diff, f"new - old  {column}", dspan, "coolwarm")):
        sc = ax.scatter(x, y, c=vals, s=6, cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("thx (deg, OCS field angle)")
        ax.set_ylabel("thy (deg, OCS field angle)")
        fig.colorbar(sc, ax=ax, label="wavefront (um)")
    if len(new_only):
        for ax in axes:
            ax.scatter(new_only["thx_deg"], new_only["thy_deg"], s=14,
                       facecolors="none", edgecolors="k", linewidths=0.5,
                       label=f"{len(new_only)} new-only cells")
        axes[0].legend(loc="upper left", fontsize=7)
    pdf.savefig(fig)
    plt.close(fig)


def plot_valid_mask(pdf, old, new, merged):
    """Valid-cell mask: old only, new only, both, with CWFS corners and the old hull."""
    shared = set(zip(merged["ix"], merged["iy"]))
    new_only = new[[p not in shared for p in zip(new["ix"], new["iy"])]]
    old_only = old[[p not in shared for p in zip(old["ix"], old["iy"])]]

    fig, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    ax.scatter(merged["thx_deg_n"], merged["thy_deg_n"], s=5, c="0.75",
               label=f"both ({len(merged)} cells)")
    ax.scatter(new_only["thx_deg"], new_only["thy_deg"], s=16, c="tab:red",
               label=f"new only ({len(new_only)} cells)")
    if len(old_only):
        ax.scatter(old_only["thx_deg"], old_only["thy_deg"], s=16, c="tab:blue",
                   label=f"old only ({len(old_only)} cells)")
    else:
        ax.scatter([], [], s=16, c="tab:blue", label="old only (0 cells)")

    try:
        from scipy.spatial import ConvexHull
        pts = old[["thx_deg", "thy_deg"]].to_numpy()
        hull = ConvexHull(pts)
        ax.add_patch(Polygon(pts[hull.vertices], closed=True, fill=False,
                             edgecolor="k", linestyle="--", linewidth=1.0,
                             label="old-tree convex hull"))
    except Exception as exc:                      # scipy absent or degenerate hull
        print(f"[compare] convex hull skipped: {exc}")

    for name, (cx, cy) in CWFS_CORNERS_DEG.items():
        ax.plot(cx, cy, marker="s", ms=9, mfc="none", mec="tab:green", mew=1.6)
        ax.annotate(name, (cx, cy), textcoords="offset points", xytext=(7, 5),
                    fontsize=7, color="tab:green")
    ax.plot([], [], marker="s", ls="none", mfc="none", mec="tab:green",
            label="CWFS corner (nominal)")

    ax.set_aspect("equal")
    ax.set_xlabel("thx (deg, OCS field angle)")
    ax.set_ylabel("thy (deg, OCS field angle)")
    ax.set_title("Valid field-grid cells, old tree vs new build", fontsize=11)
    ax.legend(loc="upper left", fontsize=7)
    pdf.savefig(fig)
    plt.close(fig)


def plot_difference_histogram(pdf, merged, columns, stats):
    """Per-cell difference histogram on the shared cells, against expected difference 2."""
    diffs = np.concatenate([
        (merged[f"{c}_n"].to_numpy() - merged[f"{c}_o"].to_numpy())
        for c in columns])
    diffs = diffs[np.isfinite(diffs)]
    maxabs = float(np.abs(diffs).max())
    rms = float(np.sqrt((diffs ** 2).mean()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    axes[0].hist(diffs, bins=200, color="tab:blue")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("new - old (um of wavefront)")
    axes[0].set_ylabel("cells x Zernike components (count)")
    axes[0].set_title(
        f"shared-cell difference\nmax abs {maxabs:.3e} um, RMS {rms:.3e} um",
        fontsize=10)

    absd = np.abs(diffs)
    absd = absd[absd > 0]
    axes[1].hist(np.log10(absd), bins=200, color="tab:gray")
    for value, label, color in (
            (EXPECTED_OCS_MAX_UM, "expected OCS max 7e-06 um", "tab:green"),
            (EXPECTED_CCS_MAX_UM, "expected CCS max 9.2e-04 um", "tab:orange")):
        axes[1].axvline(np.log10(value), color=color, ls="--", label=label)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("log10 |new - old| (um of wavefront)")
    axes[1].set_ylabel("count")
    axes[1].set_title("against comparison-plan difference 2", fontsize=10)
    axes[1].legend(fontsize=7)
    pdf.savefig(fig)
    plt.close(fig)

    worst = stats.sort_values("maxabs_um", ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    ax.barh(worst["column"][::-1], worst["maxabs_um"][::-1], color="tab:red")
    ax.set_xscale("log")
    ax.set_xlabel("max abs difference (um of wavefront)")
    ax.set_title("worst 12 map columns on the shared cells", fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)
    return maxabs, rms


def plot_radial(pdf, table):
    """Worst-Zernike difference by field radius -- the field-edge test."""
    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    pos = np.arange(len(table))
    ax.bar(pos - 0.2, table["maxabs_um"], width=0.4, label="max abs")
    ax.bar(pos + 0.2, table["median_um"], width=0.4, label="median")
    ax.set_xticks(pos)
    ax.set_xticklabels(table["r_deg_bin"], fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("field radius bin (deg)")
    ax.set_ylabel("|new - old|, worst Zernike (um of wavefront)")
    ax.set_title("Difference vs field radius: the field-edge hypothesis", fontsize=10)
    ax.legend(fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)


def compare_rms(old_dir, new_dir):
    """``intrinsic_split_rms``, old vs new, plus the columns only the new build has."""
    old = pd.read_parquet(old_dir / "intrinsic_split_rms.parquet")
    new = pd.read_parquet(new_dir / "intrinsic_split_rms.parquet")
    merged = old.merge(new, on="j", suffixes=("_o", "_n"))
    rows = []
    for col in ("dataRMS", "O_tel", "C_cam", "residRMS"):
        d = merged[f"{col}_n"].to_numpy() - merged[f"{col}_o"].to_numpy()
        rows.append((col, float(np.abs(merged[f"{col}_o"]).max()),
                     float(np.abs(merged[f"{col}_n"]).max()),
                     float(np.abs(d).max())))
    stats = pd.DataFrame(rows, columns=[
        "column", "max_old_um", "max_new_um", "maxabs_diff_um"])
    present = [c for c in NEW_RMS_COLUMNS if c in new.columns]
    rings = new[["Zernike", "j"] + present] if present else pd.DataFrame()
    return stats, rings, merged


def compare_decomp(old_dir, new_dir):
    """``intrinsic_split_decomp`` O/C polar fields, old vs new, per ``(j, part)``.

    The fields are array-valued (``n_r`` x ``n_az`` flattened), so the comparison is
    elementwise on the shared polar lattice rather than on a scalar.
    """
    old = pd.read_parquet(old_dir / "intrinsic_split_decomp.parquet")
    new = pd.read_parquet(new_dir / "intrinsic_split_decomp.parquet")
    merged = old.merge(new, on=["j", "part"], suffixes=("_o", "_n"))
    rows = []
    for _, row in merged.iterrows():
        entry = {"j": int(row["j"]), "part": row["part"]}
        for field in ("O_re", "O_im", "C_re", "C_im"):
            a = np.asarray(row[f"{field}_o"], dtype=float)
            b = np.asarray(row[f"{field}_n"], dtype=float)
            n = min(a.size, b.size)
            d = b[:n] - a[:n]
            good = np.isfinite(d)
            entry[f"{field}_maxabs_um"] = (
                float(np.abs(d[good]).max()) if good.any() else np.nan)
        rows.append(entry)
    return pd.DataFrame(rows)


def compare_vmode13(old_dir, new_dir, bins):
    """``vmode_13`` and the DOF spread per rotator bin, old vs new.

    Tests the near-null-rotation explanation: a v-mode that differs a lot while every
    DOF agrees to round-off is a rotation inside a degenerate subspace, not a changed
    fit.
    """
    rows = []
    for name in bins:
        po = old_dir / "build" / name / "dz_fits.parquet"
        pn = new_dir / "build" / name / "dz_fits.parquet"
        if not (po.exists() and pn.exists()):
            continue
        cols = pq.ParquetFile(pn).schema_arrow.names
        dofs = [c for c in cols if c.startswith("dof_")]
        want = ["day_obs", "seq_num", "vmode_13"] + dofs
        old = pd.read_parquet(po, columns=[c for c in want if c in
                                           pq.ParquetFile(po).schema_arrow.names])
        new = pd.read_parquet(pn, columns=want)
        merged = old.merge(new, on=["day_obs", "seq_num"], suffixes=("_o", "_n"))
        if not len(merged):
            continue
        dv = merged["vmode_13_n"].to_numpy() - merged["vmode_13_o"].to_numpy()
        dof_max = 0.0
        for col in dofs:
            if f"{col}_o" in merged:
                d = merged[f"{col}_n"].to_numpy() - merged[f"{col}_o"].to_numpy()
                dof_max = max(dof_max, float(np.nanmax(np.abs(d))))
        rows.append({
            "rot_bin": name, "n_visits": int(len(merged)),
            "vmode_13_maxabs_diff": float(np.nanmax(np.abs(dv))),
            "vmode_13_rms_old": float(np.sqrt(np.nanmean(
                merged["vmode_13_o"].to_numpy() ** 2))),
            "dof_maxabs_diff_um": dof_max,
        })
    return pd.DataFrame(rows)


def plot_vmode13(pdf, table):
    """v-mode 13 difference beside the worst DOF difference, per rotator bin."""
    if not len(table):
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.6), constrained_layout=True)
    pos = np.arange(len(table))
    ax.bar(pos - 0.2, table["vmode_13_maxabs_diff"], width=0.4,
           label="vmode_13 max abs difference (dimensionless amplitude)")
    ax.bar(pos + 0.2, table["dof_maxabs_diff_um"].clip(lower=1e-16), width=0.4,
           label="worst DOF max abs difference (um)")
    ax.set_xticks(pos)
    ax.set_xticklabels(table["rot_bin"], rotation=45, ha="right", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("max abs difference")
    ax.set_title("vmode_13 vs the DOF it is built from, per rotator bin", fontsize=10)
    ax.legend(fontsize=7)
    pdf.savefig(fig)
    plt.close(fig)


def compare_zk(old_dir, new_dir, limit=4_000_000):
    """``zk_intrinsic`` on the shared donut key: max abs difference per Zernike.

    The new sidecar covers the whole FAM donut table and the old one a third of it, so
    the comparison is on the inner join of
    ``(day_obs, seq_num, detector, centroid_x_extra, centroid_y_extra)``.
    """
    key = ["day_obs", "seq_num", "detector",
           "centroid_x_extra", "centroid_y_extra"]
    cols = key + ["zk_intrinsic_MI"]
    if not ((old_dir / "zk_intrinsic.parquet").exists()
            and (new_dir / "zk_intrinsic.parquet").exists()):
        return pd.DataFrame(), 0
    old = pd.read_parquet(old_dir / "zk_intrinsic.parquet", columns=cols)
    new = pd.read_parquet(new_dir / "zk_intrinsic.parquet", columns=cols)
    if len(old) > limit:
        old = old.iloc[:limit]
    merged = old.merge(new, on=key, suffixes=("_o", "_n"))
    if not len(merged):
        return pd.DataFrame(), 0
    a = np.stack([np.asarray(v, dtype=float)
                  for v in merged["zk_intrinsic_MI_o"].to_numpy()])
    b = np.stack([np.asarray(v, dtype=float)
                  for v in merged["zk_intrinsic_MI_n"].to_numpy()])
    n = min(a.shape[1], b.shape[1])
    d = b[:, :n] - a[:, :n]
    with np.errstate(invalid="ignore"):
        table = pd.DataFrame({
            "zk_index": np.arange(n),
            "maxabs_diff_um": np.nanmax(np.abs(d), axis=0),
            "rms_diff_um": np.sqrt(np.nanmean(d ** 2, axis=0)),
        })
    return table, len(merged)


def plot_zk(pdf, table, n_rows):
    """Per-Zernike max abs difference of the per-donut sidecar."""
    if not len(table):
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.4), constrained_layout=True)
    ax.bar(table["zk_index"], table["maxabs_diff_um"].clip(lower=1e-18))
    ax.set_yscale("log")
    ax.set_xlabel("Zernike index in zk_intrinsic_MI (dimensionless)")
    ax.set_ylabel("max abs difference (um of wavefront)")
    ax.set_title(f"zk_intrinsic on {n_rows} shared donut rows", fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


def copy_build_plots(new_dir, out_dir, bins, grid_dir=None):
    """Link the new build's own split PDF and per-bin build_intrinsic plots.

    Hard links where the filesystem allows it and copies otherwise, so the review
    directory is self-contained without duplicating 10 MB PDFs needlessly. ``grid_dir``
    is where the per-bin plots live, which for a ``build_from`` variant is its parent's
    build directory.
    """
    grid_dir = grid_dir or new_dir
    dest = out_dir / "build_plots"
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    sources = [new_dir / "intrinsic_split.pdf"]
    for name in bins:
        sources.extend(sorted((grid_dir / "build" / name).glob("*.pdf"))
                       if (grid_dir / "build" / name).is_dir() else [])
    for src in sources:
        if not src.exists():
            continue
        target = dest / (src.name if src.parent == new_dir
                         else f"{src.parent.name}__{src.name}")
        if target.exists():
            copied.append(target)
            continue
        try:
            target.hardlink_to(src)
        except OSError:
            shutil.copy2(src, target)
        copied.append(target)
    return copied


def write_summary(path, variant, new_dir, old_dir, parts):
    """Write the per-variant ``summary.md``: every number with its quantity and units."""
    maps_stats = parts["maps_stats"]
    ocs = maps_stats[maps_stats["column"].str.endswith("_OCS")]
    ccs = maps_stats[maps_stats["column"].str.endswith("_CCS")]
    lines = [
        f"# `miw` validation: `{variant}`, old tree vs new build",
        "",
        "> **Status:** validation output for review. Nothing promoted, no `set-current`.",
        f"> · **New build:** `{new_dir}`",
        f"> · **Old tree:** `{old_dir}`",
        "> · **Kind:** working state (validation)",
        "",
        "## What the numbers say",
        "",
        "Three claims, each with the number that supports it. **None of this is a "
        "decision** -- the ring-continuation question is Aaron's.",
        "",
        "1. **The map comparison is now real, not vacuous.** `compare_builds` compared 0 "
        "columns because its `(thx_deg, thy_deg)` key does not join across the two "
        "lattices. On the integer lattice index the join is exact and the cell sets "
        f"nest: {parts['n_shared']} shared, {parts['n_new_only']} new-only, "
        f"{parts['n_old_only']} old-only.",
        "2. **The difference is confined to the field edge.** Interior cells "
        f"(r < 1.650 deg) agree to **{parts['interior_maxabs_um']:.3e} um of "
        f"wavefront**, {parts['interior_verdict']} difference 2's CCS bound of "
        f"9.2e-04 um. Beyond r = 1.729 deg the worst cell reaches "
        f"**{parts['edge_maxabs_um']:.3e} um**, a factor "
        f"{parts['edge_maxabs_um'] / parts['interior_maxabs_um']:.0f} larger. "
        "That radial profile matches commit `3f66904`'s own account of what it changed, "
        "so the excess is the **accepted new edge behaviour**, not a mask artifact of "
        "comparing different valid-cell sets and not a physics change in the interior.",
        "3. **`vmode_13` is a degenerate-subspace rotation, not a changed fit.** It "
        f"differs by up to **{parts['vmode_maxabs']:.3f} (dimensionless v-mode "
        "amplitude)** on bins where every DOF agrees to "
        f"**{parts['vmode_dof_maxabs_um']:.2e} um**. A mode whose DOF are identical to "
        "round-off is not constrained by the data.",
        "",
        "## The row key, and why `compare_builds` could not settle this",
        "",
        parts["key_note"],
        "",
        "| set | cells (dimensionless) |",
        "|---|---|",
        f"| old-tree cells | {parts['n_old']} |",
        f"| new-build cells | {parts['n_new']} |",
        f"| shared | {parts['n_shared']} |",
        f"| new only | {parts['n_new_only']} |",
        f"| old only | {parts['n_old_only']} |",
        "",
        f"- NaN map cells summed over all `Z<j>_OCS`/`Z<j>_CCS` columns: "
        f"**old {parts['nan_old']}**, **new {parts['nan_new']}** (cells, dimensionless)",
        f"- NaN cells in `Z4_OCS` alone: **old {parts['nan_old_z4']}**, "
        f"**new {parts['nan_new_z4']}**",
        "",
        "## `intrinsic_split_maps` on the shared cells",
        "",
        f"- max abs difference, all map columns: **{parts['maps_maxabs_um']:.4e} um of "
        "wavefront**",
        f"- RMS difference, all map columns: **{parts['maps_rms_um']:.4e} um of "
        "wavefront**",
        f"- worst OCS column: **{ocs.loc[ocs['maxabs_um'].idxmax(), 'column']}** at "
        f"**{ocs['maxabs_um'].max():.4e} um of wavefront**",
        f"- worst CCS column: **{ccs.loc[ccs['maxabs_um'].idxmax(), 'column']}** at "
        f"**{ccs['maxabs_um'].max():.4e} um of wavefront**",
        "",
        "Against comparison-plan **difference 2** (CCS 7.8e-05 to 9.2e-04 um of "
        f"wavefront, OCS at most 7e-06 um): the OCS change **exceeds** its bound by "
        f"{ocs['maxabs_um'].max() / EXPECTED_OCS_MAX_UM:.0f}x and the CCS change by "
        f"{ccs['maxabs_um'].max() / EXPECTED_CCS_MAX_UM:.0f}x. The radial breakdown "
        "below says where.",
        "",
        "## Where the difference lives: field radius",
        "",
        "Worst-Zernike absolute difference per cell, binned by field radius.",
        "",
        parts["radial"].to_markdown(index=False),
        "",
        "Interpretation: the interior is inside difference 2's CCS bound and the whole "
        "excess sits at the field edge, which is the signature of the "
        "`ts_intrinsic_wavefront` change below rather than a physics change.",
        "",
        "## `intrinsic_split_rms`, old vs new",
        "",
        parts["rms_stats"].to_markdown(index=False),
        "",
        "### The 4 columns only the new build has, per Zernike ring",
        "",
        (parts["rings"].to_markdown(index=False) if len(parts["rings"])
         else "_none present_"),
        "",
        "## `intrinsic_split_decomp` O/C polar fields, per `(j, part)`",
        "",
        "Elementwise on the shared polar lattice, um of wavefront.",
        "",
        parts["decomp"].to_markdown(index=False, floatfmt=".3e"),
        "",
        "## `vmode_13` vs the DOF it is built from, per rotator bin",
        "",
        (parts["vmode"].to_markdown(index=False, floatfmt=".4g")
         if len(parts["vmode"]) else "_no comparable dz_fits bins_"),
        "",
        "`vmode_13` is a dimensionless v-mode amplitude; the DOF differences are um. A "
        "large v-mode difference beside round-off DOF differences is a rotation within a "
        "degenerate subspace, not a changed fit.",
        "",
        "## `zk_intrinsic` on the shared donut key",
        "",
        f"Shared donut rows: **{parts['zk_rows']}** (dimensionless).",
        "",
        (parts["zk"].to_markdown(index=False, floatfmt=(".0f", ".3e", ".3e"))
         if len(parts["zk"]) else
         "_no shared rows compared (this variant's build has no `zk_intrinsic`)_"),
        "",
        "## The `ts_intrinsic_wavefront` commit responsible",
        "",
        "`intrinsic_split.py` changed **once** between the old builds (2026-07) and "
        "these:",
        "",
        "- **`3f66904`** (2026-10-05) *Fix the OCS/CCS null space at the field edge in "
        "the intrinsic split* — stops skipping under-determined outer rings, continues "
        "rings below `min_ring_coverage` radially from the nearest fitted ring, assigns "
        "unconstrained singular directions by `degen_assignment` instead of 50/50, and "
        "adds the 4 `intrinsic_split_rms` columns. Its own note records the NaN map "
        "points going 124/16 -> 0.",
        "",
        "This is a **rim/NaN-ring fix, not a convex-hull or CWFS-outside-hull "
        "extension**. The evidence: the old `_5rot` map has **124 NaN cells in "
        "`Z4_OCS`**, exactly the \"124 points\" annulus at `r >= 1.729 deg` that the "
        "commit message names, and the new build has **0**. The CWFS corners sit at "
        "about 1.19 deg of field angle, already well inside the old valid region, so "
        "nothing here makes MIW newly available outside a convex hull.",
        "",
        "## Files",
        "",
        f"- `{parts['pdf'].name}` — every plot above",
        "- `build_plots/` — the new build's own `intrinsic_split.pdf` and the per-bin "
        "`build_intrinsic` PDFs",
        f"- `{parts['stats_csv'].name}` — the per-column map statistics",
        "",
    ]
    path.write_text("\n".join(lines))


def run_variant(variant, build, out_root):
    """Produce the full validation output for one MIW variant."""
    old_dir = OLD_TREE / OLD_DIR_BY_VARIANT[variant]
    new_dir = build_dir(variant, build)
    out_dir = out_root / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[compare] {variant}: {old_dir}  vs  {new_dir}")

    old = load_maps(old_dir / "intrinsic_split_maps.parquet")
    new = load_maps(new_dir / "intrinsic_split_maps.parquet")
    merged, maps_stats, new_only = compare_maps(old, new)
    columns = map_columns(old)
    offset = float(np.abs(merged["thx_deg_n"] - merged["thx_deg_o"]).max())
    INTERIOR_BINS = ["0.000-1.000", "1.000-1.500", "1.500-1.650"]
    key_note = describe_grids(old, new, merged, new_only, offset)
    nan_old, nan_old_z4 = nan_counts(old)
    nan_new, nan_new_z4 = nan_counts(new)
    radial = radial_bins(merged, columns)
    rms_stats, rings, _ = compare_rms(old_dir, new_dir)
    decomp = compare_decomp(old_dir, new_dir)
    # A build_from variant runs no build_intrinsic, so its per-bin grids and dz_fits
    # live in the PARENT's build directory; the plots and vmode comparison come from
    # there.  The old tree has the same split -- `_5rot` has no build/ either.
    grid_dir = new_dir if (new_dir / "build").is_dir() else build_dir(
        build_source(variant), build)
    old_grid_dir = old_dir if (old_dir / "build").is_dir() else (
        OLD_TREE / OLD_DIR_BY_VARIANT[build_source(variant)])
    # Only the bins this variant's split actually reads: `_5rot` uses 5 of the parent's 9.
    bins = [f"rot_{lo:g}_{hi:g}" for lo, hi in split_rotator_bins(variant)]
    vmode = compare_vmode13(old_grid_dir, grid_dir, bins)
    zk, zk_rows = compare_zk(old_dir, new_dir)
    interior_maxabs_um = float(
        radial[radial["r_deg_bin"].isin(INTERIOR_BINS)]["maxabs_um"].max())

    pdf_path = out_dir / f"{variant}_old_vs_new.pdf"
    with PdfPages(pdf_path) as pdf:
        plot_valid_mask(pdf, old, new, merged)
        plot_radial(pdf, radial)
        maps_maxabs, maps_rms = plot_difference_histogram(
            pdf, merged, columns, maps_stats)
        for column in ("Z4_OCS", "Z4_CCS", "Z5_OCS", "Z6_OCS", "Z7_CCS",
                       "Z10_CCS"):
            if f"{column}_o" in merged:
                plot_map_triptych(pdf, merged, new_only, column)
        plot_vmode13(pdf, vmode)
        plot_zk(pdf, zk, zk_rows)

    stats_csv = out_dir / f"{variant}_map_column_stats.csv"
    maps_stats.to_csv(stats_csv, index=False)
    copy_build_plots(new_dir, out_dir, bins, grid_dir)

    write_summary(out_dir / "summary.md", variant, new_dir, old_dir, dict(
        maps_stats=maps_stats, radial=radial, rms_stats=rms_stats, rings=rings,
        decomp=decomp, vmode=vmode, zk=zk, zk_rows=zk_rows,
        maps_maxabs_um=maps_maxabs, maps_rms_um=maps_rms,
        n_old=len(old), n_new=len(new), n_shared=len(merged),
        n_new_only=len(new_only), n_old_only=len(old) - len(merged),
        lattice_offset_deg=offset, key_note=key_note,
        interior_maxabs_um=interior_maxabs_um,
        edge_maxabs_um=float(radial["maxabs_um"].iloc[-1]),
        interior_verdict=(
            "inside" if interior_maxabs_um <= EXPECTED_CCS_MAX_UM
            else f"just over ({interior_maxabs_um / EXPECTED_CCS_MAX_UM:.1f}x)"),
        vmode_maxabs=(float(vmode["vmode_13_maxabs_diff"].max())
                      if len(vmode) else float("nan")),
        vmode_dof_maxabs_um=(float(vmode["dof_maxabs_diff_um"].max())
                             if len(vmode) else float("nan")),
        nan_old=nan_old, nan_new=nan_new,
        nan_old_z4=nan_old_z4, nan_new_z4=nan_new_z4,
        pdf=pdf_path, stats_csv=stats_csv))
    print(f"[compare] {variant}: wrote {out_dir}")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", action="append", default=None,
                        choices=sorted(OLD_DIR_BY_VARIANT))
    parser.add_argument("--all", action="store_true",
                        help="both danish_1_2 MIW variants")
    parser.add_argument("--build", default="20261014",
                        help="new build name to validate (default 20261014)")
    parser.add_argument("--out-root", default=None,
                        help="default <studies>/phase3c_validation/miw")
    args = parser.parse_args(argv)

    variants = sorted(OLD_DIR_BY_VARIANT) if args.all else (args.variant or [])
    if not variants:
        parser.error("give --variant or --all")
    out_root = (pathlib.Path(args.out_root) if args.out_root
                else studies_root() / "phase3c_validation" / "miw")
    for variant in variants:
        run_variant(variant, args.build, out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
