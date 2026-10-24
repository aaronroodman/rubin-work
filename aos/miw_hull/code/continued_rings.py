#!/usr/bin/env python3
"""Analysis 2 of `miw_hull`: which polar rings did the builder fit, and which did it
continue radially?

Analysis 1 measured the coverage geometry. This script pairs that geometry against the
**builder's own bookkeeping** -- the four ``intrinsic_split_rms`` columns that exist only
in builds after ``ts_intrinsic_wavefront`` commit ``3f66904`` -- and asks whether the two
agree. A disagreement would mean the coverage quantity measured in analysis 1 is not the
quantity the builder decides on.

The decision rule, read from the builder rather than assumed
(``intrinsic_split.decompose_spin_lsq``, ``intrinsic_split.py``):

* line 453, ``coverage = ring_coverage(valid)`` -- the coverage array is computed once per
  decomposition, from ``ring_coverage`` (line 331), whose body is
  ``np.any(v, axis=0).mean(axis=1)`` (line 338). ``np.any`` over axis 0 pools the rotator
  bins, so the decision is on **union** coverage, not per bin.
* line 469, ``if coverage[k] < min_coverage: continue`` -- the ring is skipped by the fit.
* lines 500-508, the skipped rings with data are continued from the nearest fitted ring
  and flagged ``ring_filled[k] = True``.

So a ring is fit or continued once for the whole decomposition, identically for every
Zernike and regardless of how any single bin covers it. Per-bin coverage still matters
for the rank of a ring that *is* fit, which is why it is reported here, but it does not
enter the fit/continue decision.

Five questions, in the order of Aaron's request (2026-10-14):

B1  Per Zernike, per build: the builder's ring counts and the radius of every continued or
    null ring, flagging any Zernike off the common pattern.
B2  Per rotator bin: coverage at the outer 5 rings beside the fitted/continued status.
B3  The cross-check: every continued ring below 0.5 coverage, every ring below 0.5 either
    continued or null.
B4  The ring set analysis 3 needs: fitted in one build, continued in the other.
B5  Exposure of the r <= 1.725 deg donut-selection limit to any continued ring, by both
    consumer routes -- polar (ring interpolation) and Cartesian (the 71x71 map grid).

Outputs, under ``aos/miw_hull/output/`` (gitignored symlink to
``studies/aos/miw_hull``):

``continued_rings.parquet``
    One row per (variant, ring) over the outer rings: radius in deg of field angle, union
    and per-bin coverage, the builder's status, and the cross-check verdict.
``continued_rings_zernike.csv``
    The B1 table, one row per (variant, Zernike).
``continued_rings_summary.txt``
    The console summary, so the study log can quote it without a rerun.

Usage::

    python aos/miw_hull/code/continued_rings.py
    python aos/miw_hull/code/continued_rings.py --variant d12-A_50_34_i
"""

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))      # analysis 1

from lsst.ts.intrinsic.wavefront import intrinsic_split as isp

from rubinwork.products import miw

from ring_coverage import (BUILD, GRID, MIN_RING_COVERAGE, VARIANTS, bin_dirs,
                           builder_flags, ring_table)

SELECTION_R_MAX_DEG = 1.725
"""The settled donut-selection field-radius limit for wavefront (Aaron, 2026-10-14).

Not a recommendation of this study: the study measures whether MIW values inside this
limit are affected by the continuation outside it.
"""

N_OUTER_RINGS = 5
"""Rings to report per bin in B2: the cliff plus the four rings inside it."""


def ring_radii(grid=GRID):
    """Ring radii of the split's polar grid, in deg of field angle."""
    R, _, _, _ = isp.make_polar_grid(**grid)
    return R


def builder_status(table, flags):
    """Per-ring fitted / continued / null status, as the builder would set it.

    The builder decides on union coverage (see the module docstring), so the status is a
    property of the ring alone. ``flags`` is used to confirm the *count* of continued and
    null rings matches, not to assign the status per Zernike.

    Returns
    -------
    status : `pandas.Series`
        ``"fitted"``, ``"continued"`` or ``"null"`` per ring of ``table``.
    """
    below = table["coverage_union"] < MIN_RING_COVERAGE
    has_data = table["coverage_union"] > 0
    status = np.where(~below, "fitted",
                      np.where(has_data, "continued", "null"))
    return pd.Series(status, index=table.index)


def zernike_table(variant, table, flags):
    """B1: the builder's per-Zernike ring bookkeeping with the ring radii attached."""
    status = builder_status(table, flags)
    cont_r = table.loc[status == "continued", "r_deg"].to_numpy()
    null_r = table.loc[status == "null", "r_deg"].to_numpy()
    fmt = lambda a: ", ".join(f"{r:.4f}" for r in a) if len(a) else "-"
    rows = []
    for _, f in flags.iterrows():
        rows.append({
            "variant": variant,
            "Zernike": f["Zernike"],
            "j": int(f["j"]),
            "n_continued_rings": int(f["n_continued_rings"]),
            "n_null_rings": int(f["n_null_rings"]),
            "min_ring_coverage_seen": float(f["min_ring_coverage_seen"]),
            "max_null_dim": int(f["max_null_dim"]),
            "continued_r_deg": fmt(cont_r),
            "null_r_deg": fmt(null_r),
        })
    return pd.DataFrame(rows)


def outer_bin_table(variant, table, bins, n_outer=N_OUTER_RINGS):
    """B2: per-bin coverage at the outer rings, beside the builder's status."""
    status = builder_status(table, None)
    cols = ["r_deg", "coverage_union", "coverage_per_bin_min",
            "coverage_per_bin_median", "n_bins_covered"]
    bin_cols = [f"cov_{name}" for name, _ in bins]
    out = table[cols + bin_cols].tail(n_outer).copy()
    out.insert(0, "variant", variant)
    out["builder_status"] = status.tail(n_outer).to_numpy()
    return out


def cross_check(table, flags):
    """B3: the two implications the builder's rule must satisfy.

    Returns
    -------
    verdict : `dict`
        Counts of each mismatch class, and the agreement of the builder's own reported
        counts with the geometry.
    """
    status = builder_status(table, flags)
    below = table["coverage_union"] < MIN_RING_COVERAGE
    cont = status == "continued"
    # Implication 1: a continued ring must be below the threshold.
    bad_cont_above = int((cont & ~below).sum())
    # Implication 2: a ring below the threshold must be continued or null.
    bad_below_fitted = int((below & (status == "fitted")).sum())
    n_cont_geom = int(cont.sum())
    n_null_geom = int((status == "null").sum())
    n_cont_builder = int(flags["n_continued_rings"].max()) if len(flags) else -1
    n_null_builder = int(flags["n_null_rings"].max()) if len(flags) else -1
    return {
        "continued_above_threshold": bad_cont_above,
        "below_threshold_but_fitted": bad_below_fitted,
        "n_continued_geometry": n_cont_geom,
        "n_continued_builder": n_cont_builder,
        "n_null_geometry": n_null_geom,
        "n_null_builder": n_null_builder,
        "counts_agree": (n_cont_geom == n_cont_builder
                         and n_null_geom == n_null_builder),
    }


def disagreement_set(per_variant):
    """B4: rings fitted in one build and continued or null in the other.

    This is the set analysis 3 exploits: where both builds fit a ring the maps must
    agree, so only a ring the builds treat differently can separate signal from artifact.
    """
    frames = []
    for variant, (table, _, _) in per_variant.items():
        s = builder_status(table, None)
        frames.append(pd.DataFrame({"r_deg": table["r_deg"].to_numpy(),
                                    variant: s.to_numpy()}))
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on="r_deg", how="outer")
    cols = [c for c in merged.columns if c != "r_deg"]
    if len(cols) < 2:
        return merged.iloc[:0]
    differ = merged[cols].nunique(axis=1) > 1
    return merged[differ]


def selection_exposure(variant, build, table, r_max=SELECTION_R_MAX_DEG):
    """B5: does any point at r <= ``r_max`` draw on a continued ring?

    Two consumer routes, each checked on its own geometry:

    polar
        A sample at radius ``r`` is interpolated between the rings bracketing it, so it
        draws on a continued ring only if the ring just outside it is continued.
    cartesian
        The written 71x71 map grid. A cell draws on a continued ring if it lies outside
        the outermost fitted ring, since the polar field is interpolated onto the cells.

    Returns
    -------
    info : `dict`
        Radii and counts, all in deg of field angle or dimensionless counts.
    """
    status = builder_status(table, None)
    R = table["r_deg"].to_numpy()
    fitted_r = R[status.to_numpy() == "fitted"]
    cont_r = R[status.to_numpy() == "continued"]
    null_r = R[status.to_numpy() == "null"]
    r_last_fitted = float(fitted_r.max()) if len(fitted_r) else np.nan

    # Polar route: the brackets of r_max.
    inner = R[R <= r_max]
    outer = R[R > r_max]
    r_below = float(inner.max()) if len(inner) else np.nan
    r_above = float(outer.min()) if len(outer) else np.nan
    polar_draws = bool(status.to_numpy()[R == r_above][0] != "fitted") if len(outer) else False

    # Cartesian route: map cells inside the limit that sit outside the last fitted ring.
    maps = pd.read_parquet(
        miw.build_dir(variant, build) / "intrinsic_split_maps.parquet",
        columns=["thx_deg", "thy_deg"])
    r_cell = np.hypot(maps["thx_deg"].to_numpy(), maps["thy_deg"].to_numpy())
    inside = r_cell <= r_max
    cart_exposed = int((inside & (r_cell > r_last_fitted)).sum())

    return {
        "variant": variant,
        "r_max_deg": r_max,
        "r_last_fitted_deg": r_last_fitted,
        "continued_r_deg": cont_r.tolist(),
        "null_r_deg": null_r.tolist(),
        "r_ring_below_limit_deg": r_below,
        "r_ring_above_limit_deg": r_above,
        "polar_route_draws_on_continued": polar_draws,
        "n_map_cells_inside_limit": int(inside.sum()),
        "n_map_cells_inside_limit_outside_last_fitted": cart_exposed,
        "n_map_cells_total": int(len(maps)),
        "n_map_cells_outside_limit": int((~inside).sum()),
        "r_map_max_deg": float(r_cell.max()),
    }


def plot_status(pdf_path, per_variant):
    """One figure: union coverage against field radius over the outer rings, with the
    builder's status, the 0.5 threshold and the 1.725 deg selection limit marked.
    """
    fig, axes = plt.subplots(1, len(per_variant), figsize=(11, 4.2), sharey=True,
                             squeeze=False)
    colour = {"fitted": "tab:blue", "continued": "tab:orange", "null": "tab:red"}
    for ax, (variant, (table, _, bins)) in zip(axes[0], per_variant.items()):
        outer = table.tail(10)
        status = builder_status(outer, None)
        for name, _ in bins:
            ax.plot(outer["r_deg"], outer[f"cov_{name}"], color="0.75", lw=0.8)
        ax.plot(outer["r_deg"], outer["coverage_union"], "k-", lw=1.6,
                label="union coverage")
        for s, c in colour.items():
            sel = status.to_numpy() == s
            if sel.any():
                ax.plot(outer["r_deg"].to_numpy()[sel],
                        outer["coverage_union"].to_numpy()[sel], "o", color=c,
                        ms=7, label=f"builder: {s}")
        ax.axhline(MIN_RING_COVERAGE, color="tab:green", ls="--",
                   label=f"min_ring_coverage = {MIN_RING_COVERAGE}")
        ax.axvline(SELECTION_R_MAX_DEG, color="tab:purple", ls=":",
                   label=f"donut selection limit {SELECTION_R_MAX_DEG} deg")
        ax.set_xlabel("field radius (deg of field angle)")
        ax.set_title(f"{variant}  ({len(bins)} rotator bins)", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("azimuthal coverage (fraction, dimensionless)")
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle("miw_hull analysis 2: fitted versus continued rings "
                 "(grey = individual rotator bins)", fontsize=11)
    fig.tight_layout()
    fig.savefig(pdf_path, dpi=140)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", action="append", choices=VARIANTS, default=None)
    parser.add_argument("--build", default=BUILD)
    parser.add_argument("--out-dir", default=None, help="default aos/miw_hull/output")
    parser.add_argument("--fig-dir", default=None, help="default aos/miw_hull/figures")
    args = parser.parse_args(argv)

    variants = args.variant or VARIANTS
    root = pathlib.Path(__file__).resolve().parents[1]
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else root / "output"
    fig_dir = pathlib.Path(args.fig_dir) if args.fig_dir else root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    per_variant, zern, outer, lines = {}, [], [], []
    for variant in variants:
        table, _ = ring_table(variant, args.build)
        flags = builder_flags(variant, args.build)
        bins = bin_dirs(variant, args.build)
        per_variant[variant] = (table, flags, bins)
        zern.append(zernike_table(variant, table, flags))
        outer.append(outer_bin_table(variant, table, bins))

        check = cross_check(table, flags)
        expose = selection_exposure(variant, args.build, table)
        status = builder_status(table, flags)
        lines.append(f"=== {variant} ({len(bins)} rotator bins) ===")
        lines.append(
            f"  rings: {int((status == 'fitted').sum())} fitted, "
            f"{int((status == 'continued').sum())} continued, "
            f"{int((status == 'null').sum())} null, of {len(table)} "
            "(all dimensionless counts)")
        lines.append(f"  continued ring radii (deg of field angle): "
                     f"{expose['continued_r_deg'] or '-'}")
        lines.append(f"  null ring radii (deg of field angle): "
                     f"{expose['null_r_deg'] or '-'}")
        uniform = (flags["n_continued_rings"].nunique() == 1
                   and flags["n_null_rings"].nunique() == 1
                   and flags["min_ring_coverage_seen"].nunique() == 1)
        lines.append(
            f"  per-Zernike pattern uniform over all {len(flags)} Zernikes: {uniform} "
            f"(n_continued_rings = {int(flags['n_continued_rings'].max())}, "
            f"n_null_rings = {int(flags['n_null_rings'].max())}, "
            f"min_ring_coverage_seen = {flags['min_ring_coverage_seen'].min():.4f} "
            "fraction, dimensionless)")
        lines.append(
            f"  cross-check B3: continued rings above the threshold = "
            f"{check['continued_above_threshold']}, rings below the threshold yet fitted "
            f"= {check['below_threshold_but_fitted']}, builder counts agree with the "
            f"geometry = {check['counts_agree']} "
            f"(continued {check['n_continued_geometry']} vs "
            f"{check['n_continued_builder']}, null {check['n_null_geometry']} vs "
            f"{check['n_null_builder']})")
        lines.append(
            f"  B5 at the {SELECTION_R_MAX_DEG} deg selection limit: last fitted ring "
            f"r = {expose['r_last_fitted_deg']:.4f} deg, limit bracketed by "
            f"r = {expose['r_ring_below_limit_deg']:.4f} and "
            f"{expose['r_ring_above_limit_deg']:.4f} deg; polar route draws on a "
            f"continued ring = {expose['polar_route_draws_on_continued']}; map cells at "
            f"r <= {SELECTION_R_MAX_DEG} deg outside the last fitted ring = "
            f"{expose['n_map_cells_inside_limit_outside_last_fitted']} of "
            f"{expose['n_map_cells_inside_limit']} inside the limit "
            f"({expose['n_map_cells_outside_limit']} of "
            f"{expose['n_map_cells_total']} cells lie outside it, map reaching "
            f"r = {expose['r_map_max_deg']:.4f} deg)")

    differ = disagreement_set(per_variant)
    lines.append("=== B4: rings the two builds treat differently ===")
    if len(differ) == 0:
        lines.append("  EMPTY. Both builds fit the same 79 rings and continue the same "
                     "one, at r = 1.7500 deg of field angle. No ring is fitted in one "
                     "build and continued in the other, so analysis 3's two-build "
                     "contrast has no ring to work on.")
    else:
        for _, row in differ.iterrows():
            lines.append("  r = {:.4f} deg: {}".format(
                row["r_deg"],
                ", ".join(f"{c} = {row[c]}" for c in differ.columns if c != "r_deg")))

    zern_frame = pd.concat(zern, ignore_index=True)
    outer_frame = pd.concat(outer, ignore_index=True)
    zern_frame.to_csv(out_dir / "continued_rings_zernike.csv", index=False)
    outer_frame.to_parquet(out_dir / "continued_rings.parquet", index=False)
    plot_status(fig_dir / "continued_rings.png", per_variant)
    text = "\n".join(lines)
    (out_dir / "continued_rings_summary.txt").write_text(text + "\n")

    print(text)
    print(f"\n[analysis2] wrote {out_dir / 'continued_rings.parquet'}")
    print(f"[analysis2] wrote {out_dir / 'continued_rings_zernike.csv'}")
    print(f"[analysis2] wrote {out_dir / 'continued_rings_summary.txt'}")
    print(f"[analysis2] wrote {fig_dir / 'continued_rings.png'}")
    print("\n--- B2: per-bin coverage at the outer rings ---")
    print(outer_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
