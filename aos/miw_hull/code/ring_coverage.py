#!/usr/bin/env python3
"""Analysis 1 of `miw_hull`: where does the MIW stop being constrained by data?

The measured intrinsic wavefront (MIW) is decomposed on polar rings. Each ring's
telescope/camera (OCS/CCS) split is a least-squares fit over the azimuth nodes that have
real donut support, so a ring whose azimuthal coverage is thin is poorly constrained and
one below ``min_ring_coverage`` is not fit at all -- since
``ts_intrinsic_wavefront`` commit ``3f66904`` it is **continued radially** from the
nearest fitted ring instead of being nulled.

This script measures the coverage that drives that decision, per rotator bin, on the
**same polar grid and with the same validity rule the builder uses**: it calls the
package's own `make_polar_grid`, `sample_maps_polar` and `ring_coverage` rather than
reimplementing the geometry, so the fractions here are the quantity
``min_ring_coverage`` is compared against and not a lookalike.

Two coverage measures are reported per ring, and they answer different questions:

``coverage_union``
    Fraction of azimuth nodes sampled by **at least one** rotator bin -- exactly
    `ring_coverage`, and what the split's per-ring design sees once every bin is pooled.
``coverage_per_bin``
    The same fraction for a **single** bin. The O/C separation needs the ring sampled at
    several distinct rotator angles, so a ring can be union-covered yet still have a
    rank-deficient design; the per-bin spread is what shows that.

Outputs, under ``aos/miw_hull/output/`` (gitignored symlink to
``studies/aos/miw_hull``):

``ring_coverage.parquet``
    One row per (variant, ring): radius in deg of field angle, the two coverage
    fractions, donut counts, and the builder's own continued/null flags joined on.
``ring_coverage.pdf``
    Coverage against field radius per bin, the union curve, the ``min_ring_coverage``
    threshold, and the robust (Huber) outward trend.

Usage::

    python aos/miw_hull/code/ring_coverage.py
    python aos/miw_hull/code/ring_coverage.py --variant d12-A_50_34_i
"""

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages


from lsst.ts.intrinsic.wavefront import intrinsic_split as isp

from rubinwork.common.utils import nmad
from rubinwork.products import miw

VARIANTS = ["d12-A_50_34_i", "d12-A_50_34_i_5rot"]
"""The two ``danish_1_2`` MIW variants (Aaron, 2026-10-14: these only)."""

BUILD = "20261014"

MIN_RING_COVERAGE = 0.5
"""``DEFAULT_SPLIT['min_ring_coverage']`` in ``run_intrinsic_split.py``.

Not in ``aos/mi_config.yaml``'s ``defaults.split``, so the runner's default applies.
A ring below this fraction is continued radially rather than fit.
"""

GRID = dict(r_min=0.06, r_max=1.75, n_r=80, n_az=180)
"""The split's polar grid, from ``aos/mi_config.yaml`` ``defaults.split``."""

HOLE_DIST_DEG = 0.06
"""``defaults.split.hole_dist``: a node farther than this from a real grid point is
invalid, so dead-detector holes do not count as coverage.
"""


def bin_dirs(variant, build):
    """Per-rotator-bin grid directories this variant's split reads.

    A ``build_from`` variant runs no ``build_intrinsic``, so its grids live in the
    parent's build directory; it also reads only a subset of the parent's bins.

    Returns
    -------
    bins : `list` of (`str`, `pathlib.Path`)
        Bin name and its directory, in the order the split reads them.
    """
    source = miw.build_source(variant)
    base = miw.build_dir(source, build)
    out = []
    for lo, hi in miw.split_rotator_bins(variant):
        name = f"rot_{lo:g}_{hi:g}"
        path = base / "build" / name / "intrinsic_grid.parquet"
        if path.exists():
            out.append((name, path))
        else:
            print(f"[coverage] {variant}: no grid for {name}, skipped")
    return out


def load_bin_maps(bins, column="z4_optical_OCS"):
    """Scattered field maps per rotator bin, in `sample_maps_polar`'s input form.

    Coverage is a property of **where donuts are**, not of which Zernike is being fit, so
    any one finite per-cell column defines the same support. ``z4_optical_OCS`` is used
    because it is finite wherever the cell has donuts.

    Returns
    -------
    maps : `list` of (`numpy.ndarray`, `numpy.ndarray`, `numpy.ndarray`)
        Per bin: ``thx_deg``, ``thy_deg``, value, restricted to finite values.
    counts : `pandas.DataFrame`
        Per bin: cells and total donuts, both dimensionless counts.
    """
    maps, rows = [], []
    for name, path in bins:
        frame = pd.read_parquet(
            path, columns=["thx_deg", "thy_deg", "n_donuts", column])
        good = np.isfinite(frame[column].to_numpy())
        sub = frame[good]
        maps.append((sub["thx_deg"].to_numpy(), sub["thy_deg"].to_numpy(),
                     sub[column].to_numpy()))
        rows.append({"rot_bin": name, "n_cells": int(len(sub)),
                     "n_donuts_total": int(frame["n_donuts"].sum())})
    return maps, pd.DataFrame(rows)


def ring_table(variant, build):
    """Per-ring coverage for one variant, on the builder's own polar grid.

    Returns
    -------
    table : `pandas.DataFrame`
        One row per ring: ``r_deg`` (deg of field angle), ``coverage_union`` and the
        per-bin coverage columns (fraction of azimuth nodes, dimensionless),
        ``coverage_per_bin_min`` / ``_median``, ``n_bins_covered``, and
        ``below_threshold``.
    counts : `pandas.DataFrame`
        Per-bin cell and donut counts.
    """
    bins = bin_dirs(variant, build)
    if not bins:
        raise RuntimeError(f"{variant}: no per-bin grids found")
    maps, counts = load_bin_maps(bins)
    R, A, X, Y = isp.make_polar_grid(**GRID)
    _, valid, _ = isp.sample_maps_polar(maps, X, Y, hole_dist=HOLE_DIST_DEG)

    union = isp.ring_coverage(valid)                 # (n_r,), pooled over bins
    per_bin = valid.mean(axis=2)                     # (n_set, n_r), one bin each
    table = pd.DataFrame({"r_deg": R, "coverage_union": union})
    for i, (name, _) in enumerate(bins):
        table[f"cov_{name}"] = per_bin[i]
    table["coverage_per_bin_min"] = per_bin.min(axis=0)
    table["coverage_per_bin_median"] = np.median(per_bin, axis=0)
    # How many bins sample the ring at all: the O/C split needs several distinct rotator
    # angles, so this is the rank-limiting count, not the union fraction.
    table["n_bins_covered"] = (per_bin > 0).sum(axis=0)
    table["below_threshold"] = table["coverage_union"] < MIN_RING_COVERAGE
    table.insert(0, "variant", variant)
    return table, counts


def builder_flags(variant, build):
    """The builder's own per-Zernike ring bookkeeping, from ``intrinsic_split_rms``.

    These four columns exist only in builds after ``3f66904``; a pre-fix build returns
    an empty frame.
    """
    path = miw.build_dir(variant, build) / "intrinsic_split_rms.parquet"
    frame = pd.read_parquet(path)
    cols = [c for c in ("n_null_rings", "max_null_dim", "n_continued_rings",
                        "min_ring_coverage_seen") if c in frame.columns]
    if not cols:
        return pd.DataFrame()
    return frame[["Zernike", "j"] + cols]


def huber_trend(x, y):
    """Robust outward slope of coverage against field radius.

    Huber RLM per the repository's fitting rule (Aaron's standing default). Returns the
    slope in fraction of azimuth per deg of field angle, its robust residual scatter,
    and the number of rings used.
    """
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 3:
        return np.nan, np.nan, int(good.sum())
    design = sm.add_constant(x[good])
    fit = sm.RLM(y[good], design, M=sm.robust.norms.HuberT()).fit()
    resid = y[good] - fit.fittedvalues
    return float(fit.params[1]), float(nmad(resid)), int(good.sum())


def plot_coverage(pdf, table, counts, variant, slope, scatter, n_used):
    """Coverage against field radius: per bin, pooled, and the threshold."""
    bin_cols = [c for c in table.columns if c.startswith("cov_")]
    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    for col in bin_cols:
        ax.plot(table["r_deg"], table[col], lw=0.8, alpha=0.55,
                label=col.replace("cov_", ""))
    ax.plot(table["r_deg"], table["coverage_union"], lw=2.4, color="k",
            label="union over bins")
    ax.axhline(MIN_RING_COVERAGE, color="tab:red", ls="--",
               label=f"min_ring_coverage = {MIN_RING_COVERAGE}")
    ax.set_xlabel("ring field radius (deg of field angle)")
    ax.set_ylabel("azimuth nodes sampled (fraction, dimensionless)")
    ax.set_title(
        f"{variant}: per-ring azimuthal coverage\n"
        f"Huber slope {slope:+.4f} fraction/deg, nMAD {scatter:.4f}, "
        f"n = {n_used} rings", fontsize=10)
    ax.legend(fontsize=6, ncol=2, loc="lower left")
    ax.set_ylim(-0.03, 1.05)
    pdf.savefig(fig)
    plt.close(fig)

    # The rank-limiting view: how many distinct rotator bins sample each ring.
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True,
                             constrained_layout=True)
    axes[0].plot(table["r_deg"], table["n_bins_covered"], lw=1.6, color="tab:purple")
    axes[0].set_ylabel("rotator bins sampling the ring\n(count, dimensionless)")
    axes[0].set_title(f"{variant}: bins per ring, and the per-bin coverage floor",
                      fontsize=10)
    axes[1].fill_between(table["r_deg"], table["coverage_per_bin_min"],
                         table["coverage_per_bin_median"], alpha=0.35,
                         color="tab:blue", label="per-bin min to median")
    axes[1].plot(table["r_deg"], table["coverage_per_bin_median"], lw=1.4,
                 color="tab:blue")
    axes[1].axhline(MIN_RING_COVERAGE, color="tab:red", ls="--",
                    label=f"min_ring_coverage = {MIN_RING_COVERAGE}")
    axes[1].set_xlabel("ring field radius (deg of field angle)")
    axes[1].set_ylabel("azimuth sampled, single bin\n(fraction, dimensionless)")
    axes[1].legend(fontsize=7)
    pdf.savefig(fig)
    plt.close(fig)


def summarize(table, counts, flags, variant):
    """Lines for the console and the study log: the radii that matter, with units."""
    out = [f"=== {variant} ==="]
    below = table[table["below_threshold"]]
    if len(below):
        first = below["r_deg"].min()
        out.append(
            f"  first ring below min_ring_coverage={MIN_RING_COVERAGE}: "
            f"r = {first:.4f} deg of field angle "
            f"({len(below)} of {len(table)} rings below)")
    else:
        out.append(f"  no ring below min_ring_coverage={MIN_RING_COVERAGE} "
                   f"(0 of {len(table)} rings)")
    full = table[table["coverage_union"] >= 0.999]
    if len(full):
        out.append(f"  last fully covered ring (union >= 0.999): "
                   f"r = {full['r_deg'].max():.4f} deg of field angle")
    out.append(f"  union coverage at the outermost ring "
               f"(r = {table['r_deg'].max():.4f} deg): "
               f"{table['coverage_union'].iloc[-1]:.4f} (fraction, dimensionless)")
    out.append(f"  per-bin coverage floor at that ring: "
               f"{table['coverage_per_bin_min'].iloc[-1]:.4f} (fraction)")
    out.append(f"  bins sampling the outermost ring: "
               f"{int(table['n_bins_covered'].iloc[-1])} of {len(counts)}")
    if len(flags):
        seen = flags["min_ring_coverage_seen"].min()
        cont = flags["n_continued_rings"].max()
        out.append(f"  builder reports: up to {int(cont)} continued ring(s) per Zernike, "
                   f"min_ring_coverage_seen = {seen:.4f} (fraction, dimensionless)")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", action="append", choices=VARIANTS, default=None)
    parser.add_argument("--build", default=BUILD)
    parser.add_argument("--out-dir", default=None,
                        help="default aos/miw_hull/output")
    args = parser.parse_args(argv)

    variants = args.variant or VARIANTS
    out_dir = (pathlib.Path(args.out_dir) if args.out_dir
               else pathlib.Path(__file__).resolve().parents[1] / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    tables, lines = [], []
    pdf_path = out_dir / "ring_coverage.pdf"
    with PdfPages(pdf_path) as pdf:
        for variant in variants:
            table, counts = ring_table(variant, args.build)
            flags = builder_flags(variant, args.build)
            slope, scatter, n_used = huber_trend(
                table["r_deg"].to_numpy(), table["coverage_union"].to_numpy())
            plot_coverage(pdf, table, counts, variant, slope, scatter, n_used)
            tables.append(table)
            lines += summarize(table, counts, flags, variant)
            lines.append(f"  bins: {len(counts)}; cells and donuts per bin:")
            for _, row in counts.iterrows():
                lines.append(f"    {row.rot_bin}: {row.n_cells} cells, "
                             f"{row.n_donuts_total} donuts (both dimensionless)")

    combined = pd.concat(tables, ignore_index=True)
    combined.to_parquet(out_dir / "ring_coverage.parquet", index=False)
    print("\n".join(lines))
    print(f"\n[coverage] wrote {out_dir / 'ring_coverage.parquet'}")
    print(f"[coverage] wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
