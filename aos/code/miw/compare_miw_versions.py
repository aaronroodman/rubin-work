"""Compare two Measured Intrinsic Wavefront (MIW) builds term by term, as a PDF.

One page per pupil Zernike Noll term, three field maps across the page: the first MIW,
the second MIW, and their difference (second minus first). The two MIW panels share one
colour scale so their amplitudes can be read against each other directly; the difference
panel gets its own scale, set from the 2nd to 98th percentile of the difference.

Both builds must be sampled on the same field grid — `intrinsic_split` writes a fixed
grid, so two builds of the same `rotator_select` share it — and the script asserts that
rather than interpolating.

Usage
-----
    python code/miw/compare_miw_versions.py \
        --miw-a output/miw/danish_1_2_A_50_34_i_5rot/intrinsic_split_maps.parquet \
        --miw-b output/miw/danish_1_3_test_A_50_34_i_5rot/intrinsic_split_maps.parquet \
        --label-a "Danish 1.2 (paired)" --label-b "Danish 1.3 blitz (unpaired)" \
        --out-dir output/miw/danish_1_2_vs_1_3

Key arguments: `--coord` selects the OCS (telescope-fixed, default) or CCS
(camera-fixed) component; `--js` overrides the Noll terms plotted.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # aos/code

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm

from common.utils import nmad
from miw_io import JS_DEFAULT

# Field-map rendering, matching code/static_optics/camera_gravity_maps.py so MIW maps
# look the same wherever they are drawn.
CMAP = "RdBu_r"
MARKER_SIZE = 6

# Percentile envelope for the difference colour scale, per the comparison request:
# the range covers the central 96 % of the difference values.
DIFF_PCT_LO = 2.0
DIFF_PCT_HI = 98.0

# Percentile for the shared MIW colour scale, on |value| pooled over both builds.
MIW_PCT = 98.0

# Field radius in deg inside which the colour scales are computed. The outermost ring of
# the MIW grid carries the known convex-hull edge defect, present in both builds: across
# the 1.70-1.75 deg step the Z5 OCS difference RMS jumps from 0.0499 to 0.1109 um of
# wavefront and reaches 0.6441 um, which is an artefact of the hull, not a wavefront
# difference. Those 240 of 3969 points would otherwise set the colour range and flatten
# the real structure. Every point is still PLOTTED -- only the scale is computed inside
# this radius, and points beyond it saturate.
SCALE_R_MAX_DEG = 1.70


def _symmetric_scale(values, pct_lo, pct_hi):
    """Symmetric colour limit spanning a percentile interval of `values`.

    A diverging colormap centred on zero needs a symmetric range, so the limit is the
    larger absolute end of the interval. That covers the requested percentile interval
    while keeping zero at the colormap's white point.

    Parameters
    ----------
    values : `array_like`
        Values in µm of wavefront; NaNs ignored.
    pct_lo, pct_hi : `float`
        Percentile bounds, 0 to 100.

    Returns
    -------
    vmax : `float`
        Colour limit in µm of wavefront, used as (-vmax, +vmax). Falls back to a small
        positive value when the input is all-NaN or identically zero, so `TwoSlopeNorm`
        cannot be handed a degenerate range.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1e-3
    lo, hi = np.percentile(v, [pct_lo, pct_hi])
    vmax = max(abs(lo), abs(hi))
    return float(vmax) if vmax > 0 else 1e-3


def _panel(ax, thx, thy, v, title, vmax):
    """Draw one field map; returns the mappable for the colour bar."""
    sc = ax.scatter(thx, thy, c=v, s=MARKER_SIZE, cmap=CMAP,
                    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    return sc


def _stats_line(v):
    """Robust and standard scatter of a field map, in µm of wavefront."""
    finite = np.isfinite(v)
    n = int(finite.sum())
    if n == 0:
        return "no finite points"
    rms = float(np.sqrt(np.nanmean(np.asarray(v, float) ** 2)))
    return (f"n={n}, RMS={rms:.4f}, nMAD={nmad(np.asarray(v, float)[finite]):.4f} "
            f"$\\mu$m wf")


def compare(miw_a, miw_b, label_a, label_b, out_pdf, coord="OCS", js=JS_DEFAULT,
            scale_r_max_deg=SCALE_R_MAX_DEG):
    """Write the per-term comparison PDF and return a per-term summary table.

    Parameters
    ----------
    miw_a, miw_b : `pandas.DataFrame`
        `intrinsic_split_maps` tables, on the same field grid.
    label_a, label_b : `str`
        Panel titles naming each build.
    out_pdf : `str` or `pathlib.Path`
        Destination PDF.
    coord : {'OCS', 'CCS'}, optional
        Which frame's columns to compare. OCS (default) is telescope-fixed.
    js : `iterable` [`int`], optional
        Pupil Zernike Noll indices, one page each.
    scale_r_max_deg : `float`, optional
        Field radius in deg inside which the colour scales are computed, excluding the
        convex-hull edge ring. Points beyond it are still plotted, and saturate. Pass a
        value above the grid's outer radius to include every point in the scale.

    Returns
    -------
    summary : `pandas.DataFrame`
        One row per Noll term: the RMS of each build, of the difference, and the number
        of field points where both builds are finite. All wavefront values are in µm of
        wavefront.
    """
    thx = miw_a["thx_deg"].to_numpy(float)
    thy = miw_a["thy_deg"].to_numpy(float)
    r_field = np.hypot(thx, thy)
    inner = r_field <= scale_r_max_deg
    n_outer = int((~inner).sum())

    rows = []
    with PdfPages(str(out_pdf)) as pdf:
        for j in js:
            col = f"Z{j}_{coord}"
            if col not in miw_a.columns or col not in miw_b.columns:
                continue
            va = miw_a[col].to_numpy(float)
            vb = miw_b[col].to_numpy(float)
            diff = vb - va

            # One scale for the two MIW panels so their amplitudes compare directly.
            # Both scales come from the interior only -- see SCALE_R_MAX_DEG.
            vmax_miw = _symmetric_scale(np.concatenate([va[inner], vb[inner]]),
                                        100.0 - MIW_PCT, MIW_PCT)
            vmax_diff = _symmetric_scale(diff[inner], DIFF_PCT_LO, DIFF_PCT_HI)

            # Explicit axes rectangles: three map panels and two colour bars. Letting
            # fig.colorbar(ax=[...]) steal space from a list of axes puts the shared bar
            # over the middle panel once the title block is made room for.
            fig = plt.figure(figsize=(14.0, 4.8))
            w, h, y0 = 0.235, 0.62, 0.06
            ax_a = fig.add_axes([0.035, y0, w, h])
            ax_b = fig.add_axes([0.285, y0, w, h])
            cax1 = fig.add_axes([0.532, y0 + 0.06, 0.011, h - 0.12])
            ax_d = fig.add_axes([0.660, y0, w, h])
            cax2 = fig.add_axes([0.907, y0 + 0.06, 0.011, h - 0.12])

            sc_a = _panel(ax_a, thx, thy, va,
                          f"{label_a}\n{_stats_line(va)}", vmax_miw)
            _panel(ax_b, thx, thy, vb,
                   f"{label_b}\n{_stats_line(vb)}", vmax_miw)
            sc_d = _panel(ax_d, thx, thy, diff,
                          f"difference: B - A\n{_stats_line(diff)}",
                          vmax_diff)

            # One bar for the two MIW panels, which share a scale; the difference keeps
            # its own.
            cb1 = fig.colorbar(sc_a, cax=cax1)
            cb1.set_label(f"Z{j} {coord} ($\\mu$m of wavefront)", fontsize=8)
            cb1.ax.tick_params(labelsize=7)
            cb2 = fig.colorbar(sc_d, cax=cax2)
            cb2.set_label(f"difference ($\\mu$m of wavefront), "
                          f"{DIFF_PCT_LO:g}-{DIFF_PCT_HI:g}%", fontsize=8)
            cb2.ax.tick_params(labelsize=7)

            both = np.isfinite(va) & np.isfinite(vb)
            fig.suptitle(
                f"MIW comparison — Z{j}, {coord} frame   "
                f"A = {label_a},  B = {label_b}\n"
                f"field angle in deg; {int(both.sum())} field points in common; "
                f"colour scales set inside {scale_r_max_deg:g} deg "
                f"({n_outer} outer points plotted but saturating)",
                fontsize=10, y=0.985, va="top")
            pdf.savefig(fig)
            plt.close(fig)

            fin_in = inner & np.isfinite(diff)
            rows.append(dict(
                noll_j=j,
                rms_a_um_wf=float(np.sqrt(np.nanmean(va ** 2))),
                rms_b_um_wf=float(np.sqrt(np.nanmean(vb ** 2))),
                rms_diff_um_wf=float(np.sqrt(np.nanmean(diff ** 2))),
                nmad_diff_um_wf=float(nmad(diff[np.isfinite(diff)])),
                # Interior only, excluding the convex-hull edge ring.
                rms_diff_inner_um_wf=float(np.sqrt(np.nanmean(diff[fin_in] ** 2))),
                n_common_points=int(both.sum()),
                n_inner_points=int(fin_in.sum()),
                vmax_miw_um_wf=vmax_miw,
                vmax_diff_um_wf=vmax_diff,
            ))

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--miw-a", required=True,
                   help="intrinsic_split_maps parquet of the reference build")
    p.add_argument("--miw-b", required=True,
                   help="intrinsic_split_maps parquet of the build being compared")
    p.add_argument("--label-a", default="MIW A")
    p.add_argument("--label-b", default="MIW B")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-name", default="miw_compare",
                   help="basename for the PDF and the summary parquet")
    p.add_argument("--coord", default="OCS", choices=["OCS", "CCS"])
    p.add_argument("--scale-r-max-deg", type=float, default=SCALE_R_MAX_DEG,
                   help="field radius in deg inside which colour scales are computed; "
                        "excludes the convex-hull edge ring (default %(default)g). "
                        "Pass 99 to scale on every point.")
    p.add_argument("--js", default=None,
                   help="comma-separated Noll indices; default is the FAM 21-term set")
    args = p.parse_args()

    js = (tuple(int(s) for s in args.js.split(",")) if args.js else JS_DEFAULT)

    a = pd.read_parquet(args.miw_a)
    b = pd.read_parquet(args.miw_b)

    if len(a) != len(b):
        raise SystemExit(f"field grids differ in length: {len(a)} vs {len(b)} rows — "
                         "the two builds are not on a common grid")
    dthx = np.abs(a["thx_deg"].to_numpy(float) - b["thx_deg"].to_numpy(float))
    dthy = np.abs(a["thy_deg"].to_numpy(float) - b["thy_deg"].to_numpy(float))
    if np.nanmax(dthx) > 1e-9 or np.nanmax(dthy) > 1e-9:
        raise SystemExit("field grids are not identical row-for-row "
                         f"(max |dthx| = {np.nanmax(dthx):.3e} deg, "
                         f"max |dthy| = {np.nanmax(dthy):.3e} deg)")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / f"{args.out_name}.pdf"
    out_parquet = out_dir / f"{args.out_name}_summary.parquet"

    summary = compare(a, b, args.label_a, args.label_b, out_pdf,
                      coord=args.coord, js=js,
                      scale_r_max_deg=args.scale_r_max_deg)
    summary.to_parquet(out_parquet, index=False)

    print(f"wrote {out_pdf} ({len(summary)} pages)")
    print(f"wrote {out_parquet}")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
