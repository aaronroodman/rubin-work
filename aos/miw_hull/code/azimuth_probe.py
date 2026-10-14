#!/usr/bin/env python3
"""Analysis 1b of `miw_hull`: walk the azimuth at a fixed field radius and check returns.

Aaron, 2026-10-14: at a radius of exactly **1.725 deg of field angle**, step the azimuth
from 0 to 360 deg in 1 deg steps and check the value returned for the OCS map of **every**
Zernike and the CCS map of **Z4**; plot against azimuth and flag any NaN or
identically-zero return.

**This exercises the real delivery path, not a reimplementation.** Both consumer routes a
caller can take are probed, each through its own `ts_intrinsic_wavefront` function:

``maps`` route
    `intrinsic_split.polar_field_to_points` applied to the polar O/C fields on the
    decomposition's own ``X``/``Y`` grid -- the function that **wrote**
    ``intrinsic_split_maps.parquet``, including the ``nearest_within`` rim fill added by
    commit ``3f66904``. This is what a consumer reading the handoff maps gets.
``decomp`` route
    `intrinsic_split.reconstruct_at`, which combines O and C at a given camera rotator
    angle, reached through `rubinwork.miw_corner.load_decomposition`. This is what
    ``value_added``'s ``--intrinsic miw`` path gets.

The two differ by construction: the maps route is rotator-independent (O and C are stored
separately and the written map columns are the two components), while the decomp route
evaluates ``A(theta) = O + exp(i*n_spin*s*theta) * roll(C, ...)`` at one rotator angle.
Probing both is the point -- a defect in one and not the other localizes itself.

Zero returns are classified, because **some zeros are correct**. A Zernike with
``ocs_only`` true in ``intrinsic_split_rms`` has its camera term forced to 0, so its CCS
map is identically zero by design: in ``d12-A_50_34_i_5rot`` that is 20 of 21 CCS columns,
leaving Z4 as the only meaningful one (which is why the request names Z4 only), while
``d12-A_50_34_i`` splits all 21. A zero is reported as a **defect** only where the
configuration says the term should be fit.

Outputs, under ``aos/miw_hull/output/``:

``azimuth_probe.parquet``
    One row per (variant, route, component, Zernike, azimuth): the value in µm of
    wavefront and its flags.
``azimuth_probe.pdf``
    Value against azimuth per Zernike, OCS all and CCS Z4, both routes, with NaN and
    zero spans marked.
``azimuth_probe_summary.csv``
    Per (variant, route, component, Zernike): finite count, NaN count, exact-zero count,
    min/max/RMS in µm of wavefront.

Usage::

    python aos/miw_hull/code/azimuth_probe.py
    python aos/miw_hull/code/azimuth_probe.py --radius-deg 1.725 --step-deg 1.0
"""

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repo root

from lsst.ts.intrinsic.wavefront import intrinsic_split as isp

from rubinwork import miw_corner
from rubinwork.products import miw

VARIANTS = ["d12-A_50_34_i", "d12-A_50_34_i_5rot"]
BUILD = "20261014"

RADIUS_DEG = 1.725
"""Field radius of the azimuthal cut, deg of field angle (Aaron's request).

Context from analysis 1: this sits between the last well-covered ring
(``r = 1.7286 deg``, union azimuthal coverage 0.9333) and the grid half-width 1.726 deg,
so the cut is **inside** the constrained region and just inside the one under-covered
ring at ``r = 1.7500 deg``.
"""

STEP_DEG = 1.0
"""Azimuth step, deg. At r = 1.725 deg a 1 deg step is 0.0301 deg of arc, against the
map lattice step of 0.0493 deg, so the cut oversamples the lattice by about 1.6x --
structure finer than the lattice is the interpolant's, not new measurement.
"""

RADIAL_STEP_DEG = (1.75 - 0.06) / (80 - 1)
"""The split's radial grid step, deg. Passed as ``nearest_within`` exactly as
``run_intrinsic_split`` does, so the rim fill behaves identically here.
"""


def probe_points(radius_deg, step_deg):
    """Cartesian field coordinates of the azimuthal cut.

    Returns
    -------
    az_deg : `numpy.ndarray`
        Azimuth of each sample, deg, from 0 up to but excluding 360.
    thx, thy : `numpy.ndarray`
        Field coordinates, deg of field angle.
    """
    az = np.arange(0.0, 360.0, step_deg)
    rad = np.deg2rad(az)
    return az, radius_deg * np.cos(rad), radius_deg * np.sin(rad)


def ocs_only_map(variant, build):
    """``{Zernike noll index: ocs_only}`` from ``intrinsic_split_rms``.

    Identifies which camera (CCS) terms are forced to zero by configuration, so a zero
    return there is by design rather than a defect.
    """
    rms = pd.read_parquet(
        miw.build_dir(variant, build) / "intrinsic_split_rms.parquet")
    if "ocs_only" not in rms.columns:
        return {}
    return {int(j): bool(flag)
            for j, flag in zip(rms["j"], rms["ocs_only"])}


def probe_maps_route(dec, thx, thy):
    """Sample O and C per Zernike with the function that wrote the handoff maps.

    Uses `intrinsic_split.polar_field_to_points` on the decomposition's own polar grid,
    with ``nearest_within`` set to the radial step -- the same call
    ``run_intrinsic_split`` makes.

    Returns
    -------
    values : `dict`
        ``(component, noll) -> numpy.ndarray`` of µm of wavefront, component in
        ``{"OCS", "CCS"}``.
    """
    out = {}
    for i, j in enumerate(dec["noll"]):
        for comp, field in (("OCS", dec["O_pol"][i]), ("CCS", dec["C_pol"][i])):
            out[(comp, int(j))] = isp.polar_field_to_points(
                field.real, dec["X"], dec["Y"], thx, thy,
                nearest_within=RADIAL_STEP_DEG)
    return out


def probe_decomp_route(dec, thx, thy, rot_deg):
    """Sample the reconstructed field at one rotator angle, via `reconstruct_at`.

    This is the ``value_added`` / `miw_corner` path: O and C are combined at the camera
    rotator angle first, then interpolated to the points. Reported as the ``TOTAL``
    component, since that is what the caller receives -- a single wavefront, not a
    two-component split.
    """
    out = {}
    theta = np.deg2rad(rot_deg)
    A = dec["A"]
    for i, j in enumerate(dec["noll"]):
        one = dict(O_pol=dec["O_pol"][i], C_pol=dec["C_pol"][i],
                   n_spin=int(dec["n_spin"][i]), s=int(dec["s"][i]))
        field = isp.reconstruct_at(one, theta, A)
        out[("TOTAL", int(j))] = isp.polar_field_to_points(
            field.real, dec["X"], dec["Y"], thx, thy,
            nearest_within=RADIAL_STEP_DEG)
    return out


def flag_rows(variant, route, values, az, ocs_only, radius_deg, rot_deg):
    """Long-form rows with the NaN / zero flags, one per sample."""
    rows = []
    for (comp, j), vals in sorted(values.items()):
        by_design = comp == "CCS" and ocs_only.get(j, False)
        for a, v in zip(az, vals):
            finite = bool(np.isfinite(v))
            rows.append({
                "variant": variant, "route": route, "component": comp, "j": j,
                "r_deg": radius_deg, "rot_deg": rot_deg, "az_deg": float(a),
                "value_um": float(v) if finite else np.nan,
                "is_nan": not finite,
                "is_exact_zero": bool(finite and v == 0.0),
                "zero_by_design": by_design,
            })
    return rows


def summarize(frame):
    """Per (variant, route, component, Zernike) counts and ranges, with units."""
    out = []
    keys = ["variant", "route", "component", "j"]
    for key, grp in frame.groupby(keys, sort=True):
        vals = grp["value_um"].to_numpy()
        good = vals[np.isfinite(vals)]
        out.append({
            "variant": key[0], "route": key[1], "component": key[2], "j": key[3],
            "n_samples": len(grp),
            "n_nan": int(grp["is_nan"].sum()),
            "n_exact_zero": int(grp["is_exact_zero"].sum()),
            "zero_by_design": bool(grp["zero_by_design"].iloc[0]),
            "min_um": float(good.min()) if good.size else np.nan,
            "max_um": float(good.max()) if good.size else np.nan,
            "rms_um": float(np.sqrt((good ** 2).mean())) if good.size else np.nan,
            "ptp_um": float(good.max() - good.min()) if good.size else np.nan,
        })
    return pd.DataFrame(out)


def plot_route(pdf, frame, variant, route, component, title_extra=""):
    """Value against azimuth, one panel per Zernike, NaN and zero spans marked."""
    sel = frame[(frame["variant"] == variant) & (frame["route"] == route)
                & (frame["component"] == component)]
    if not len(sel):
        return
    js = sorted(sel["j"].unique())
    ncol = 3
    nrow = int(np.ceil(len(js) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.3 * nrow),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, j in zip(axes, js):
        one = sel[sel["j"] == j].sort_values("az_deg")
        az = one["az_deg"].to_numpy()
        val = one["value_um"].to_numpy()
        ax.plot(az, val, lw=1.1, color="tab:blue")
        nan_az = az[one["is_nan"].to_numpy()]
        if nan_az.size:
            for a in nan_az:
                ax.axvline(a, color="tab:red", lw=0.6, alpha=0.6)
        zero = one["is_exact_zero"].to_numpy()
        by_design = bool(one["zero_by_design"].iloc[0])
        note = ""
        if zero.all():
            note = " — ALL ZERO (by design)" if by_design else " — ALL ZERO (DEFECT)"
        elif zero.any():
            note = f" — {int(zero.sum())} zero samples"
        if nan_az.size:
            note += f" — {nan_az.size} NaN"
        ax.set_title(f"Z{j}{note}", fontsize=8,
                     color="tab:red" if (nan_az.size or
                                         (zero.any() and not by_design)) else "k")
        ax.tick_params(labelsize=7)
    for ax in axes[len(js):]:
        ax.set_visible(False)
    for ax in axes[max(0, len(js) - ncol):len(js)]:
        ax.set_xlabel("azimuth (deg)", fontsize=8)
    axes[0].set_ylabel("wavefront (µm)", fontsize=8)
    fig.suptitle(f"{variant} — {route} route, {component} — "
                 f"r = {RADIUS_DEG} deg of field angle{title_extra}", fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


def plot_overlay(pdf, frame, variant):
    """The two routes on one axis for Z4, where both are meaningful."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, comp, route in ((axes[0], "OCS", "maps"), (axes[1], "TOTAL", "decomp")):
        one = frame[(frame["variant"] == variant) & (frame["route"] == route)
                    & (frame["component"] == comp) & (frame["j"] == 4)]
        if len(one):
            one = one.sort_values("az_deg")
            ax.plot(one["az_deg"], one["value_um"], lw=1.4)
        ax.set_xlabel("azimuth (deg)")
        ax.set_ylabel("wavefront (µm)")
        ax.set_title(f"Z4 {comp}, {route} route", fontsize=9)
    fig.suptitle(f"{variant} — Z4 at r = {RADIUS_DEG} deg of field angle, both routes",
                 fontsize=10)
    pdf.savefig(fig)
    plt.close(fig)


def radius_scan(dec, step_deg, radii=(1.700, 1.7250, 1.7286, 1.7500, 1.7512, 1.760)):
    """NaN count against probe radius, with and without the rim fill.

    A negative control: a probe that reports "no NaN" is only meaningful if it can
    detect one. It also localizes a **trap for any consumer sampling near the rim**.

    The polar grid has ``n_az`` azimuth nodes, so its azimuthal spacing is
    ``360 / n_az`` deg -- 2.0 deg for the standard ``n_az = 180``. On the **outermost**
    ring there is no triangle farther out to interpolate within, so a probe step finer
    than the node spacing asks for points outside the convex hull at every azimuth that
    falls between two nodes, and those return NaN. That is the probe oversampling the
    lattice, not missing data: the stored polar field has no NaN on that ring.

    Returns
    -------
    table : `pandas.DataFrame`
        Per radius: NaN counts out of the sample count, both with ``nearest_within``
        set to the radial step and with it off, plus the polar field's own NaN count on
        the nearest ring.
    """
    az = np.arange(0.0, 360.0, step_deg)
    rad = np.deg2rad(az)
    field = dec["O_pol"][0].real              # Z4 OCS: finite wherever the ring is fit
    n_az = field.shape[1]
    rows = []
    for r in radii:
        thx, thy = r * np.cos(rad), r * np.sin(rad)
        with_fill = isp.polar_field_to_points(
            field, dec["X"], dec["Y"], thx, thy, nearest_within=RADIAL_STEP_DEG)
        no_fill = isp.polar_field_to_points(
            field, dec["X"], dec["Y"], thx, thy, nearest_within=None)
        ring = int(np.argmin(np.abs(
            np.hypot(dec["X"][:, 0], dec["Y"][:, 0]) - r)))
        rows.append({
            "r_deg": r, "n_samples": len(az),
            "n_nan_with_rim_fill": int(np.sum(~np.isfinite(with_fill))),
            "n_nan_no_rim_fill": int(np.sum(~np.isfinite(no_fill))),
            "nearest_ring_index": ring,
            "n_nan_in_polar_field_on_that_ring": int(
                np.sum(~np.isfinite(field[ring]))),
            "polar_az_step_deg": 360.0 / n_az,
        })
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", action="append", choices=VARIANTS, default=None)
    parser.add_argument("--build", default=BUILD)
    parser.add_argument("--radius-deg", type=float, default=RADIUS_DEG)
    parser.add_argument("--step-deg", type=float, default=STEP_DEG)
    parser.add_argument("--rot-deg", type=float, default=0.0,
                        help="camera rotator angle for the decomp route, deg")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    variants = args.variant or VARIANTS
    out_dir = (pathlib.Path(args.out_dir) if args.out_dir
               else pathlib.Path(__file__).resolve().parents[1] / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    az, thx, thy = probe_points(args.radius_deg, args.step_deg)
    print(f"[probe] r = {args.radius_deg} deg of field angle, {len(az)} azimuth samples "
          f"at {args.step_deg} deg steps")

    rows = []
    scans = []
    for variant in variants:
        dec_path = miw.build_dir(variant, args.build) / "intrinsic_split_decomp.parquet"
        dec = miw_corner.load_decomposition(dec_path)
        flags = ocs_only_map(variant, args.build)
        scan = radius_scan(dec, args.step_deg)
        scan.insert(0, "variant", variant)
        scans.append(scan)
        rows += flag_rows(variant, "maps", probe_maps_route(dec, thx, thy),
                          az, flags, args.radius_deg, np.nan)
        rows += flag_rows(variant, "decomp",
                          probe_decomp_route(dec, thx, thy, args.rot_deg),
                          az, flags, args.radius_deg, args.rot_deg)

    frame = pd.DataFrame(rows)
    frame.to_parquet(out_dir / "azimuth_probe.parquet", index=False)
    summary = summarize(frame)
    summary.to_csv(out_dir / "azimuth_probe_summary.csv", index=False)
    scan_table = pd.concat(scans, ignore_index=True)
    scan_table.to_csv(out_dir / "azimuth_probe_radius_scan.csv", index=False)

    pdf_path = out_dir / "azimuth_probe.pdf"
    with PdfPages(pdf_path) as pdf:
        for variant in variants:
            plot_route(pdf, frame, variant, "maps", "OCS", " (all Zernikes)")
            plot_route(pdf, frame, variant, "maps", "CCS", " (all Zernikes)")
            plot_route(pdf, frame, variant, "decomp", "TOTAL",
                       f" (rotator {args.rot_deg:g} deg)")
            plot_overlay(pdf, frame, variant)

    # ---- the report Aaron asked for: what came back NaN or identically zero ----
    print()
    for variant in variants:
        print(f"=== {variant} ===")
        sub = summary[summary["variant"] == variant]
        nan_any = sub[sub["n_nan"] > 0]
        print(f"  NaN returns: "
              f"{'NONE' if not len(nan_any) else str(len(nan_any)) + ' series affected'}")
        for _, r in nan_any.iterrows():
            print(f"    {r['route']:>6} {r['component']:>5} Z{r['j']}: "
                  f"{r['n_nan']} of {r['n_samples']} samples NaN")
        zero_bad = sub[(sub["n_exact_zero"] == sub["n_samples"])
                       & (~sub["zero_by_design"])]
        zero_ok = sub[(sub["n_exact_zero"] == sub["n_samples"])
                      & (sub["zero_by_design"])]
        print(f"  identically-zero series, BY DESIGN (ocs_only): {len(zero_ok)}"
              + (f" -> {sorted('Z%d' % j for j in zero_ok['j'].unique())}"
                 if len(zero_ok) else ""))
        print(f"  identically-zero series, DEFECT: "
              f"{'NONE' if not len(zero_bad) else len(zero_bad)}")
        for _, r in zero_bad.iterrows():
            print(f"    {r['route']:>6} {r['component']:>5} Z{r['j']}")
        part = sub[(sub["n_exact_zero"] > 0)
                   & (sub["n_exact_zero"] < sub["n_samples"])]
        print(f"  partially-zero series: "
              f"{'NONE' if not len(part) else len(part)}")
        for _, r in part.iterrows():
            print(f"    {r['route']:>6} {r['component']:>5} Z{r['j']}: "
                  f"{r['n_exact_zero']} of {r['n_samples']} samples exactly zero")
        live = sub[(sub["component"] != "CCS") | (~sub["zero_by_design"])]
        live = live[live["n_exact_zero"] < live["n_samples"]]
        if len(live):
            print(f"  live series peak-to-peak over azimuth, µm of wavefront: "
                  f"min {live['ptp_um'].min():.4e}, max {live['ptp_um'].max():.4e}")

    print("\n=== negative control: NaN against probe radius (Z4 OCS, maps route) ===")
    print("A finer probe step than the polar grid's azimuthal node spacing asks for")
    print("points outside the hull on the OUTERMOST ring -- probe geometry, not missing")
    print("data. The last column is the stored field's own NaN count on that ring.")
    print(scan_table.to_string(index=False))

    print(f"\n[probe] wrote {out_dir / 'azimuth_probe.parquet'}")
    print(f"[probe] wrote {out_dir / 'azimuth_probe_radius_scan.csv'}")
    print(f"[probe] wrote {out_dir / 'azimuth_probe_summary.csv'}")
    print(f"[probe] wrote {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
