"""Compare two Measured Intrinsic Wavefront (MIW) builds as image quality and by region.

Companion to `compare_miw_versions.py`, which draws the two builds term by term as field
maps.  This script answers the questions that a *pupil-model* comparison adds on top of
that: what the difference is worth as delivered image quality, and whether it sits where
the changed optic acts.

Three things, for one pair of builds:

1. **Inferred full width at half maximum (FWHM), in arcsec.**  Each field point's pupil
   Zernike vector is converted with ts_wep `convertZernikesToPsfWidth` and
   quadrature-summed over Noll 4 and up, giving the wavefront's own contribution to the
   PSF width at that point.  Reported per build and for the difference wavefront.  Zero
   on this axis means no wavefront, not a perfect PSF — it adds in quadrature on top of
   the delivered image quality.
2. **Where the difference sits in the field**, in radial annuli.  The v1000 pupil model
   adds the M1 outer and inner baffles, whose `ClearCircle` surfaces at radius 4.165 m
   sit 15 mm inside M1's 4.18 m rim, so a baffle signature is expected to grow toward
   the field edge rather than be spread uniformly.
3. **Which pupil Zernike terms carry the difference.**  A pupil-edge effect loads the
   high-order radial terms — spherical Noll 11 and 22 — rather than spreading evenly, so
   the per-term split is the pupil-edge diagnostic available from a field-map product.
   Reported as each term's share of the total difference power.

The outermost field ring carries the convex-hull edge defect, present in both builds and
carried deliberately, so the radial profile reports it as its own annulus rather than
letting it contaminate an inner one.  `--edge-r-deg` sets where that ring starts.

Usage
-----
    python code/miw/compare_pupil_models.py \
        --miw-a output/miw/danish_1_3_test_A_50_34_i_5rot/intrinsic_split_maps.parquet \
        --miw-b output/miw/danish_1_3_v1000_A_50_34_i_5rot/intrinsic_split_maps.parquet \
        --label-a "legacy pupil" \
        --label-b "v1000 pupil" \
        --out-dir output/miw/danish_1_3_legacy_vs_v1000

Key arguments: `--coord` selects the OCS (telescope-fixed, default) or CCS (camera-fixed)
component; `--edge-r-deg` sets the radius at which the convex-hull edge ring begins.

Notes
-----
Needs `lsst.ts.wep.utils.convertZernikesToPsfWidth`, so this is RSP/USDF only.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # aos/code

from common.utils import nmad  # noqa: E402
from miw_io import JS_DEFAULT  # noqa: E402

# Field radius in deg at which the convex-hull edge ring begins.  Inside this the MIW
# grid is interpolated between real visits; outside, the hull is extrapolating and the
# known edge defect lives.  Matches `compare_miw_versions.py`'s scale cut.
EDGE_R_DEG = 1.70

# Inner annulus edges in deg for the radial profile, below EDGE_R_DEG.
ANNULUS_EDGES_DEG = (0.0, 0.6, 1.0, 1.35, 1.55, EDGE_R_DEG)

# Pupil Zernike Noll indices whose difference share is called out as the pupil-edge
# diagnostic: the rotationally symmetric high-order radial terms.  Noll 11 is primary
# spherical and Noll 22 secondary spherical; a mask change at the pupil rim loads these.
SPHERICAL_JS = (11, 22)


def load_maps(path, coord, js):
    """Field map of one MIW build as an (n_pos, n_j) array, with the field positions.

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        An `intrinsic_split_maps.parquet` written by the `intrinsic_split` rule.
    coord : `str`
        ``'OCS'`` (telescope-fixed) or ``'CCS'`` (camera-fixed); selects the
        ``Z<j>_<coord>`` columns.
    js : `iterable` [`int`]
        Pupil Zernike Noll indices to read, in the returned column order.

    Returns
    -------
    Z : `numpy.ndarray`, (n_pos, n_j)
        Wavefront coefficient per field point per Noll term, µm of wavefront.
    pos : `numpy.ndarray`, (n_pos, 2)
        Field position, deg, in the frame named by `coord`.
    js_found : `list` [`int`]
        The Noll indices actually present, aligned to `Z`'s columns.

    Raises
    ------
    SystemExit
        If none of the requested Noll columns are present.
    """
    d = pd.read_parquet(path)
    js_found = [int(j) for j in js if f"Z{int(j)}_{coord}" in d.columns]
    if not js_found:
        raise SystemExit(f"no Z<j>_{coord} columns in {path}")
    Z = np.column_stack([d[f"Z{j}_{coord}"].to_numpy(float) for j in js_found])
    pos = np.column_stack([d["thx_deg"].to_numpy(float), d["thy_deg"].to_numpy(float)])
    return Z, pos, js_found


def fwhm_per_point(Z, js):
    """Inferred FWHM contribution per field point, arcsec.

    Parameters
    ----------
    Z : `numpy.ndarray`, (n_pos, n_j)
        Pupil Zernike coefficients per field point, µm of wavefront.
    js : `iterable` [`int`]
        Noll indices aligned to `Z`'s columns.

    Returns
    -------
    fwhm : `numpy.ndarray`, (n_pos,)
        Quadrature-summed FWHM contribution of Noll 4 and up, arcsec.

    Notes
    -----
    Delegates the conversion and the padding to `aos_fwhm.zj_to_fwhm`, so this agrees
    with the bounce and corner studies' FWHM axis rather than defining a second one.
    """
    from lsst.ts.wep.utils import convertZernikesToPsfWidth

    from aos_fwhm import zj_to_fwhm
    return zj_to_fwhm(np.asarray(Z, float), list(js), convertZernikesToPsfWidth)


def radial_profile(diff, pos, edges_deg, edge_r_deg=EDGE_R_DEG):
    """Difference amplitude per field annulus.

    Parameters
    ----------
    diff : `numpy.ndarray`, (n_pos, n_j)
        Difference wavefront, build B minus build A, µm of wavefront.
    pos : `numpy.ndarray`, (n_pos, 2)
        Field positions, deg.
    edges_deg : `iterable` [`float`]
        Inner annulus edges in deg, ascending, the last being `edge_r_deg`.
    edge_r_deg : `float`, optional
        Radius in deg at which the convex-hull edge ring begins; everything outside
        becomes one final annulus, reported separately.

    Returns
    -------
    prof : `pandas.DataFrame`
        One row per annulus: `r_lo_deg`, `r_hi_deg`, `n_points`, `rms_um_wf` (the
        root-mean-square of the difference over the annulus's points and Noll terms, µm
        of wavefront), `is_hull_edge` (bool).
    """
    r = np.hypot(pos[:, 0], pos[:, 1])
    edges = list(edges_deg) + [float(np.nanmax(r)) + 1e-9]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo) & (r < hi)
        if not m.any():
            continue
        rows.append(dict(
            r_lo_deg=float(lo),
            r_hi_deg=float(hi),
            n_points=int(m.sum()),
            rms_um_wf=float(np.sqrt(np.nanmean(diff[m] ** 2))),
            is_hull_edge=bool(lo >= edge_r_deg),
        ))
    return pd.DataFrame(rows)


def per_term_share(diff, js, inside):
    """Each Noll term's share of the difference power, inside the hull.

    Parameters
    ----------
    diff : `numpy.ndarray`, (n_pos, n_j)
        Difference wavefront, µm of wavefront.
    js : `iterable` [`int`]
        Noll indices aligned to `diff`'s columns.
    inside : `numpy.ndarray`, (n_pos,)
        Boolean mask of field points inside the convex-hull edge ring.

    Returns
    -------
    share : `pandas.DataFrame`
        One row per Noll term, sorted by `frac_power` descending: `noll_j`,
        `rms_um_wf` (µm of wavefront), `nmad_um_wf` (robust scatter, µm of wavefront),
        and `frac_power` (dimensionless, that term's squared amplitude over the sum over
        all terms — a **power** fraction, not amplitude).
    """
    rows = []
    ms = np.nanmean(diff[inside] ** 2, axis=0)
    tot = float(np.nansum(ms))
    for i, j in enumerate(js):
        rows.append(dict(
            noll_j=int(j),
            rms_um_wf=float(np.sqrt(ms[i])),
            nmad_um_wf=float(nmad(diff[inside, i])),
            frac_power=float(ms[i] / tot) if tot > 0 else np.nan,
        ))
    return (pd.DataFrame(rows)
            .sort_values("frac_power", ascending=False, kind="mergesort")
            .reset_index(drop=True))


def report(res, label_a, label_b, coord):
    """Print the comparison, units carried on every number."""
    print(f"\n=== MIW pupil-model comparison, {coord} ===")
    print(f"  A: {label_a}")
    print(f"  B: {label_b}")
    print(f"  {res['n_pos']} field points, {res['n_j']} pupil Zernike Noll terms; "
          f"{res['n_inside']} points inside the convex-hull edge ring at "
          f"{res['edge_r_deg']:g} deg")

    print("\n  Inferred FWHM over the focal plane, arcsec "
          "(Noll 4+ quadrature, median over field points inside the hull;\n"
          "   this is the wavefront's own contribution, which adds in quadrature on "
          "top of delivered image quality):")
    print(f"    {label_a:28s} {res['fwhm_a_arcsec']:.4f} arcsec")
    print(f"    {label_b:28s} {res['fwhm_b_arcsec']:.4f} arcsec")
    print(f"    difference wavefront B-A    {res['fwhm_diff_arcsec']:.4f} arcsec")
    print(f"    change in FWHM, B minus A   {res['dfwhm_arcsec']:+.4f} arcsec")

    print(f"\n  Wavefront amplitude inside the hull, µm of wavefront (RMS over field "
          f"points and Noll terms):")
    print(f"    {label_a:28s} {res['rms_a_um_wf']:.4f} µm of wavefront")
    print(f"    {label_b:28s} {res['rms_b_um_wf']:.4f} µm of wavefront")
    print(f"    difference B-A              {res['rms_diff_um_wf']:.4f} µm of wavefront")
    print(f"    ratio difference over A     {res['rel_diff']:.4f} "
          f"(dimensionless, difference RMS over build-A RMS, both amplitudes)")

    print("\n  Difference by field annulus (is the change at the field edge?):")
    print(f"    {'r_lo':>6s} {'r_hi':>6s} {'n_pts':>6s} {'diff RMS':>10s}   note")
    print(f"    {'deg':>6s} {'deg':>6s} {'':>6s} {'µm of wf':>10s}")
    for _, p in res["profile"].iterrows():
        note = "convex-hull edge ring (defect, both builds)" if p["is_hull_edge"] else ""
        print(f"    {p['r_lo_deg']:6.2f} {p['r_hi_deg']:6.2f} "
              f"{p['n_points']:6d} {p['rms_um_wf']:10.4f}   {note}")

    print("\n  Difference by pupil Zernike term, inside the hull "
          "(frac_power is dimensionless,\n"
          "   that term's squared amplitude over the sum over all terms — POWER, "
          "not amplitude):")
    print(f"    {'Noll j':>6s} {'diff RMS':>10s} {'diff nMAD':>10s} {'frac_power':>11s}")
    print(f"    {'':>6s} {'µm of wf':>10s} {'µm of wf':>10s} {'':>11s}")
    for _, s in res["share"].head(10).iterrows():
        print(f"    {int(s['noll_j']):6d} {s['rms_um_wf']:10.4f} "
              f"{s['nmad_um_wf']:10.4f} {s['frac_power']:11.4f}")
    sph = res["share"][res["share"]["noll_j"].isin(SPHERICAL_JS)]
    if len(sph):
        tot = float(sph["frac_power"].sum())
        names = ", ".join(f"Noll {int(j)}" for j in sph["noll_j"])
        print(f"\n    Spherical terms ({names}) carry {tot:.4f} of the difference "
              f"power\n    (dimensionless); a pupil-rim change loads these rather "
              f"than spreading evenly.")


def compare(path_a, path_b, coord, js, edge_r_deg, annulus_edges):
    """Run the full comparison, returning every reported quantity in a dict."""
    Za, pos_a, js_a = load_maps(path_a, coord, js)
    Zb, pos_b, js_b = load_maps(path_b, coord, js)
    if js_a != js_b:
        raise SystemExit(f"Noll terms differ: {js_a} vs {js_b}")
    if Za.shape != Zb.shape or not np.allclose(pos_a, pos_b, atol=1e-9):
        raise SystemExit("the two builds are not on the same field grid; "
                         "intrinsic_split writes a fixed grid per rotator_select, so "
                         "check that both builds share one")

    diff = Zb - Za
    r = np.hypot(pos_a[:, 0], pos_a[:, 1])
    inside = r < edge_r_deg

    fa = fwhm_per_point(Za, js_a)
    fb = fwhm_per_point(Zb, js_a)
    fd = fwhm_per_point(diff, js_a)

    rms_a = float(np.sqrt(np.nanmean(Za[inside] ** 2)))
    rms_d = float(np.sqrt(np.nanmean(diff[inside] ** 2)))
    return dict(
        n_pos=len(pos_a), n_j=len(js_a), n_inside=int(inside.sum()),
        edge_r_deg=float(edge_r_deg),
        fwhm_a_arcsec=float(np.nanmedian(fa[inside])),
        fwhm_b_arcsec=float(np.nanmedian(fb[inside])),
        fwhm_diff_arcsec=float(np.nanmedian(fd[inside])),
        dfwhm_arcsec=float(np.nanmedian(fb[inside]) - np.nanmedian(fa[inside])),
        rms_a_um_wf=rms_a,
        rms_b_um_wf=float(np.sqrt(np.nanmean(Zb[inside] ** 2))),
        rms_diff_um_wf=rms_d,
        rel_diff=float(rms_d / rms_a) if rms_a > 0 else np.nan,
        profile=radial_profile(diff, pos_a, annulus_edges, edge_r_deg=edge_r_deg),
        share=per_term_share(diff, js_a, inside),
        fwhm_a_per_point=fa, fwhm_b_per_point=fb, pos=pos_a, inside=inside,
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--miw-a", required=True,
                   help="intrinsic_split_maps.parquet of the baseline build")
    p.add_argument("--miw-b", required=True,
                   help="intrinsic_split_maps.parquet of the comparison build")
    p.add_argument("--label-a", default="MIW A")
    p.add_argument("--label-b", default="MIW B")
    p.add_argument("--out-dir", default=None,
                   help="write the radial profile and per-term summary parquets here")
    p.add_argument("--coord", default="OCS", choices=["OCS", "CCS"])
    p.add_argument("--edge-r-deg", type=float, default=EDGE_R_DEG,
                   help="field radius in deg at which the convex-hull edge ring "
                        "begins (default %(default)g)")
    p.add_argument("--js", default=None,
                   help="comma-separated pupil Zernike Noll indices "
                        "(default: the build's 21 terms)")
    args = p.parse_args()

    js = ([int(x) for x in args.js.split(",")] if args.js else list(JS_DEFAULT))
    edges = [e for e in ANNULUS_EDGES_DEG if e < args.edge_r_deg] + [args.edge_r_deg]

    res = compare(args.miw_a, args.miw_b, args.coord, js, args.edge_r_deg, edges)
    report(res, args.label_a, args.label_b, args.coord)

    if args.out_dir:
        out = pathlib.Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        res["profile"].to_parquet(out / f"pupil_radial_profile_{args.coord}.parquet",
                                  index=False)
        res["share"].to_parquet(out / f"pupil_term_share_{args.coord}.parquet",
                                index=False)
        print(f"\n  wrote {out}/pupil_radial_profile_{args.coord}.parquet")
        print(f"  wrote {out}/pupil_term_share_{args.coord}.parquet")


if __name__ == "__main__":
    main()
