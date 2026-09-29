"""Check the MIW build's per-visit optical state against the allowed DOF range.

The Measured Intrinsic Wavefront (MIW) build recovers a 50-degree-of-freedom (DOF)
optical state per visit with a truncated singular-value decomposition (SVD) keeping 34
v-modes.  Truncation is that recovery's only regularizer: nothing stops a recovered
amplitude exceeding the stroke the mirror or hexapod can physically reach, so the
subtracted state — and therefore the MIW itself — may be unphysical.  The
`regularized_inversion` study in `smatrix/` defines the allowed range `r_j` per DOF and
the Range-Bounded Recovery (RBR) penalty that enforces it; this script only *measures*
how far the existing MIW build's states sit outside `r_j`, and changes nothing.

Per DOF it reports the fraction of visits with `|d_j| > r_j`, and flags every DOF where
that fraction reaches a threshold (default 5 % of all visits).

The per-visit DOF come from the build's own `dz_fits.parquet`, one per rotator bin under
`<miw>/build/rot_*/`, written by `run_build_intrinsic.py` from the final iteration's
`svd.dof(A_last)`.  They are the exact states the build subtracted, not a re-derivation.
The `r_j` vector is back-derived from the same SVD normalization weights the recovery
already uses, so no new input enters the comparison.

Usage
-----
    python code/miw/check_dof_ranges.py \
        --build-dir output/miw/danish_1_3_test_A_50_34_i/build \
        --label "Danish 1.3 blitz (unpaired)" \
        --out-dir output/miw/dof_ranges

Key arguments: `--rotator-select` restricts to the five in-family rotator bins the
`_5rot` split decomposes (default is every bin present); `--frac-threshold` sets what
counts as a significant number of visits.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # aos/code

# The allowed-range vector and the RBR solver live in the `regularized_inversion` study
# under smatrix/, where the method is derived.  Reached by path insert, as bounce_lib
# does, rather than copied -- a second copy would be free to drift.
_SM = pathlib.Path(__file__).resolve().parents[3] / "smatrix" / "code"
for _p in (_SM / "regularized_inversion", _SM):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Noll indices the FAM build fits: Z4-Z26 omitting Z20 and Z21.
from miw_io import JS_DEFAULT  # noqa: E402

# The five in-family rotator bins the `_5rot` entries decompose, in the
# `rot_<lo>_<hi>` directory-name form the build writes.  April-9 sits solely in the
# +-30/+-45 bins, so selecting these excludes that out-of-family epoch by angle.
ROT_5 = ("rot_-65_-55", "rot_-20_-10", "rot_-3_3", "rot_10_20", "rot_55_65")

# Fraction of visits outside +-r_j at which a DOF is called significant.  0.05 is
# dimensionless, visits outside over all visits.
FRAC_THRESHOLD = 0.05

# The MIW build's own SVD truncation, from mi_config.yaml: all 50 DOF, 34 v-modes kept.
N_DOF = 50
N_KEEP = 34
K_MIN = 1
K_MAX = 6


def load_ranges(js=JS_DEFAULT, n_dof=N_DOF, n_keep=N_KEEP,
                k_min=K_MIN, k_max=K_MAX, ofc_normalization_yaml=None):
    """Allowed range `r_j` per DOF, with the DOF labels and units.

    Parameters
    ----------
    js : `iterable` [`int`], optional
        Pupil Zernike Noll indices the sensitivity matrix is built over; must match
        the MIW build so the SVD, and hence the normalization weights, are the same.
    n_dof, n_keep : `int`, optional
        The build's DOF count and retained v-mode count.
    k_min, k_max : `int`, optional
        Field-order range of the Double Zernike (DZ) basis.
    ofc_normalization_yaml : `str`, optional
        Normalization yaml; `None` takes the ts_config_mttcs default, which is what
        `mi_config.yaml` leaves it at.

    Returns
    -------
    r : `numpy.ndarray`, (n_dof,)
        Allowed range per DOF, µm for translations and bending-mode amplitudes,
        arcsec for hexapod rotations.
    labels : `list` [`str`]
        DOF names, e.g. `M2_dz`, `B1_1`.
    units : `list` [`str`]
        Unit string per DOF, matching `r`.
    svd : `lsst.ts.intrinsic.wavefront.ofc_svd.OFCSvd`
        The decomposition `r` was derived from, returned so the wavefront accounting
        uses the same one rather than rebuilding it.

    Notes
    -----
    Needs `lsst.ts.ofc`, so this is RSP/USDF only.
    """
    from lsst.ts.intrinsic.wavefront.ofc_svd import build_ofc_svd
    import regularized_inversion as ri

    svd = build_ofc_svd(list(js), int(k_min), int(k_max), int(n_keep),
                        n_dof=int(n_dof),
                        ofc_normalization_yaml=ofc_normalization_yaml)
    labels, units = svd.dof_labels()
    return ri.dof_range_vector(svd), list(labels), list(units), svd


def load_build_dof(build_dir, rotator_select=None):
    """Per-visit recovered DOF from a MIW build's per-rotator-bin `dz_fits.parquet`.

    Parameters
    ----------
    build_dir : `str` or `pathlib.Path`
        A build directory holding `rot_<lo>_<hi>/dz_fits.parquet` subdirectories.
    rotator_select : `iterable` [`str`], optional
        Rotator-bin directory names to keep, e.g. `ROT_5`.  `None` keeps every bin
        found.

    Returns
    -------
    df : `pandas.DataFrame`
        One row per visit: `day_obs`, `seq_num`, `rot_bin`, `n_donuts`, `bad_fit`, and
        one `dof_<label>` column per DOF in that DOF's own unit.

    Raises
    ------
    SystemExit
        If no `dz_fits.parquet` is found, or a requested rotator bin is missing.
    """
    build_dir = pathlib.Path(build_dir)
    found = sorted(p for p in build_dir.glob("rot_*/dz_fits.parquet"))
    if not found:
        raise SystemExit(f"no rot_*/dz_fits.parquet under {build_dir}")
    by_bin = {p.parent.name: p for p in found}
    if rotator_select is not None:
        missing = [b for b in rotator_select if b not in by_bin]
        if missing:
            raise SystemExit(f"rotator bins not built under {build_dir}: {missing}")
        keep = list(rotator_select)
    else:
        keep = sorted(by_bin)

    frames = []
    for b in keep:
        d = pd.read_parquet(by_bin[b])
        cols = ([c for c in ("day_obs", "seq_num", "n_donuts", "bad_fit")
                 if c in d.columns]
                + [c for c in d.columns if c.startswith("dof_")])
        sub = d[cols].copy()
        sub.insert(0, "rot_bin", b)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def range_stats(dof_df, r, labels, units, frac_threshold=FRAC_THRESHOLD):
    """Per-DOF statistics of the recovered amplitudes against the allowed range.

    Parameters
    ----------
    dof_df : `pandas.DataFrame`
        From `load_build_dof`; the `dof_<label>` columns are read.
    r : `array_like`, (n_dof,)
        Allowed range per DOF, in that DOF's own unit.
    labels, units : `list` [`str`]
        DOF names and units, aligned to `r`.
    frac_threshold : `float`, optional
        Dimensionless fraction (visits outside over all visits) at which a DOF is
        flagged.

    Returns
    -------
    stats : `pandas.DataFrame`
        One row per DOF, sorted by `frac_outside` descending.  Columns: `dof_index`,
        `dof_label`, `unit`, `range_r_j` (in `unit`), `n_visits`, `n_outside`,
        `frac_outside` (dimensionless), `median_abs_ratio`, `p95_abs_ratio`,
        `max_abs_ratio` (all dimensionless, |d_j| over r_j), `median_abs_dof` and
        `max_abs_dof` (in `unit`), and `significant` (bool).

    Notes
    -----
    Rows with a non-finite DOF are excluded from that DOF's counts, so `n_visits` is
    per DOF rather than global.
    """
    r = np.asarray(r, dtype=float)
    rows = []
    for i, (lab, unit) in enumerate(zip(labels, units)):
        col = f"dof_{lab}"
        if col not in dof_df.columns:
            continue
        d = dof_df[col].to_numpy(float)
        fin = np.isfinite(d)
        n = int(fin.sum())
        ratio = np.abs(d[fin]) / r[i]
        n_out = int((ratio > 1.0).sum())
        rows.append(dict(
            dof_index=i,
            dof_label=lab,
            unit=unit,
            range_r_j=float(r[i]),
            n_visits=n,
            n_outside=n_out,
            frac_outside=(n_out / n if n else np.nan),
            median_abs_ratio=float(np.median(ratio)) if n else np.nan,
            p95_abs_ratio=float(np.percentile(ratio, 95.0)) if n else np.nan,
            max_abs_ratio=float(ratio.max()) if n else np.nan,
            median_abs_dof=float(np.median(np.abs(d[fin]))) if n else np.nan,
            max_abs_dof=float(np.abs(d[fin]).max()) if n else np.nan,
        ))
    stats = pd.DataFrame(rows)
    stats["significant"] = stats["frac_outside"] >= float(frac_threshold)
    return stats.sort_values("frac_outside", ascending=False,
                             kind="mergesort").reset_index(drop=True)


def wavefront_excess(dof_df, svd, r, labels):
    """How much of the subtracted state's wavefront comes from over-range DOF.

    The count of over-range DOF says how often the range is violated but not how much
    wavefront rides on the violation.  This clips each recovered amplitude into
    `[-r_j, +r_j]`, forward-propagates both the full and the clipped state through the
    same rank-limited sensitivity matrix the recovery inverted, and compares them.  The
    clipped state is not a fit — clipping is not what Range-Bounded Recovery does — so
    this is an accounting of the wavefront at stake, not a proposed alternative MIW.

    Parameters
    ----------
    dof_df : `pandas.DataFrame`
        From `load_build_dof`.
    svd : `lsst.ts.intrinsic.wavefront.ofc_svd.OFCSvd`
        The build's SVD, supplying the forward operator and the normalization weights.
    r : `array_like`, (n_dof,)
        Allowed range per DOF, in that DOF's own unit.
    labels : `list` [`str`]
        DOF names aligned to `r`.

    Returns
    -------
    out : `dict`
        `rms_full_um_wf` and `rms_excess_um_wf`, each the median over visits of the
        root-mean-square over the Double Zernike (k, j) grid in µm of wavefront, and
        `frac_excess_amplitude`, their dimensionless ratio (over-range wavefront RMS
        over full-state wavefront RMS, both amplitudes).
    """
    import regularized_inversion as ri

    r = np.asarray(r, dtype=float)
    # dW = S @ (d / w) for physical DOF d and normalization weights w.
    S = ri.forward_operator(svd)
    w = np.asarray(svd.normalization_weights, dtype=float)

    D = np.column_stack([dof_df[f"dof_{L}"].to_numpy(float) for L in labels])
    D_clip = np.clip(D, -r[None, :], r[None, :])
    dW_full = (D / w[None, :]) @ S.T
    dW_clip = (D_clip / w[None, :]) @ S.T

    rms_full = np.sqrt(np.nanmean(dW_full ** 2, axis=1))
    rms_exc = np.sqrt(np.nanmean((dW_full - dW_clip) ** 2, axis=1))
    return dict(
        rms_full_um_wf=float(np.median(rms_full)),
        rms_excess_um_wf=float(np.median(rms_exc)),
        frac_excess_amplitude=float(np.median(rms_exc / rms_full)),
    )


def report(stats, label, n_visits, frac_threshold=FRAC_THRESHOLD, wf=None):
    """Print the assessment, units carried on every column."""
    sig = stats[stats["significant"]]
    print(f"\n=== {label}: recovered DOF against the allowed range r_j ===")
    print(f"  {n_visits} visits; a DOF is flagged when at least "
          f"{100.0 * frac_threshold:g} % of its visits have |d_j| > r_j")
    print(f"  {len(sig)} of {len(stats)} DOF are flagged\n")

    hdr = (f"  {'idx':>3s} {'DOF':8s} {'r_j':>11s} {'unit':>6s} "
           f"{'n_out':>6s} {'frac':>7s} {'med|d|/r':>9s} {'p95|d|/r':>9s} "
           f"{'max|d|/r':>9s} {'med|d|':>11s} {'max|d|':>11s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, s in stats.iterrows():
        mark = "*" if s["significant"] else " "
        print(f" {mark}{s['dof_index']:3d} {s['dof_label']:8s} "
              f"{s['range_r_j']:11.4g} {s['unit']:>6s} "
              f"{s['n_outside']:6d} {s['frac_outside']:7.3f} "
              f"{s['median_abs_ratio']:9.3f} {s['p95_abs_ratio']:9.3f} "
              f"{s['max_abs_ratio']:9.3f} "
              f"{s['median_abs_dof']:11.4g} {s['max_abs_dof']:11.4g}")
    print("\n  r_j, med|d| and max|d| are in the DOF's own unit (column `unit`); "
          "the three\n  ratio columns and `frac` are dimensionless "
          "(|d_j| over r_j, and visits outside\n  over all visits).  "
          "* marks a flagged DOF.")

    if len(sig):
        grp = {"M2 hexapod": range(0, 5), "Camera hexapod": range(5, 10),
               "M1M3 bending": range(10, 30), "M2 bending": range(30, 50)}
        print("\n  Flagged DOF by group (count flagged of count in group):")
        for name, idx in grp.items():
            n_in = int(stats["dof_index"].isin(list(idx)).sum())
            n_sig = int(sig["dof_index"].isin(list(idx)).sum())
            print(f"    {name:16s} {n_sig:2d} of {n_in:2d}")

    if wf is not None:
        print(f"\n  Wavefront at stake, median over visits of the RMS over the DZ "
              f"(k, j) grid:\n"
              f"    full recovered state            "
              f"{wf['rms_full_um_wf']:.4f} µm of wavefront\n"
              f"    part from over-range amplitudes "
              f"{wf['rms_excess_um_wf']:.4f} µm of wavefront\n"
              f"    ratio                           "
              f"{wf['frac_excess_amplitude']:.4f} (dimensionless, over-range "
              f"wavefront RMS over full-state wavefront RMS, both amplitudes)")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build-dir", required=True, action="append",
                   help="MIW build directory holding rot_*/dz_fits.parquet; "
                        "repeat to compare builds")
    p.add_argument("--label", action="append", default=None,
                   help="label per --build-dir, in the same order")
    p.add_argument("--out-dir", default=None,
                   help="write a per-DOF summary parquet per build here")
    p.add_argument("--rotator-select", default="5rot",
                   choices=["5rot", "all"],
                   help="'5rot' keeps the five in-family rotator bins the _5rot "
                        "split decomposes; 'all' keeps every bin built "
                        "(default %(default)s)")
    p.add_argument("--frac-threshold", type=float, default=FRAC_THRESHOLD,
                   help="dimensionless fraction of visits outside +-r_j at which a "
                        "DOF is flagged (default %(default)g)")
    p.add_argument("--n-dof", type=int, default=N_DOF)
    p.add_argument("--n-keep", type=int, default=N_KEEP)
    args = p.parse_args()

    labels_in = args.label or []
    if labels_in and len(labels_in) != len(args.build_dir):
        raise SystemExit(f"{len(args.build_dir)} --build-dir but "
                         f"{len(labels_in)} --label")

    rot = ROT_5 if args.rotator_select == "5rot" else None

    r, dof_labels, dof_units, svd = load_ranges(n_dof=args.n_dof,
                                                n_keep=args.n_keep)
    print(f"allowed range r_j over {len(r)} DOF: "
          f"{np.nanmin(r):.4g} to {np.nanmax(r):.4g} "
          f"(µm or arcsec, per DOF unit; smallest is "
          f"{dof_labels[int(np.nanargmin(r))]}, largest "
          f"{dof_labels[int(np.nanargmax(r))]})")

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    for i, bd in enumerate(args.build_dir):
        lab = labels_in[i] if labels_in else pathlib.Path(bd).parent.name
        dof_df = load_build_dof(bd, rotator_select=rot)
        stats = range_stats(dof_df, r, dof_labels, dof_units,
                            frac_threshold=args.frac_threshold)
        wf = wavefront_excess(dof_df, svd, r, dof_labels)
        print(f"\nbuild {bd}")
        print(f"  rotator bins: {sorted(dof_df['rot_bin'].unique())}")
        report(stats, lab, len(dof_df), frac_threshold=args.frac_threshold, wf=wf)
        if out_dir is not None:
            name = pathlib.Path(bd).parent.name
            out = out_dir / f"dof_ranges_{name}.parquet"
            stats.to_parquet(out, index=False)
            print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
