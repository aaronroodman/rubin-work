"""Compare two MIW builds' recovered optical state, on the visits they have in common.

A Measured Intrinsic Wavefront (MIW) build recovers a per-visit 50-degree-of-freedom
(DOF) optical state and subtracts it before averaging what remains into the intrinsic.
Two builds that differ in the wavefronts they were built from will differ in that
subtracted state as well as in the MIW itself, and whether they do is a result in its own
right: a change that moves the MIW but leaves the optical state alone is acting on the
static wavefront, while one that moves both is partly being absorbed by the fit.

This compares the two per-visit states **on the `(day_obs, seq_num)` visits common to
both builds**, so the difference is per visit and paired rather than a difference of two
samples.  It reports, per DOF and per v-mode, the median and robust scatter of the
paired difference, and scales each against the allowed range `r_j` where that is defined.

Reads each build's own `build/rot_*/dz_fits.parquet`, written by `run_build_intrinsic.py`
from the final iteration's `svd.dof(A_last)` — the exact states the builds subtracted.

Usage
-----
    python code/miw/compare_build_dof.py \
        --build-a output/miw/danish_1_3_test_A_50_34_i/build \
        --build-b output/miw/danish_1_3_v1000_A_50_34_i/build \
        --label-a "legacy pupil" \
        --label-b "v1000 pupil" \
        --out-dir output/miw/danish_1_3_legacy_vs_v1000

Key arguments: `--rotator-select` restricts to the five in-family rotator bins the
`_5rot` split decomposes (the default, matching the canonical product).

Notes
-----
`r_j` comes from `smatrix/code/regularized_inversion`, so this needs `lsst.ts.ofc` —
RSP/USDF only.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))  # aos/code
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))      # this study

from common.utils import nmad  # noqa: E402
from check_dof_ranges import ROT_5, load_build_dof, load_ranges  # noqa: E402

# Number of v-modes the 50/34 build keeps, so the vmode_<i> columns run 1..34.
N_KEEP = 34

KEY = ["day_obs", "seq_num"]


def load_vmodes(build_dir, rotator_select=None, n_keep=N_KEEP):
    """Per-visit v-mode amplitudes from a build's per-rotator-bin `dz_fits.parquet`.

    Parameters
    ----------
    build_dir : `str` or `pathlib.Path`
        A build directory holding `rot_<lo>_<hi>/dz_fits.parquet`.
    rotator_select : `iterable` [`str`], optional
        Rotator-bin directory names to keep; `None` keeps every bin found.
    n_keep : `int`, optional
        Number of retained v-modes, giving columns `vmode_1` .. `vmode_<n_keep>`.

    Returns
    -------
    df : `pandas.DataFrame`
        `day_obs`, `seq_num` and one `vmode_<i>` column per retained mode,
        dimensionless (the v-mode amplitudes the build recovered).
    """
    build_dir = pathlib.Path(build_dir)
    found = sorted(p for p in build_dir.glob("rot_*/dz_fits.parquet"))
    if not found:
        raise SystemExit(f"no rot_*/dz_fits.parquet under {build_dir}")
    by_bin = {p.parent.name: p for p in found}
    keep = list(rotator_select) if rotator_select is not None else sorted(by_bin)
    missing = [b for b in keep if b not in by_bin]
    if missing:
        raise SystemExit(f"rotator bins not built under {build_dir}: {missing}")

    frames = []
    for b in keep:
        d = pd.read_parquet(by_bin[b])
        cols = KEY + [f"vmode_{i}" for i in range(1, int(n_keep) + 1)
                      if f"vmode_{i}" in d.columns]
        frames.append(d[cols].copy())
    return pd.concat(frames, ignore_index=True)


def paired_diff(a, b, value_cols):
    """Inner-join two per-visit frames on `(day_obs, seq_num)` and difference them.

    Parameters
    ----------
    a, b : `pandas.DataFrame`
        Per-visit frames, each carrying `day_obs`, `seq_num` and `value_cols`.
    value_cols : `list` [`str`]
        Columns to difference, as `b` minus `a`.

    Returns
    -------
    diff : `pandas.DataFrame`
        `day_obs`, `seq_num` and one column per `value_cols` holding B minus A.
    n_a, n_b, n_common : `int`
        Visit counts in A, in B, and in the inner join.
    """
    a = a.drop_duplicates(subset=KEY)
    b = b.drop_duplicates(subset=KEY)
    m = a.merge(b, on=KEY, suffixes=("_a", "_b"), how="inner")
    out = m[KEY].copy()
    for c in value_cols:
        out[c] = m[f"{c}_b"].to_numpy(float) - m[f"{c}_a"].to_numpy(float)
    return out, len(a), len(b), len(m)


def summarize(diff, a, b, value_cols, labels=None, units=None, r=None):
    """Per-quantity summary of the paired difference and of each build's own level.

    Parameters
    ----------
    diff : `pandas.DataFrame`
        From `paired_diff`.
    a, b : `pandas.DataFrame`
        The two per-visit frames, for each build's own median amplitude.
    value_cols : `list` [`str`]
        Columns summarized.
    labels, units : `list` [`str`], optional
        Display name and unit per column; default is the column name and `''`.
    r : `array_like`, optional
        Allowed range per column, same unit as the column, for the ratio columns.

    Returns
    -------
    out : `pandas.DataFrame`
        One row per quantity, sorted by `abs_median_diff` descending.  Columns:
        `name`, `unit`, `median_a`, `median_b` (median over visits of the signed value,
        in `unit`), `median_diff` and `nmad_diff` (B minus A, in `unit`),
        `abs_median_diff` (in `unit`), and where `r` is given `range_r_j` (in `unit`)
        plus `median_diff_over_r` (dimensionless, median difference over allowed range).
    """
    labels = list(labels) if labels is not None else list(value_cols)
    units = list(units) if units is not None else [""] * len(value_cols)
    rows = []
    for i, c in enumerate(value_cols):
        d = diff[c].to_numpy(float)
        row = dict(
            name=labels[i],
            unit=units[i],
            median_a=float(np.nanmedian(a[c].to_numpy(float))),
            median_b=float(np.nanmedian(b[c].to_numpy(float))),
            median_diff=float(np.nanmedian(d)),
            nmad_diff=float(nmad(d)),
            abs_median_diff=float(abs(np.nanmedian(d))),
        )
        if r is not None:
            row["range_r_j"] = float(r[i])
            row["median_diff_over_r"] = (float(abs(np.nanmedian(d)) / r[i])
                                         if r[i] > 0 else np.nan)
        rows.append(row)
    return (pd.DataFrame(rows)
            .sort_values("abs_median_diff", ascending=False, kind="mergesort")
            .reset_index(drop=True))


def report(title, summ, n_a, n_b, n_common, label_a, label_b, top=12,
           has_range=False):
    """Print one summary table, units carried on every column."""
    print(f"\n=== {title} ===")
    print(f"  A: {label_a} ({n_a} visits)")
    print(f"  B: {label_b} ({n_b} visits)")
    print(f"  {n_common} visits common to both; every difference below is per visit, "
          f"B minus A")
    cols = (f"  {'name':10s} {'unit':>6s} {'med A':>11s} {'med B':>11s} "
            f"{'med diff':>11s} {'nMAD diff':>11s}")
    if has_range:
        cols += f" {'r_j':>11s} {'|diff|/r_j':>11s}"
    print(f"\n{cols}")
    print("  " + "-" * (len(cols) - 2))
    for _, s in summ.head(top).iterrows():
        line = (f"  {s['name']:10s} {s['unit']:>6s} "
                f"{s['median_a']:11.4g} {s['median_b']:11.4g} "
                f"{s['median_diff']:11.4g} {s['nmad_diff']:11.4g}")
        if has_range:
            line += f" {s['range_r_j']:11.4g} {s['median_diff_over_r']:11.4f}"
        print(line)
    print(f"\n  Sorted by |median difference|, top {min(top, len(summ))} of "
          f"{len(summ)}.  med A, med B, med diff, nMAD diff and r_j are in the\n"
          f"  column `unit`; |diff|/r_j is dimensionless (median difference over "
          f"allowed range).")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build-a", required=True,
                   help="baseline build dir holding rot_*/dz_fits.parquet")
    p.add_argument("--build-b", required=True,
                   help="comparison build dir holding rot_*/dz_fits.parquet")
    p.add_argument("--label-a", default="build A")
    p.add_argument("--label-b", default="build B")
    p.add_argument("--out-dir", default=None,
                   help="write the DOF and v-mode summary parquets here")
    p.add_argument("--rotator-select", default="5rot", choices=["5rot", "all"],
                   help="'5rot' keeps the five in-family rotator bins the _5rot split "
                        "decomposes (default %(default)s)")
    p.add_argument("--n-keep", type=int, default=N_KEEP)
    p.add_argument("--top", type=int, default=12,
                   help="rows printed per table (default %(default)d)")
    args = p.parse_args()

    rot = ROT_5 if args.rotator_select == "5rot" else None

    r, dof_labels, dof_units, _ = load_ranges(n_keep=args.n_keep)

    dof_a = load_build_dof(args.build_a, rotator_select=rot)
    dof_b = load_build_dof(args.build_b, rotator_select=rot)
    dof_cols = [f"dof_{L}" for L in dof_labels if f"dof_{L}" in dof_a.columns]
    keep = [i for i, L in enumerate(dof_labels) if f"dof_{L}" in dof_a.columns]
    d_diff, na, nb, nc = paired_diff(dof_a, dof_b, dof_cols)
    dof_summ = summarize(d_diff, dof_a, dof_b, dof_cols,
                         labels=[dof_labels[i] for i in keep],
                         units=[dof_units[i] for i in keep],
                         r=np.asarray(r, float)[keep])
    report("Recovered DOF, paired difference on common visits", dof_summ,
           na, nb, nc, args.label_a, args.label_b, top=args.top, has_range=True)

    vm_a = load_vmodes(args.build_a, rotator_select=rot, n_keep=args.n_keep)
    vm_b = load_vmodes(args.build_b, rotator_select=rot, n_keep=args.n_keep)
    vm_cols = [c for c in vm_a.columns if c.startswith("vmode_")]
    v_diff, nva, nvb, nvc = paired_diff(vm_a, vm_b, vm_cols)
    vm_summ = summarize(v_diff, vm_a, vm_b, vm_cols,
                        labels=vm_cols,
                        units=["dimensionless"] * len(vm_cols))
    report("Recovered v-modes, paired difference on common visits", vm_summ,
           nva, nvb, nvc, args.label_a, args.label_b, top=args.top)

    if args.out_dir:
        out = pathlib.Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        dof_summ.to_parquet(out / "build_dof_compare.parquet", index=False)
        vm_summ.to_parquet(out / "build_vmode_compare.parquet", index=False)
        print(f"\n  wrote {out}/build_dof_compare.parquet")
        print(f"  wrote {out}/build_vmode_compare.parquet")


if __name__ == "__main__":
    main()
