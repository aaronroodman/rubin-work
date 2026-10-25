"""How the loose ``visit_quality_pass`` cut (`rubinwork.products.fam_tables.quality`)
acts on each ``fam_tables`` variant, and what the DZ fits look like after it.

For each variant's build, prints:

- visits failing each of the three cuts, and all together;
- ``bad_fit`` counts among the fitted visits, split by reason (``n_donuts`` < 200, or
  some |DZ coefficient| > 2.0 µm), for the k=1..3 and k=1..6 fits;
- for the k=1..6 fit, the distribution of each visit's largest |DZ coefficient|, and
  the visits whose coefficients are robust outliers (more than 5 nMAD from the median
  across visits, per coefficient), with the coefficient and its value.

Run from the repo root::

    python -m rubinwork.products.fam_tables.qa.evaluate_quality_cut --build 20261025
"""

import argparse
import re

import numpy as np
import pandas as pd

from rubinwork.common.utils import nmad
from rubinwork.products.fam_tables import quality, reader

VARIANTS = ("danish_1_2", "danish_1_3", "danish_1_3_test", "danish_1_3_v1000")
BAD_FIT_THRESHOLD_UM = 2.0
BAD_FIT_MIN_DONUTS = 200
OUTLIER_NMAD = 5.0


def coeff_columns(columns, prefix):
    pat = re.compile(rf"^{prefix}_z(\d+)_c(\d+)$")
    return [c for c in columns if pat.match(c)]


def evaluate(variant, build):
    d = reader.build_dir(variant, build)
    v = pd.read_parquet(d / "visits.parquet",
                        columns=["day_obs", "seq_num", "n_donuts",
                                 "n_detectors_with_min_donuts", "median_blur_arcsec",
                                 "visit_quality_pass"])
    f = pd.read_parquet(d / "fits.parquet")
    print(f"\n## {variant} @ {build}\n")
    n = len(v)
    cuts = {
        f"n_donuts <= {quality.MIN_DONUTS_EXCLUSIVE}":
            ~(v.n_donuts > quality.MIN_DONUTS_EXCLUSIVE),
        f"n_detectors_with_min_donuts <= {quality.MIN_DETECTORS_EXCLUSIVE}":
            ~(v.n_detectors_with_min_donuts > quality.MIN_DETECTORS_EXCLUSIVE),
        f"median_blur_arcsec >= {quality.MAX_BLUR_ARCSEC_EXCLUSIVE} (or NaN)":
            ~(v.median_blur_arcsec < quality.MAX_BLUR_ARCSEC_EXCLUSIVE),
    }
    print(f"visits: {n} (count)")
    for name, fail in cuts.items():
        print(f"  fail {name}: {int(fail.sum())}")
    print(f"  fail any (visit_quality_pass False): {int((~v.visit_quality_pass).sum())}; "
          f"pass: {int(v.visit_quality_pass.sum())}")
    print(f"fitted visits (fits.parquet rows): {len(f)}")

    for prefix in ("z1toz3", "z1toz6"):
        cols = coeff_columns(f.columns, prefix)
        big = f[cols].abs().gt(BAD_FIT_THRESHOLD_UM).any(axis=1)
        few = f["n_donuts_1"] < BAD_FIT_MIN_DONUTS if "n_donuts_1" in f else big & False
        print(f"  {prefix}_bad_fit: {int(f[f'{prefix}_bad_fit'].sum())} "
              f"(|DZ| > {BAD_FIT_THRESHOLD_UM} µm: {int(big.sum())}; "
              f"n_donuts < {BAD_FIT_MIN_DONUTS}: {int(few.sum())})")
    print(f"  bad_fit (either): {int(f['bad_fit'].sum())}")

    cols = coeff_columns(f.columns, "z1toz6")
    a = f[cols].abs()
    vmax = a.max(axis=1)
    q = np.nanpercentile(vmax, [50, 90, 99, 100])
    print(f"  k=1..6 largest |DZ| per visit [µm]: median {q[0]:.2f}, 90% {q[1]:.2f}, "
          f"99% {q[2]:.2f}, max {q[3]:.2f}")
    for t in (1.0, 2.0, 3.0, 5.0):
        print(f"    visits with largest |DZ| > {t} µm: {int((vmax > t).sum())}")
    print(f"  most frequent arg-max coefficient among |DZ| > {BAD_FIT_THRESHOLD_UM} µm: "
          f"{a[vmax > BAD_FIT_THRESHOLD_UM].idxmax(axis=1).value_counts().head(6).to_dict()}")

    x = f[cols]
    z = (x - x.median()) / x.apply(lambda s: nmad(s.dropna().to_numpy()))
    out = z.abs().gt(OUTLIER_NMAD)
    n_out = out.sum(axis=1)
    print(f"  visits with any coefficient > {OUTLIER_NMAD:g} nMAD from the visit median: "
          f"{int((n_out > 0).sum())}; with > 10 such coefficients: {int((n_out > 10).sum())}")
    worst = (z.abs().max(axis=1)).sort_values(ascending=False).head(12).index
    rows = []
    for i in worst:
        c = z.loc[i].abs().idxmax()
        b = v.set_index(["day_obs", "seq_num"]).loc[(f.at[i, "day_obs"], f.at[i, "seq_num"])]
        rows.append(dict(day_obs=f.at[i, "day_obs"], seq_num=f.at[i, "seq_num"],
                         coeff=c, value_um=round(float(x.at[i, c]), 3),
                         nmad_units=round(float(z.at[i, c]), 1),
                         n_outlier_coeffs=int(n_out[i]),
                         largest_abs_dz_um=round(float(vmax[i]), 2),
                         bad_fit=bool(f.at[i, "bad_fit"]),
                         blur_arcsec=round(float(b.median_blur_arcsec), 2),
                         n_det=int(b.n_detectors_with_min_donuts),
                         n_donuts=int(b.n_donuts)))
    print("  most extreme visits (value_um in µm; nmad_units dimensionless):")
    print(pd.DataFrame(rows).to_string(index=False))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--build", required=True)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = p.parse_args(argv)
    for variant in args.variants:
        evaluate(variant, args.build)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
