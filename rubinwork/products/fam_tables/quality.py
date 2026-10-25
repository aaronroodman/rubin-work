"""The per-visit quality cut that defines ``visit_quality_pass`` in every ``fam_tables``
build.

A loose cut, applied before the Double Zernike (DZ) fit: it removes only visits that
cannot be used at all. Each consumer applies its own tighter cuts at use, from the metric
columns that travel in ``visits.parquet`` and ``fits.parquet``.

A visit passes when all three hold:

- ``n_donuts`` > 500 (donuts in the visit, count);
- ``n_detectors_with_min_donuts`` > 100 (detectors with at least 3 donuts, count);
- ``median_blur_arcsec`` < 1.8 (median donut blur, arcsec).
"""

import numpy as np

__all__ = ["MIN_DONUTS_EXCLUSIVE", "MIN_DETECTORS_EXCLUSIVE", "MAX_BLUR_ARCSEC_EXCLUSIVE",
           "CUT_DESCRIPTION", "visit_quality_pass", "apply"]

MIN_DONUTS_EXCLUSIVE = 500
MIN_DETECTORS_EXCLUSIVE = 100
MAX_BLUR_ARCSEC_EXCLUSIVE = 1.8

CUT_DESCRIPTION = (f"n_donuts > {MIN_DONUTS_EXCLUSIVE}, "
                   f"n_detectors_with_min_donuts > {MIN_DETECTORS_EXCLUSIVE}, "
                   f"median_blur_arcsec < {MAX_BLUR_ARCSEC_EXCLUSIVE}")


def visit_quality_pass(n_donuts, n_detectors_with_min_donuts, median_blur_arcsec):
    """The ``visit_quality_pass`` flag for each visit.

    Parameters
    ----------
    n_donuts : `array_like`
        Donuts per visit, count.
    n_detectors_with_min_donuts : `array_like`
        Detectors with at least the per-detector donut floor, count.
    median_blur_arcsec : `array_like`
        Median donut blur per visit, arcsec. NaN fails.

    Returns
    -------
    passes : `numpy.ndarray` [`bool`]
        True where the visit passes all three cuts.
    """
    n_donuts = np.asarray(n_donuts, dtype=float)
    n_det = np.asarray(n_detectors_with_min_donuts, dtype=float)
    blur = np.asarray(median_blur_arcsec, dtype=float)
    with np.errstate(invalid="ignore"):
        return ((n_donuts > MIN_DONUTS_EXCLUSIVE) & (n_det > MIN_DETECTORS_EXCLUSIVE)
                & (blur < MAX_BLUR_ARCSEC_EXCLUSIVE))


def apply(visits_path):
    """Rewrite ``visit_quality_pass`` in a ``visits.parquet`` with this module's cut.

    Only that column and the table metadata change; every other column keeps its
    schema. In the astropy metadata the threshold keys ``run_mktable`` writes
    (``min_donuts_per_visit``, ``min_detectors_per_visit``, ``max_median_blur_arcsec``)
    are replaced by ``visit_quality_cut``, since they no longer describe the flag.

    Parameters
    ----------
    visits_path : `str` or `pathlib.Path`
        The table, rewritten in place.

    Returns
    -------
    n_pass, n_total : `int`
        Visits passing, and visits in the table (count).
    """
    import re

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(str(visits_path))
    passes = visit_quality_pass(table["n_donuts"].to_numpy(zero_copy_only=False),
                                table["n_detectors_with_min_donuts"].to_numpy(
                                    zero_copy_only=False),
                                table["median_blur_arcsec"].to_numpy(zero_copy_only=False))
    i = table.schema.get_field_index("visit_quality_pass")
    table = table.set_column(i, "visit_quality_pass", pa.array(passes, pa.bool_()))
    meta = dict(table.schema.metadata or {})
    yml = meta.get(b"table_meta_yaml")
    if yml is not None:
        text = re.sub(r"(?m)^- \{(min_donuts_per_visit|min_detectors_per_visit|"
                      r"max_median_blur_arcsec|visit_quality_cut):.*\}\n?", "",
                      yml.decode())
        if "meta: !!omap" not in text:
            text = text.rstrip("\n") + "\nmeta: !!omap\n"
        text = text.rstrip("\n") + f"\n- {{visit_quality_cut: '{CUT_DESCRIPTION}'}}\n"
        meta[b"table_meta_yaml"] = text.encode()
    meta[b"visit_quality_cut"] = CUT_DESCRIPTION.encode()
    pq.write_table(table.replace_schema_metadata(meta), str(visits_path))
    return int(passes.sum()), len(passes)


def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m rubinwork.products.fam_tables.quality",
        description="Rewrite visit_quality_pass in visits.parquet files: " + CUT_DESCRIPTION)
    p.add_argument("visits", nargs="+", help="visits.parquet files, rewritten in place")
    for path in p.parse_args(argv).visits:
        n_pass, n = apply(path)
        print(f"{path}: {n_pass} of {n} visits pass ({CUT_DESCRIPTION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
