"""Summarize every corner visit of one night into per-CCD and per-raft rows, and list
the nights.

Command line, used by the Snakefile::

    python -m rubinwork.products.cwfs_science.builders.build_night \\
        --variant d13-blitz_rso1071 --night 20260315 \\
        --out-ccd ccd_20260315.parquet --out-raft raft_20260315.parquet
    python -m rubinwork.products.cwfs_science.builders.build_night \\
        --variant d13-blitz_rso1071 --list-nights
"""

import argparse
import sys

import numpy as np
import pandas as pd

from .. import reader
from .summarize import READ_COLUMNS, summarize_visit

__all__ = ["butler_for", "nights", "build_night", "main"]

REPO = "/repo/main"


def butler_for(variant):
    """A Butler on the variant's collections."""
    from lsst.daf.butler import Butler
    return Butler(REPO, collections=list(reader.variant_config(variant)["collections"]))


def nights(variant, butler=None):
    """Sorted ``day_obs`` values that have a corner table.

    Returns
    -------
    nights : `list` [`int`]
    """
    cfg = reader.variant_config(variant)
    butler = butler or butler_for(variant)
    refs = butler.query_datasets(cfg["dataset_type"], limit=None, find_first=True)
    return sorted({int(r.dataId["day_obs"]) for r in refs})


def _visit_records(butler, night):
    """``{visit: record}`` for one night's visits."""
    return {int(r.id): r for r in butler.query_dimension_records(
        "visit", where=f"visit.day_obs = {int(night)}", limit=None)}


def _quantity(meta, key, unit):
    v = meta.get(key)
    if v is None:
        return np.nan
    return float(v.to_value(unit)) if hasattr(v, "to_value") else float(v)


def build_night(variant, night, butler=None, log=print):
    """Per-CCD summary rows for every corner visit of one night.

    Parameters
    ----------
    variant : `str`
        Variant name.
    night : `int`
        ``day_obs``.
    butler : `lsst.daf.butler.Butler`, optional
        On the variant's collections; made by default.
    log : callable, optional

    Returns
    -------
    ccd, raft : `pandas.DataFrame`
        As `summarize.summarize_visit`, with visit columns: ``day_obs``, ``seq_num``,
        ``visit``, ``band``, ``physical_filter``, ``science_program``,
        ``observation_reason``, ``mjd`` (TAI, day), ``exp_time`` (s), ``alt``, ``az``,
        ``rot_tel_pos`` and ``rot_sky`` (deg).
    """
    import astropy.units as u

    cfg = reader.variant_config(variant)
    butler = butler or butler_for(variant)
    where = f"instrument = 'LSSTCam' and visit.day_obs = {int(night)}"
    refs = butler.query_datasets(cfg["dataset_type"], where=where, limit=None,
                                 find_first=True)
    zrefs = {}
    for r in butler.query_datasets(cfg["zernikes_dataset_type"], where=where,
                                   limit=None, find_first=True, explain=False):
        zrefs.setdefault(int(r.dataId["visit"]), []).append(r)
    records = _visit_records(butler, night)
    ccd, raft, n_fail, n_noz = [], [], 0, 0
    for ref in sorted(refs, key=lambda r: int(r.dataId["visit"])):
        visit = int(ref.dataId["visit"])
        try:
            tab = butler.get(ref, parameters={"columns": list(READ_COLUMNS),
                                              "strip_astropy_meta_yaml": False})
        except Exception as exc:
            n_fail += 1
            log(f"[cwfs_science] {visit}: read failed: {exc}")
            continue
        if len(tab) == 0:
            continue
        rec, meta = records.get(visit), tab.meta
        info = dict(
            day_obs=int(night), seq_num=visit % 100000, visit=visit,
            band=str(ref.dataId.get("band", "")),
            physical_filter=str(ref.dataId.get("physical_filter", "")),
            science_program=str(getattr(rec, "science_program", "") or ""),
            observation_reason=str(getattr(rec, "observation_reason", "") or ""),
            mjd=float(meta["date"].tai.mjd) if "date" in meta else np.nan,
            exp_time=_quantity(meta, "exposure_time", u.s),
            alt=_quantity(meta, "boresight_alt", u.deg),
            az=_quantity(meta, "boresight_az", u.deg),
            rot_tel_pos=_quantity(meta, "rot_tel_pos", u.deg),
            rot_sky=_quantity(meta, "boresight_rot_angle", u.deg))
        zernikes = {int(r.dataId["detector"]): butler.get(r) for r in zrefs.get(visit, [])}
        n_noz += not zernikes
        c, r = summarize_visit(tab, cfg["noll"], zernikes, info)
        ccd.extend(c)
        raft.extend(r)
    log(f"[cwfs_science] {night}: {len(refs)} visits, {n_fail} unreadable, "
        f"{n_noz} without zernikes, {len(ccd)} CCD rows, {len(raft)} raft rows (counts)")
    return pd.DataFrame(ccd), pd.DataFrame(raft)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m rubinwork.products.cwfs_science.builders"
                                ".build_night", description=__doc__.split("\n")[0])
    p.add_argument("--variant", required=True)
    p.add_argument("--night", type=int)
    p.add_argument("--out-ccd")
    p.add_argument("--out-raft")
    p.add_argument("--list-nights", action="store_true")
    a = p.parse_args(argv)
    if a.list_nights:
        print("\n".join(str(n) for n in nights(a.variant)))
        return 0
    if a.night is None or not a.out_ccd or not a.out_raft:
        p.error("--night, --out-ccd and --out-raft are required unless --list-nights")
    ccd, raft = build_night(a.variant, a.night)
    ccd.to_parquet(a.out_ccd, index=False)
    raft.to_parquet(a.out_raft, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
