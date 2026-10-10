"""Does every collection in ``variants.yaml`` still resolve, and how much does it hold?

The pre-flight check for the phase 3c rebuild of
``notes/status/organization_plan.md``: before rebuilding every product into the new
tree, confirm that each registered variant's chunk collections and each
``wfs_collections`` entry still exist in their Butler repo, and count the datasets
in the chunk's ``day_obs`` range so the run can be sized.

Counts are **datasets** of the aggregate dataset type, one per Full Array Mode (FAM)
visit for the FAM aggregates and one per in-focus exposure (or per detector, for the
per-detector aggregates) for the corner-wavefront-sensor (CWFS) ones, so they size
the rebuild rather than describing the science content.

Usage::

    python -m rubinwork.products.fam_tables.refbuild.check_collections
    python -m rubinwork.products.fam_tables.refbuild.check_collections --variant danish_1_2
    python -m rubinwork.products.fam_tables.refbuild.check_collections --json out.json
"""

import argparse
import collections
import json
import sys

from .. import reader

__all__ = ["FAM_DATASET", "BLITZ_DATASET", "check_variant", "main"]

FAM_DATASET = "aggregateAOSVisitTableRaw"
"""Default FAM aggregate dataset type, one dataset per FAM visit."""

BLITZ_DATASET = "donutBlitzFamResults"
"""Blitz-variant FAM dataset type, one dataset per FAM visit."""


def _butler(repo, cache={}):
    """A cached read-only Butler for one repo.

    Parameters
    ----------
    repo : `str`
        Butler repo path or alias.

    Returns
    -------
    butler : `lsst.daf.butler.Butler`
        The butler.
    """
    if repo not in cache:
        from lsst.daf.butler import Butler
        cache[repo] = Butler(repo, instrument="LSSTCam")
    return cache[repo]


def _probe(repo, collection, dataset_type, day_obs_min=None, day_obs_max=None):
    """Resolve one collection and count its datasets in a ``day_obs`` range.

    Parameters
    ----------
    repo : `str`
        Butler repo.
    collection : `str`
        Collection name.
    dataset_type : `str`
        Dataset type to count.
    day_obs_min, day_obs_max : `int`, optional
        Inclusive ``day_obs`` bounds, as ``YYYYMMDD``. No bound by default.

    Returns
    -------
    result : `dict`
        ``resolves`` (`bool`), ``n_datasets`` (count, dimensionless, or `None` if
        the collection does not resolve), ``n_day_obs`` (count of distinct nights),
        ``day_obs_seen`` (min and max ``day_obs`` found, or `None`), and ``error``
        (the message, when something failed).
    """
    out = {"repo": repo, "collection": collection, "dataset_type": dataset_type,
           "resolves": False, "n_datasets": None, "n_day_obs": None,
           "day_obs_seen": None, "error": None}
    try:
        butler = _butler(repo)
    except Exception as exc:
        out["error"] = f"butler {repo}: {type(exc).__name__}: {exc}"
        return out
    try:
        butler.registry.queryCollections(collection)
    except Exception as exc:
        out["error"] = f"collection: {type(exc).__name__}: {exc}"
        return out
    out["resolves"] = True

    where = []
    bind = {}
    if day_obs_min is not None:
        where.append("exposure.day_obs >= dmin")
        bind["dmin"] = int(day_obs_min)
    if day_obs_max is not None:
        where.append("exposure.day_obs <= dmax")
        bind["dmax"] = int(day_obs_max)
    try:
        refs = list(butler.registry.queryDatasets(
            dataset_type, collections=collection, findFirst=True,
            where=" AND ".join(where) if where else "", bind=bind or None))
    except Exception as exc:
        # A repo that does not define the dataset type at all is a real finding,
        # not a crash: record it and carry on to the next collection.
        out["error"] = f"queryDatasets: {type(exc).__name__}: {exc}"
        return out
    nights = set()
    for ref in refs:
        value = ref.dataId.get("exposure") or ref.dataId.get("visit")
        if value is not None:
            nights.add(int(str(value)[:8]))
    out["n_datasets"] = len(refs)
    out["n_day_obs"] = len(nights)
    out["day_obs_seen"] = [min(nights), max(nights)] if nights else None
    return out


def check_variant(variant):
    """Probe every collection of one variant.

    Parameters
    ----------
    variant : `str`
        Variant name.

    Returns
    -------
    result : `dict`
        ``variant``, ``chunks`` (one probe per chunk, each with its ``day_obs``
        range) and ``wfs_collections`` (one probe per entry).
    """
    cfg = reader.variant_config(variant)
    repo = cfg["butler_repo"]
    base_collection = cfg["fam_collections"][0]
    blitz = cfg.get("builder") == "blitz"
    fam_dataset = BLITZ_DATASET if blitz else FAM_DATASET

    chunks = []
    if blitz:
        # No chunks: the whole collection is built in one pass, bounded by
        # day_obs_min/max.
        chunks.append(dict(
            chunk=f"{cfg['day_obs_min']}_{cfg['day_obs_max']}",
            day_obs=[cfg["day_obs_min"], cfg["day_obs_max"]],
            **_probe(repo, base_collection, fam_dataset,
                     cfg["day_obs_min"], cfg["day_obs_max"])))
    for entry in cfg.get("chunks", []):
        dmin, dmax = entry["day_obs"]
        chunks.append(dict(
            chunk=f"{dmin}_{dmax}", day_obs=[dmin, dmax],
            **_probe(entry.get("butler_repo", repo),
                     entry.get("collection", base_collection),
                     fam_dataset, dmin, dmax)))

    # The CWFS collections are probed UNBOUNDED, not over day_obs_min/max: on a
    # composite variant such as danish_1_2 those bounds cover only the 2026
    # chunks, and refitWcs_2025 is a 2025 collection that would then count 0.
    wfs = []
    for name, entry in (cfg.get("wfs_collections") or {}).items():
        if isinstance(entry, str):
            entry = {"collection": entry}
        wfs.append(dict(
            name=name,
            **_probe(entry.get("butler_repo", repo), entry["collection"],
                     entry.get("dataset_type", FAM_DATASET))))
    return {"variant": variant, "chunks": chunks, "wfs_collections": wfs}


def _print_rows(title, rows, key):
    """Print one probe table.

    Parameters
    ----------
    title : `str`
        Section heading.
    rows : `list` [`dict`]
        Probe results.
    key : `str`
        Which field names the row (``"chunk"`` or ``"name"``).
    """
    print(f"\n  {title}")
    if not rows:
        print("    none")
        return
    width = max(len(str(r[key])) for r in rows)
    for r in rows:
        status = "ok" if r["resolves"] and r["error"] is None else "FAIL"
        count = "-" if r["n_datasets"] is None else f"{r['n_datasets']:,}"
        seen = ("-" if not r["day_obs_seen"]
                else f"{r['day_obs_seen'][0]}..{r['day_obs_seen'][1]}")
        print(f"    {str(r[key]):<{width}}  {status:<4}  "
              f"{count:>7} datasets  {r['n_day_obs'] or 0:>3} nights  "
              f"day_obs seen {seen}  {r['repo']}")
        if r["error"]:
            print(f"      error: {r['error']}")
        if r["resolves"] and r["n_datasets"] == 0:
            print(f"      collection resolves but holds 0 datasets of "
                  f"{r['dataset_type']} in range: {r['collection']}")


def main(argv=None):
    """Command line entry point.

    Returns
    -------
    status : `int`
        0 when every collection resolved with no error, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        prog="python -m rubinwork.products.fam_tables.refbuild.check_collections",
        description="Check that every variants.yaml collection still resolves, "
                    "and count its datasets per chunk.")
    parser.add_argument("--variant", action="append", default=None,
                        help="Variant to check; repeatable. Every registered "
                             "variant by default.")
    parser.add_argument("--json", default=None,
                        help="Also write the full results to this JSON file")
    args = parser.parse_args(argv)

    variants = args.variant or reader.variants()
    results = []
    for variant in variants:
        print(f"\n=== {variant} ===")
        result = check_variant(variant)
        results.append(result)
        _print_rows("chunks (FAM)", result["chunks"], "chunk")
        _print_rows("wfs_collections (CWFS)", result["wfs_collections"], "name")
        total = sum(r["n_datasets"] or 0 for r in result["chunks"])
        print(f"\n  total FAM visits over all chunks: {total:,} "
              "(count, dimensionless)")

    print("\n=== summary ===")
    per_variant = collections.OrderedDict()
    failures = []
    for result in results:
        per_variant[result["variant"]] = sum(
            r["n_datasets"] or 0 for r in result["chunks"])
        for row in result["chunks"] + result["wfs_collections"]:
            if not row["resolves"] or row["error"]:
                failures.append((result["variant"], row))
    for variant, total in per_variant.items():
        print(f"  {variant}: {total:,} FAM visits (count, dimensionless)")
    if failures:
        print(f"\n  {len(failures)} collection(s) did not resolve cleanly:")
        for variant, row in failures:
            print(f"    {variant}: {row['collection']} in {row['repo']}")
            print(f"      {row['error']}")
    else:
        print("\n  every collection resolved")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\n  wrote {args.json}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
