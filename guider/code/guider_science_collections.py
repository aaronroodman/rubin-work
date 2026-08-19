#!/usr/bin/env python3
"""Map each dayObs to the processed collection holding its science stars.

For each dayObs, queries the Butler for ``single_visit_star`` under a parent
(chained) collection and records the concrete RUN collection(s) that hold it,
plus the number of visits found. Writes a YAML dayObs -> collection mapping the
guider adjacent-CCD extractor can reuse, and reports nights with no processed
data. Run once on the RSP.

    python guider_science_collections.py --day-obs 20260702 20260703 ... \
        [--repo /repo/main] [--parent LSSTCam/runs/nightlyValidation]
"""
from __future__ import annotations

import argparse
import os


def parseArgs(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--day-obs", type=int, nargs="+", required=True, help="dayObs values (YYYYMMDD).")
    p.add_argument("--repo", default="/repo/main", help="Butler repository.")
    p.add_argument("--parent", default="LSSTCam/runs/nightlyValidation",
                   help="Parent (chained) collection to search.")
    p.add_argument("--dataset", default="single_visit_star", help="Dataset type to look for.")
    p.add_argument("--out", default=os.path.join(here, "guider_science_collections.yaml"))
    return p.parse_args(argv)


def main(argv=None):
    args = parseArgs(argv)
    from lsst.daf.butler import Butler

    butler = Butler(args.repo)
    mapping = {}
    print(f"{'dayObs':>10}  {'nVisits':>7}  run collection(s)")
    for dayObs in args.day_obs:
        try:
            refs = list(butler.registry.queryDatasets(
                args.dataset, collections=args.parent,
                where="instrument='LSSTCam' AND visit.day_obs = dayObs",
                bind={"dayObs": dayObs}, findFirst=True))
        except Exception as exc:  # noqa: BLE001
            print(f"{dayObs:>10}  query failed: {exc}")
            mapping[dayObs] = {"collection": None, "n_visits": 0, "error": str(exc)}
            continue
        visits = sorted({r.dataId["visit"] for r in refs})
        runs = sorted({r.run for r in refs})
        # Prefer the parent chained collection for reuse; record the runs too.
        mapping[dayObs] = {
            "collection": args.parent if visits else None,
            "runs": runs,
            "n_visits": len(visits),
        }
        run_str = ", ".join(runs) if runs else "(none -- not processed?)"
        print(f"{dayObs:>10}  {len(visits):>7}  {run_str}")

    try:
        import yaml
        with open(args.out, "w") as fh:
            yaml.safe_dump(mapping, fh, sort_keys=True)
        print(f"\nwrote {args.out}")
    except Exception as exc:  # noqa: BLE001
        print(f"\n(could not write YAML: {exc}); dict below:\n{mapping}")


if __name__ == "__main__":
    main()
