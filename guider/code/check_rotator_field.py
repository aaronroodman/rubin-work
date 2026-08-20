#!/usr/bin/env python
"""Discover + verify the physical camera rotator angle for science visits.

Run on the RSP (where ConsDB resolves). For a range of visits it prints a table:
  seq_num | ROTPA | ROTCOORD | boresightRotAngle | physical_rotator_angle | diff

  * ROTPA / ROTCOORD  -- from the preliminary_visit_image FITS metadata.
  * boresightRotAngle -- visitInfo (the SKY position angle; typically == ROTPA).
  * physical_rotator_angle -- ConsDB cdb_lsstcam.visit1_quicklook (canonical
    physical camera rotator, degrees).

Use the table to lock down the relation between the header ROTPA and the ConsDB
physical rotator (units, sign, offset) BEFORE wiring the chosen source into
extract_adjacent_psf_moments.visitRotDeg().

    python check_rotator_field.py --day-obs 20260709 \
        --collection LSSTCam/runs/nightlyValidation/68 \
        --seq-min 800 --seq-max 830 --detector 94
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day-obs", type=int, required=True)
    ap.add_argument("--collection", required=True,
                    help="processed-science run holding preliminary_visit_image")
    ap.add_argument("--seq-min", type=int, default=0)
    ap.add_argument("--seq-max", type=int, default=10**9)
    ap.add_argument("--limit", type=int, default=25,
                    help="max visits to probe headers for (default 25)")
    ap.add_argument("--detector", type=int, default=94,
                    help="a science detector present in the visits (default 94)")
    ap.add_argument("--repo", default="/repo/main")
    ap.add_argument("--consdb-url", default="auto",
                    help="ConsDB URL ('auto' picks in-pod vs external; see aos_trim)")
    args = ap.parse_args()

    from lsst.daf.butler import Butler
    butler = Butler(args.repo)

    # ---- ConsDB: physical_rotator_angle for the night (canonical) -----------
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "aos", "code"))
    from aos_trim import make_consdb_client
    cdb = make_consdb_client(args.consdb_url)
    q = f"""
        SELECT v1.visit_id, ql.physical_rotator_angle
        FROM cdb_lsstcam.visit1 v1
        LEFT JOIN cdb_lsstcam.visit1_quicklook ql ON v1.visit_id = ql.visit_id
        WHERE v1.day_obs = {args.day_obs}
    """
    cdf = cdb.query(q).to_pandas().set_index("visit_id")
    print(f"ConsDB: {len(cdf)} visits for day_obs={args.day_obs}")

    # visit_id = day_obs * 100000 + seq_num
    base = args.day_obs * 100000
    visits = sorted(v for v in cdf.index
                    if args.seq_min <= (v - base) <= args.seq_max)[:args.limit]

    hdr = f"{'seq':>6} {'ROTPA':>11} {'ROTCOORD':>9} {'boresight':>11} " \
          f"{'consdb_rot':>11} {'consdb-ROTPA':>13}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for v in visits:
        kw = dict(collections=args.collection, instrument="LSSTCam",
                  visit=int(v), detector=args.detector)
        rotpa = coord = bore = None
        try:
            md = butler.get("preliminary_visit_image.metadata", **kw)
            rotpa = md.get("ROTPA")
            coord = md.get("ROTCOORD")
        except Exception:
            pass
        try:
            vi = butler.get("preliminary_visit_image.visitInfo", **kw)
            bore = vi.boresightRotAngle.asDegrees()
        except Exception:
            pass
        crot = cdf.loc[v, "physical_rotator_angle"]

        def f(x):
            return f"{x:11.4f}" if isinstance(x, (int, float)) else f"{str(x):>11}"

        diff = (crot - rotpa) if (isinstance(rotpa, (int, float))
                                  and isinstance(crot, (int, float))) else None
        print(f"{v - base:>6} {f(rotpa)} {str(coord):>9} {f(bore)} "
              f"{f(crot)} {f(diff):>13}")


if __name__ == "__main__":
    main()
