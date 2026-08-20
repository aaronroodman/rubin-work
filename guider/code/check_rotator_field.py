#!/usr/bin/env python
"""Discover + verify the physical camera rotator angle for a science visit.

Run on the RSP (where ConsDB resolves). Prints, for one visit:
  * every rotation-ish key in the preliminary_visit_image FITS metadata,
  * visitInfo.boresightRotAngle (the SKY position angle -- NOT the rotator),
  * ConsDB cdb_lsstcam.visit1_quicklook.physical_rotator_angle (the canonical
    physical camera rotator, degrees).

Compare the header candidate(s) against the ConsDB value to lock down the field
name, units (deg vs rad), sign, and any 2*pi / 360 offset BEFORE wiring the
chosen key into extract_adjacent_psf_moments.visitRotDeg().

    python check_rotator_field.py --visit 2026070900808 \
        --collection LSSTCam/runs/nightlyValidation/68 --detector 94
"""
import argparse
import math
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--visit", type=int, required=True)
    ap.add_argument("--collection", required=True,
                    help="processed-science run holding preliminary_visit_image")
    ap.add_argument("--detector", type=int, default=94,
                    help="a science detector present in the visit (default 94)")
    ap.add_argument("--repo", default="/repo/main")
    ap.add_argument("--consdb-url", default="auto",
                    help="ConsDB URL ('auto' picks in-pod vs external; see aos_trim)")
    args = ap.parse_args()

    from lsst.daf.butler import Butler
    butler = Butler(args.repo)
    kw = dict(collections=args.collection, instrument="LSSTCam",
              visit=args.visit, detector=args.detector)

    # ---- 1. FITS metadata: dump every rotation-ish key -----------------------
    print(f"\n=== preliminary_visit_image.metadata  visit={args.visit} "
          f"det={args.detector} ===")
    try:
        md = butler.get("preliminary_visit_image.metadata", **kw)
        keys = list(md.keys())
        hits = [k for k in keys
                if any(t in k.upper() for t in ("ROT", "NASMYTH", "INST_PA", "PA"))]
        if not hits:
            print("  (no ROT/NASMYTH/PA keys found; dumping all keys instead)")
            hits = keys
        for k in hits:
            print(f"  {k:24s} = {md[k]!r}")
    except Exception as e:
        print(f"  metadata get failed: {type(e).__name__}: {e}", file=sys.stderr)

    # ---- 2. visitInfo.boresightRotAngle (reference; NOT the rotator) ---------
    print("\n=== visitInfo (reference) ===")
    try:
        vi = butler.get("preliminary_visit_image.visitInfo", **kw)
        print(f"  boresightRotAngle = {vi.boresightRotAngle.asDegrees():.6f} deg "
              f"(SKY position angle, not the mechanical rotator)")
    except Exception as e:
        print(f"  visitInfo get failed: {type(e).__name__}: {e}", file=sys.stderr)

    # ---- 3. ConsDB physical_rotator_angle (canonical) -----------------------
    print("\n=== ConsDB cdb_lsstcam.visit1_quicklook.physical_rotator_angle ===")
    try:
        sys.path.insert(0, "../../aos/code")
        from aos_trim import make_consdb_client
        cdb = make_consdb_client(args.consdb_url)
        q = f"""
            SELECT v1.visit_id, ql.physical_rotator_angle
            FROM cdb_lsstcam.visit1 v1
            LEFT JOIN cdb_lsstcam.visit1_quicklook ql ON v1.visit_id = ql.visit_id
            WHERE v1.visit_id = {args.visit}
        """
        df = cdb.query(q).to_pandas()
        if len(df):
            rot = float(df["physical_rotator_angle"].iloc[0])
            print(f"  physical_rotator_angle = {rot:.6f}  (expected: degrees)")
            print("\n  --- compare against your chosen header key above ---")
            print(f"  as-is:            {rot:.4f}")
            print(f"  if header is rad: {math.degrees(rot):.4f} deg")
            print(f"  mod 360:          {rot % 360:.4f}")
        else:
            print("  no ConsDB row for this visit_id")
    except Exception as e:
        print(f"  ConsDB query failed: {type(e).__name__}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
