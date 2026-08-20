#!/usr/bin/env python
"""Verify the physical camera rotator angle computed from visitInfo.

Run on the RSP (where ConsDB resolves). The physical camera rotator (rotTelPos)
is NOT boresightRotAngle; it is derived from the two boresight angles
(cf. aos/code/run_wfs_fam_compare.py):

    rotTelPos = parallacticAngle - boresightRotAngle - pi/2

For a range of visits this prints a table comparing that visitInfo-derived value
against ConsDB cdb_lsstcam.visit1_quicklook.physical_rotator_angle (the canonical
physical rotator, degrees), so we can lock down sign / offset / 2*pi wrap BEFORE
wiring the visitInfo source into extract_adjacent_psf_moments.visitRotDeg().

    python check_rotator_field.py --day-obs 20260709 \
        --collection LSSTCam/runs/nightlyValidation/68 \
        --seq-min 730 --seq-max 790 --detector 94
"""
import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
HALFPI = math.pi / 2.0


def wrap180(x):
    """Wrap a degree value to (-180, 180]."""
    return (x + 180.0) % 360.0 - 180.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day-obs", type=int, required=True)
    ap.add_argument("--collection", required=True,
                    help="processed-science run holding preliminary_visit_image")
    ap.add_argument("--seq-min", type=int, default=730)
    ap.add_argument("--seq-max", type=int, default=790)
    ap.add_argument("--limit", type=int, default=40,
                    help="max visits to probe (default 40)")
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

    def f(x, w=10):
        return f"{x:{w}.4f}" if isinstance(x, (int, float)) else f"{str(x):>{w}}"

    hdr = (f"{'seq':>5} {'parAng':>10} {'boresight':>10} {'rtp_vi':>10} "
           f"{'consdb':>10} {'c-rtp':>10} {'c-rtp(w)':>10}")
    print("\n" + hdr)
    print("-" * len(hdr))
    diffs = []
    for v in visits:
        kw = dict(collections=args.collection, instrument="LSSTCam",
                  visit=int(v), detector=args.detector)
        par = bore = rtp = None
        try:
            vi = butler.get("preliminary_visit_image.visitInfo", **kw)
            par = vi.boresightParAngle.asDegrees()
            bore = vi.boresightRotAngle.asDegrees()
            rtp = math.degrees(vi.boresightParAngle.asRadians()
                               - vi.boresightRotAngle.asRadians() - HALFPI)
        except Exception as e:
            print(f"{v - base:>5}  visitInfo failed: {type(e).__name__}: {e}")
            continue
        crot = float(cdf.loc[v, "physical_rotator_angle"])
        d = crot - rtp
        dw = wrap180(d)
        diffs.append(dw)
        print(f"{v - base:>5} {f(par)} {f(bore)} {f(rtp)} {f(crot)} {f(d)} {f(dw)}")

    if diffs:
        import statistics
        print(f"\nconsdb - rtp_vi (wrapped): mean={statistics.mean(diffs):.4f}  "
              f"stdev={statistics.pstdev(diffs):.4f}  n={len(diffs)}")
        print("If stdev ~ 0, the physical rotator = rtp_vi + (that constant mean offset).")


if __name__ == "__main__":
    main()
