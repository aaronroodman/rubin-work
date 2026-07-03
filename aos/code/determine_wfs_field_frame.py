#!/usr/bin/env python3
"""determine_wfs_field_frame — establish how ts_wep per-detector `zernikes`
field angles (intra_field/extra_field) map to the aggregate table's
thx/thy_{OCS,CCS,NW}_{intra,extra}, so the ai_donut reader can build OCS
positions WITHOUT guessing the coord/sign convention.

Run on a collection that has BOTH `zernikes` (per-detector) AND
`aggregateAOSVisitTableRaw` (with thx/thy positions) — e.g. the paired_3mm
collection.  For a few visits it joins the two on donut id and reports, per
frame:
  * which thx/thy_<frame>_intra the raw `intra_field` matches (residual + unit ratio)
  * if none match directly, the rotation angle alpha such that
    R(alpha).field ~= thx/thy_OCS_intra, compared to rotTelPos / rotAngle /
    parallacticAngle to identify which angle + sign it is
  * how the paired thx_OCS relates to the intra/extra pair (mean? == extra?)
  * confirmation of the native frame of the Z4..Z26 columns vs zk_{OCS,CCS,NW}

RSP only (Butler).  Read-only; writes nothing.
"""
import argparse
import numpy as np

CORNER_DETS = [191, 195, 199, 203]                 # extra-focal SW0 corner detector ids
DET_ID_TO_NAME = {191: 'R00_SW0', 195: 'R04_SW0', 199: 'R40_SW0', 203: 'R44_SW0'}
NOLL = list(range(4, 20)) + [22, 23, 24, 25, 26]    # `zernikes` Z columns (no 20,21)


def _rot(v, ang_rad):
    c, s = np.cos(ang_rad), np.sin(ang_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _int(x):
    """int, or None if masked/NaN."""
    if x is np.ma.masked or (np.ma.is_masked(x)):
        return None
    try:
        return int(x)
    except (ValueError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', default='/repo/main')
    ap.add_argument('--collection', required=True,
                    help='collection with BOTH zernikes and aggregateAOSVisitTableRaw (e.g. paired_3mm)')
    ap.add_argument('--visits', type=int, nargs='+', default=None,
                    help='visit ids to test; default: first 3 with an aggregate table')
    args = ap.parse_args()
    from lsst.daf.butler import Butler
    b = Butler(args.repo, collections=args.collection)

    visits = args.visits
    if not visits:
        refs = list(b.registry.queryDatasets('aggregateAOSVisitTableRaw',
                                             collections=args.collection))
        allv = sorted({int(r.dataId['visit']) for r in refs})
        # spread across the collection so rotTelPos varies (OCS vs CCS degenerate near 0)
        idx = np.unique(np.linspace(0, len(allv) - 1, min(6, len(allv))).astype(int))
        visits = [allv[i] for i in idx]
    print(f'testing visits: {visits}\n')

    for vid in visits:
        try:
            agg = b.get('aggregateAOSVisitTableRaw', visit=vid, collections=args.collection)
        except Exception as e:
            print(f'visit {vid}: no aggregate ({e})'); continue
        meta = agg.meta
        rot = {k: meta.get(k) for k in ('rotTelPos', 'rotAngle', 'parallacticAngle', 'skyAngle')}
        print(f'==== visit {vid}  angles(deg?): {rot} ====')
        # index the aggregate by (detector-name, intra_id, extra_id)
        aidx = {}
        for row in agg:
            ii, ee = _int(row['intra_donut_id']), _int(row['extra_donut_id'])
            if ii is None or ee is None:
                continue
            aidx[(str(row['detector']), ii, ee)] = row

        for det in CORNER_DETS:
            try:
                zt = b.get('zernikes', visit=vid, detector=det, collections=args.collection)
            except Exception:
                continue
            name = DET_ID_TO_NAME[det]
            for zr in zt:
                ii, ee = _int(zr['intra_donut_id']), _int(zr['extra_donut_id'])
                if ii is None or ee is None:
                    continue
                arow = aidx.get((name, ii, ee))
                if arow is None:
                    continue
                fld = np.asarray(zr['intra_field'], float).ravel()[:2]   # native field (intra)
                print(f'  det {det} donut {key[1]}/{key[2]}: intra_field={fld}')
                for fr in ('CCS', 'OCS', 'NW'):
                    try:
                        tgt = np.array([float(arow[f'thx_{fr}_intra']), float(arow[f'thy_{fr}_intra'])])
                    except Exception:
                        continue
                    d = tgt - fld
                    ratio = np.linalg.norm(tgt) / (np.linalg.norm(fld) + 1e-30)
                    print(f'      thx/thy_{fr}_intra={tgt}  resid={d}  |tgt|/|fld|={ratio:.4f}')
                # fit rotation field->OCS
                try:
                    ocs = np.array([float(arow['thx_OCS_intra']), float(arow['thy_OCS_intra'])])
                    a = np.arctan2(fld[0] * ocs[1] - fld[1] * ocs[0],
                                   fld[0] * ocs[0] + fld[1] * ocs[1])   # angle field->OCS
                    print(f'      => R(alpha).field~=OCS  alpha={np.degrees(a):+.3f} deg  '
                          f'check R(a).field={_rot(fld, a)}')
                except Exception:
                    pass
                # paired thx_OCS vs intra/extra
                try:
                    pair = np.array([float(arow['thx_OCS']), float(arow['thy_OCS'])])
                    ins = np.array([float(arow['thx_OCS_intra']), float(arow['thy_OCS_intra'])])
                    exs = np.array([float(arow['thx_OCS_extra']), float(arow['thy_OCS_extra'])])
                    print(f'      thx_OCS(paired)={pair}  mean(intra,extra)={(ins + exs) / 2}  extra={exs}')
                except Exception:
                    pass
                # native Z frame check
                try:
                    zc = {fr: np.asarray(arow[f'zk_{fr}'], float).ravel() for fr in ('OCS', 'CCS', 'NW')}
                    znat = np.array([float(zr[f'Z{j}']) for j in NOLL])
                    for fr, zz in zc.items():
                        n = min(len(znat), len(zz))
                        print(f'      Z-native vs zk_{fr}: rms={np.sqrt(np.nanmean((znat[:n] - zz[:n])**2)):.4g}')
                except Exception as e:
                    print(f'      (Z-frame check skipped: {e})')
                break            # one donut per detector is enough
        print()


if __name__ == '__main__':
    main()
