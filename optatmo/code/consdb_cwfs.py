"""ConsDB corner-WFS Zernikes + per-visit metadata for optatmo, REUSING the
rubin-work/aos ConsDB code:

  * aos_trim.make_consdb_client         -- the generic ConsDB client
  * aos_state.fetch_corner_zernikes_consdb / ZK_NOLL / CORNERS
                                        -- per-corner total-OPD Zernikes from
                                           cdb_<instr>.ccdvisit1_quicklook

For a night where the `danish` aggregateAOSVisitTableRaw is NOT available at the
USDF (e.g. 20260706), this is the corner-WFS source.  It writes, per visit:

  data/cwfs_<visit>.parquet   detector, thx_OCS, thy_OCS (rad), ztot_<i>  --
      same schema extract_cwfs.py produces, so plot_data_model is unchanged
      (ztot_<i> is the total OPD for Noll ZK_NOLL[i], Z4..Z26 excl Z20,Z21).
  data/visitmeta_<visit>.parquet   visit, rot_deg, alt_deg, az_deg, band  --
      night-agnostic rotator/alt/az for fit_optatmo + plot_data_model.

Corner field positions come from cameraGeom (the SW0 detector centres in CCS)
rotated to OCS by the rotator (frames.rotate_field), matching the moment frame.

VALIDATE on 20260513 first: both this ConsDB path and the danish cwfs parquet
exist there, so `--validate` compares them to pin the corner-Zernike frame/sign
before trusting a new night.  RUN ON USDF (ConsDB + cameraGeom).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

# reuse the aos ConsDB code (sibling package rubin-work/aos/code)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', 'aos', 'code'))
from aos_trim import make_consdb_client                       # noqa: E402
from aos_state import fetch_corner_zernikes_consdb, ZK_NOLL   # noqa: E402
import frames                                                 # noqa: E402

# corner wavefront-sensor SW0 detector ids ONLY; names come from LsstCam so we
# never hardcode a detector->raft mapping (that could itself mislabel a corner).
CORNER_DETS = [191, 195, 199, 203]

# candidate visit1 column names for altitude / azimuth (pick whichever exists)
ALT_CANDS = ['altitude', 's_alt', 'altitude_start', 'elevation', 'sky_altitude']
AZ_CANDS = ['azimuth', 's_az', 'azimuth_start', 'sky_azimuth']


def fetch_visit_meta(cdb, visit_ids, instrument='lsstcam'):
    """Per-visit rotator (physical_rotator_angle) + alt/az/band from ConsDB."""
    ids = ','.join(str(int(v)) for v in visit_ids)
    q = f"""SELECT v1.*, ql.physical_rotator_angle
            FROM cdb_{instrument}.visit1 v1
            LEFT JOIN cdb_{instrument}.visit1_quicklook ql
                 ON v1.visit_id = ql.visit_id
            WHERE v1.visit_id IN ({ids})"""
    df = cdb.query(q).to_pandas()
    alt = next((c for c in ALT_CANDS if c in df.columns), None)
    az = next((c for c in AZ_CANDS if c in df.columns), None)
    if alt is None or az is None:
        print(f'  (alt/az column not found; visit1 has: {sorted(df.columns)[:40]})')
    out = pd.DataFrame({
        'visit': df['visit_id'].astype(int),
        'rot_deg': pd.to_numeric(df['physical_rotator_angle'], errors='coerce'),
        'alt_deg': pd.to_numeric(df[alt], errors='coerce') if alt else np.nan,
        'az_deg': pd.to_numeric(df[az], errors='coerce') if az else np.nan,
        'band': df['band'] if 'band' in df.columns else '?'})
    return out.set_index('visit')


def corner_names(cam=None):
    """{det_id: LsstCam detector name} for the corner SW0 sensors."""
    if cam is None:
        from lsst.obs.lsst import LsstCam
        cam = LsstCam.getCamera()
    return {d: cam[int(d)].getName() for d in CORNER_DETS}


def corner_ocs_positions(rot_deg, sign=1, cam=None):
    """OCS field positions (rad) of the corner SW0 sensors at this rotator,
    keyed by the LsstCam detector name (no hardcoded raft mapping)."""
    from lsst.afw.cameraGeom import PIXELS, FIELD_ANGLE
    import lsst.geom as geom
    if cam is None:
        from lsst.obs.lsst import LsstCam
        cam = LsstCam.getCamera()
    pos = {}
    for det in CORNER_DETS:
        d = cam[int(det)]
        c = d.getBBox().getCenter()
        fa = d.getTransform(PIXELS, FIELD_ANGLE).applyForward(
            geom.Point2D(c.getX(), c.getY()))
        # cameraGeom FIELD_ANGLE is DVCS; swap to CCS (thx_CCS = field_y) before
        # the rotator rotation to OCS, matching the star-moment path in
        # data_fit.load_and_prep.  See frames.dvcs_to_ccs_field.
        cx, cy = frames.dvcs_to_ccs_field(fa.getX(), fa.getY())
        tx, ty = frames.rotate_field(cx, cy,
                                     np.deg2rad(rot_deg), sign)   # CCS->OCS, rad
        pos[d.getName()] = (float(tx), float(ty))
    return pos


def write_visit(cdb, visit, out_dir, meta, zk, sign=1, cam=None):
    """Write cwfs_<visit>.parquet (corner totals + OCS positions) + visitmeta."""
    m = meta.loc[visit]
    m.to_frame().T.to_parquet(f'{out_dir}/visitmeta_{visit}.parquet')
    if visit not in zk.index:
        print(f'  {visit}: no corner Zernikes in ConsDB'); return
    row = zk.loc[visit]
    pos = corner_ocs_positions(float(m['rot_deg']), sign=sign, cam=cam)
    recs = []
    for name, (tx, ty) in pos.items():
        rec = {'detector': name, 'thx_OCS': tx, 'thy_OCS': ty, 'snr': np.nan}
        for i, noll in enumerate(ZK_NOLL):
            col = f'z{noll}_{name}'
            rec[f'ztot_{i}'] = float(row[col]) if col in row else np.nan
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_parquet(f'{out_dir}/cwfs_{visit}.parquet')
    print(f'  {visit}: rot={float(m["rot_deg"]):.2f} alt={float(m["alt_deg"]):.2f} '
          f'az={float(m["az_deg"]):.2f} -> cwfs_{visit}.parquet ({len(df)} corners)')


def validate(out_dir, visit):
    """Compare ConsDB vs danish cwfs parquet for a visit (frame/sign check)."""
    cd = pd.read_parquet(f'{out_dir}/cwfs_{visit}.parquet')
    dn = pd.read_parquet(f'{out_dir}/cwfs_{visit}_danish.parquet')
    cd['corner'] = cd.detector.str[:3]; dn['corner'] = dn.detector.str[:3]
    print(f'  validate {visit}: ConsDB vs danish, ztot per corner (Z4,Z11,Z22):')
    for c in ['R00', 'R04', 'R40', 'R44']:
        a = cd[cd.corner == c]; b = dn[dn.corner == c]
        if not len(a) or not len(b):
            continue
        i4, i11, i22 = ZK_NOLL.index(4), ZK_NOLL.index(11), ZK_NOLL.index(22)
        print(f'    {c}: ConsDB Z4={a.iloc[0].ztot_0:+.3f} '
              f'danish Z4={float(b.iloc[0].ztot_0):+.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--visits', type=int, nargs='+', required=True)
    ap.add_argument('--instrument', default='lsstcam')
    ap.add_argument('--out-dir', default='data')
    ap.add_argument('--sign', type=int, default=1, help='CCS->OCS rotation sign')
    ap.add_argument('--consdb-url', default='auto',
                    help="ConsDB endpoint; 'auto' picks the in-pod host inside "
                         "the RSP and the external tokened endpoint on S3DF "
                         "(sdfiana/slacrd batch). Default 'auto'.")
    ap.add_argument('--token-file', default=None,
                    help='RSP access-token file for the external endpoint '
                         '(default ~/.lsst/consdb_token, else $ACCESS_TOKEN)')
    ap.add_argument('--validate', action='store_true',
                    help='after writing, compare each visit against the danish '
                         'cwfs_<visit>_danish.parquet (frame/sign check)')
    args = ap.parse_args()

    from lsst.obs.lsst import LsstCam
    cam = LsstCam.getCamera()
    corners = corner_names(cam)                 # {det_id: LsstCam name}
    print('corner SW0 detectors (LsstCam names):', corners)
    cdb = make_consdb_client(url=args.consdb_url, token_file=args.token_file)
    meta = fetch_visit_meta(cdb, args.visits, args.instrument)
    zk = fetch_corner_zernikes_consdb(cdb, args.visits, instrument=args.instrument,
                                      corners=corners)
    for v in args.visits:
        write_visit(cdb, v, args.out_dir, meta, zk, sign=args.sign, cam=cam)
        if args.validate:
            import os
            if os.path.exists(f'{args.out_dir}/cwfs_{v}_danish.parquet'):
                validate(args.out_dir, v)
            else:
                print(f'  validate {v}: no cwfs_{v}_danish.parquet in '
                      f'{args.out_dir}; skipping')


if __name__ == '__main__':
    main()
