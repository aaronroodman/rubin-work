"""Shared pieces of pupil pass 6: run paths, the rotator angle, the side list and the masks.

Sides are the 188 pass 5 sides (`pass5_common.sides`), with their raw-pixel positions
from the pass 3 and pass 4 samples. Each side carries two rotator angles:

- ``rtp_visitinfo_deg``: boresightParAngle − boresightRotAngle − 90 deg from the raw
  visitInfo (`fit_donuts.rtp_rad`), what passes 1–5 used;
- ``rtp_phys_deg``: ConsDB ``cdb_lsstcam.visit1_quicklook.physical_rotator_angle``, the
  study's rtp from pass 6 on.

`with_rtp` sets ``rtp_deg`` to one of them and recomputes the OCS field azimuth.

Masks: ``v4.02g`` (pass 2, `obscuration_v402g`) and ``v4.02s`` (pass 6 step 4,
`pass6_silhouette`: v4.02g with the camera-silhouette circle refitted to the Batoid inner
edge at 1.3–1.75 deg).
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass5_common as pc  # noqa: E402  (blitz path first)
import obscuration_v402g as og  # noqa: E402

STUDY = 'aos/giant_donut_pupil'
RUN_LABEL = 'pupil6'
RUN_DATE = '20261023'
STUDY_DIR = pc.p3.STUDY_DIR
RUN = STUDY_DIR / 'output' / f'{RUN_DATE}_{RUN_LABEL}'
PASS5_RUN = STUDY_DIR / 'output' / '20261022_pupil5'
CALIB_COLLECTION = 'LSSTCam/calib'
RAW_COLLECTION = 'LSSTCam/raw/all'
VISITS = {'extra': 2025102300337, 'intra': 2025102300340}
# Pass 5 free-blur medians (step 7), arcsec FWHM; the simulations' blur.
SIM_FWHM_ARCSEC = {'extra': 1.00, 'intra': 1.60}
CAD_VANE_WIDTH_M = 0.05


def wrap(a):
    return pc.wrap(a)


def run_dir():
    from rubinwork.common import study_run
    return study_run(STUDY, RUN_LABEL, date=RUN_DATE).path


# ---------------------------------------------------------------- rotator

def rotator_table(refresh=False):
    """physical_rotator_angle (ConsDB) and the visitInfo rtp per visit, deg."""
    path = RUN / 'rotator.parquet'
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    from rubinwork.common.telemetry_clients import make_consdb_client
    ids = ','.join(str(v) for v in VISITS.values())
    q = ('select visit_id, physical_rotator_angle from cdb_lsstcam.visit1_quicklook '
         f'where visit_id in ({ids})')
    t = make_consdb_client().query(q).to_pandas()
    t = t.rename(columns={'visit_id': 'visit', 'physical_rotator_angle': 'rtp_phys_deg'})
    s = pc.sides().drop_duplicates('visit').set_index('visit').rtp_deg
    t['rtp_visitinfo_deg'] = t.visit.map(s)
    t['phys_minus_visitinfo_deg'] = t.rtp_phys_deg - t.rtp_visitinfo_deg
    t['side'] = t.visit.map({v: k for k, v in VISITS.items()})
    RUN.mkdir(parents=True, exist_ok=True)
    t.to_parquet(path)
    t.to_csv(RUN / 'rotator.csv', index=False, float_format='%.6f')
    return t


# ---------------------------------------------------------------- sides

def sides():
    """The 188 pass 5 sides with raw positions and both rotator angles."""
    s = pc.sides()
    s = s[s.error == ''].reset_index(drop=True)
    pos = []
    for run in (pc.PASS4_RUN, pc.PASS3_RUN):
        p = pd.read_parquet(run / 'sample.parquet')[['sample_id', 'side', 'x_pix', 'y_pix']]
        pos.append(p)
    pos = pd.concat(pos).drop_duplicates(['sample_id', 'side'])
    s = s.merge(pos, on=['sample_id', 'side'], how='left')
    rot = rotator_table().set_index('visit')
    s['rtp_visitinfo_deg'] = s.rtp_deg
    s['rtp_phys_deg'] = s.visit.map(rot.rtp_phys_deg)
    return s


def with_rtp(s, which):
    """Copy of `s` with ``rtp_deg`` from ``rtp_<which>_deg`` and the OCS field azimuth."""
    s = s.copy()
    s['rtp_deg'] = s[f'rtp_{which}_deg']
    r = np.radians(s.rtp_deg.to_numpy())
    x, y = s.field_x_ccs_deg.to_numpy(), s.field_y_ccs_deg.to_numpy()
    s['field_az_ocs_deg'] = np.mod(np.degrees(np.arctan2(np.sin(r) * x + np.cos(r) * y,
                                                         np.cos(r) * x - np.sin(r) * y)), 360.0)
    s['rtp_source'] = which
    return s


# ---------------------------------------------------------------- masks

def mask_name_s(offsets_mm):
    d, c, m = offsets_mm
    return f'RubinObsc_v4.02s_r_det{d:+.1f}_cam{c:+.1f}_m2{m:+.1f}.yaml'


def mask_path(variant, offsets_m):
    if variant == 'v4.02g':
        return og.mask_path(pc.p3.MASK_RUN, offsets_m)
    mm = tuple(round(o * 1e3, 1) + 0.0 for o in offsets_m)
    return RUN / 'masks' / mask_name_s(mm)


_MASKS = {}


def mask_params(variant, offsets_m, vane_width_m=None):
    """danish mask_params for a side; `vane_width_m` replaces every Spider_3D width."""
    key = (variant, tuple(offsets_m))
    if key not in _MASKS:
        _MASKS[key] = yaml.safe_load(mask_path(variant, offsets_m).read_text())
    p = _MASKS[key]
    if vane_width_m is None:
        return p
    p = dict(p)
    p['Spider_3D'] = [dict(v, width=float(vane_width_m)) for v in p['Spider_3D']]
    return p
