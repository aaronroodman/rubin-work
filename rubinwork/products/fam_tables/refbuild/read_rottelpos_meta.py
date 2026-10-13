"""Read rotTelPos per visit from the donut-table metadata, cheaply.

Phase 3b item 1's before/after check: compare the `rotator_angle` a `fam_tables`
visits.parquet holds against the `rotTelPos` the pipeline rotated with.

The aggregateAOSVisitTableRaw datasets are ~1.1 GB ECSV text files, so a butler.get
costs about 393 s each. The scalar metadata sits in a `meta: !!omap` block in the first
~60 `#` lines, as one-key maps. `yaml.safe_load` refuses it -- the block carries `!!omap`
and `!numpy.ndarray` tags -- so match the scalars directly.
"""
import re
import time
import numpy as np
import pandas as pd
from lsst.daf.butler import Butler

COLL = 'LSSTCam/runs/aos/fam/danish_1_2_0/wep_17_7_0/dv_4_7_0/bin_x2/paired/refitWcs'
REF = ('/sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/danish_1_2'
       '/chunks/20251116_20251130/visits.parquet')
MAX_HEADER_LINES = 400
WANT = ('rotTelPos', 'rotAngle', 'parallacticAngle', 'visit')
PATTERN = re.compile(r'^#\s*-\s*\{(\w+):\s*([^}]*)\}')


def header_scalars(path, want=WANT):
    """The named scalar metadata entries, from the leading ECSV comment block only."""
    found = {}
    with open(path, 'r') as fh:
        for i, line in enumerate(fh):
            if i >= MAX_HEADER_LINES or not line.startswith('#'):
                break
            m = PATTERN.match(line)
            if m and m.group(1) in want:
                found[m.group(1)] = m.group(2).strip()
            if len(found) == len(want):
                break
    return found


def _f(d, k):
    try:
        return float(d[k])
    except (KeyError, TypeError, ValueError):
        return np.nan


v = pd.read_parquet(REF, columns=['day_obs', 'seq_num', 'rotator_angle', 'skyAngle'])
b = Butler('/repo/main', instrument='LSSTCam', collections=[COLL])
rows, t0 = [], time.time()
for i, r in enumerate(v.itertuples(index=False), 1):
    meta = {}
    try:
        uri = b.getURI('aggregateAOSVisitTableRaw',
                       day_obs=int(r.day_obs), seq_num=int(r.seq_num))
        meta = header_scalars(uri.ospath)
    except Exception as e:
        print(f'FAIL {r.day_obs} {r.seq_num} {type(e).__name__}: {str(e)[:70]}',
              flush=True)
    rows.append(dict(day_obs=int(r.day_obs), seq_num=int(r.seq_num),
                     old_rotator_angle_deg=float(r.rotator_angle),
                     skyAngle_rad=float(r.skyAngle),
                     rotTelPos_rad=_f(meta, 'rotTelPos'),
                     rotAngle_rad=_f(meta, 'rotAngle'),
                     parallactic_rad=_f(meta, 'parallacticAngle')))
    if i % 10 == 0:
        print(f'  {i}/{len(v)} in {time.time()-t0:.0f} s', flush=True)

d = pd.DataFrame(rows)
raw = np.degrees(d.rotTelPos_rad)
d['rotTelPos_raw_deg'] = raw
d['rotTelPos_deg'] = (raw + 180.0) % 360.0 - 180.0
d['diff_deg'] = d.rotTelPos_deg - d.old_rotator_angle_deg
# the identity: rotTelPos = parallactic - rotAngle - 90 deg, wrapped
ident = (np.degrees(d.parallactic_rad) - np.degrees(d.rotAngle_rad) - 90.0)
d['identity_deg'] = (ident + 180.0) % 360.0 - 180.0
d.to_parquet('/tmp/rotcheck/rot_before_d12.parquet')
print(d[['day_obs', 'seq_num', 'old_rotator_angle_deg', 'rotTelPos_raw_deg',
         'rotTelPos_deg', 'diff_deg', 'identity_deg']].to_string(index=False))
print(f'\nelapsed {time.time()-t0:.0f} s; visits {len(d)}; '
      f'rotTelPos missing {int(d.rotTelPos_rad.isna().sum())}')
print('n needing the wrap (|raw| > 180 deg):', int((raw.abs() > 180).sum()))
print('n with |diff| > 1e-6 deg (rotTelPos vs old rotator_angle):',
      int((d.diff_deg.abs() > 1e-6).sum()))
print('max |diff| deg (camera rotator angle):', float(d.diff_deg.abs().max()))
print('max |rotTelPos - (parallactic - rotAngle - 90)| deg:',
      float((d.rotTelPos_deg - d.identity_deg).abs().max()))
print('\nvisits with |diff| > 1e-6 deg:')
big = d[d.diff_deg.abs() > 1e-6]
print(big[['day_obs', 'seq_num', 'old_rotator_angle_deg', 'rotTelPos_deg',
           'diff_deg']].to_string(index=False))
