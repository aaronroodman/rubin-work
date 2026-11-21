"""Copy the A-B-A bounce-test points used by the elevation-pair comparison into a CSV.

One-off transcription, approved 2026-11-20: a study does not read another study's output,
so the few points needed are frozen here with their source. Source: `aos/bounce` run
``20261102_aba_optical_state`` (notebook ``bounce_aba_optical_state_v2``, commit e379f29),
``bounce_aba_optical_state.csv``. Each point is B minus the 70 deg reference A at B
elevation, from RBR 50/50; DZ in µm of wavefront, v1 dimensionless, DOF in µm (tilts in deg;
the source's arcsec label is stale).

Run as ``python extract_bounce_points.py <source csv>``; writes ``bounce_points_20261102.csv``
beside this script.
"""
import pathlib
import sys

import pandas as pd

SOURCE = 'aos/bounce output/20261102_aba_optical_state/bounce_aba_optical_state.csv (commit e379f29)'
HEX = ['M2_dz', 'M2_dx', 'M2_dy', 'M2_rx', 'M2_ry', 'Cam_dz', 'Cam_dx', 'Cam_dy', 'Cam_rx', 'Cam_ry']
#: Bounce name -> elevation_pairs quantity, for v1, DZ(1,4..8) and the standard_22 DOF.
NAME_MAP = {'v1': 'v1',
            **{f'DZ(1,{j})': f'dz1_{j}' for j in range(4, 9)},
            **{n: f'dof{k}' for k, n in enumerate(HEX)},
            **{f'B1_{i}': f'dof{9 + i}' for i in range(1, 8)},
            **{f'B2_{i}': f'dof{29 + i}' for i in range(1, 6)}}

src = pd.read_csv(sys.argv[1])
out = src[src['name'].isin(NAME_MAP)].copy()
out['quantity'] = out['name'].map(NAME_MAP)
out['unit'] = out['unit'].replace({'arcsec': 'deg', 'μm': 'um'})
out = out.rename(columns={'b_value_deg': 'b_elev_deg', 'name': 'bounce_name'})
out = out[['night', 'leg', 'b_elev_deg', 'quantity', 'bounce_name', 'unit', 'delta', 'err']]
path = pathlib.Path(__file__).resolve().parent / 'bounce_points_20261102.csv'
with open(path, 'w') as f:
    f.write(f'# source: {SOURCE}\n# delta = B - A (A = 70 deg), RBR 50/50 basis\n')
    out.to_csv(f, index=False, float_format='%.6g')
print(f'{len(out)} rows, {out["quantity"].nunique()} quantities -> {path}')
