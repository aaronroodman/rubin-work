"""Check that the telemetry sidecar reaches fits.parquet as well as visits.parquet.

Phase 3b item 6: ``combine_fits`` dropped every sidecar column, because
``merge_sidecars`` had ``visits.parquet`` as its only target. Run this after a
``fam_tables`` build to confirm the columns are there.

The phase 2 reference ``fits.parquet`` is the **broken** state — it carries 13 of the
385 sidecar columns — so this does not compare the two for equality.

The check is:
  (a) every column the reference already had is identical at zero tolerance;
  (b) the added columns equal the telemetry sidecar, joined on (day_obs, seq_num);
  (c) visits.parquet is unchanged at zero tolerance (it already had them).
"""
from pathlib import Path
import pandas as pd, numpy as np

REF = Path('/sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/danish_1_2')
SCR = Path('/sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/_scratch_3b/item6')
CH = 'chunks/20251116_20251130'
KEY = ['day_obs', 'seq_num']

def differing(a, b, cols):
    out = []
    for c in cols:
        x, y = a[c], b[c]
        try:
            if x.dtype.kind in 'fc' or y.dtype.kind in 'fc':
                if not np.array_equal(np.asarray(x, float), np.asarray(y, float),
                                      equal_nan=True):
                    out.append(c)
            elif not x.equals(y):
                out.append(c)
        except Exception:
            if not x.astype(str).equals(y.astype(str)):
                out.append(c)
    return out

fail = 0
for rel in (CH + '/fits.parquet', 'fits.parquet'):
    a = pd.read_parquet(REF / rel).sort_values(KEY).reset_index(drop=True)
    b = pd.read_parquet(SCR / rel).sort_values(KEY).reset_index(drop=True)
    lost = sorted(set(a.columns) - set(b.columns))
    added = sorted(set(b.columns) - set(a.columns))
    kept = [c for c in a.columns]
    d = differing(a, b[kept], kept)
    cam_a = sum(1 for c in a.columns if c.startswith('cam_'))
    cam_b = sum(1 for c in b.columns if c.startswith('cam_'))
    print(f'{rel}: {len(a)} rows')
    print(f'  columns {len(a.columns)} -> {len(b.columns)}; '
          f'cam_* {cam_a} -> {cam_b} (dimensionless counts)')
    print(f'  columns lost: {len(lost)}' + (f' {lost[:8]}' if lost else ''))
    print(f'  columns added: {len(added)}')
    print(f'  pre-existing columns differing: {len(d)}'
          + (f' -> {d[:8]}' if d else ' (zero tolerance)'))
    fail += len(lost) + len(d)

    # (b) the added columns must equal the sidecar
    src = (SCR / CH / 'telemetry.parquet' if rel.startswith('chunks')
           else SCR / CH / 'telemetry.parquet')
    tel = pd.read_parquet(src)
    j = b[KEY + added].merge(tel[KEY + added], on=KEY, how='left',
                             suffixes=('_got', '_exp'))
    bad = []
    for c in added:
        x = np.asarray(j[c + '_got'], float) if j[c + '_got'].dtype.kind in 'fc' \
            else j[c + '_got'].astype(str)
        y = np.asarray(j[c + '_exp'], float) if j[c + '_exp'].dtype.kind in 'fc' \
            else j[c + '_exp'].astype(str)
        same = (np.array_equal(x, y, equal_nan=True)
                if isinstance(x, np.ndarray) and x.dtype.kind == 'f' else (x == y).all())
        if not same:
            bad.append(c)
    print(f'  added columns matching the sidecar: {len(added) - len(bad)}/{len(added)}'
          + (f'; MISMATCH {bad[:6]}' if bad else ''))
    fail += len(bad)

for rel in (CH + '/visits.parquet', 'visits.parquet'):
    a = pd.read_parquet(REF / rel).sort_values(KEY).reset_index(drop=True)
    b = pd.read_parquet(SCR / rel).sort_values(KEY).reset_index(drop=True)
    assert set(a.columns) == set(b.columns), 'visits column set moved'
    d = differing(a, b[a.columns], list(a.columns))
    print(f'{rel}: {len(a)} rows, {len(a.columns)} cols, differing: {len(d)}'
          + (f' -> {d[:8]}' if d else ' (zero tolerance)'))
    fail += len(d)

print('\nRESULT:', 'PASS' if fail == 0 else f'FAIL ({fail})')
