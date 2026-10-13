"""Check that the registered ``fam_variant`` rebuilds identically from its own path.

Phase 3b items 3 and 4: ``build_fam_dz.resolve_paths`` built the FAM tables directory
from the **long** ``param_set`` key, a layout the tree lost when ``dir_name`` arrived, so
the registered variant could not be rebuilt as recorded. Run ``build_fam_dz`` against a
**scratch** copy of the reference database with no ``--aos-output`` override, then run
this.

``fam_dz`` and ``fam_variant`` must match the reference at zero tolerance apart from
``computed_at`` / ``created_at`` (build-time timestamps) and ``fits_path`` -- the last
because the fix is precisely that the recorded path now resolves.
"""
import duckdb, numpy as np, pandas as pd

REF = ('/sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/value_added/ref'
       '/aos_efd.duckdb')
SCR = ('/sdf/group/rubin/u/roodman/LSST/rubin-work/_phase2_refbuild/_scratch_3b/item34'
       '/aos_efd.duckdb')
EXPECTED_DIFF = {'computed_at', 'created_at', 'fits_path'}
KEY = ['visit_id', 'fam_variant_id']

a = duckdb.connect(REF, read_only=True).execute('select * from fam_dz').df()
b = duckdb.connect(SCR, read_only=True).execute('select * from fam_dz').df()
print(f'fam_dz rows: reference {len(a)}, rebuilt {len(b)} (dimensionless counts)')
print(f'fam_dz columns: {len(a.columns)} vs {len(b.columns)}')
assert set(a.columns) == set(b.columns), set(a.columns) ^ set(b.columns)
a = a.sort_values(KEY).reset_index(drop=True)
b = b.sort_values(KEY)[a.columns].reset_index(drop=True)

diff, ok = [], 0
for c in a.columns:
    x, y = a[c], b[c]
    same = True
    if x.dtype == object and len(x) and isinstance(x.iloc[0], (list, np.ndarray)):
        for u, w in zip(x, y):
            if not np.array_equal(np.asarray(u, float), np.asarray(w, float),
                                  equal_nan=True):
                same = False
                break
    elif x.dtype.kind in 'fc':
        same = np.array_equal(np.asarray(x, float), np.asarray(y, float),
                              equal_nan=True)
    else:
        same = x.astype(str).equals(y.astype(str))
    (ok := ok + 1) if same else diff.append(c)

unexpected = [c for c in diff if c not in EXPECTED_DIFF]
print(f'  identical columns: {ok} of {len(a.columns)}')
print(f'  differing: {diff} (expected to differ: {sorted(EXPECTED_DIFF)})')
print(f'  UNEXPECTED differing: {unexpected or "none"}')

ra = duckdb.connect(REF, read_only=True).execute('select * from fam_variant').df()
rb = duckdb.connect(SCR, read_only=True).execute('select * from fam_variant').df()
print(f'\nfam_variant rows: {len(ra)} vs {len(rb)}')
print('  reference fits_path:', ra.fits_path.iloc[0])
print('  rebuilt   fits_path:', rb.fits_path.iloc[0])
from pathlib import Path
print('  rebuilt path resolves on disk:', Path(rb.fits_path.iloc[0]).exists())
print('  reference path resolves from the repo root:',
      Path('/sdf/home/r/roodman/notebooks/rubin-work/value_added'
           ).joinpath(ra.fits_path.iloc[0]).resolve().exists())
rowdiff = [c for c in ra.columns
           if c not in EXPECTED_DIFF
           and not ra[c].astype(str).equals(rb[c].astype(str))]
print('  fam_variant columns differing beyond the expected:', rowdiff or 'none')

print('\nRESULT:', 'PASS' if not unexpected and not rowdiff else 'FAIL')
