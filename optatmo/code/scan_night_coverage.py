"""Per-visit CCD coverage for a night's in-focus exposures (fast, registry only).

Counts distinct detectors with a preliminary_visit_image per visit (= processed
CCDs, which equals the single_visit_star coverage), to find the best-covered
in-focus exposures. Run on the USDF: python code/scan_night_coverage.py [day_obs]
"""
import sys
from collections import defaultdict
from lsst.daf.butler import Butler

DAY = int(sys.argv[1]) if len(sys.argv) > 1 else 20260513
COLL = 'LSSTCam/runs/nightlyValidation'
b = Butler('/repo/main')

per_visit = defaultdict(set)
for r in b.registry.queryDatasets('preliminary_visit_image', collections=COLL,
                                  where=f"instrument='LSSTCam' and exposure.day_obs={DAY}"):
    per_visit[r.dataId['visit']].add(r.dataId['detector'])

counts = sorted(((v, len(d)) for v, d in per_visit.items()),
                key=lambda t: -t[1])
print(f'day {DAY}: {len(counts)} in-focus visits with preliminary_visit_image')
print(f'{"visit":>14} {"seq":>5} {"nCCD":>5}')
for v, n in counts:
    print(f'{v:>14} {v % 1000:>5} {n:>5}')
if counts:
    ns = [n for _, n in counts]
    import numpy as np
    print(f'\nnCCD: max {max(ns)}, median {int(np.median(ns))}, min {min(ns)}; '
          f'visits with >=180 CCDs: {sum(n >= 180 for n in ns)}')
