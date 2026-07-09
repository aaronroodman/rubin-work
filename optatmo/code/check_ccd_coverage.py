"""Check per-CCD coverage of the PSF-star dataset for the in-focus visits.

For each in-focus visit: how many of the 189 science CCDs have single_visit_star
sources (and point sources), vs how many have a preliminary_visit_image — to see
whether the ~96/189 coverage is missing source processing or missing images
(relevant to using focal-plane focus/CCD-height diversity for the Z4 degeneracy).

Run on the USDF from rubin-work/optatmo:  python code/check_ccd_coverage.py
"""
import numpy as np
from lsst.daf.butler import Butler
from lsst.obs.lsst import LsstCam

COLL = 'LSSTCam/runs/nightlyValidation'
DAY = 20260513
VISITS = [2026051300031, 2026051300034]

b = Butler('/repo/main')
n_sci = len([d for d in LsstCam.getCamera()
             if d.getType().name == 'SCIENCE'])
print(f'LSSTCam science CCDs: {n_sci}')

for V in VISITS:
    st = b.get('single_visit_star', collections=COLL,
               instrument='LSSTCam', visit=V).to_pandas()
    det_all = st.detector.nunique()
    ps = st[st.detect_isPrimary & (st.extendedness < 0.5)
            & (st.psfFlux / st.psfFluxErr > 80) & (~st.pixelFlags_saturated)]
    det_ps = ps.detector.nunique()
    # how many detectors have a calibrated image?
    img = list(b.registry.queryDatasets('preliminary_visit_image', collections=COLL,
                                         where=f"instrument='LSSTCam' and visit={V}"))
    det_img = len({r.dataId['detector'] for r in img})
    have = set(ps.detector.unique())
    missing = sorted(set(range(189)) - have)
    print(f'\nvisit {V}:')
    print(f'  detectors with single_visit_star sources : {det_all}/189')
    print(f'  detectors with good point sources (>0)   : {det_ps}/189')
    print(f'  detectors with preliminary_visit_image   : {det_img}/189')
    print(f'  # point sources per detector: median '
          f'{int(ps.groupby("detector").size().median())}, '
          f'min {int(ps.groupby("detector").size().min())}')
    print(f'  missing detector ids ({len(missing)}): {missing[:40]}')
