"""Night 3, part A: are the 2026-11 BLOCK-T746 ±4 mm camera steps giant donuts?

The value_added database ends before these nights and the EFD was unreachable, so the
candidates come from the ConsDB ``focus_z`` (camera hexapod z, mm) alone. The M2 step is
unknown; the donut radius tells a 4 + 4 mm giant (science outer radius about 338 pixels)
from a camera-only step. Each exposure goes through `night2_review.image_task`: minimal
4×4-binned ISR on its four detectors, the radii, star density and the final-rule count.

Usage, from the repo root::

    python aos/giant_donut_pupil/code/night3_check.py
"""
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import night2_review as nr  # noqa: E402
from rubinwork.common import study_run  # noqa: E402

STUDY = 'aos/giant_donut_pupil'

# visit, side, camera step from the in-focus reference (mm, ConsDB focus_z), RA, Dec (deg)
CANDIDATES = [
    (2026110700366, 'intra', -3.80, 55.23, -44.94),
    (2026110700388, 'extra', +3.99, 55.23, -44.94),
    (2026110800014, 'intra', -4.00, 37.50, -45.00),
    (2026110800036, 'extra', +3.94, 37.50, -45.00),
    (2026110800147, 'extra', +3.75, 37.50, -45.00),
    (2026110800152, 'intra', -4.01, 37.50, -45.00),
    (2026110800168, 'intra', -4.14, 37.50, -45.00),
    (2026110800190, 'extra', +3.98, 37.50, -45.00),
]


def main():
    run = study_run(STUDY, 'night3_check').path
    rows = [nr.image_task((f'{v}', v, side, 4.0, ra, dec, str(run)))
            for v, side, _, ra, dec in CANDIDATES]
    t = pd.DataFrame(rows).merge(
        pd.DataFrame(CANDIDATES, columns=['visit', 'side', 'cam_step_mm', 's_ra', 's_dec']),
        on=['visit', 'side'])
    t.to_parquet(run / 'images.parquet')
    t.to_csv(run / 'images.csv', index=False, float_format='%.4g')
    print('done', run, flush=True)


if __name__ == '__main__':
    main()
