"""The ``atmo_sim_moments`` product: moments measured on ``atmo_sim`` visits.

Per simulated visit, two sets of tables:

- the ``guider_moments`` builder's three tables (``moments``, ``stars``, ``metrics``),
  produced by running that builder unchanged on the ``atmo_sim`` Butler collection with
  the ``guider_moments`` v6 tracker and moment settings. Same schema and frames as the
  data: arcsec**2, each guide sensor's Data Visualization Coordinate System (DVCS) frame;
- ``ccd_psf``: one row per science CCD, GalSim HSM adaptive moments in the CCD pixel
  frame, with the Rubin Consolidated Database (ConsDB) ``ccdvisit1_quicklook`` columns
  ``psf_sigma``, ``psf_ixx``, ``psf_iyy``, ``psf_ixy`` (pix, pix**2), so it reads like
  `rubinwork.common.psf_moments_consdb`.

Partitioned by visit, ``products/atmo_sim_moments/v1/visits/<expId>/``.

Build one visit, after ``atmo_sim``::

    python -m rubinwork.products.atmo_sim_moments.build --day-obs 20260706 --seq-num 688

Read::

    from rubinwork.products import atmo_sim_moments
    mom = atmo_sim_moments.load_table("moments", 20260706, 688)
"""

from .reader import (  # noqa: F401
    PRODUCT,
    PRODUCT_VARIANT,
    TABLES,
    VISITS_DIR,
    load_table,
    variant_config,
    variants,
    visit_dir,
    visits,
)
