"""The ``fitted_smatrix`` product: double-Zernike sensitivity matrices from Danish fits of
ray-traced donuts, in the ts_ofc formats.

The ts_ofc v13 matrix is the in-focus optical path difference (OPD). These are instead the
response of the Zernikes a Danish fit set reports: ray-traced batoid donuts at 940 field
points x 50 DOF x 5 amplitudes (the ``smatrix/sparse_fit`` scan), with each fitted
Zernike's slope fitted over the full-array mode (FAM) field. One variant per fit set:
``dense21``, ``sparse_no22`` and ``sparse_with22``.

A build holds:

``smatrix_dz_31_29_50_r_<variant>.npy`` and ``.fits``
    (31 field k, 29 pupil Noll j, 50 DOF), µm of wavefront per µm of DOF (per deg for
    the tilts), ts_ofc DOF order and sign.
``noll.txt``
    the fitted pupil Noll indices; other pupil rows are zero.
``ofc/``
    the v13 OFC config directory with ``sensitivity_matrix/lsst_sensitivity_dz_31_29_50.yaml``
    replaced.

Read a build::

    from rubinwork.products import fitted_smatrix
    smatrix, noll = fitted_smatrix.load("sparse_no22")
    se = fitted_smatrix.state_estimator("sparse_no22", dof_set="all_50")

Build one, from the repository root::

    python -m rubinwork.products.fitted_smatrix.builders.build --variant sparse_no22
"""

from .reader import (PRODUCT, build_dir, load, ofc_config_dir, state_estimator,
                     variant_config, variants)

__all__ = ["PRODUCT", "build_dir", "load", "ofc_config_dir", "state_estimator",
           "variant_config", "variants"]
