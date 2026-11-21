"""The ``batoid_intrinsic`` product: the wavefront of a batoid optical model, in focus and
at the donut defocus configurations.

Per band, on a Gauss-Legendre science field grid and on a grid over the corner
wavefront-sensor (CWFS) half-chips:

- ``gq_infocus``: in-focus optical path difference (OPD) Zernikes, ``batoid.zernikeGQ``;
- ``ta``: defocused transverse-aberration Zernikes, ``batoid.zernikeTA``, the reference
  ts_wep and Danish build their donut model from;
- ``gq_defocus``: defocused OPD Zernikes.

The defocus configurations are ``fam`` (camera ±1.5 mm, science grid), ``cwfs``
(detector ±1.5 mm, each corner half-chip on its own side) and ``giant`` (M2 and camera
±4 mm, science grid). The grid is fixed in the Camera Coordinate System (CCS); the table
carries field angles and Zernikes in CCS and in the Optical Coordinate System (OCS),
OCS = R(+rotator angle) CCS, at the variant's ``rotator_angle_deg``.

A build holds:

``zernikes.parquet``
    one row per band, kind, configuration, side and field point; Noll 1..jmax, µm, CCS
    and OCS.
``dz.fits``
    double-Zernike fits (OCS, as the optical-state fits) of each science-grid set, with their residual RMS.

Read a build::

    from rubinwork.products import batoid_intrinsic
    df = batoid_intrinsic.load("v4_speed", band="r", kind="ta", config="fam")

Build one, from the repository root::

    python -m rubinwork.products.batoid_intrinsic.builders.build --variant v4_speed
"""

from .reader import (KINDS, PRODUCT, build_dir, evaluate_dz, load, load_dz, noll_columns,
                     variant_config, variants)

__all__ = ["KINDS", "PRODUCT", "build_dir", "evaluate_dz", "load", "load_dz",
           "noll_columns", "variant_config", "variants"]
