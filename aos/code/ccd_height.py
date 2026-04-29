"""CCD focal-plane height map helpers.

Reads the LSST focal-plane height-map FITS file produced by SLAC metrology
(per-sensor BinTableHDUs with X_CCS / Y_CCS / Z_CCS_MEASURED / Z_CCS_MODEL),
provides a KNN interpolator height(fpx, fpy), evaluates per-donut
focal-plane coordinates via cameraGeom, and converts a local piston
(height) into a defocus-Zernike (Z4) contribution using the empirical
`HEIGHT_TO_Z4_UM_PER_MM` = 15 μm/mm constant (Guillem's estimate).

These helpers were originally inline in ``aos/intrinsics_checkZ4.ipynb``;
they are extracted here so that other notebooks (e.g. the
build_measured_intrinsic flow) can import them.

Notes
-----
* The FITS reader applies the X<->Y swap from the CCS frame to the
  focal-plane convention: ``fpx = Y_CCS`` and ``fpy = X_CCS``.  This
  mirrors the convention used elsewhere in the repo (e.g.
  ``plot_Z_FAM_August-archive``).
* `get_height_interpolator` returns a callable that takes (fpx, fpy)
  arrays in mm and returns the height in whatever units `Z_CCS_*` stores
  in the FITS file.  Aaron's existing code labels that variable as
  ``height_mm`` and applies the 15 μm/mm constant directly, so we keep
  that interpretation here.
"""

import re
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table


# Empirical conversion: μm of Z4 wavefront defocus per mm of local piston.
# Source: Guillem's estimate (≈ 0.15 μm Z4 per 10 μm of local height).
HEIGHT_TO_Z4_UM_PER_MM = 15.0


# ----------------------------------------------------------------------
# Pixel -> focal-plane coordinates (per donut)
# ----------------------------------------------------------------------

def compute_fp_coords(donut_df, camera,
                      x_col='centroid_x_intra',
                      y_col='centroid_y_intra',
                      det_col='detector'):
    """Per-donut focal-plane (mm) coordinates via cameraGeom.

    Groups donuts by detector and calls ``common.camera_utils.pixel_to_focal``
    once per sensor (vectorised over all donuts on that sensor).

    Parameters
    ----------
    donut_df : pandas.DataFrame or astropy.QTable
        Must contain `x_col`, `y_col`, and `det_col`.
    camera : lsst.afw.cameraGeom.Camera
    x_col, y_col, det_col : str

    Returns
    -------
    fpx_mm, fpy_mm : ndarray
    """
    # Lazy import — the LSST stack is only available on RSP
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from common.camera_utils import pixel_to_focal  # noqa: E402

    det_names = np.asarray(donut_df[det_col]).astype(str)
    x_pix = np.asarray(donut_df[x_col], dtype=float)
    y_pix = np.asarray(donut_df[y_col], dtype=float)
    fpx = np.full_like(x_pix, np.nan)
    fpy = np.full_like(y_pix, np.nan)
    for det in camera:
        name = det.getName()
        mask = (det_names == name)
        if not np.any(mask):
            continue
        fx, fy = pixel_to_focal(x_pix[mask], y_pix[mask], det)
        fpx[mask] = fx
        fpy[mask] = fy
    return fpx, fpy


# ----------------------------------------------------------------------
# Height-map I/O + KNN interpolator
# ----------------------------------------------------------------------

def make_metrology_table(file, rsid=None):
    """Build a per-point focal-plane metrology table from the height map.

    Each per-sensor BinTableHDU is concatenated into one table with
    columns ``fpx, fpy, z_mod, z_meas, det``.  Note the x/y swap from
    CCS to focal-plane convention: ``fpx = Y_CCS``, ``fpy = X_CCS``.

    Parameters
    ----------
    file : str or Path
        FITS file path (e.g. ``LSST_FP_cold_b_measurement_4col_bysurface.fits``).
    rsid : str, optional
        Restrict to a single raft-sensor identifier (e.g. ``'R22S11'``).

    Returns
    -------
    astropy.table.Table
    """
    rows = []
    with fits.open(str(file)) as hdulist:
        for hdu in hdulist:
            if not isinstance(hdu, fits.BinTableHDU):
                continue
            extname = hdu.header.get('EXTNAME', '')
            if rsid is not None and extname != rsid:
                continue
            if rsid is None and not re.fullmatch(r'R\d\dS\d\d', extname):
                continue
            det_label = re.sub(r'(R\d\d)(S\d\d)', r'\1_\2', extname)
            tab = Table(hdu.data)
            for x, y, z_mod, z_meas in zip(
                tab['X_CCS'], tab['Y_CCS'],
                tab['Z_CCS_MODEL'], tab['Z_CCS_MEASURED'],
            ):
                rows.append([y, x, z_mod, z_meas, det_label])
    return Table(rows=rows, names=['fpx', 'fpy', 'z_mod', 'z_meas', 'det'])


def get_height_interpolator(metrology_table, k=3,
                            weight_type='distance',
                            ztype='measured'):
    """KNN interpolator for focal-plane height.

    Returns ``interp_func(fpx, fpy)`` which accepts ndarrays and returns
    the height (in the FITS file's native Z_CCS unit; see module
    docstring).
    """
    from sklearn.neighbors import KNeighborsRegressor  # local import

    x = np.column_stack((metrology_table['fpx'], metrology_table['fpy']))
    if ztype == 'measured':
        y = np.asarray(metrology_table['z_meas'], dtype=float)
    elif ztype == 'model':
        y = np.asarray(metrology_table['z_mod'], dtype=float)
    else:
        raise ValueError("ztype must be 'measured' or 'model'")
    knn = KNeighborsRegressor(n_neighbors=k, weights=weight_type)
    knn.fit(x, y)

    def interp_func(fpx, fpy):
        fpx = np.atleast_1d(np.asarray(fpx, dtype=float))
        fpy = np.atleast_1d(np.asarray(fpy, dtype=float))
        pts = np.column_stack((fpx, fpy))
        return knn.predict(pts)

    return interp_func


def height_to_z4(height_mm, factor=HEIGHT_TO_Z4_UM_PER_MM):
    """Convert local CCD piston (mm) to a defocus-Z4 contribution (μm)."""
    return factor * np.asarray(height_mm, dtype=float)


# ----------------------------------------------------------------------
# Per-CCD x<->y transpose helpers (used to mirror the pipeline's
# 'intrinsic_transpose_bug' when computing the height contribution that
# matches the *intrinsic* Zernike tabulation rather than the data.)
# ----------------------------------------------------------------------

def ccd_centers_fp(camera):
    """Return {detector_name: (cx_mm, cy_mm)} for every detector in `camera`."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from common.camera_utils import pixel_to_focal  # noqa: E402

    centers = {}
    for det in camera:
        c = det.getBBox().getCenter()
        cx, cy = pixel_to_focal(np.array([c.getX()]),
                                np.array([c.getY()]), det)
        centers[det.getName()] = (float(cx[0]), float(cy[0]))
    return centers


def transpose_around_ccd_centers(fpx_mm, fpy_mm, det_names, camera):
    """Per-CCD x<->y transpose around each detector's center.

    Mirrors the pipeline's per-CCD intrinsic Zernike calculation bug:
    relative to the detector center, the x and y offsets are swapped.

    Parameters
    ----------
    fpx_mm, fpy_mm : ndarray (n_donuts,)
        Focal-plane coordinates (already in DVCS mm) for every donut.
    det_names : array-like (n_donuts,) of str
    camera : lsst.afw.cameraGeom.Camera

    Returns
    -------
    fpx_swap, fpy_swap : ndarray (n_donuts,)
    """
    centers = ccd_centers_fp(camera)
    fpx_arr = np.asarray(fpx_mm, dtype=float)
    fpy_arr = np.asarray(fpy_mm, dtype=float)
    fpx_swap = np.full_like(fpx_arr, np.nan)
    fpy_swap = np.full_like(fpy_arr, np.nan)
    det_arr = np.asarray(det_names).astype(str)
    for name, (cx, cy) in centers.items():
        mask = (det_arr == name)
        if not np.any(mask):
            continue
        dx = fpx_arr[mask] - cx
        dy = fpy_arr[mask] - cy
        fpx_swap[mask] = cx + dy
        fpy_swap[mask] = cy + dx
    return fpx_swap, fpy_swap
