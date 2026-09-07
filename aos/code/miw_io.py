"""Read the Measured Intrinsic Wavefront (MIW) field maps from parquet.

The pipeline's `intrinsic_split` step writes the MIW as a sampled field grid: columns
``thx_deg``, ``thy_deg`` and ``Z<j>_{OCS,CCS}``, one row per grid cell. `load_miw` reads
that form, which is what every analysis treating the MIW *as data* wants — surface fits,
back-projection, focal-plane maps.

The MIW also exists in a Butler as an `lsst.ip.isr.IntrinsicZernikes` calibration
(dataset type ``intrinsicZernikes``, one per detector, written by the external package's
``ingest_calib_tables.py``). That form is an interpolator queried at arbitrary field
positions, with a per-detector CCS contribution, and it is **deliberately not wrapped
here**: its only consumer is `notebooks/cwfs/aos_miw_cwfs_intrinsic_check.ipynb`, which exists to verify
that the ingested calibration reproduces what ts_wep computes, and so must call
`getIntrinsicZernikes` directly rather than through a wrapper that could mask the
behaviour under test.
"""

import numpy as np
import pandas as pd

# Pupil (annular) Zernike Noll indices carried by the FAM MIW maps: Z4-Z26 omitting
# Z20 and Z21, 21 terms.
JS_DEFAULT = tuple(j for j in range(4, 27) if j not in (20, 21))


def load_miw(source, stride=1, coord='OCS', js=JS_DEFAULT, require=None):
    """Load an MIW field map from parquet as a grid of Zernike coefficients.

    Parameters
    ----------
    source : `str` or `pathlib.Path`
        Path to an ``intrinsic_split_maps`` parquet.
    stride : `int`, optional
        Keep every `stride`-th surviving grid row. 1 (default) keeps all.
    coord : {'OCS', 'CCS'}, optional
        Which frame's columns to read. **OCS** (default) is the telescope-fixed
        component, and is what the static-optics and back-projection analyses want; CCS
        is the camera-fixed component, which rotates with the rotator.
    js : `iterable` [`int`], optional
        Pupil Zernike Noll indices to extract. Defaults to the FAM set, Z4-Z26 omitting
        Z20 and Z21.
    require : `iterable` [`int`] or `None`, optional
        Which Zernikes must be finite for a row to be kept. `None` (default) requires
        all of `js`. Pass a subset to keep rows complete only in the terms an analysis
        actually uses — e.g. ``require=(5, 6, 7, 8)`` for an astigmatism/coma study,
        which keeps 108 extra field points in the current maps where ``Z4_OCS``
        (defocus, CCD-height sensitive) is non-finite but Z5-Z8 are fine.

    Returns
    -------
    pts : `numpy.ndarray`, shape (N, 2)
        Field positions, in degrees, as (thx, thy).
    zk : `numpy.ndarray`, shape (N, max(js) + 1)
        Zernike coefficients in µm of wavefront, **Noll-indexed**: column `j` holds
        Z\\ :sub:`j`, so columns 0-3 (and 20, 21 for the default set) are zero.
    df : `pandas.DataFrame`
        The surviving rows, with `thx_deg`, `thy_deg` and the ``Z<j>_<coord>`` columns.

    Raises
    ------
    KeyError
        If the map lacks any requested ``Z<j>_<coord>`` column.

    Notes
    -----
    Rows with a non-finite value in any *required* Zernike are dropped, so every returned
    row is complete across `require`. Columns outside `require` may still hold NaN, so a
    caller that widens its Zernike use later must widen `require` too.

    The canonical MIW product for downstream use is the ``_5rot``
    ``intrinsic_split_maps.parquet`` with its **OCS** columns; the frozen, version-tracked
    copy is ``aos/calibration/miw/intrinsic_split_maps_v1.parquet``.
    """
    js = list(js)
    df = pd.read_parquet(source)
    cols = [f'Z{j}_{coord}' for j in js]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f'MIW map {source} is missing {len(missing)} column(s), '
                       f'first few: {missing[:4]}')

    req_cols = [f'Z{j}_{coord}' for j in (js if require is None else list(require))]
    ok = np.all(np.isfinite(df[req_cols].to_numpy()), axis=1)
    df = df[ok].iloc[::stride].reset_index(drop=True)

    pts = np.column_stack([df['thx_deg'].to_numpy(), df['thy_deg'].to_numpy()])
    zk = np.zeros((len(df), max(js) + 1))
    for j in js:
        zk[:, j] = df[f'Z{j}_{coord}'].to_numpy()
    return pts, zk, df
