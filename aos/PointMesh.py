"""
PointMesh: A 2D scattered-data interpolation and visualization toolkit.

This module provides the ``PointMesh`` class, which encapsulates a cloud of
(x, y, z) points on a focal-plane-like domain, builds gridded interpolations
with several selectable methods, and supports mesh arithmetic (differencing,
merging, fitting tip/tilt/piston offsets, etc.).

Supported interpolation methods
-------------------------------
* ``sbs``     -- scipy SmoothBivariateSpline
* ``rbf``     -- scipy RBF (radial basis function)
* ``grid``    -- scipy griddata (linear triangulation)
* ``idw``     -- inverse-distance-weighted via kNN (built-in implementation)
* ``tmean``   -- per-grid truncated mean (single scalar per grid)
* ``bmedian`` -- binned median (one value per grid cell)

Aaron Roodman (C) SLAC National Accelerator Laboratory, Stanford University
2012-2026.  Modernised rewrite 2026.
"""

from __future__ import annotations

import bisect
from typing import Optional, Sequence, Tuple

import numpy as np
import numpy.lib.index_tricks as itricks
from scipy.interpolate import SmoothBivariateSpline, RBFInterpolator, griddata
from scipy.spatial import cKDTree
from scipy import stats
import statsmodels.api as sm

try:
    from matplotlib import cm, colors
    from matplotlib import pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper: Inverse Distance Weighted interpolator
# ---------------------------------------------------------------------------

class IDWInterp:
    """Inverse-distance-weighted interpolator using k-nearest neighbours.

    Parameters
    ----------
    x : numpy.ndarray
        1-D array of x coordinates of the known points.
    y : numpy.ndarray
        1-D array of y coordinates of the known points.
    z : numpy.ndarray
        1-D array of values at the known points.
    kNN : int, optional
        Number of nearest neighbours to use (default 4).
    power : float, optional
        Power parameter for the inverse-distance weight ``w = 1/d^power``
        (default 1).
    epsilon : float, optional
        Small constant added to all distances to avoid division by zero and
        to provide a smoothing length scale (default 0.0).

    Notes
    -----
    The IDW formula (for ``power=1``) is::

        value = sum(z_i / d_i) / sum(1 / d_i)

    where the sums run over the ``kNN`` nearest neighbours and ``d_i`` is the
    Euclidean distance (plus ``epsilon``) from the query point to neighbour *i*.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        kNN: int = 4,
        power: float = 1,
        epsilon: float = 0.0,
    ):
        self.kNN = kNN
        self.power = power
        self.epsilon = epsilon
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.z = np.asarray(z, dtype=np.float64)
        xy = np.column_stack([self.x, self.y])
        self.kdtree = cKDTree(xy)
        self._usekNN = min(self.kNN, len(self.z))

    def ev(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate the interpolant at query points.

        Parameters
        ----------
        x, y : numpy.ndarray
            1-D arrays of query coordinates.

        Returns
        -------
        numpy.ndarray
            Interpolated values, same length as *x*.
        """
        xy = np.column_stack([np.asarray(x, dtype=np.float64),
                              np.asarray(y, dtype=np.float64)])
        d, idx = self.kdtree.query(xy, self._usekNN)
        # Ensure 2-D even when _usekNN == 1
        if d.ndim == 1:
            d = d[:, np.newaxis]
            idx = idx[:, np.newaxis]
        z_nn = self.z[idx]
        d = d + self.epsilon
        w = 1.0 / np.power(d, self.power)
        return (z_nn * w).sum(axis=1) / w.sum(axis=1)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

_VALID_METHODS = {"sbs", "rbf", "grid", "idw", "tmean", "bmedian"}


class PointMesh:
    """A 2-D scattered-data mesh with gridded interpolation.

    The mesh holds a point cloud ``(x, y, z)`` with optional per-point
    weights ``w``, builds an interpolation on a regular Cartesian grid
    covering a circular focal-plane region, and provides utilities for
    mesh arithmetic, fitting, and plotting.

    Parameters
    ----------
    x : numpy.ndarray
        1-D array of x coordinates.
    y : numpy.ndarray
        1-D array of y coordinates.
    z : numpy.ndarray
        1-D array of values (e.g. wavefront error) at each point.
    w : numpy.ndarray or None, optional
        1-D array of per-point weights.  Defaults to all ones.
    radius : float, optional
        Focal-plane radius used for the grid domain and vignetting mask
        (default 1.75, in degrees).
    n_bins : int, optional
        Number of grid cells along each axis (default 72).
    method : str, optional
        Interpolation method (default ``'bmedian'``).  One of
        ``'sbs'``, ``'rbf'``, ``'grid'``, ``'idw'``, ``'tmean'``,
        ``'bmedian'``.
    method_val : sequence or None, optional
        Extra parameters passed to the interpolation method.  For ``'idw'``
        this is ``(kNN, epsilon)``; for ``'sbs'`` the smoothing factor *s*
        can be supplied as a single-element sequence.
    title : str, optional
        Descriptive title stored with the mesh.

    Attributes
    ----------
    x, y, z, w : numpy.ndarray
        The point-cloud data.
    radius : float
        Domain radius.
    n_bins : int
        Grid resolution along each axis.
    method : str
        Current interpolation method name.
    method_val : sequence or None
        Current method parameters.
    interp_values : numpy.ndarray or None
        2-D ``(n_bins, n_bins)`` array of interpolated values on the grid
        (set after :pymeth:`make_interpolation` runs).
    interp_present : bool
        Whether an interpolation has been built.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        w: Optional[np.ndarray] = None,
        radius: float = 1.75,
        n_bins: int = 72,
        method: str = "bmedian",
        method_val: Optional[Sequence] = None,
        title: str = "",
    ):
        self.x = np.asarray(x, dtype=np.float64).ravel()
        self.y = np.asarray(y, dtype=np.float64).ravel()
        self.z = np.asarray(z, dtype=np.float64).ravel()
        if w is not None:
            self.w = np.asarray(w, dtype=np.float64).ravel()
        else:
            self.w = np.ones_like(self.z)

        self.radius = float(radius)
        self.n_bins = int(n_bins)
        self.title = title

        # Interpolation state
        self.method = method
        self.method_val = method_val
        self.interp_present = False

        # Grid arrays (filled by make_grid / make_interpolation)
        self.x_grid: Optional[np.ndarray] = None
        self.y_grid: Optional[np.ndarray] = None
        self.x_edge: Optional[np.ndarray] = None
        self.y_edge: Optional[np.ndarray] = None
        self.interp_values: Optional[np.ndarray] = None

        # Per-method cached objects
        self._interp_spline = None
        self._interp_rbf = None
        self._interp_idw: Optional[IDWInterp] = None
        self._interp_tmean: float = 0.0
        self._interp_tstd: float = 0.0
        self._interp_bmedian: Optional[np.ndarray] = None
        self._interp_bnentry: Optional[np.ndarray] = None
        self._interp_bmad: Optional[np.ndarray] = None

        if method in _VALID_METHODS:
            self.interp_present = True
            self.make_interpolation(self.method, self.method_val)

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_table(
        cls,
        table,
        x_col: str,
        y_col: str,
        z_col: str,
        w_col: Optional[str] = None,
        **kwargs,
    ) -> "PointMesh":
        """Create a PointMesh from an astropy QTable (or any column-indexable table).

        Parameters
        ----------
        table : astropy.table.QTable or similar
            Input table.
        x_col, y_col, z_col : str
            Column names for x, y, and z data.
        w_col : str or None, optional
            Column name for weights.  If ``None``, unit weights are used.
        **kwargs
            Forwarded to the ``PointMesh`` constructor (e.g. ``radius``,
            ``n_bins``, ``method``).

        Returns
        -------
        PointMesh
        """
        x = np.asarray(table[x_col])
        y = np.asarray(table[y_col])
        z = np.asarray(table[z_col])
        w = np.asarray(table[w_col]) if w_col is not None else None
        return cls(x, y, z, w=w, **kwargs)

    # ------------------------------------------------------------------
    # Pickle support
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        """Return picklable state, omitting non-picklable interpolation objects."""
        keys = [
            "x", "y", "z", "w",
            "radius", "n_bins", "method", "method_val", "title",
        ]
        return {k: self.__dict__[k] for k in keys}

    def __setstate__(self, state: dict):
        """Restore from pickled state, rebuilding interpolation."""
        for k, v in state.items():
            setattr(self, k, v)
        self.interp_present = False
        self.x_grid = None
        self.y_grid = None
        self.x_edge = None
        self.y_edge = None
        self.interp_values = None
        self._interp_spline = None
        self._interp_rbf = None
        self._interp_idw = None
        self._interp_tmean = 0.0
        self._interp_tstd = 0.0
        self._interp_bmedian = None
        self._interp_bnentry = None
        self._interp_bmad = None
        if self.method in _VALID_METHODS:
            self.interp_present = True
            self.make_interpolation(self.method, self.method_val)

    # ------------------------------------------------------------------
    # Grid construction
    # ------------------------------------------------------------------

    @staticmethod
    def make_grid(
        n_bins: int, radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build a regular Cartesian grid of cell centres and edges.

        Parameters
        ----------
        n_bins : int
            Number of cells along each axis.
        radius : float
            Half-width of the grid domain; the grid spans
            ``[-radius, radius]`` in both x and y.

        Returns
        -------
        x_grid : numpy.ndarray
            2-D array of cell-centre x coordinates, shape ``(n_bins, n_bins)``.
        y_grid : numpy.ndarray
            2-D array of cell-centre y coordinates.
        x_edge : numpy.ndarray
            2-D array of cell-edge x coordinates, shape ``(n_bins+1, n_bins+1)``.
        y_edge : numpy.ndarray
            2-D array of cell-edge y coordinates.
        """
        lo, hi = -radius, radius
        delta = (hi - lo) / n_bins
        cen_lo = lo + 0.5 * delta
        cen_hi = hi - 0.5 * delta

        y_grid, x_grid = itricks.mgrid[
            cen_lo : cen_hi : 1j * n_bins,
            cen_lo : cen_hi : 1j * n_bins,
        ]
        y_edge, x_edge = itricks.mgrid[
            lo : hi : 1j * (n_bins + 1),
            lo : hi : 1j * (n_bins + 1),
        ]
        return x_grid, y_grid, x_edge, y_edge

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def make_interpolation(
        self, method: str = "bmedian", method_val: Optional[Sequence] = None
    ) -> None:
        """Build the interpolation on the focal-plane grid.

        Parameters
        ----------
        method : str
            Interpolation method name.
        method_val : sequence or None, optional
            Extra parameters for the method.
        """
        self.method = method
        self.method_val = method_val
        n = self.n_bins
        r = self.radius

        x_grid, y_grid, x_edge, y_edge = self.make_grid(n, r)
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.x_edge = x_edge
        self.y_edge = y_edge

        # Reset cached objects
        self._interp_spline = None
        self._interp_rbf = None
        self._interp_idw = None
        self._interp_tmean = 0.0
        self._interp_tstd = 0.0
        self._interp_bmedian = None
        self._interp_bnentry = None
        self._interp_bmad = None

        x_data = self.x
        y_data = self.y
        z_data = self.z
        npts = len(z_data)

        lo, hi = -r, r

        if npts < 5:
            self.interp_values = np.zeros((n, n))
            self.interp_present = False
            return

        flat_x = x_grid.ravel()
        flat_y = y_grid.ravel()

        if method == "sbs":
            s_val = method_val[0] if method_val is not None and len(method_val) > 0 else 1.0e6
            if npts > 600:
                kx, ky = 4, 4
            elif npts >= 100:
                kx, ky = 3, 3
            elif npts > 9:
                kx, ky = 2, 2
            else:
                kx, ky = 1, 1
                s_val = 1.0e7
            self._interp_spline = SmoothBivariateSpline(
                x_data, y_data, z_data,
                bbox=[lo, hi, lo, hi], kx=kx, ky=ky, s=s_val,
            )
            self.interp_values = self._interp_spline.ev(flat_x, flat_y).reshape((n, n))

        elif method == "rbf":
            xy_data = np.column_stack([x_data, y_data])
            self._interp_rbf = RBFInterpolator(xy_data, z_data)
            xy_query = np.column_stack([flat_x, flat_y])
            self.interp_values = self._interp_rbf(xy_query).reshape((n, n))

        elif method == "tmean":
            zstd = stats.tstd(z_data)
            zmean = stats.tmean(z_data)
            lo_cut = zmean - 3.0 * zstd
            hi_cut = zmean + 3.0 * zstd
            self._interp_tmean = stats.tmean(z_data, (lo_cut, hi_cut))
            self._interp_tstd = stats.tstd(z_data, (lo_cut, hi_cut))
            self.interp_values = self._interp_tmean * np.ones((n, n))

        elif method == "bmedian":
            bmed = np.zeros((n, n))
            bnentry = np.zeros((n, n))
            bmad = np.zeros((n, n))
            xbin = np.digitize(x_data, x_edge[0, :]) - 1
            ybin = np.digitize(y_data, y_edge[:, 0]) - 1
            for i in range(n):
                for j in range(n):
                    mask = (xbin == i) & (ybin == j)
                    z_here = z_data[mask]
                    nentry = len(z_here)
                    if nentry >= 1:
                        med = np.median(z_here)
                        bmed[j, i] = med
                        bnentry[j, i] = nentry
                        bmad[j, i] = np.median(np.abs(z_here - med))
            self._interp_bmedian = bmed
            self._interp_bnentry = bnentry
            self._interp_bmad = bmad
            self.interp_values = bmed.copy()

        elif method == "grid":
            Z = griddata(
                (x_data, y_data), z_data, (flat_x, flat_y), method="linear"
            )
            self.interp_values = Z.reshape((n, n))

        elif method == "idw":
            if method_val is not None and len(method_val) >= 2:
                kNN = int(method_val[0])
                eps = float(method_val[1])
            else:
                kNN = 4
                eps = 1.0
            self._interp_idw = IDWInterp(x_data, y_data, z_data, kNN=kNN, epsilon=eps)
            self.interp_values = self._interp_idw.ev(flat_x, flat_y).reshape((n, n))

        else:
            self.interp_values = np.zeros((n, n))

    # ------------------------------------------------------------------
    # Point-wise evaluation
    # ------------------------------------------------------------------

    def do_interp(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate the current interpolation at arbitrary query points.

        Parameters
        ----------
        x, y : numpy.ndarray
            Query coordinates (1-D arrays or scalars).

        Returns
        -------
        numpy.ndarray
            Interpolated values at the query locations.
        """
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))

        if self.method == "sbs":
            if self._interp_spline is not None:
                return self._interp_spline.ev(x, y)
            return np.zeros_like(x)

        if self.method == "rbf":
            if self._interp_rbf is not None:
                xy = np.column_stack([x, y])
                return self._interp_rbf(xy)
            return np.zeros_like(x)

        if self.method == "idw":
            if self._interp_idw is not None:
                return self._interp_idw.ev(x, y)
            return np.zeros_like(x)

        if self.method == "tmean":
            return self._interp_tmean * np.ones_like(x)

        if self.method == "bmedian":
            if self._interp_bmedian is not None and self.x_edge is not None:
                xbin = np.clip(np.digitize(x, self.x_edge[0, :]) - 1, 0, self.n_bins - 1)
                ybin = np.clip(np.digitize(y, self.y_edge[:, 0]) - 1, 0, self.n_bins - 1)
                return self._interp_bmedian[ybin, xbin]
            return np.zeros_like(x)

        if self.method == "grid":
            # griddata must be recomputed each time
            return griddata(
                (self.x, self.y), self.z, (x, y), method="linear"
            )

        return np.zeros_like(x)

    # ------------------------------------------------------------------
    # Mesh fitting / comparison
    # ------------------------------------------------------------------

    def fit_mesh(
        self, other: "PointMesh", method: str = "OLS"
    ):
        """Fit offset + tip + tilt between this mesh and another.

        Computes ``z_diff = other.z - self.do_interp(other.x, other.y)``
        and fits the model ``z_diff = thetax * y + thetay * x + delta``
        using ordinary or robust least squares.

        Parameters
        ----------
        other : PointMesh
            The other mesh whose discrete points are compared against this
            mesh's interpolation.
        method : str, optional
            ``'OLS'`` for ordinary least squares or ``'RLM'`` for robust
            linear model (default ``'OLS'``).

        Returns
        -------
        results : statsmodels results object or None
            The fitted model results (``results.params`` contains
            ``[thetax, thetay, delta]``).  ``None`` if there are fewer
            than 4 points.
        z_diff : numpy.ndarray
            Residual vector.
        x_col : numpy.ndarray
            x coordinates of the comparison points.
        y_col : numpy.ndarray
            y coordinates of the comparison points.

        Notes
        -----
        When ``method='RLM'``, the robust weights from the fit are written
        into ``other.w``.
        """
        interp_at_other = self.do_interp(other.x, other.y)
        z_diff = other.z - interp_at_other
        x_col = other.x.copy()
        y_col = other.y.copy()

        npts = len(z_diff)
        if npts <= 3:
            return None, z_diff, x_col, y_col

        # Design matrix: z_diff = thetax*y + thetay*x + delta
        a_matrix = np.column_stack([y_col, x_col, np.ones(npts)])

        if method == "OLS":
            model = sm.OLS(z_diff, a_matrix)
        elif method == "RLM":
            model = sm.RLM(z_diff, a_matrix)
        else:
            raise ValueError(f"Unknown fit method '{method}'; use 'OLS' or 'RLM'.")

        try:
            results = model.fit()
        except Exception:
            results = None

        if results is not None and method == "RLM":
            other.w = results.weights.copy()

        return results, z_diff, x_col, y_col

    def diff_mesh(self, other: "PointMesh"):
        """Compute the difference between another mesh's points and this interpolation.

        Parameters
        ----------
        other : PointMesh
            The other mesh.

        Returns
        -------
        z_diff : numpy.ndarray
            ``other.z - self.do_interp(other.x, other.y)``.
        x_col : numpy.ndarray
            x coordinates.
        y_col : numpy.ndarray
            y coordinates.
        """
        interp_at_other = self.do_interp(other.x, other.y)
        z_diff = other.z - interp_at_other
        return z_diff, other.x.copy(), other.y.copy()

    # ------------------------------------------------------------------
    # Mesh manipulation
    # ------------------------------------------------------------------

    def adjust_mesh(
        self,
        thetax: float,
        thetay: float,
        delta: float,
        angle_conversion: float = 1.0,
    ) -> None:
        """Remove a rotation and piston offset from the z values.

        Applies::

            z_new = z - (thetay / angle_conversion * x
                         + thetax / angle_conversion * y + delta)

        and rebuilds the interpolation.

        Parameters
        ----------
        thetax : float
            Rotation about the x axis (tip).
        thetay : float
            Rotation about the y axis (tilt).
        delta : float
            Piston offset.
        angle_conversion : float, optional
            Divisor applied to thetax/thetay before multiplication
            (default 1.0, i.e. no conversion).
        """
        self.z = self.z - (
            self.x * thetay / angle_conversion
            + self.y * thetax / angle_conversion
            + delta
        )
        if self.interp_present:
            self.make_interpolation(self.method, self.method_val)

    def merge_mesh(self, other: "PointMesh") -> None:
        """Merge another mesh's points into this one and rebuild interpolation.

        Parameters
        ----------
        other : PointMesh
            The mesh whose points will be appended.
        """
        self.x = np.concatenate([self.x, other.x])
        self.y = np.concatenate([self.y, other.y])
        self.z = np.concatenate([self.z, other.z])
        self.w = np.concatenate([self.w, other.w])
        if self.interp_present:
            self.make_interpolation(self.method, self.method_val)

    def subtract_mesh(self, other: "PointMesh") -> "PointMesh":
        """Create a new mesh equal to ``self - other`` evaluated on this grid.

        Both meshes are evaluated on ``self``'s interpolation grid and the
        difference is stored as the point cloud of a new ``PointMesh``
        (using ``method='grid'``).

        Parameters
        ----------
        other : PointMesh
            The mesh to subtract.

        Returns
        -------
        PointMesh
            New mesh with the difference values.
        """
        if self.x_grid is None:
            self.make_interpolation(self.method, self.method_val)
        xx = self.x_grid.ravel()
        yy = self.y_grid.ravel()
        z_self = self.do_interp(xx, yy)
        z_other = other.do_interp(xx, yy)
        z_diff = z_self - z_other
        return PointMesh(
            xx, yy, z_diff,
            radius=self.radius,
            n_bins=self.n_bins,
            method="grid",
        )

    def cull_mesh(self, min_weight: float) -> None:
        """Remove points with weight below a threshold and rebuild interpolation.

        Parameters
        ----------
        min_weight : float
            Minimum weight; points with ``w <= min_weight`` are discarded.
        """
        keep = self.w > min_weight
        self.x = self.x[keep]
        self.y = self.y[keep]
        self.z = self.z[keep]
        self.w = self.w[keep]
        if self.interp_present:
            self.make_interpolation(self.method, self.method_val)

    def redo_interp(
        self, method: str, method_val: Optional[Sequence] = None
    ) -> None:
        """Change the interpolation method and rebuild.

        Parameters
        ----------
        method : str
            New interpolation method name.
        method_val : sequence or None, optional
            New method parameters.
        """
        self.method = method
        self.method_val = method_val
        self.interp_present = False
        if method in _VALID_METHODS:
            self.interp_present = True
            self.make_interpolation(method, method_val)

    # ------------------------------------------------------------------
    # Weight calculation
    # ------------------------------------------------------------------

    def calc_wgt_dist_to_interp(self, max_dist: float = 0.1) -> None:
        """Set weights based on distance from data to interpolation.

        Points whose absolute residual ``|z - interp(x,y)|`` exceeds
        *max_dist* receive weight 0; others receive weight 1.

        Parameters
        ----------
        max_dist : float, optional
            Maximum allowed absolute residual (default 0.1).
        """
        if not self.interp_present or len(self.z) == 0:
            return
        resid = np.abs(self.z - self.do_interp(self.x, self.y))
        self.w = np.where(resid < max_dist, 1.0, 0.0)

    def calc_wgt_nsig(self, nsigma: float = 3.0) -> None:
        """Set weights by number of sigma from the truncated mean.

        Only meaningful when the current interpolation method is ``'tmean'``.

        Parameters
        ----------
        nsigma : float, optional
            Sigma threshold (default 3.0).
        """
        if len(self.z) == 0 or self._interp_tstd <= 0.0:
            return
        n_sig = np.abs(self.z - self._interp_tmean) / self._interp_tstd
        self.w = np.where(n_sig < nsigma, 1.0, 0.0)

    def calc_wgt_nmad(self, k_nn: int = 100, nmad_cut: float = 4.0) -> np.ndarray:
        """Set weights using the number-of-MAD criterion with k-nearest neighbours.

        For each point, the MAD (median absolute deviation) and median of
        its ``k_nn`` nearest neighbours are computed. Points whose
        deviation from the local median exceeds ``nmad_cut * MAD`` receive
        weight 0.

        Parameters
        ----------
        k_nn : int, optional
            Number of nearest neighbours (default 100).
        nmad_cut : float, optional
            nMAD threshold (default 4.0).

        Returns
        -------
        numpy.ndarray
            Array of nMAD values for every point (useful for diagnostics).
        """
        npts = len(self.z)
        if npts == 0:
            return np.array([])

        xy = np.column_stack([self.x, self.y])
        tree = cKDTree(xy)
        use_k = min(k_nn, npts)

        nmad_arr = np.empty(npts)
        new_w = np.ones(npts)
        for i in range(npts):
            _, idx = tree.query(xy[i], use_k)
            z_nn = self.z[idx]
            med = np.median(z_nn)
            mad = np.median(np.abs(z_nn - med))
            if mad > 0:
                nmad_arr[i] = (self.z[i] - med) / mad
            else:
                nmad_arr[i] = 0.0
            if np.abs(nmad_arr[i]) >= nmad_cut:
                new_w[i] = 0.0

        self.w = new_w
        return nmad_arr

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_2d(
        self,
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
        title: Optional[str] = None,
        cmap=None,
        figsize: Tuple[float, float] = (10, 8),
        ax=None,
    ):
        """Make a 2-D colour plot of the interpolated grid, masked outside the radius.

        Parameters
        ----------
        zmin, zmax : float or None, optional
            Colour scale limits.  ``None`` means auto-range from the data.
        title : str or None, optional
            Plot title.  Defaults to ``self.title``.
        cmap : matplotlib colormap or None, optional
            Colour map (default ``viridis``).
        figsize : tuple, optional
            Figure size in inches (default ``(10, 8)``).
        ax : matplotlib Axes or None, optional
            If provided, plot into this axes instead of creating a new figure.

        Returns
        -------
        fig : matplotlib Figure
        ax : matplotlib Axes
        cset : matplotlib mappable
            The pcolormesh artist (useful for adding a colorbar externally).
        """
        if plt is None:
            raise RuntimeError("matplotlib is not available.")

        if cmap is None:
            cmap = cm.viridis

        if self.x_grid is None or self.interp_values is None:
            raise RuntimeError(
                "No interpolation present; call make_interpolation() first."
            )

        # Mask outside focal-plane radius
        r_cen = np.sqrt(self.x_grid ** 2 + self.y_grid ** 2)
        z_masked = np.ma.masked_where(r_cen > self.radius, self.interp_values)

        if zmin is None:
            zmin = float(np.nanmin(z_masked))
        if zmax is None:
            zmax = float(np.nanmax(z_masked))
        norm = colors.Normalize(vmin=zmin, vmax=zmax)

        created_fig = ax is None
        if created_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        cset = ax.pcolormesh(
            self.x_edge, self.y_edge, z_masked,
            cmap=cmap, norm=norm, shading="flat",
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)
        ax.set_xlim(-self.radius, self.radius)
        ax.set_ylim(-self.radius, self.radius)

        if title is None:
            title = self.title
        if title:
            ax.set_title(title, fontsize=16)

        if created_fig:
            fig.colorbar(cset, ax=ax)

        return fig, ax, cset

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def npoints(self) -> int:
        """Number of points in the point cloud."""
        return len(self.z)

    def __repr__(self) -> str:
        return (
            f"PointMesh(npoints={self.npoints}, method='{self.method}', "
            f"radius={self.radius}, n_bins={self.n_bins})"
        )
