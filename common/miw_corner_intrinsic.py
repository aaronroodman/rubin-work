"""Measured Intrinsic Wavefront (MIW) evaluated at the four corner-wavefront-sensor points.

The optical state recovered from the corner wavefront sensors (CWFS) is the measured optical
path difference (OPD) minus an assumed intrinsic wavefront, so the intrinsic is what defines
the state rather than being an implementation detail. Two routes exist: the batoid ray-trace
prediction from ``lsst.ts.ofc``, and the MIW measured from Full Array Mode (FAM) data. This
module supplies the second one, as the ``miw_lookup`` callable that
``common/scripts/build_optical_state.py`` takes for ``--intrinsic miw``.

The MIW is stored as a rotator-angle decomposition: each Zernike term carries a
telescope-fixed component in the Observatory Coordinate System (OCS) and a camera-fixed
component in the Camera Coordinate System (CCS), the latter rotating with the camera rotator.
`lsst.ts.intrinsic.wavefront.intrinsic_split.reconstruct_at` combines them at a given rotator
angle to give the intrinsic field over the focal plane, which is then interpolated to the
field points wanted.

Two properties of this problem make the evaluation direct rather than requiring the
pseudo-donut table and ``run_make_intrinsic_sidecar.py`` route:

- The four corner field points are **fixed** in the OCS -- they are `ts_ofc`'s
  ``sample_points`` for the corner sensors -- so the barycentric interpolation weights onto
  the decomposition's polar grid are computed once for all visits.
- Reconstruction costs about 1 ms per rotator angle for all 21 Zernike terms, so every
  distinct rotator angle in a multi-year sample is reconstructed exactly, with no
  interpolation over rotator angle.

Zernike basis: `aos_state.ZK_NOLL`, the 21 Noll indices 4 to 26 excluding 20 and 21, which is
the basis used throughout this repository and is exactly the set the MIW decomposition
carries.

Notes
-----
The MIW carries no band dependence: it is measured from FAM data taken in a single band,
named in the build (``pathA_50_34_i_5rot`` is i band), and applies as measured to every band.
The batoid intrinsic is band-dependent, differing by about 0.014 µm of wavefront in Zernike 4
between g and i at the corners. So a truss-temperature or elevation slope fitted per band
carries a band-dependent intrinsic offset on the batoid route and a band-independent one on
the MIW route. That shifts intercepts between the two routes, and is the expected difference
rather than a defect.

**The detector heights are already in the MIW.** The physical height of each detector above
the focal surface is a camera-fixed property, so the MIW fit puts it in the camera-fixed CCS
component, and `reconstruct_at` combines that with the telescope-fixed OCS component --
counter-rotating the CCS part into the OCS frame at the given rotator angle -- to give the
full intrinsic. Evaluating that combination at the corner field points therefore accounts for
the corner sensors' heights on the intrinsic side, with no separate height term. At the
corners the CCS Zernike 4 component runs from +0.013 to +0.066 µm of wavefront, reaching 1.83
times the OCS component at ``R40_SW0``, and it carries ``n_spin = 0, s = 1`` -- a pure
camera-fixed pattern, which is the height signature.

`corner_z4_height_um` computes a standalone per-sensor height term from the `batoid_rubin`
height maps, for comparison against what the CCS component implies. It is **not** added by
default and `MiwCornerLookup`'s ``add_ccd_height`` should normally stay false: the standalone
term is -0.055 to -0.093 µm of wavefront at the corners, comparable to the CCS component
itself, so adding it double-counts the heights.
"""

import pathlib

import numpy as np
from scipy.spatial import Delaunay

# The MIW build this repository uses. A different build is a different intrinsic and so a
# different optical-state variant, recorded in `state_variant.intrinsic_ref`.
DEFAULT_PARAM_SET = 'fam_danish_1_2_0_wep17_6_1_refitWCS_bin2x'
DEFAULT_MI_NAME = 'pathA_50_34_i_5rot'


def decomp_path(mi_name=DEFAULT_MI_NAME, param_set=DEFAULT_PARAM_SET, output_root=None):
    """Path to a MIW build's rotator decomposition.

    Parameters
    ----------
    mi_name : `str`, optional
        MIW build name, e.g. ``'pathA_50_34_i_5rot'``.
    param_set : `str`, optional
    output_root : `str` or `pathlib.Path`, optional
        Defaults to ``aos/output`` in this repository.

    Returns
    -------
    path : `pathlib.Path`
        The ``intrinsic_split_decomp.parquet`` for that build.
    """
    if output_root is None:
        output_root = pathlib.Path(__file__).resolve().parents[1] / 'aos' / 'output'
    return (pathlib.Path(output_root) / param_set / mi_name
            / 'intrinsic_split_decomp.parquet')


def load_decomposition(path):
    """Read a MIW rotator decomposition.

    Parameters
    ----------
    path : `str` or `pathlib.Path`
        An ``intrinsic_split_decomp.parquet``.

    Returns
    -------
    dec : `dict`
        ``noll`` [`list` of `int`], ``O_pol`` and ``C_pol`` (complex, shape
        ``(n_zk, n_r, n_az)``), ``n_spin``, ``s``, ``part``, ``A`` (the polar amplitude grid),
        ``X`` and ``Y`` (field grid [deg], shape ``(n_r, n_az)``).

    Notes
    -----
    The file holds one row per ``(Noll index, part)``, with the complex OCS and CCS fields
    flattened into real and imaginary columns in C order.
    """
    from astropy.table import Table

    dt = Table.read(str(path), format='parquet')
    n_r = int(dt.meta['n_r'])
    n_az = int(dt.meta['n_az'])

    def _field(row, re_col, im_col):
        return (np.asarray(dt[re_col][row], dtype=float)
                + 1j * np.asarray(dt[im_col][row], dtype=float)).reshape(n_r, n_az)

    n = len(dt)
    return dict(
        noll=[int(j) for j in dt['j']],
        O_pol=np.stack([_field(i, 'O_re', 'O_im') for i in range(n)]),
        C_pol=np.stack([_field(i, 'C_re', 'C_im') for i in range(n)]),
        n_spin=np.asarray(dt['n_spin'], dtype=int),
        s=np.asarray(dt['s'], dtype=int),
        part=np.asarray(dt['part'], dtype=int),
        A=np.asarray(dt.meta['A'], dtype=float),
        X=np.asarray(dt.meta['X'], dtype=float).reshape(n_r, n_az),
        Y=np.asarray(dt.meta['Y'], dtype=float).reshape(n_r, n_az),
    )


def corner_field_points(ofc_version=None):
    """Field angles of the four corner wavefront sensors.

    Parameters
    ----------
    ofc_version : `str`, optional
        Accepted for symmetry with the batoid route and ignored: the corner sample points are
        fixed instrument geometry and do not vary between `ts_ofc` configuration versions.

    Returns
    -------
    pts : `numpy.ndarray`
        Shape ``(4, 2)``, ``(thx, thy)`` [deg] in the OCS, ordered as
        `aos_state.SENSOR_NAMES`.

    Notes
    -----
    These are `ts_ofc`'s ``sample_points``, the same interface the batoid route uses, so both
    intrinsic routes are evaluated at identical field positions.
    """
    import aos_state
    from lsst.ts.ofc import OFCData

    ofcd = OFCData('lsst')
    return np.array([np.asarray(ofcd.sample_points[s], dtype=float)
                     for s in aos_state.SENSOR_NAMES])


def _barycentric_weights(tri, pts):
    """Barycentric weights of `pts` in the triangulation `tri`.

    Parameters
    ----------
    tri : `scipy.spatial.Delaunay`
    pts : `numpy.ndarray`
        Shape ``(n, 2)``.

    Returns
    -------
    simp : `numpy.ndarray`
        Containing simplex per point, -1 where outside the hull.
    verts : `numpy.ndarray`
        Shape ``(n, 3)``, vertex indices.
    bary : `numpy.ndarray`
        Shape ``(n, 3)``, weights summing to one.
    """
    simp = tri.find_simplex(pts)
    verts = np.zeros((len(pts), 3), dtype=int)
    bary = np.zeros((len(pts), 3))
    inside = simp >= 0
    if inside.any():
        s = simp[inside]
        T = tri.transform[s]
        d = pts[inside] - T[:, 2]
        b2 = np.einsum('ijk,ik->ij', T[:, :2], d)
        bary[inside] = np.column_stack([b2, 1.0 - b2.sum(axis=1)])
        verts[inside] = tri.simplices[s]
    return simp, verts, bary


def corner_z4_height_um(ofc_version=None, height_map_dir=None):
    """Defocus equivalent of each corner sensor's physical height [µm of wavefront].

    Parameters
    ----------
    ofc_version : `str`, optional
    height_map_dir : `str`, optional
        Override for the `batoid_rubin` height-map directory.

    Returns
    -------
    z4 : `numpy.ndarray`
        Shape ``(4,)``, µm of wavefront, ordered as `aos_state.SENSOR_NAMES`. NaN entries
        where the height is unavailable.

    Notes
    -----
    The corner sensors are split, with the intra-focal half on ``SW1`` and the extra-focal
    half on ``SW0``; the height taken here is the mean of the two halves at the sensor
    centre, matching ``run_make_intrinsic_sidecar.py --wfs-corner-height``.

    This is a **diagnostic**, not a term to add to the MIW. The heights are camera-fixed and
    the MIW already carries them in its CCS component, so adding this on top double-counts
    them. It is useful for checking that the two routes to the same physical quantity are of
    comparable size.
    """
    import aos_state

    try:
        import pandas as pd
        from lsst.obs.lsst import LsstCam
        from lsst.ts.intrinsic.wavefront.ccd_height import (compute_ccd_heights,
                                                            HEIGHT_TO_Z4_UM_PER_MM)
    except ImportError as exc:
        raise RuntimeError(f'corner CCD height needs the AOS environment: {exc}') from exc

    cam = LsstCam.getCamera()
    pts = corner_field_points(ofc_version)
    # compute_ccd_heights works from detector name and centroid pixel position; the sensor
    # centre is the right evaluation point for a field-point intrinsic.
    rows = []
    for name, (thx, thy) in zip(aos_state.SENSOR_NAMES, pts):
        det = cam[name]
        cx, cy = det.getBBox().getCenter()
        rows.append(dict(detector=name, centroid_x_intra=cx, centroid_y_intra=cy,
                         centroid_x_extra=cx, centroid_y_extra=cy))
    df = pd.DataFrame(rows)
    kw = dict(source='batoid_rubin', height_map_dir=height_map_dir,
              metrology_fits=None, factor=HEIGHT_TO_Z4_UM_PER_MM)
    dfi = df.copy()
    dfi['detector'] = df['detector'].str.replace('SW0', 'SW1')
    hi = np.asarray(compute_ccd_heights(dfi, cam, **kw)['ccd_height_intra'], float)
    he = np.asarray(compute_ccd_heights(df, cam, **kw)['ccd_height_extra'], float)
    with np.errstate(invalid='ignore'):
        return HEIGHT_TO_Z4_UM_PER_MM * np.nanmean(np.vstack([hi, he]), axis=0)


class MiwCornerLookup:
    """MIW intrinsic at the four corner sensors, as a function of rotator angle.

    Reconstructs the MIW field at each distinct rotator angle and interpolates it to the four
    fixed corner field points, returning the corner-major vector
    ``build_optical_state.measured_deviation`` expects.

    Parameters
    ----------
    mi_name : `str`, optional
        MIW build name.
    param_set : `str`, optional
    output_root : `str` or `pathlib.Path`, optional
    ofc_version : `str`, optional
        Passed to `corner_field_points`, so both intrinsic routes use the same field points.
    add_ccd_height : `bool`, optional
        Add a standalone per-sensor height-equivalent defocus to the Zernike 4 term. Default
        false, and normally left false: the detector heights are camera-fixed and so are
        already carried by the CCS component that `field_at` reconstructs, making this a
        double count. Available only for comparing the standalone height term against what
        the CCS component implies.
    height_map_dir : `str`, optional
    round_deg : `int`, optional
        Rotator angles are grouped after rounding to this many decimal places, which
        deduplicates identical pointings without measurably changing the angle. Reconstruction
        is about 1 ms per angle, so this is a small saving rather than a necessity.

    Notes
    -----
    Rotator-angle sign and frame follow `intrinsic_split.reconstruct_at`, which rotates the
    camera-fixed CCS component into the OCS before the field interpolation. The angle passed
    in must be the ConsDB ``physical_rotator_angle``, in deg, which is what
    `build_optical_state.visit_metadata` supplies.

    Examples
    --------
    >>> lookup = MiwCornerLookup()                            # doctest: +SKIP
    >>> intr = lookup(visit_ids, rot_angles, aos_state.ZK_NOLL)   # doctest: +SKIP
    """

    def __init__(self, mi_name=DEFAULT_MI_NAME, param_set=DEFAULT_PARAM_SET,
                 output_root=None, ofc_version=None, add_ccd_height=False,
                 height_map_dir=None, round_deg=3):
        self.mi_name = mi_name
        self.param_set = param_set
        self.path = decomp_path(mi_name, param_set, output_root)
        if not self.path.exists():
            raise FileNotFoundError(
                f'no MIW decomposition at {self.path}; build the MIW for param_set '
                f'{param_set!r}, build {mi_name!r} before using the miw intrinsic route')
        self.dec = load_decomposition(self.path)
        self.noll = list(self.dec['noll'])
        self.points = corner_field_points(ofc_version)
        self.round_deg = int(round_deg)

        # The corner field points are fixed, so triangulate and weight once.
        X, Y = self.dec['X'], self.dec['Y']
        tri = Delaunay(np.column_stack([X.ravel(), Y.ravel()]))
        self._simp, self._verts, self._bary = _barycentric_weights(tri, self.points)
        n_out = int((self._simp < 0).sum())
        if n_out:
            raise ValueError(
                f'{n_out} of 4 corner field points fall outside the MIW field coverage '
                f'(grid reaches {np.hypot(X, Y).max():.3f} deg, corners sit at '
                f'{np.hypot(*self.points.T).max():.3f} deg); the intrinsic would be an '
                f'extrapolation')

        self.z4_height_um = None
        if add_ccd_height:
            self.z4_height_um = corner_z4_height_um(ofc_version, height_map_dir)
        self._cache = {}

    def field_at(self, rot_deg):
        """MIW at the four corner points for one rotator angle.

        Parameters
        ----------
        rot_deg : `float`
            Camera rotator angle [deg].

        Returns
        -------
        z : `numpy.ndarray`
            Shape ``(4, n_zk)``, µm of wavefront, rows ordered as `aos_state.SENSOR_NAMES`
            and columns in this build's Noll order.
        """
        from lsst.ts.intrinsic.wavefront import intrinsic_split as isp

        key = round(float(rot_deg), self.round_deg)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        theta = np.deg2rad(key)
        d = self.dec
        out = np.empty((4, len(self.noll)))
        for ij in range(len(self.noll)):
            recon = isp.reconstruct_at(
                {'O_pol': d['O_pol'][ij], 'C_pol': d['C_pol'][ij],
                 'n_spin': int(d['n_spin'][ij]), 's': int(d['s'][ij])}, theta, d['A'])
            fld = recon.real if d['part'][ij] == 0 else recon.imag
            flat = np.ascontiguousarray(fld).ravel()
            out[:, ij] = np.einsum('ij,ij->i', flat[self._verts], self._bary)
        if self.z4_height_um is not None and 4 in self.noll:
            j4 = self.noll.index(4)
            h = self.z4_height_um
            out[:, j4] = np.where(np.isfinite(h), out[:, j4] + h, out[:, j4])
        self._cache[key] = out
        return out

    def __call__(self, visit_ids, rot_angles, zk_noll):
        """Intrinsic vector per visit, in the layout `measured_deviation` expects.

        Parameters
        ----------
        visit_ids : `array_like` [`int`]
            Used only for its length; the MIW depends on rotator angle, not on the visit.
        rot_angles : `array_like` [`float`]
            Camera rotator angle per visit [deg].
        zk_noll : `list` [`int`]
            Noll indices wanted, in the sensitivity matrix's row order.

        Returns
        -------
        intr : `numpy.ndarray`
            Shape ``(n_visits, 4 * len(zk_noll))``, µm of wavefront, corner-major: all
            Zernikes of the first corner, then the second, matching
            ``[f'z{z}_{c}' for c in SENSOR_NAMES for z in zk_noll]``. NaN where the rotator
            angle is not finite.

        Raises
        ------
        ValueError
            If `zk_noll` asks for a Noll index this MIW build does not carry.
        """
        missing = [z for z in zk_noll if z not in self.noll]
        if missing:
            raise ValueError(
                f'MIW build {self.mi_name!r} carries Noll {self.noll} and cannot supply '
                f'{missing}; the corner analysis basis is aos_state.ZK_NOLL')
        cols = [self.noll.index(z) for z in zk_noll]
        rot = np.asarray(rot_angles, dtype=float)
        n_z = len(zk_noll)
        out = np.full((len(rot), 4 * n_z), np.nan)
        for i, r in enumerate(rot):
            if not np.isfinite(r):
                continue
            out[i] = self.field_at(r)[:, cols].reshape(-1)
        return out
