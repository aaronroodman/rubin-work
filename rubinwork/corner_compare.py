"""Corner wavefront sensor (CWFS) against full-array-mode (FAM) wavefront, corner by corner.

For each FAM triplet and each corner sensor, the median in-focus CWFS Zernike is set
against the FAM wavefront interpolated in azimuth to the corner, from the FAM donuts in
an outer field annulus. Both are the raw measured wavefront (``zk_<coord>``) of the same
triplet, matched on the CWFS ``fam_seq_num`` and the FAM ``seq_num``.

Used by the ``wfs_corner_compare`` product, which writes the matched table and the
per-(corner, Zernike) statistics, and by the ``aos/cwfs`` study, which plots them.
"""
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rubinwork.common.utils import nmad

__all__ = ["CORNER_RAFTS", "DEFAULTS", "TABLE_COLUMNS", "STATS_COLUMNS", "ang_dist",
           "azimuth_interp", "fit_metrics", "match_corners", "corner_stats", "pivot"]

CORNER_RAFTS = ("R00", "R04", "R40", "R44")

DEFAULTS = dict(coord="OCS", r_min_deg=1.5178, r_max_deg=1.725, interp="gp",
                fourier_m=4, gp_bin_deg=15.0, gp_restarts=10, wedge_half_deg=30.0, min_fam=3)
"""Comparison settings: frame, FAM annulus [deg of field radius], interpolation method
(``gp``, ``fourier`` or ``wedge-median``), Fourier harmonics, GP azimuth bin [deg], GP
optimizer restarts, wedge half-width [deg], and the minimum FAM donuts per wedge or ring.

Ten restarts, not two: on one Danish 1.2 triplet, 11 of 126 fits (count) with 2
restarts stopped at a worse optimum, moving the interpolated value by up to 0.26 µm of
wavefront; with 10, none did."""

TABLE_COLUMNS = ("day_obs", "seq_num", "corner", "defocus", "j", "fam_interp",
                 "cwfs_median", "corner_az_deg", "rotator_angle", "alt_deg", "az_deg",
                 "mjd", "science_program")

STATS_COLUMNS = ("corner", "defocus", "j", "n", "r", "rho", "slope", "off", "rms")


def ang_dist(a, b):
    """Smallest angular separation on the circle [deg]."""
    return np.abs((np.asarray(a) - b + 180.0) % 360.0 - 180.0)


def _fourier_curve(az, z, m_max):
    """Least-squares truncated Fourier series in azimuth with `m_max` harmonics."""
    def design(a):
        a = np.radians(np.atleast_1d(np.asarray(a, float)))
        cols = [np.ones_like(a)]
        for m in range(1, m_max + 1):
            cols += [np.cos(m * a), np.sin(m * a)]
        return np.column_stack(cols)
    coef = np.linalg.lstsq(design(az), z, rcond=None)[0]
    return lambda a: design(a) @ coef


def _gp_curve(az, z, bin_deg, restarts=10):
    """Gaussian process on the azimuth circle, fitted to `bin_deg`-binned medians.

    The kernel bounds are wide because many Zernikes are nearly constant around the ring
    or nearly noiseless; at narrower bounds the optimizer parks on a bound. The
    restarts are seeded, so a rebuild gives the same numbers.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    centers, medians = [], []
    for lo in np.arange(0, 360, bin_deg):
        m = (az >= lo) & (az < lo + bin_deg)
        if m.any():
            centers.append(lo + bin_deg / 2)
            medians.append(np.nanmedian(z[m]))

    def circ(a):
        a = np.radians(np.atleast_1d(a))
        return np.c_[np.cos(a), np.sin(a)]

    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.5, (0.02, 10.0))
              + WhiteKernel(0.01, (1e-8, 1.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=restarts,
                                      random_state=0).fit(circ(centers), medians)
    return lambda a: gp.predict(circ(a))


def azimuth_interp(az, z, corner_az, interp="gp", fourier_m=4, gp_bin_deg=15.0,
                   gp_restarts=10, wedge_half_deg=30.0, min_fam=3):
    """One FAM Zernike interpolated in azimuth to the corner azimuths.

    Parameters
    ----------
    az : `numpy.ndarray`
        FAM donut focal-plane azimuths [deg].
    z : `numpy.ndarray`
        The Zernike coefficient at those donuts [µm of wavefront].
    corner_az : `array_like`
        Corner azimuths to interpolate to [deg].
    interp : {"gp", "fourier", "wedge-median"}, optional
        A smooth ring fit (``gp``, ``fourier``) or the median of the donuts within
        `wedge_half_deg` of each corner.
    fourier_m, gp_bin_deg, gp_restarts, wedge_half_deg, min_fam : optional
        See `DEFAULTS`.

    Returns
    -------
    values : `numpy.ndarray`
        The interpolated coefficient at each corner [µm of wavefront]; NaN where too few
        donuts.
    curve : callable or `None`
        The ring fit as a function of azimuth [deg], or `None` for ``wedge-median``.
    """
    ca = np.atleast_1d(np.asarray(corner_az, float))
    if interp == "wedge-median":
        vals = np.array([np.nanmedian(z[ang_dist(az, c) <= wedge_half_deg])
                         if (ang_dist(az, c) <= wedge_half_deg).sum() >= min_fam
                         else np.nan for c in ca])
        return vals, None
    fin = np.isfinite(z)
    if fin.sum() < max(min_fam, fourier_m * 2 + 2):
        return np.full(len(ca), np.nan), None
    if interp == "gp":
        curve = _gp_curve(az[fin], z[fin], gp_bin_deg, gp_restarts)
    else:
        curve = _fourier_curve(az[fin], z[fin], fourier_m)
    return np.asarray(curve(ca), float).ravel(), curve


def fit_metrics(x, y):
    """Robust fit of CWFS (`y`) against FAM (`x`), over the finite pairs.

    Returns
    -------
    metrics : `dict`
        ``n`` (pairs, count); ``r`` (Pearson) and ``rho`` (Spearman), dimensionless;
        ``slope`` (dimensionless) and ``off`` [µm of wavefront] of a Huber fit
        (statsmodels ``RLM`` with ``HuberT``); ``rms``, the nMAD of the residuals about
        it [µm of wavefront].
    """
    import statsmodels.api as sm

    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return dict(n=int(m.sum()), r=np.nan, rho=np.nan, slope=np.nan, off=np.nan,
                    rms=np.nan)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    rlm = sm.RLM(y, sm.add_constant(x), M=sm.robust.norms.HuberT()).fit()
    off, slope = float(rlm.params[0]), float(rlm.params[1])
    return dict(n=len(x), r=float(np.corrcoef(x, y)[0, 1]),
                rho=float(spearmanr(x, y).statistic), slope=slope, off=off,
                rms=float(nmad(y - (slope * x + off))))


def _corner_azimuth(az_deg):
    """Circular median of donut azimuths [deg]."""
    a = np.radians(az_deg)
    return np.degrees(np.arctan2(np.nanmedian(np.sin(a)), np.nanmedian(np.cos(a)))) % 360.0


def match_corners(fam, cwfs, visits, noll, settings=None, log=print):
    """The matched CWFS-against-FAM table, one row per (triplet, corner, Zernike).

    Parameters
    ----------
    fam : `pandas.DataFrame`
        FAM donuts: ``day_obs``, ``seq_num``, ``zk_<coord>``, ``thx_<coord>``,
        ``thy_<coord>`` (field angles in rad).
    cwfs : `pandas.DataFrame`
        In-focus CWFS donuts: ``detector``, ``day_obs``, ``fam_seq_num``, the same three
        wavefront columns, and ``wfs_offset`` (-1 intra, 0 extra) when unpaired.
    visits : `pandas.DataFrame`
        FAM visits: ``day_obs``, ``seq_num`` and any of ``rotator_angle``, ``alt``,
        ``az`` (rad), ``mjd``, ``science_program``.
    noll : `list` [`int`]
        Noll indices of the ``zk`` columns, in order.
    settings : `dict`, optional
        Overrides of `DEFAULTS`.
    log : callable, optional
        Progress printer.

    Returns
    -------
    table : `pandas.DataFrame`
        `TABLE_COLUMNS`: ``fam_interp`` and ``cwfs_median`` [µm of wavefront, in
        ``coord``], ``corner_az_deg`` [deg], ``rotator_angle`` as in `visits`,
        ``alt_deg`` and ``az_deg`` [deg]. Rows where both values are NaN are dropped.
    """
    s = {**DEFAULTS, **(settings or {})}
    zc, txc, tyc = f"zk_{s['coord']}", f"thx_{s['coord']}", f"thy_{s['coord']}"
    ikw = {k: s[k] for k in ("interp", "fourier_m", "gp_bin_deg", "gp_restarts",
                             "wedge_half_deg", "min_fam")}

    cw_zk = np.stack(cwfs[zc].values).astype(float)
    cw_az = np.degrees(np.arctan2(cwfs[tyc].astype(float), cwfs[txc].astype(float))) % 360
    cw_det = cwfs["detector"].astype(str).values
    corners = sorted(d for d in set(cw_det)
                     if d[:3] in CORNER_RAFTS and d[-3:] in ("SW0", "SW1"))
    if "wfs_offset" in cwfs.columns:
        off = cwfs.assign(_d=cw_det).groupby("_d")["wfs_offset"].median()
        defocus = {d: "intra" if float(off.get(d, 0)) < 0 else "extra" for d in corners}
    else:
        defocus = {d: "" for d in corners}

    fr = np.degrees(np.hypot(fam[txc].astype(float), fam[tyc].astype(float)))
    fam = fam[(fr >= s["r_min_deg"]) & (fr <= s["r_max_deg"])].reset_index(drop=True)
    fam_zk = np.stack(fam[zc].values).astype(float)
    fam_az = np.degrees(np.arctan2(fam[tyc].astype(float), fam[txc].astype(float))) % 360
    fam_grp = fam.groupby(["day_obs", "seq_num"]).indices
    cw_grp = cwfs.groupby(["day_obs", "fam_seq_num"]).indices
    triplets = sorted(set(cw_grp) & set(fam_grp))
    log(f"{len(triplets)} matched triplets (count), corners {corners}, "
        f"interpolation {s['interp']}")

    side_cols = [c for c in ("rotator_angle", "alt", "az", "mjd", "science_program")
                 if c in visits.columns]
    side = visits.set_index(["day_obs", "seq_num"])[side_cols]
    side = side[~side.index.duplicated()]

    rows = []
    for t, key in enumerate(triplets):
        ci = np.asarray(cw_grp[key], int)
        fi = np.asarray(fam_grp[key], int)
        caz = np.full(len(corners), np.nan)
        cmed = np.full((len(corners), len(noll)), np.nan)
        for c, det in enumerate(corners):
            cc = ci[cw_det[ci] == det]
            if len(cc):
                cmed[c] = np.nanmedian(cw_zk[cc], axis=0)
                caz[c] = _corner_azimuth(cw_az[cc])
        finterp = np.full((len(corners), len(noll)), np.nan)
        ok = np.isfinite(caz)
        if ok.any():
            for jp in range(len(noll)):
                finterp[ok, jp] = azimuth_interp(fam_az[fi], fam_zk[fi, jp], caz[ok],
                                                 **ikw)[0]
        sc = side.loc[key] if key in side.index else None

        def get(name):
            return sc[name] if sc is not None and name in sc.index else np.nan

        alt, az = float(get("alt")), float(get("az"))
        meta = dict(rotator_angle=float(get("rotator_angle")),
                    # visits.parquet carries alt/az in rad.
                    alt_deg=np.degrees(alt) if abs(alt) <= 2 * np.pi else alt,
                    az_deg=np.degrees(az) if abs(az) <= 2 * np.pi else az,
                    mjd=float(get("mjd")),
                    science_program=str(get("science_program")) if sc is not None
                    and "science_program" in sc.index else "")
        for c, det in enumerate(corners):
            for jp, j in enumerate(noll):
                if np.isfinite(finterp[c, jp]) or np.isfinite(cmed[c, jp]):
                    rows.append(dict(day_obs=int(key[0]), seq_num=int(key[1]),
                                     corner=det, defocus=defocus[det], j=int(j),
                                     fam_interp=float(finterp[c, jp]),
                                     cwfs_median=float(cmed[c, jp]),
                                     corner_az_deg=float(caz[c]), **meta))
        if (t + 1) % 100 == 0:
            log(f"  {t + 1} of {len(triplets)} triplets")
    return pd.DataFrame(rows, columns=list(TABLE_COLUMNS))


def corner_stats(table):
    """`fit_metrics` per (corner, Zernike) of a matched table.

    Returns
    -------
    stats : `pandas.DataFrame`
        `STATS_COLUMNS`, one row per (corner, j).
    """
    if "defocus" not in table.columns:
        table = table.assign(defocus="")
    rows = []
    for (corner, j), g in table.groupby(["corner", "j"], sort=True):
        rows.append(dict(corner=corner, defocus=g["defocus"].iloc[0], j=int(j),
                         **fit_metrics(g["fam_interp"].to_numpy(float),
                                       g["cwfs_median"].to_numpy(float))))
    return pd.DataFrame(rows, columns=list(STATS_COLUMNS))


def pivot(table, column):
    """A matched-table column as an array over (triplet, corner, Zernike).

    Returns
    -------
    values : `numpy.ndarray`
        Shape ``(n_triplets, n_corners, n_j)``, NaN where absent.
    triplets, corners, js : `list`
        The axis labels, sorted.
    """
    triplets = sorted(set(zip(table.day_obs, table.seq_num)))
    corners = sorted(table.corner.unique())
    js = sorted(int(j) for j in table.j.unique())
    ti = {k: i for i, k in enumerate(triplets)}
    out = np.full((len(triplets), len(corners), len(js)), np.nan)
    it = np.array([ti[k] for k in zip(table.day_obs, table.seq_num)])
    ic = np.searchsorted(corners, table.corner.values)
    ij = np.searchsorted(js, table.j.values)
    out[it, ic, ij] = table[column].to_numpy(float)
    return out, triplets, corners, js
