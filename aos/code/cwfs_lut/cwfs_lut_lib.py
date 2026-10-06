"""Elevation and rotator-angle dependence of the open-loop optical state, from the corner sensors.

A look-up table (LUT) predicts the degree-of-freedom (DOF) correction the telescope needs as a
function of pointing, so the loop starts near the right state. This study measures that
dependence from the corner wavefront sensors (CWFS) over the whole science survey, rather than
from a dedicated Full Array Mode (FAM) campaign, and compares it against the AOS bounce test.

The quantity modelled is the stored **open-loop** DOF vector, ``Deviation - Trim``: the state
that would have been present with the loop open, which is what a LUT has to supply. It is read
from `dof_olr` via ``efd_db.optical_state(..., wide=True)`` as ``dof0_olr`` to ``dof49_olr``.

Two deliberate choices, both of which cost something:

**Absolute trends, not within-night paired differences.** Science visits sweep elevation and
rotator angle widely inside a single night -- `day_obs` 20260713 covers elevation 23.68 to 82.43
deg and rotator -79.43 to +78.52 deg -- so pairing would discard most of the available leverage.
The cost is that an absolute elevation trend **confounds gravity with thermal drift** that
tracks elevation through the observing pattern. That is a stated caveat of every elevation
number here, not something the fit removes.

**Both intrinsic routes.** The Deviation is ``OPD - intrinsic``, so the assumed intrinsic shifts
every recovered DOF. The measured intrinsic wavefront (MIW) differs from the batoid ray-trace
prediction by 0.0547 µm of wavefront at the corners even at rotator angle 0 deg, rising to
0.0796 at +60 deg, and the resulting open-loop M2 hexapod dx median moves by +236.57 µm -- more
than the term's own value. Neither route is the reference; the spread between them is part of
the result.

Units, and the one trap in this study
-------------------------------------
The 50-element DOF vector is stored with **the four hexapod tilts (DOF 3, 4, 8, 9) in deg** and
everything else in µm. `lsst.ts.intrinsic.wavefront.ofc_svd.DOF_UNITS_50` labels those same four
**arcsec**, and the bounce-test tables follow that convention. So any comparison against the
bounce test needs 3600 arcsec/deg applied to exactly those four entries and nothing on the other
46. `to_bounce_units` does this; the mismatch is otherwise silent, leaving the dominant decentre
and bending terms correct and corrupting only the tilts.

Scope of the bounce-test comparison
-----------------------------------
Limited to the **hexapod decentres**. The bounce test retrieves a full-focal-plane Double
Zernike field from FAM data; this study has four corner field points. Different field sampling
and a different retrieval, so the high-order mirror bending modes are not comparable. The
decentres are the bounce test's dominant terms and the testable claim.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'value_added' / 'code'))

from common.utils import nmad                                    # noqa: E402

#: Tilt entries of the 50-element DOF vector: M2 hexapod rx/ry then camera hexapod rx/ry. Stored
#: in deg; `ofc_svd.DOF_UNITS_50` and the bounce test call the same four arcsec.
HEX_TILT_DOF = (3, 4, 8, 9)

#: arcsec per deg, applied to `HEX_TILT_DOF` alone when comparing against the bounce test.
DEG_TO_ARCSEC = 3600.0

#: The ts_ofc 50-DOF layout: DOF 0-4 are the M2 hexapod, DOF 5-9 the camera hexapod, each as
#: (dz, dx, dy, rx, ry). DOF 10-29 are M1M3 bending, 30-49 M2 bending.
DOF_LABELS = {
    0: ('M2 hexapod dz', 'µm'), 1: ('M2 hexapod dx', 'µm'), 2: ('M2 hexapod dy', 'µm'),
    3: ('M2 hexapod rx', 'deg'), 4: ('M2 hexapod ry', 'deg'),
    5: ('camera hexapod dz', 'µm'), 6: ('camera hexapod dx', 'µm'),
    7: ('camera hexapod dy', 'µm'),
    8: ('camera hexapod rx', 'deg'), 9: ('camera hexapod ry', 'deg'),
}

#: DOF the bounce-test comparison is restricted to: the six hexapod decentres and pistons, which
#: are its dominant terms and are retrieved comparably from four field points.
BOUNCE_COMPARABLE_DOF = (0, 1, 2, 5, 6, 7)

#: Variants carrying the two intrinsic routes at fixed solver (both unconstrained 50/34). There
#: is no range-bounded recovery arm on the MIW route.
INTRINSIC_VARIANTS = {'batoid': 'v50_34__batoid__consdb_v1',
                      'miw': 'v50_34__miw__consdb_v1'}

#: Range-bounded recovery, physically realizable on 99.5% of visits. Reported alongside the
#: unconstrained pair because a LUT must command a reachable state.
RBR_VARIANT = 'v50_34_rbr__batoid__consdb_v1'

#: Minimum visits in a fit before a slope is reported.
MIN_VISITS = 200


def dof_label(j, bounce_units=False):
    """Name and unit of one DOF entry.

    Parameters
    ----------
    j : `int`
        DOF index, 0-based in the ts_ofc 50-element layout.
    bounce_units : `bool`, optional
        Report the tilts in arcsec, the bounce-test convention, rather than the stored deg.

    Returns
    -------
    name : `str`
    unit : `str`
    """
    if j in DOF_LABELS:
        name, unit = DOF_LABELS[j]
        if bounce_units and j in HEX_TILT_DOF:
            unit = 'arcsec'
        return name, unit
    if j < 30:
        return f'M1M3 bending {j - 9}', 'µm'
    return f'M2 bending {j - 29}', 'µm'


def to_bounce_units(dof, copy=True):
    """Convert a stored DOF vector or array into the bounce test's units.

    Parameters
    ----------
    dof : `numpy.ndarray`
        Shape ``(50,)`` or ``(n, 50)``, stored units: µm, with `HEX_TILT_DOF` in deg.
    copy : `bool`, optional
        Work on a copy, the default.

    Returns
    -------
    out : `numpy.ndarray`
        Same shape, with `HEX_TILT_DOF` scaled by `DEG_TO_ARCSEC` and the other 46 untouched.

    Raises
    ------
    ValueError
        If the last axis is not 50 long -- a 34-element v-mode vector passed here by mistake
        would otherwise be silently mangled.
    """
    a = np.asarray(dof, float)
    if a.shape[-1] != 50:
        raise ValueError(f'expected a 50-element DOF axis, got {a.shape[-1]}; the v-mode '
                         f'vectors are 34 long and must not be scaled')
    out = a.copy() if copy else a
    out[..., list(HEX_TILT_DOF)] *= DEG_TO_ARCSEC
    return out


def olr_columns(n_dof=50):
    """Stored open-loop DOF column names, in index order.

    Parameters
    ----------
    n_dof : `int`, optional

    Returns
    -------
    cols : `list` [`str`]
        ``['dof0_olr', ..., 'dof49_olr']``.
    """
    return [f'dof{j}_olr' for j in range(n_dof)]


def huber_trend(x, y, min_n=MIN_VISITS, min_span=5.0):
    """Robust straight-line fit of `y` against `x`.

    Parameters
    ----------
    x, y : `array_like` [`float`]
        Paired samples; non-finite pairs are dropped.
    min_n : `int`, optional
        Minimum finite pairs before a slope is returned.
    min_span : `float`, optional
        Minimum range of `x`, in `x`'s units, before a slope is returned. A fit over a narrow
        span extrapolates badly and its slope is not reportable.

    Returns
    -------
    res : `dict`
        ``slope`` [`y` unit per `x` unit], ``intercept`` [`y` unit], ``slope_err`` (formal,
        same unit as the slope), ``n``, ``span`` [`x` unit], ``resid_nmad`` [`y` unit], and
        ``pearson_r`` / ``spearman_rho`` (both dimensionless).

    Notes
    -----
    Huber robust linear (`statsmodels.RLM` with `HuberT`), the repository default for AOS
    correlations: the open-loop DOF carry outliers from visits whose recovery is poorly
    conditioned, which least squares would chase. ``slope_err`` is the formal RLM standard
    error and is known to understate the true uncertainty here -- successive visits are
    correlated, so the effective sample is smaller than ``n``.
    """
    import statsmodels.api as sm
    from scipy.stats import pearsonr, spearmanr

    xa = np.asarray(x, float)
    ya = np.asarray(y, float)
    m = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[m], ya[m]
    nan = dict(slope=np.nan, intercept=np.nan, slope_err=np.nan, n=int(m.sum()),
               span=np.nan, resid_nmad=np.nan, pearson_r=np.nan, spearman_rho=np.nan)
    if len(xa) < min_n:
        return nan
    span = float(xa.max() - xa.min())
    if span < min_span:
        return dict(nan, span=span)

    X = sm.add_constant(xa)
    fit = sm.RLM(ya, X, M=sm.robust.norms.HuberT()).fit()
    resid = ya - fit.predict(X)
    return dict(slope=float(fit.params[1]), intercept=float(fit.params[0]),
                slope_err=float(fit.bse[1]), n=len(xa), span=span,
                resid_nmad=float(nmad(resid)),
                pearson_r=float(pearsonr(xa, ya)[0]),
                spearman_rho=float(spearmanr(xa, ya)[0]))


def trend_table(df, angle_col, dof_indices=None, bounce_units=False, verbose=True):
    """Open-loop DOF trend against one pointing angle, per DOF.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Carrying `angle_col` and the ``dof*_olr`` columns.
    angle_col : `str`
        ``'elevation_deg'`` or ``'rotator_angle_deg'``.
    dof_indices : `iterable` [`int`], optional
        DOF to fit, defaulting to the ten rigid-body entries.
    bounce_units : `bool`, optional
        Report the four tilt DOF in arcsec rather than the stored deg, for comparison against
        the bounce test.
    verbose : `bool`, optional

    Returns
    -------
    tab : `pandas.DataFrame`
        One row per DOF: ``dof``, ``name``, ``unit``, ``slope`` [DOF unit per deg of angle],
        ``slope_err``, ``intercept``, ``n``, ``span`` [deg], ``resid_nmad``, ``pearson_r`` and
        ``spearman_rho``.
    """
    if dof_indices is None:
        dof_indices = range(10)
    if angle_col not in df.columns:
        raise KeyError(f'{angle_col} absent; optical_state carries elevation_deg and '
                       f'rotator_angle_deg from the 090b185 schema onward')

    rows = []
    for j in dof_indices:
        col = f'dof{j}_olr'
        if col not in df.columns:
            continue
        y = df[col].to_numpy(float)
        if bounce_units and j in HEX_TILT_DOF:
            y = y * DEG_TO_ARCSEC
        name, unit = dof_label(j, bounce_units=bounce_units)
        rows.append(dict(dof=j, name=name, unit=unit,
                         **huber_trend(df[angle_col].to_numpy(float), y)))
    tab = pd.DataFrame(rows)

    if verbose:
        print(f'open-loop DOF against {angle_col}, Huber robust linear, '
              f'n={len(df)} visits')
        print(f'{"DOF":<22s} {"slope":>14s} {"per deg":>10s} {"Pearson r":>10s} '
              f'{"Spearman":>9s}')
        for _, r in tab.iterrows():
            if not np.isfinite(r['slope']):
                print(f'{r["name"]:<22s} {"--":>14s} (n={int(r["n"])}, span '
                      f'{r["span"]:.1f} deg)')
                continue
            print(f'{r["name"]:<22s} {r["slope"]:>+14.4g} {r["unit"] + "/deg":>10s} '
                  f'{r["pearson_r"]:>+10.3f} {r["spearman_rho"]:>+9.3f}')
        print('  slope is DOF unit per deg of pointing angle; r and rho dimensionless')
    return tab


def intrinsic_spread(tab_batoid, tab_miw, verbose=True):
    """Compare the fitted trends between the two intrinsic routes.

    Parameters
    ----------
    tab_batoid, tab_miw : `pandas.DataFrame`
        `trend_table` results on `INTRINSIC_VARIANTS`, same angle and same DOF set.
    verbose : `bool`, optional

    Returns
    -------
    cmp : `pandas.DataFrame`
        ``dof``, ``name``, ``unit``, both slopes, their difference, and ``ratio`` (MIW over
        batoid, dimensionless) where the batoid slope is non-zero.

    Notes
    -----
    A static intrinsic offset shifts an intercept, not a slope, so a **slope** that differs
    between routes means the rotator-angle-dependent part of the MIW-batoid difference is
    entering the trend. For the elevation trend that part is not static in the fit, because
    rotator angle and elevation are correlated through the observing pattern.
    """
    cmp = tab_batoid[['dof', 'name', 'unit', 'slope', 'slope_err']].merge(
        tab_miw[['dof', 'slope', 'slope_err']], on='dof', suffixes=('_batoid', '_miw'))
    cmp['slope_diff'] = cmp['slope_miw'] - cmp['slope_batoid']
    with np.errstate(divide='ignore', invalid='ignore'):
        cmp['ratio'] = np.where(cmp['slope_batoid'] != 0,
                                cmp['slope_miw'] / cmp['slope_batoid'], np.nan)
    if verbose:
        print('intrinsic route comparison, slope per deg of pointing angle:')
        print(f'{"DOF":<22s} {"batoid":>12s} {"MIW":>12s} {"difference":>12s} {"unit":>8s}')
        for _, r in cmp.iterrows():
            print(f'{r["name"]:<22s} {r["slope_batoid"]:>+12.4g} {r["slope_miw"]:>+12.4g} '
                  f'{r["slope_diff"]:>+12.4g} {r["unit"] + "/deg":>8s}')
    return cmp
