#!/usr/bin/env python3
"""Correlate the uniform-focus Double Zernike term DZ(k=1, j=4) with TMA truss temperature.

DZ(k=1, j=4) is the focal-plane-uniform part of the Zernike Z4 (defocus) wavefront: the
k=1 focal Zernike is a constant over the field, so this single coefficient is the mean
defocus of the whole focal plane, in µm of wavefront. Telescope Mount Assembly (TMA) truss
temperature is known to correlate with focus changes, and this script asks how much of
DZ(1,4) it actually explains -- separately for the drift between Full Array Mode (FAM)
sequences and the large swings seen within a single night.

Alongside it the script builds v-mode 1 of the Optical Feedback Control (OFC) 22-degree-of-
freedom (DOF) / 12-v-mode scheme in three cumulative forms, so that the measured focus can
be compared against the commanded optical state:

- **LUT**       -- the hexapod look-up table baseline, from ``lut_dof0`` and ``lut_dof5``.
- **LUT+Trim**  -- plus the accumulated Trim offset (``dof0``, ``dof5`` and the three
  mirror bending terms). A physical hexapod position is LUT + Trim; neither term alone is
  the position, and on the camera-hexapod dz axis the two are strongly anti-correlated, so
  a v-mode from the Trim alone has the wrong variance rather than merely a missing offset.
- **LUT+Trim+Deviation** -- plus the v-mode 1 amplitude of the *measured* Double Zernike
  (DZ), obtained by projecting the raw DZ fit onto the same OFC singular-value
  decomposition (SVD) that defines the v-modes. This is the closest thing to the true
  optical state: commanded position plus the departure the wavefront actually measured.

The commanded part uses per-DOF coefficients from the ts_ofc StateEstimator:

    v1_cmd = C_CamHexdz * (lut_dof5 + dof5) + C_M2Hexdz * (lut_dof0 + dof0)
             + C_M1M3B3 * dof12 + C_M2B5 * dof34 + C_M2B4 * dof33

The mirror-mode LUT is omitted deliberately: it has never been changed from the
mirror-laboratory values, and the three mirror bending terms carry only 0.063 of v-mode
1's normalized weight in total (the two hexapod dz axes carry 0.759 and 0.650).

The Deviation sign is not fixed a priori -- the measured wavefront could add to or oppose
the commanded state -- so both signs are evaluated against truss temperature and the one
giving the tighter residual is used for the cumulative panel, with the alternative
reported in the summary table.

Every DZ(1,4) panel carries a secondary axis in µm of effective hexapod dz. The factor is
derived from the SVD rather than fitted: v-mode 1 is almost purely DZ(1,4), so dividing the
mean of the camera- and M2-hexapod dz coefficients by ``U_eff[(1,4), 0]`` gives about
-1110 µm of hexapod dz per µm of wavefront of DZ(1,4).

The last two pages close the loop on the thermal interpretation. Chaining the v-mode-1
sensitivity to truss temperature through the v-mode-1 hexapod dz coefficient gives about
107 µm of hexapod dz per °C, against 94 µm per °C for a 7835 mm steel truss at 12 ppm per
°C -- agreement to 14%, so the commanded focus tracking is consistent with a thermally
expanding truss. The truss length is taken from the LTS-213 assembly drawing, page 1 of
which is appended when available.

Fits are Huber M-estimators (`statsmodels` RLM with `HuberT`), and both Pearson r and
Spearman rho are reported, per the repository's convention for AOS correlations.

Needs `lsst.ts.ofc` and `lsst.ts.intrinsic.wavefront` for the v-mode SVD -- RSP only.

Usage:
    python code/correlations/run_dz14_truss.py --param-set <ps>
    python code/correlations/run_dz14_truss.py --param-set <ps> --dz-prefix z1toz3
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm
from astropy.table import QTable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))   # repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # aos/code
from common.utils import nmad, alt_to_deg  # noqa: E402
from aos_state import DOF22  # noqa: E402  canonical 22-DOF index list

from lsst.ts.intrinsic.wavefront import ofc_svd as osv  # noqa: E402

# v-mode 1 of the OFC 22_12 scheme, as v-mode amplitude per unit DOF, from the ts_ofc
# StateEstimator (notebooks/smatrix_vmode/vmode_dof_ts_ofc.ipynb). Units 1/µm.
# The normalized weight of each term is given for context: the two hexapod dz axes carry
# 0.759 and 0.650, the three mirror bending modes only 0.033, 0.029 and 0.001.
V1_HEX = {5: -0.0008915,    # camera hexapod dz [1/µm], normalized weight -0.759
          0: -0.0009104}    # M2 hexapod dz     [1/µm], normalized weight -0.650
V1_BEND = {12: +0.1172,     # M1M3 bending mode 3 [1/µm], normalized weight +0.033
           34: +0.1142,     # M2 bending mode 5   [1/µm], normalized weight +0.029
           33: +0.001562}   # M2 bending mode 4   [1/µm], normalized weight +0.001

# A consecutive-visit change in v-mode 1 larger than this (dimensionless) is a real slew
# rather than tracking drift. The distribution of |delta v1_lut| between consecutive
# same-night visits is strongly bimodal -- median 0.00012, 75th percentile 0.0048, 90th
# percentile 0.153 -- so this threshold sits in the empty gap between plateaus and slews.
LUT_STEP = 0.05
TRIM_STEP = 0.01     # same idea for the Trim, which steps at closed-loop alignments
PANELS_PER_PAGE = 6
# A night is shown if it contains at least one contiguous block of this many FAM triplets,
# matching the block definition in code/coadd/run_fam_coadd_miw.py: a new block starts
# whenever day_obs changes or the seq_num step away from the triplet spacing of 3.
MIN_CONTIG_TRIPLETS = 12
TRIPLET_STEP = 3

# The hexapod look-up table was updated on or about 2025-12-09; v-mode 1 from the LUT is
# taken as settled after this date. MJD 61018.0 is 2025-12-09.
LUT_FIX_DAY_OBS = 20251209
LUT_FIX_MJD = 61018.0

# Mean of the two hexapod dz coefficients of v-mode 1, in v-mode amplitude per µm of
# hexapod dz. The camera and M2 values differ by only 2.1%, so a single effective dz is a
# meaningful secondary axis on the DZ(1,4) plots.
V1_PER_UM_DZ = 0.5 * (abs(V1_HEX[5]) + abs(V1_HEX[0]))   # [1/µm], = 9.0095e-4

# Thermal cross-check of the focus--temperature chain. The truss length is taken from the
# LTS-213 telescope assembly drawing, from the elevation axis to the top of the lower top-end
# right light baffle, and steel is used for the coefficient of thermal expansion.
TRUSS_LENGTH_MM = 7835.0
STEEL_CTE_PER_C = 12e-6

# Page 1 of the LTS-213 telescope assembly drawing, included in the PDF to show where the
# truss length is measured. Not in this repository (it is a controlled document, and large);
# pass --lts213-pdf to point at a copy elsewhere.
LTS213_DEFAULT = '~/Documents/LSST/LTS-213.pdf'
LTS213_URL = 'https://docushare.lsst.org/docushare/dsweb/Get/LTS-213'


def constants_page(pdf, df, dz_to_um, trim_slope=None):
    """A text page of the conversion constants and the thermal cross-check.

    Chains the measured v-mode-1 sensitivity to truss temperature through the v-mode-1
    hexapod dz coefficient to get an equivalent hexapod motion per °C, and compares that
    against the thermal expansion of a steel truss of the LTS-213 length.

    Parameters
    ----------
    dz_to_um : `float`
        Effective hexapod dz in µm per µm of wavefront of DZ(k=1, j=4); negative.
    trim_slope : `float`, optional
        Huber slope of v-mode 1 from LUT+Trim against truss temperature, in dimensionless
        v-mode amplitude per °C. Taken from the lower Trim population, which holds 1367 of
        the 1591 visits with a truss temperature.
    """
    um_per_c = trim_slope / V1_PER_UM_DZ if trim_slope else np.nan
    exp_um_per_c = TRUSS_LENGTH_MM * 1e3 * STEEL_CTE_PER_C
    dz14_per_c = um_per_c / abs(dz_to_um) if np.isfinite(dz_to_um) else np.nan

    lines = [
        ('h', 'Measured chain: truss temperature to focus'),
        ('r', 'd(v1) / d(truss T)',
         f'{trim_slope:+.5f} per deg C',
         'dimensionless v-mode 1 amplitude per deg C; Huber slope of the lower Trim '
         'population (n=1367)'),
        ('r', 'v1 per hexapod dz',
         f'{V1_PER_UM_DZ:.4e} per um',
         'mean of the camera-hexapod dz and M2-hexapod dz coefficients of v-mode 1, '
         'which agree to 2.1%'),
        ('r', 'DZ(1,4) to hexapod dz',
         f'{dz_to_um:+.1f} um per um',
         'um of effective hexapod dz per um of wavefront of DZ(k=1, j=4); from '
         'U_eff[(1,4),0] = -0.999919'),
        ('b', 'implied focus sensitivity',
         f'{um_per_c:.1f} um of hexapod dz per deg C',
         'the first two rows divided: (d v1 / d T) / (v1 per um of dz)'),
        ('r', 'equivalently, in wavefront',
         f'{dz14_per_c:.4f} um of wavefront per deg C',
         'the same sensitivity expressed as DZ(k=1, j=4)'),
        ('h', 'Thermal cross-check: steel truss expansion'),
        ('r', 'truss length',
         f'{TRUSS_LENGTH_MM:.0f} mm',
         'LTS-213, elevation axis to the top of the lower top-end right light baffle; '
         f'see {LTS213_URL}'),
        ('r', 'coefficient of thermal expansion',
         f'{STEEL_CTE_PER_C * 1e6:.0f} ppm per deg C',
         'steel'),
        ('b', 'predicted expansion',
         f'{exp_um_per_c:.0f} um per deg C',
         'length times coefficient of thermal expansion'),
        ('h', 'Comparison'),
        ('b', 'measured / predicted',
         f'{um_per_c / exp_um_per_c:.2f} (dimensionless)',
         f'{um_per_c:.0f} um per deg C measured against {exp_um_per_c:.0f} um per deg C '
         'from the steel truss'),
        ('r', 'implied effective length',
         f'{um_per_c / STEEL_CTE_PER_C / 1e3:.0f} mm',
         f'the length a steel member would need to give {um_per_c:.0f} um per deg C'),
        ('r', 'implied coefficient of expansion',
         f'{um_per_c / (TRUSS_LENGTH_MM * 1e3) * 1e6:.1f} ppm per deg C',
         f'the coefficient a {TRUSS_LENGTH_MM:.0f} mm member would need instead'),
    ]

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.955, 'Conversion constants: uniform defocus, hexapod dz and truss '
             'temperature', ha='center', fontsize=13)
    fig.text(0.5, 0.925, 'the commanded focus sensitivity to truss temperature agrees '
             'with a thermally expanding steel truss to '
             f'{abs(um_per_c / exp_um_per_c - 1) * 100:.0f}%',
             ha='center', fontsize=10, style='italic')
    y = 0.875
    for kind, *rest in lines:
        if kind == 'h':
            y -= 0.016
            fig.text(0.06, y, rest[0], fontsize=11, weight='bold')
            fig.text(0.06, y - 0.011, '_' * 118, fontsize=7, color='0.6')
            y -= 0.040
            continue
        name, value, note = rest
        weight = 'bold' if kind == 'b' else 'normal'
        color = 'C0' if kind == 'b' else 'k'
        fig.text(0.08, y, name, fontsize=10, weight=weight)
        fig.text(0.42, y, value, fontsize=10, weight=weight, color=color,
                 family='monospace')
        fig.text(0.08, y - 0.019, note, fontsize=8, color='0.35')
        y -= 0.049

    fig.text(0.06, 0.045,
             'All v-mode amplitudes are dimensionless; the v-mode-1 basis vector is '
             '99.98% DZ(k=1, j=4), so a\nunit v-mode amplitude is very nearly one um of '
             'wavefront of uniform defocus. The sign convention\nis that a positive '
             'DZ(k=1, j=4) corresponds to a negative hexapod dz.',
             fontsize=8, color='0.35')
    pdf.savefig(fig)
    plt.close(fig)
    return dict(kind='constants', x='truss', y='dz14',
                trim_slope_per_c=float(trim_slope) if trim_slope else np.nan,
                v1_per_um_dz=float(V1_PER_UM_DZ), dz14_to_um_hex=float(dz_to_um),
                um_hex_per_c=float(um_per_c), truss_length_mm=float(TRUSS_LENGTH_MM),
                steel_cte_per_c=float(STEEL_CTE_PER_C),
                thermal_um_per_c=float(exp_um_per_c),
                measured_over_thermal=float(um_per_c / exp_um_per_c))


def lts213_page(pdf, path):
    """Append page 1 of the LTS-213 telescope assembly drawing, if it is available.

    The drawing shows where the truss length used in `constants_page` is measured. It is not
    in this repository, so a missing file is reported and skipped rather than raising.

    Parameters
    ----------
    path : `str` or `None`
        Path to the LTS-213 PDF; ``~`` is expanded.

    Returns
    -------
    ok : `bool`
        True if the drawing page was added.
    """
    if not path:
        return False
    p = pathlib.Path(path).expanduser()
    if not p.is_file():
        print(f'  LTS-213 drawing not found at {p} -- skipping that page')
        return False
    try:
        import fitz
    except ImportError:
        print('  pymupdf not available -- skipping the LTS-213 drawing page')
        return False
    doc = fitz.open(str(p))
    pix = doc[0].get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.88])
    ax.imshow(img[:, :, :3] if pix.n >= 3 else img[:, :, 0], cmap=None)
    ax.axis('off')
    fig.suptitle('LTS-213 telescope assembly, page 1\n'
                 f'the truss length used above, {TRUSS_LENGTH_MM:.0f} mm, runs from the '
                 'elevation axis to the top of the\nlower top-end right light baffle',
                 fontsize=10)
    fig.text(0.5, 0.030,
             'The dimension callouts render as mojibake: the drawing embeds Identity-H '
             'Arial subsets whose font programs neither\nMuPDF nor Ghostscript can parse, '
             'so CIDs fall through as raw codes. The geometry and the notes are unaffected; '
             'read\nthe dimensions from the source document.',
             ha='center', fontsize=7, color='0.35')
    fig.text(0.5, 0.008, LTS213_URL, ha='center', fontsize=7, color='0.35')
    pdf.savefig(fig)
    plt.close(fig)
    return True


def robust_line(x, y):
    """Huber-RLM line plus Pearson and Spearman statistics for one (x, y) pair.

    Parameters
    ----------
    x, y : `array_like`
        Paired samples in any units; NaN pairs are dropped. The returned slope carries
        units of y per unit of x.

    Returns
    -------
    stats : `dict` or `None`
        ``n``, ``intercept``, ``slope``, ``slope_err`` (Huber RLM), ``pearson_r``,
        ``pearson_p``, ``spearman_rho``, ``spearman_p``, and ``resid_nmad`` (robust
        scatter of the residual, in units of y). `None` if fewer than 10 finite pairs.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 10:
        return None
    x, y = x[m], y[m]
    X = sm.add_constant(x)
    r = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return dict(n=int(x.size), intercept=float(r.params[0]), slope=float(r.params[1]),
                slope_err=float(r.bse[1]), pearson_r=float(pr), pearson_p=float(pp),
                spearman_rho=float(sr), spearman_p=float(sp),
                resid_nmad=float(nmad(y - r.predict(X))))


def label_runs(df, col, step, by='day_obs'):
    """Integer run id per visit: a new night or a jump in `col` starts a new run.

    Used to group visits that share one look-up table (LUT) plateau or one Trim setting,
    so a trace can be coloured by which commanded state was in force.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Must be sorted by ``(day_obs, seq_num)``.
    col : `str`
        Column to watch for jumps, in its own units.
    step : `float`
        A consecutive-visit change larger than this, in the units of `col`, starts a new
        run.
    by : `str`, optional
        Column whose change also forces a new run, normally ``day_obs``.

    Returns
    -------
    run_id : `numpy.ndarray`
        Run id per row, counting from 1. NaN values in `col` never start a run.
    """
    v = df[col].to_numpy(float)
    b = df[by].to_numpy()
    out = np.empty(len(df), dtype=int)
    cur = 0
    for i in range(len(df)):
        if i == 0 or b[i] != b[i - 1]:
            cur += 1
        elif np.isfinite(v[i]) and np.isfinite(v[i - 1]) and abs(v[i] - v[i - 1]) > step:
            cur += 1
        out[i] = cur
    return out


def label_contiguous_blocks(df):
    """Integer block id per triplet, using the coadd study's contiguity definition.

    A new block starts whenever ``day_obs`` changes or the ``seq_num`` step differs from
    the FAM triplet spacing of 3, matching ``code/coadd/run_fam_coadd_miw.py`` so that
    "contiguous block of N triplets" means the same thing in both studies.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Must be sorted by ``(day_obs, seq_num)``.

    Returns
    -------
    block : `numpy.ndarray`
        Block id per row, counting from 1.
    """
    new = ((df['day_obs'] != df['day_obs'].shift())
           | ((df['seq_num'] - df['seq_num'].shift()) != TRIPLET_STEP))
    return new.cumsum().to_numpy()


def add_dz_hexapod_axis(ax, dz_to_um):
    """Add a right-hand axis expressing DZ(k=1, j=4) as an effective hexapod dz.

    v-mode 1 is almost purely DZ(k=1, j=4) -- the singular vector puts -0.99992 of its
    DZ content there -- and the camera and M2 hexapod dz coefficients of v-mode 1 agree
    to 2.1%, so one effective dz stands for both.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axes whose y-axis is DZ(k=1, j=4) in µm of wavefront.
    dz_to_um : `float`
        Effective hexapod dz in µm per µm of wavefront of DZ(1,4).
    """
    sec = ax.secondary_yaxis('right', functions=(lambda v: v * dz_to_um,
                                                 lambda v: v / dz_to_um))
    sec.set_ylabel('effective hexapod dz [um]', fontsize=9)
    return sec


def project_v1(df, prefix, iZs, k_min, k_max):
    """v-mode 1 amplitude of the measured DZ, projected onto the OFC 22_12 SVD.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Must carry the DZ coefficient columns ``{prefix}_z{j}_c{k}`` in µm of wavefront.
    prefix : `str`
        DZ column prefix, e.g. ``z1toz6``.
    iZs : `list` of `int`
        Noll indices of the pupil Zernikes the DZ fit used.
    k_min, k_max : `int`
        Focal-Zernike order range of the DZ fit.

    Returns
    -------
    v1 : `numpy.ndarray`
        v-mode 1 amplitude per visit, dimensionless.
    dz_to_um : `float`
        Effective hexapod dz in µm per µm of wavefront of DZ(k=1, j=4), for the secondary
        axis. NaN if (k=1, j=4) is outside the fit's focal-order range.
    """
    svd = osv.build_ofc_svd(iZs, int(k_min), int(k_max), 12, n_dof=DOF22)
    W = np.full((len(df), len(svd.kj_grid)), np.nan)
    missing = 0
    for ci, (k, j) in enumerate(svd.kj_grid):
        col = f'{prefix}_z{j}_c{k}'
        if col in df.columns:
            W[:, ci] = df[col].to_numpy(float)
        else:
            missing += 1
    if missing:
        print(f'  WARNING: {missing}/{len(svd.kj_grid)} DZ columns absent from the fit')

    # DZ(1,4) content of a unit-amplitude v-mode 1, in µm of wavefront: the effective
    # hexapod dz per µm of wavefront of DZ(1,4) follows from it and the mean coefficient.
    i14 = [i for i, (k, j) in enumerate(svd.kj_grid) if (k, j) == (1, 4)]
    dz_to_um = np.nan
    if i14:
        u14 = float(svd.U_eff[i14[0], 0])
        if u14 != 0.0:
            dz_to_um = 1.0 / (V1_PER_UM_DZ * u14)
    return svd.vmodes(svd.project_amplitudes(W))[:, 0], dz_to_um


def load(param_set, dz_prefix, output_root='output'):
    """Join the DZ fit, the truss temperature and the commanded DOF, one row per visit.

    Returns
    -------
    df : `pandas.DataFrame`
        Quality-passing FAM visits with ``dz14`` (µm of wavefront), ``truss`` (deg C),
        ``elev`` and ``rot`` (deg), ``mjd`` (days), the four cumulative v-mode 1 variants
        ``v1_lut``, ``v1_lut_trim``, ``v1_dev`` and ``v1_total`` (all dimensionless), and
        the run labels ``lut_run`` and ``trim_run``.
    sign : `int`
        The Deviation sign (+1 or -1) chosen for ``v1_total``.
    both : `dict`
        Residual nMAD of DZ(1,4) against each candidate sign, for reporting.
    dz_to_um : `float`
        Effective hexapod dz in µm per µm of wavefront of DZ(k=1, j=4).
    """
    base = pathlib.Path(output_root) / param_set
    k_min, k_max = 1, int(dz_prefix.split('toz')[-1])
    dz_col = f'{dz_prefix}_z4_c1'

    fits = pd.read_parquet(base / 'fits.parquet')
    keep = [c for c in fits.columns
            if c.startswith(f'{dz_prefix}_z') or c in (
                'day_obs', 'seq_num', 'mjd', 'alt', 'rotator_angle', 'band',
                'visit_quality_pass', f'{dz_prefix}_bad_fit',
                'tma_truss_temp_pxpy', 'tma_truss_temp_mxmy')]
    fits = fits[keep]

    dof_cols = ([f'dof{k}' for k in list(V1_HEX) + list(V1_BEND)]
                + [f'lut_dof{k}' for k in V1_HEX])
    vis = pd.read_parquet(base / 'visits.parquet',
                          columns=['day_obs', 'seq_num'] + dof_cols)
    df = fits.merge(vis, on=['day_obs', 'seq_num'], how='left', validate='one_to_one')

    df = df[(~df[f'{dz_prefix}_bad_fit'].astype(bool))
            & df['visit_quality_pass'].astype(bool)].copy()
    df = df.sort_values(['day_obs', 'seq_num']).reset_index(drop=True)

    df['dz14'] = df[dz_col]
    df['truss'] = 0.5 * (df['tma_truss_temp_pxpy'] + df['tma_truss_temp_mxmy'])
    # `alt` arrives in radians from this source; alt_to_deg auto-detects and converts.
    df['elev'] = alt_to_deg(df['alt'].to_numpy(float))
    df['rot'] = df['rotator_angle'].to_numpy(float)

    # v-mode 1 from the commanded DOF. The hexapod dz axes need LUT + Trim to be a
    # physical position; the mirror bending modes use Trim alone (see module docstring).
    df['v1_lut'] = sum(c * df[f'lut_dof{k}'] for k, c in V1_HEX.items())
    df['v1_trim'] = (sum(c * df[f'dof{k}'] for k, c in V1_HEX.items())
                     + sum(c * df[f'dof{k}'] for k, c in V1_BEND.items()))
    df['v1_lut_trim'] = df['v1_lut'] + df['v1_trim']

    # The Deviation term: v-mode 1 of the measured DZ itself.
    iZs = [int(j) for j in
           np.asarray(QTable.read(str(base / 'visits.parquet'))['nollIndices'][0]).tolist()]
    df['v1_dev'], dz_to_um = project_v1(df, dz_prefix, iZs, k_min, k_max)

    # Choose the Deviation sign empirically: whichever makes DZ(1,4) vs the cumulative
    # v-mode the tighter relation. Both are reported so the choice stays visible.
    both = {}
    for s in (+1, -1):
        f = robust_line(df['v1_lut_trim'] + s * df['v1_dev'], df['dz14'])
        both[s] = f['resid_nmad'] if f else np.inf
    sign = min(both, key=both.get)
    df['v1_total'] = df['v1_lut_trim'] + sign * df['v1_dev']

    df['lut_run'] = label_runs(df, 'v1_lut', LUT_STEP)
    df['trim_run'] = label_runs(df, 'v1_trim', TRIM_STEP)
    df['block'] = label_contiguous_blocks(df)
    return df, sign, both, dz_to_um


def scatter_page(pdf, rows, df, xc, yc, xlabel, ylabel, title, subtitle=None,
                 kind='pooled', color='C0', cbar=None, cbar_label=None,
                 fit=True, dz_to_um=None):
    """One full-page scatter, optionally with a Huber-RLM line and its statistics.

    Parameters
    ----------
    fit : `bool`, optional
        Draw the Huber line and annotate the statistics. Set False for a plot shown for
        information only.
    dz_to_um : `float`, optional
        If given, add a right-hand axis converting the y-axis from µm of wavefront of
        DZ(k=1, j=4) to an effective hexapod dz in µm.
    """
    fig, ax = plt.subplots(figsize=(8.5, 8.0))
    x = df[xc].to_numpy(float)
    y = df[yc].to_numpy(float)
    if cbar is not None:
        c = df[cbar].to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
        sc = ax.scatter(x[m], y[m], c=c[m], s=7, alpha=0.75, cmap='viridis')
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(cbar_label or cbar, fontsize=9)
    else:
        ax.plot(x, y, '.', ms=4, alpha=0.45, color=color)

    f = robust_line(x, y) if fit else None
    if f is not None:
        xs = np.linspace(np.nanmin(x), np.nanmax(x), 50)
        ax.plot(xs, f['intercept'] + f['slope'] * xs, 'r-', lw=2.0,
                label=(f"Huber RLM slope {f['slope']:+.4g} +/- {f['slope_err']:.3g}\n"
                       f"Pearson r {f['pearson_r']:+.3f}\n"
                       f"Spearman rho {f['spearman_rho']:+.3f}\n"
                       f"n = {f['n']}\n"
                       f"residual nMAD {f['resid_nmad']:.4g}"))
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        rows.append(dict(kind=kind, x=xc, y=yc, **f))
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_title(title + (f'\n{subtitle}' if subtitle else ''), fontsize=11)
    if xc == 'mjd':
        ax.axvline(LUT_FIX_MJD, color='r', lw=1.2, ls='--')
        ax.text(LUT_FIX_MJD, ax.get_ylim()[1], ' hexapod LUT updated 2025-12-09',
                color='r', fontsize=8, va='top', ha='left')
    if dz_to_um is not None and np.isfinite(dz_to_um):
        add_dz_hexapod_axis(ax, dz_to_um)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
    return f


def lut_validation_page(pdf, df):
    """The hexapod-LUT v-mode 1 against elevation, rotator angle and time.

    The hexapod LUT is two-dimensional in (elevation, rotator angle), so neither
    one-dimensional projection is expected to be a single curve. The fourth panel is the
    two-dimensional map itself, restricted to the epoch after the LUT was last updated so
    that a single LUT version is in force.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    t0 = float(np.nanmin(df['mjd']))
    days = df['mjd'].to_numpy(float) - t0

    ax = axes[0, 0]
    sc = ax.scatter(df['elev'], df['v1_lut'], c=days, s=7, cmap='viridis', alpha=0.8)
    fig.colorbar(sc, ax=ax).set_label(f'days since MJD {t0:.1f}', fontsize=8)
    ax.set_xlabel('elevation [deg]', fontsize=9)
    ax.set_ylabel('v1 from LUT [dimensionless]', fontsize=9)
    ax.set_title('v1 LUT vs elevation: one curve per LUT version', fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    sc = ax.scatter(df['rot'], df['v1_lut'], c=days, s=7, cmap='viridis', alpha=0.8)
    fig.colorbar(sc, ax=ax).set_label(f'days since MJD {t0:.1f}', fontsize=8)
    ax.set_xlabel('camera rotator angle [deg]', fontsize=9)
    ax.set_ylabel('v1 from LUT [dimensionless]', fontsize=9)
    ax.set_title('v1 LUT vs rotator angle: the LUT is 2D, so structure is expected',
                 fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(df['mjd'], df['v1_lut'], '.', ms=3, alpha=0.6, color='C0')
    ax.axvline(LUT_FIX_MJD, color='r', lw=1.2, ls='--')
    ax.text(LUT_FIX_MJD, ax.get_ylim()[1], ' 2025-12-09', color='r', fontsize=8,
            va='top', ha='left')
    ax.set_xlabel('MJD [days]', fontsize=9)
    ax.set_ylabel('v1 from LUT [dimensionless]', fontsize=9)
    ax.set_title('v1 LUT time history; the LUT was updated on or about 2025-12-09',
                 fontsize=9)
    ax.grid(alpha=0.3)

    # The LUT is two-dimensional, so map it in (elevation, rotator angle) directly. One
    # epoch only, after the last LUT update, so a single LUT version is in force.
    ax = axes[1, 1]
    late = df[(df['day_obs'] >= LUT_FIX_DAY_OBS) & np.isfinite(df['v1_lut'])
              & np.isfinite(df['elev']) & np.isfinite(df['rot'])]
    if len(late) > 20:
        e_bins = np.arange(np.floor(late['elev'].min() / 5) * 5,
                           np.ceil(late['elev'].max() / 5) * 5 + 5, 5.0)
        r_bins = np.arange(np.floor(late['rot'].min() / 10) * 10,
                           np.ceil(late['rot'].max() / 10) * 10 + 10, 10.0)
        s = late['v1_lut'].to_numpy(float)
        cnt, _, _ = np.histogram2d(late['elev'], late['rot'], bins=[e_bins, r_bins])
        tot, _, _ = np.histogram2d(late['elev'], late['rot'], bins=[e_bins, r_bins],
                                   weights=s)
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = np.where(cnt > 0, tot / cnt, np.nan)
        pm = ax.pcolormesh(r_bins, e_bins, np.ma.masked_invalid(mean),
                           cmap='RdYlBu_r', shading='flat')
        fig.colorbar(pm, ax=ax).set_label('mean v1 from LUT [dimensionless]', fontsize=8)
        ax.set_title(f'v1 LUT over (elevation, rotator), day_obs >= {LUT_FIX_DAY_OBS}\n'
                     f'n={len(late)} visits, {int((cnt > 0).sum())} of {cnt.size} '
                     f'cells occupied (5 deg x 10 deg bins)', fontsize=8)
    ax.set_xlabel('camera rotator angle [deg]', fontsize=9)
    ax.set_ylabel('elevation [deg]', fontsize=9)

    fig.suptitle('Validation of the hexapod LUT v-mode 1\n'
                 'v1 LUT is built from lut_dof5 (camera hexapod dz) and lut_dof0 '
                 '(M2 hexapod dz)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    pdf.savefig(fig)
    plt.close(fig)


def split_offset_populations(df, xc='truss', yc='v1_trim'):
    """Separate two populations offset in `yc` at fixed `xc`, by a gap in the residual.

    The v-mode-1 Trim against truss temperature falls into two bands with a similar slope
    and a large offset. A single date boundary cannot define them -- three nights contain
    visits from both bands -- so the split is made on the residual about one common Huber
    line, cut at the widest empty gap in that residual's distribution. The date intervals
    each population covers are then reported rather than assumed.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Must carry `xc`, `yc` and ``day_obs``.
    xc, yc : `str`, optional
        Column names; the residual and its cut carry the units of `yc`.

    Returns
    -------
    split : `dict` or `None`
        ``cut`` (residual value of the split, units of `yc`), ``gap`` (width of the empty
        gap, same units), boolean mask ``hi``, the two per-population fits ``fit_hi`` /
        ``fit_lo``, and ``days_hi`` / ``days_lo`` (sorted `day_obs` lists). `None` if the
        residual shows no clear gap.
    """
    m = np.isfinite(df[xc]) & np.isfinite(df[yc])
    d = df[m]
    if len(d) < 60:
        return None
    f = robust_line(d[xc], d[yc])
    if f is None:
        return None
    res = (d[yc] - (f['intercept'] + f['slope'] * d[xc])).to_numpy(float)

    # Widest empty gap in the sorted residual, ignoring the sparse tails so a single
    # outlier cannot define the split.
    s = np.sort(res)
    lo_i, hi_i = int(0.02 * s.size), int(0.98 * s.size)
    seg = s[lo_i:hi_i]
    gaps = np.diff(seg)
    gi = int(np.argmax(gaps))
    gap = float(gaps[gi])
    # A real bimodality has a gap far wider than the typical spacing between points.
    if gap < 10 * float(np.median(gaps[gaps > 0])):
        return None
    cut = float(0.5 * (seg[gi] + seg[gi + 1]))

    hi = pd.Series(res > cut, index=d.index)
    fh = robust_line(d[xc][hi], d[yc][hi])
    fl = robust_line(d[xc][~hi], d[yc][~hi])
    if fh is None or fl is None:
        return None
    return dict(cut=cut, gap=gap, hi=hi, fit_hi=fh, fit_lo=fl,
                days_hi=sorted(int(v) for v in d['day_obs'][hi].unique()),
                days_lo=sorted(int(v) for v in d['day_obs'][~hi].unique()))


def trim_population_pages(pdf, rows, df, split):
    """v1 Trim vs truss temperature, pooled and then split into the two populations."""
    scatter_page(pdf, rows, df, 'truss', 'v1_trim',
                 'mean TMA truss temperature [deg C]',
                 'v1 from Trim alone [dimensionless]',
                 'v-mode 1 from the Trim alone vs TMA truss temperature',
                 'two populations with the same slope and different intercept; '
                 'the single pooled line below fits neither',
                 kind='trim_pooled', color='C2')

    if split is None:
        return
    hi = split['hi'].reindex(df.index, fill_value=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, (sel, lbl, tag, col, days) in zip(axes, [
            (hi, 'upper population', 'trim_hi', 'C1', split['days_hi']),
            (~hi, 'lower population', 'trim_lo', 'C0', split['days_lo'])]):
        d = df[sel]
        ax.plot(d['truss'], d['v1_trim'], '.', ms=4, alpha=0.5, color=col)
        f = robust_line(d['truss'], d['v1_trim'])
        if f is not None:
            xs = np.linspace(np.nanmin(d['truss']), np.nanmax(d['truss']), 50)
            ax.plot(xs, f['intercept'] + f['slope'] * xs, 'r-', lw=2.0)
            ax.set_title(f"{lbl}\nslope {f['slope']:+.4g} +/- {f['slope_err']:.3g} "
                         f"per deg C, intercept {f['intercept']:+.4g}\n"
                         f"Pearson r {f['pearson_r']:+.3f}, "
                         f"Spearman rho {f['spearman_rho']:+.3f}, n={f['n']}\n"
                         f"{len(days)} nights, {days[0]} to {days[-1]}", fontsize=8)
            rows.append(dict(kind=tag, x='truss', y='v1_trim', **f))
        ax.set_xlabel('mean TMA truss temperature [deg C]', fontsize=10)
        ax.set_ylabel('v1 from Trim alone [dimensionless]', fontsize=10)
        ax.grid(alpha=0.3)
        # Zoom each panel to its own population, clipping the few far outliers so the
        # band's structure is visible rather than compressed to a line.
        v = d['v1_trim'].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size > 20:
            lo, hi_q = np.percentile(v, [0.5, 99.5])
            pad = 0.08 * max(hi_q - lo, 1e-6)
            ax.set_ylim(lo - pad, hi_q + pad)
    fh, fl = split['fit_hi'], split['fit_lo']
    both_days = sorted(set(split['days_hi']) & set(split['days_lo']))
    fig.suptitle(
        'The two Trim populations, split on the residual about a common Huber line\n'
        f'offset {fh["intercept"] - fl["intercept"]:+.4g} (dimensionless) at fixed '
        f'temperature, across an empty residual gap of {split["gap"]:.3g}; '
        f'{len(both_days)} nights contain both, so no single date defines the split',
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    pdf.savefig(fig)
    plt.close(fig)


def night_trace_pages(pdf, df, dz_to_um=None, min_contig=MIN_CONTIG_TRIPLETS):
    """DZ(1,4) and truss temperature against seq_num, for every qualifying night.

    A night qualifies if it contains at least one contiguous block of
    ``MIN_CONTIG_TRIPLETS`` FAM triplets, using the same block definition as the coadd
    study. That is a criterion on the observing pattern, not on any value of DZ(1,4), so
    the figure is not selected on the result. Marker colour groups visits sharing one
    look-up table (LUT) plateau; a colour change is a telescope slew or a LUT change, and
    an open triangle flags a visit where the Trim also stepped, which is a closed-loop
    alignment.

    Returns
    -------
    days : `numpy.ndarray`
        The qualifying ``day_obs`` values, in order, one panel each.
    """
    block_max = df.groupby(['day_obs', 'block']).size().groupby('day_obs').max()
    days = np.array(sorted(block_max[block_max >= min_contig].index))
    n_truss = df.groupby('day_obs')['truss'].count()
    n_no_truss = int(sum(1 for d in days if n_truss.get(d, 0) == 0))
    print(f'  nights with a contiguous block of {min_contig}+ FAM triplets: '
          f'{len(days)} of {len(block_max)} ({n_no_truss} of them with no truss '
          f'temperature)')
    if len(days) == 0:
        return np.array([], dtype=int)
    palette = plt.get_cmap('tab10').colors

    for start in range(0, len(days), PANELS_PER_PAGE):
        chunk = days[start:start + PANELS_PER_PAGE]
        fig, axes = plt.subplots(3, 2, figsize=(13, 9))
        for ax, day in zip(axes.ravel(), chunk):
            d = df[df['day_obs'] == day].sort_values('seq_num')
            runs = list(dict.fromkeys(d['lut_run']))
            for ri, r in enumerate(runs):
                s = d[d['lut_run'] == r]
                col = palette[ri % len(palette)]
                # Break the connecting line across seq_num gaps, so a straight segment
                # never implies visits that were not taken.
                sn = s['seq_num'].to_numpy(float)
                y = s['dz14'].to_numpy(float).copy()
                if len(sn) > 1:
                    y_plot = np.where(np.r_[False, np.diff(sn) > 20], np.nan, y)
                    ax.plot(sn, y_plot, '-', lw=0.8, color=col, alpha=0.8)
                ax.plot(sn, y, 'o', ms=3.5, color=col)
            # mark the visits where the Trim stepped: closed-loop alignments
            tr = d['trim_run'].to_numpy()
            step = np.zeros(len(d), bool)
            step[1:] = tr[1:] != tr[:-1]
            if step.any():
                ax.plot(d['seq_num'].to_numpy()[step], d['dz14'].to_numpy()[step],
                        'k^', ms=6, mfc='none', mew=1.2)
            ax.set_ylabel('DZ(1,4) [um of wf]', fontsize=7, color='C0')
            ax.tick_params(labelsize=6)
            if dz_to_um is not None and np.isfinite(dz_to_um):
                # truss occupies the inner right-hand spine, so the effective-dz scale
                # goes on a spine offset further out to the right.
                sec = ax.secondary_yaxis(
                    1.20, functions=(lambda v: v * dz_to_um, lambda v: v / dz_to_um))
                sec.set_ylabel('eff. hex dz [um]', fontsize=6)
                sec.tick_params(labelsize=5)
            ax2 = ax.twinx()
            sn_all = d['seq_num'].to_numpy(float)
            tr_plot = d['truss'].to_numpy(float).copy()
            if len(sn_all) > 1:
                tr_plot = np.where(np.r_[False, np.diff(sn_all) > 20], np.nan, tr_plot)
            ax2.plot(sn_all, tr_plot, '-', lw=1.2, color='r', alpha=0.8)
            ax2.set_ylabel('truss [deg C]', fontsize=7, color='r')
            ax2.tick_params(labelsize=6, colors='r')
            tr_v = d['truss'].to_numpy(float)
            dz_v = d['dz14'].to_numpy(float)
            dz_rng = (np.nanmax(dz_v) - np.nanmin(dz_v)) if np.isfinite(dz_v).any() else np.nan
            tr_rng = (np.nanmax(tr_v) - np.nanmin(tr_v)) if np.isfinite(tr_v).any() else np.nan
            ax.set_title(f'day_obs {day}: {len(d)} visits, {len(runs)} LUT groups\n'
                         f'DZ(1,4) range {dz_rng:.3f} um of wf, '
                         f'truss range {tr_rng:.3f} deg C', fontsize=7)
            ax.set_xlabel('seq_num', fontsize=7)
            ax.grid(alpha=0.3)
        for ax in axes.ravel()[len(chunk):]:
            ax.axis('off')

        handles = [
            Line2D([], [], color=palette[0], marker='o', ls='-', ms=4,
                   label='DZ(k=1,j=4), left axis; one colour per LUT plateau'),
            Line2D([], [], color=palette[1], marker='o', ls='-', ms=4,
                   label='next LUT plateau (colour changes at a slew or LUT change)'),
            Line2D([], [], color='r', ls='-', lw=1.5,
                   label='mean TMA truss temperature, right axis'),
            Line2D([], [], color='k', marker='^', ls='none', ms=7, mfc='none', mew=1.2,
                   label='Trim step: closed-loop alignment'),
        ]
        fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=7,
                   frameon=True, bbox_to_anchor=(0.5, 0.005))
        page = start // PANELS_PER_PAGE + 1
        n_pages = int(np.ceil(len(days) / PANELS_PER_PAGE))
        fig.suptitle(
            f'Nights with a contiguous block of {min_contig}+ FAM triplets '
            f'(page {page} of {n_pages}): DZ(k=1,j=4) vs seq_num\n'
            'large focus swings occur with no matching truss-temperature motion; '
            'a nan truss range means the night has no truss reading at all',
            fontsize=10)
        fig.tight_layout(rect=[0, 0.055, 1, 0.92], w_pad=4.0)
        pdf.savefig(fig)
        plt.close(fig)
    return days


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--param-set', required=True)
    ap.add_argument('--dz-prefix', default='z1toz6', choices=['z1toz3', 'z1toz6'],
                    help='which focal-order DZ fit to read (default z1toz6)')
    ap.add_argument('--output-root', default='output')
    ap.add_argument('--output-dir', default=None,
                    help='default <output-root>/<param_set>/correlations')
    ap.add_argument('--min-contig-triplets', type=int, default=MIN_CONTIG_TRIPLETS,
                    help='a night is shown if it has a contiguous block of this many '
                         'FAM triplets (default %(default)s)')
    ap.add_argument('--lts213-pdf', default=LTS213_DEFAULT,
                    help='LTS-213 telescope assembly drawing, page 1 of which is appended '
                         'after the constants page; skipped if absent '
                         f'(default %(default)s, from {LTS213_URL})')
    args = ap.parse_args()

    out = pathlib.Path(args.output_dir or
                       f'{args.output_root}/{args.param_set}/correlations')
    out.mkdir(parents=True, exist_ok=True)

    df, sign, both, dz_to_um = load(args.param_set, args.dz_prefix, args.output_root)
    print(f'quality-passing FAM visits: {len(df)}')
    print(f"  with truss temperature:   {int(df['truss'].notna().sum())}")
    print(f"  with hexapod LUT:         {int(df['v1_lut'].notna().sum())}")
    print(f'  Deviation sign chosen: {sign:+d}  '
          f"(residual nMAD of DZ(1,4): {both[+1]:.4f} um of wavefront for +1, "
          f'{both[-1]:.4f} for -1)')
    for c, unit in [('dz14', 'um of wavefront'), ('truss', 'deg C'),
                    ('elev', 'deg'), ('rot', 'deg'),
                    ('v1_lut', 'dimensionless'), ('v1_trim', 'dimensionless'),
                    ('v1_lut_trim', 'dimensionless'), ('v1_dev', 'dimensionless'),
                    ('v1_total', 'dimensionless')]:
        v = df[c].to_numpy(float)
        v = v[np.isfinite(v)]
        print(f'  {c:12s} [{unit:15s}] n={v.size:5d} median {np.median(v):+.4f} '
              f'nMAD {nmad(v):.4f}')

    rows = [dict(kind='deviation_sign', x='v1_total', y='dz14', n=int(len(df)),
                 resid_nmad=float(both[sign]),
                 resid_nmad_plus=float(both[+1]), resid_nmad_minus=float(both[-1]),
                 deviation_sign=int(sign))]
    pdf_path = out / f'dz14_truss_{args.dz_prefix}.pdf'
    with PdfPages(str(pdf_path)) as pdf:
        # (a)-(d) the four requested scatters, one per page, cumulative in v-mode 1
        scatter_page(pdf, rows, df, 'truss', 'dz14',
                     'mean TMA truss temperature [deg C]',
                     'DZ(k=1, j=4) [um of wavefront]',
                     '(a) measured uniform defocus vs TMA truss temperature',
                     'DZ(k=1,j=4) is the focal-plane-uniform part of Z4 defocus',
                     kind='pooled', dz_to_um=dz_to_um)
        scatter_page(pdf, rows, df, 'truss', 'v1_lut',
                     'mean TMA truss temperature [deg C]',
                     'v1 from LUT [dimensionless]',
                     '(b) v-mode 1 from the hexapod LUT alone vs TMA truss temperature',
                     'the LUT is a model of elevation and temperature, so a trend here '
                     'is by construction', kind='pooled', color='C1')
        scatter_page(pdf, rows, df, 'truss', 'v1_lut_trim',
                     'mean TMA truss temperature [deg C]',
                     'v1 from LUT+Trim [dimensionless]',
                     '(c) v-mode 1 from LUT + Trim vs TMA truss temperature',
                     'LUT + Trim is the physical commanded hexapod position',
                     kind='pooled', color='C2')
        scatter_page(pdf, rows, df, 'truss', 'v1_total',
                     'mean TMA truss temperature [deg C]',
                     f'v1 from LUT+Trim{"+" if sign > 0 else "-"}Deviation '
                     '[dimensionless]',
                     '(d) v-mode 1 from LUT + Trim + Deviation vs TMA truss temperature',
                     f'Deviation is v-mode 1 of the measured DZ, entering with sign '
                     f'{sign:+d} (chosen by residual scatter)', kind='pooled', color='C3')

        # LUT validation
        lut_validation_page(pdf, df)

        # v1 LUT vs v1 Trim, coloured by time
        t0 = float(np.nanmin(df['mjd']))
        df['days'] = df['mjd'] - t0
        scatter_page(pdf, rows, df, 'v1_trim', 'v1_lut',
                     'v1 from Trim alone [dimensionless]',
                     'v1 from LUT alone [dimensionless]',
                     'v-mode 1 from the LUT vs from the Trim, coloured by time',
                     'a slope near -1 means the Trim undoes the LUT, so neither term '
                     'alone is the hexapod position', kind='lut_vs_trim',
                     cbar='days', cbar_label=f'days since MJD {t0:.1f}')

        # the two Trim populations
        split = split_offset_populations(df)
        if split is not None:
            fh, fl = split['fit_hi'], split['fit_lo']
            both_days = sorted(set(split['days_hi']) & set(split['days_lo']))
            print(f"\n  two Trim populations, split at residual {split['cut']:+.4f} "
                  f"across an empty gap of {split['gap']:.4f} (dimensionless)")
            print(f"    upper: n={fh['n']:5d}, intercept {fh['intercept']:+.4f}, "
                  f"slope {fh['slope']:+.4f} per deg C, {len(split['days_hi'])} nights "
                  f"{split['days_hi'][0]}-{split['days_hi'][-1]}")
            print(f"    lower: n={fl['n']:5d}, intercept {fl['intercept']:+.4f}, "
                  f"slope {fl['slope']:+.4f} per deg C, {len(split['days_lo'])} nights "
                  f"{split['days_lo'][0]}-{split['days_lo'][-1]}")
            print(f"    offset {fh['intercept'] - fl['intercept']:+.4f} (dimensionless) "
                  f"at fixed temperature")
            print(f"    nights containing BOTH populations: "
                  f"{both_days if both_days else 'none'}")
            rows.append(dict(kind='trim_split', x='truss', y='v1_trim',
                             n=fh['n'] + fl['n'], resid_cut=split['cut'],
                             resid_gap=split['gap'],
                             intercept_offset=fh['intercept'] - fl['intercept'],
                             n_days_both=len(both_days)))
        trim_population_pages(pdf, rows, df, split)

        # DZ(1,4) time history, for information only
        scatter_page(pdf, rows, df, 'mjd', 'dz14', 'MJD [days]',
                     'DZ(k=1, j=4) [um of wavefront]',
                     'DZ(k=1,j=4) time history',
                     'shown for information; no fit', fit=False,
                     dz_to_um=dz_to_um, color='C4')

        # per-night traces
        pick = night_trace_pages(pdf, df, dz_to_um, args.min_contig_triplets)
        print(f'\n  trace pages show day_obs: {", ".join(str(d) for d in pick)}')

        # last pages: the conversion constants and the LTS-213 drawing they refer to
        trim_slope = split['fit_lo']['slope'] if split is not None else None
        crow = constants_page(pdf, df, dz_to_um, trim_slope)
        rows.append(crow)
        print(f"\n  focus sensitivity implied by the chain: "
              f"{crow['um_hex_per_c']:.1f} um of hexapod dz per deg C, against "
              f"{crow['thermal_um_per_c']:.0f} um per deg C for a "
              f"{TRUSS_LENGTH_MM:.0f} mm steel truss at "
              f"{STEEL_CTE_PER_C * 1e6:.0f} ppm per deg C "
              f"(ratio {crow['measured_over_thermal']:.2f}, dimensionless)")
        lts213_page(pdf, args.lts213_pdf)

    print(f'\nSaved: {pdf_path}')
    sm_path = out / f'dz14_truss_{args.dz_prefix}_summary.parquet'
    pd.DataFrame(rows).to_parquet(sm_path)
    print(f'Saved: {sm_path}  ({len(rows)} fit rows)')


if __name__ == '__main__':
    main()
