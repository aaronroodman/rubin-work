"""Fit a giant donut with danish, driven exactly as ts_wep's blitz drives it.

Answers item 7's Q1 -- does the blitz forward model fit an 8 mm donut at all --
and then step 4, the v3.14-against-v1000 pupil comparison, without running the
butler pipeline. The pipeline needs a reference catalog and matched calibrations
that this engineering night does not have; the question is about the *forward
model*, so this calls blitz's own factory builder and danish model directly on
stamps cut from the raws.

Everything that defines the fit is taken from blitz rather than reimplemented:
`_INSTRUMENT` for the pupil radii, ``maskParams``, focal length and pixel size,
`_telescope_for_offsets` for the defocused telescope, and `batoid.zernikeTA` for
the reference wavefront. The one thing this module adds is the ability to swap
the pupil model, which blitz has no config field for.

Two representations of the pupil
--------------------------------
blitz carries the pupil **twice**, and a credible comparison has to swap both:

1. The analytic mask danish uses -- `Instrument.radius`,
   `Instrument.obscuration` and `Instrument.maskParams`.
2. A real batoid telescope, ``LSST_{band}.yaml``, hardcoded in
   `donutBlitzFamTask` and `donutBlitzMonolithTask`. It supplies the reference
   wavefront and is shifted along z to apply the defocus.

``LSST_r.yaml`` is identical to ``Rubin_v3.14_r.yaml`` in pupil geometry
(``pupilSize`` 8.36 m, ``pupilObscuration`` 0.612), so blitz today is
self-consistently v3.14. Swapping only the ``maskParams`` would leave the
reference wavefront on the old model and confound the comparison, so
`pupil_override` changes both together.

Note that v1000 keeps ``pupilObscuration: 0.612`` as a nominal value even though
its inner edge moves outward by 25.3 mm. The traced boundary, not that number, is
what `gen_mask_params` fits and what this module uses.

The defocus
-----------
These exposures are a symmetric 4 + 4 mm split -- 4000 um on the M2 hexapod and
4000 um on the camera hexapod -- established from the Trim. blitz expresses that
as an offset triplet ``(detector, camera, m2)`` in meters, signed positive
extra-focal, so the pair is ``(0, +4e-3, +4e-3)`` and ``(0, -4e-3, -4e-3)``.

Run as a script::

    python wfs/code/giant_donuts/fit_giant_donut.py \\
        --detector R22_S10 --pupil-model v3.14 --fwhm-max 1.5
"""
import argparse
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# Default location of the blitz checkout. The shared ts_wep is on `develop`, which
# does not carry the blitz prototype, so the path is explicit rather than relying
# on whatever `setup ts_wep` happens to have selected.
BLITZ_PYTHON = '/sdf/home/r/roodman/u/LSST/packages/ts_wep_blitz/python'

# This insert must happen before *any* `lsst` import, which is why it sits at
# module scope rather than inside `_import_blitz`.  `lsst` and `lsst.ts` are
# namespace packages: the first import of `lsst.ts` freezes its `__path__` from
# whatever `sys.path` held at that moment, and the shared EUPS ts_wep (on
# `develop`, no blitz subpackage) is already on PYTHONPATH.  A later insert is
# then silently ignored -- `lsst.ts.wep` resolves to the shared checkout and
# `lsst.ts.wep.blitz` does not exist.  Deleting the modules from `sys.modules`
# afterwards does not fix it, because the frozen parent `__path__` is what the
# finder consults.
if BLITZ_PYTHON not in sys.path:
    sys.path.insert(0, BLITZ_PYTHON)

# day_obs 20251023, BLOCK-T626, r band, 60 s.
EXPOSURES = {'extra': 2025102300337, 'intra': 2025102300340}

# Signed z offsets (detector, camera, M2) in meters, extra-focal.  The 4 + 4 mm
# split measured from the Trim; intra-focal is the negation, as blitz does it.
EXTRA_OFFSETS_M = (0.0, 4.0e-3, 4.0e-3)

BUTLER_REPO = '/repo/main'
RAW_COLLECTION = 'LSSTCam/raw/all'
BAND = 'r'

# Noll indices blitz fits by default: 4..19 and 22..26.  Z11 (spherical) is the
# quantity of interest and is inside this set.
NOLL_INDICES = list(range(4, 20)) + list(range(22, 27))

NOLL_Z11 = 11

__all__ = [
    'EXPOSURES', 'EXTRA_OFFSETS_M', 'NOLL_INDICES', 'NOLL_Z11',
    'pupil_override', 'fit_stamp',
]


def _import_blitz(blitz_python=BLITZ_PYTHON):
    """Import the blitz modules from the side-branch checkout.

    Parameters
    ----------
    blitz_python : `str`, optional
        ``python`` directory of the ts_wep checkout carrying blitz.

    Returns
    -------
    modules : `tuple`
        ``(utils, wavefrontFittingTask)`` modules.

    Notes
    -----
    Getting this import right is fiddly, and getting it *wrong* can silently mix
    two ts_wep versions, so the steps are deliberate.

    ``lsst`` and ``lsst.ts`` are namespace packages, but ``lsst.ts.wep`` is a
    regular package with an ``__init__.py``. So once anything imports
    ``lsst.ts.wep`` -- and importing ``lsst.daf.butler`` is enough to freeze
    ``lsst.ts`` -- the shared EUPS checkout on ``PYTHONPATH`` wins, its
    ``__path__`` is fixed to a single directory, and no later `sys.path` insert
    or `importlib.invalidate_caches` can displace it. In a Jupyter kernel that
    has already happened before the first cell runs.

    Inserting the checkout into ``lsst.ts.wep.__path__`` makes ``blitz``
    importable, but on its own it would leave `Instrument` and the rest of
    ts_wep coming from the *shared* checkout while ``blitz`` comes from this one.
    That mix is worse than an ImportError, because it is silent. So the
    sibling modules blitz depends on are dropped from `sys.modules` and the
    checkout is put first, and the result is then verified to come from here.
    """
    if blitz_python not in sys.path:
        sys.path.insert(0, blitz_python)

    wep_dir = str(pathlib.Path(blitz_python) / 'lsst' / 'ts' / 'wep')

    # Drop the already-imported ts_wep tree so blitz's sibling imports resolve
    # against this checkout rather than being inherited from the shared one.
    for name in [m for m in sys.modules if m.startswith('lsst.ts.wep')]:
        del sys.modules[name]

    # Re-resolving is not enough on its own: `lsst.ts` is a namespace package
    # whose __path__ was frozen by the first `lsst` import, and that frozen list
    # -- not sys.path -- is what the finder searches for `wep`. Put this
    # checkout's `lsst/ts` at the front of it.
    # `__path__` here is a `_NamespacePath`, which has no `insert`, so assign a
    # plain list. That also pins it: it stops being recomputed from `sys.path`,
    # which is what we want, since recomputation is what kept restoring the
    # shared checkout.
    import lsst.ts as ts_pkg
    ts_dir = str(pathlib.Path(blitz_python) / 'lsst' / 'ts')
    if ts_dir not in list(ts_pkg.__path__):
        ts_pkg.__path__ = [ts_dir] + list(ts_pkg.__path__)

    from lsst.ts.wep.blitz import utils as blitz_utils
    from lsst.ts.wep.blitz import wavefrontFittingTask as wf_task

    # Compare resolved paths: ~/u is a symlink to /sdf/data/rubin/user/roodman, so
    # the imported file's real path does not contain `blitz_python` literally.
    want = pathlib.Path(blitz_python).resolve()
    for module in (blitz_utils, wf_task, sys.modules['lsst.ts.wep.instrument']):
        got = pathlib.Path(module.__file__).resolve()
        if want not in got.parents:
            raise RuntimeError(
                f'{module.__name__} imported from {got}, not from {blitz_python}. '
                'Mixing two ts_wep checkouts would make the fit unattributable, '
                'so this is fatal.'
            )
    return blitz_utils, wf_task


def pupil_override(blitz_utils, model, mask_params_file=None):
    """Point blitz's instrument and telescope at one pupil model.

    Mutates the module-level `_INSTRUMENT` and `_CALIB_STORE` that blitz's
    factory builder reads, which is the only way in: neither the pupil model nor
    the ``maskParams`` is a config field.

    Parameters
    ----------
    blitz_utils : `module`
        ``lsst.ts.wep.blitz.utils``.
    model : {'v3.14', 'v1000'}
        Pupil model to install.
    mask_params_file : `str`, optional
        YAML file of generated ``maskParams``, as written by
        `gen_mask_params`. Required for ``'v1000'``; ignored for ``'v3.14'``,
        which uses the shipped values.

    Returns
    -------
    summary : `dict`
        What was installed: the batoid model name, ``pupilSize`` in meters, the
        traced inner edge as a normalised radius (dimensionless), and the
        ``maskParams`` source.
    """
    import batoid

    if model not in ('v3.14', 'v1000'):
        raise ValueError(f"model must be 'v3.14' or 'v1000', not {model!r}")

    telescope_name = 'Rubin_v3.14_r' if model == 'v3.14' else 'Rubin_v1000_r'
    telescope = batoid.Optic.fromYaml(telescope_name + '.yaml')
    blitz_utils._CALIB_STORE['telescope'] = telescope
    # Shifted telescopes are memoized per offset triplet, so a stale entry built
    # from the previous model would silently survive the swap.
    blitz_utils._CALIB_STORE.pop('telescope_by_offsets', None)

    instrument = blitz_utils._INSTRUMENT
    if model == 'v1000':
        if mask_params_file is None:
            raise ValueError('v1000 needs --mask-params from gen_mask_params.py')
        with open(mask_params_file) as f:
            mask_params = yaml.safe_load(f)['maskParams']
        instrument.maskParams = mask_params

        # `radius` and `obscuration` come from policy/instruments/LsstCam.yaml,
        # not from the batoid model, so they stay at the v3.14 values (4.18 m and
        # 0.612) unless set here.  Leaving them is not a harmless inconsistency:
        # danish then takes R_outer/R_inner from one model and the mask circles
        # from another, and the fit absorbs the mismatch into Z4 -- measured as
        # extra-focal Z4 of +1.44 um against +0.64 um when consistent.  Take both
        # from the traced M1 edges, which is what the mask circles use.
        outer_m = float(np.polyval(mask_params['M1Baffle1']['outer']['radius'], 0.0))
        inner_m = float(np.polyval(mask_params['M1']['inner']['radius'], 0.0))
        instrument.diameter = 2.0 * outer_m
        instrument.obscuration = inner_m / outer_m
        source = mask_params_file
    else:
        source = 'shipped policy:instruments/LsstCam.yaml'

    return {
        'model': model,
        'telescope': telescope_name,
        'pupil_size_m': float(telescope.pupilSize),
        'instrument_radius_m': float(instrument.radius),
        'instrument_obscuration': float(instrument.obscuration),
        'mask_params_source': source,
    }


def fit_stamp(stamp, thx_rad, thy_rad, offsets_m, blitz_utils, wf_task,
              noll_indices=None, binning=2, fwhm_max=1.5, fwhm_min=0.1,
              fwhm_fixed=None, spiders=False, rtp_deg=0.0, max_nfev=200,
              tol=1e-3):
    """Fit one donut stamp with danish, through blitz's forward model.

    Parameters
    ----------
    stamp : `numpy.ndarray`
        Background-subtracted donut stamp, in electrons.
    thx_rad, thy_rad : `float`
        Field angle components, in radians, CCS.
    offsets_m : `tuple` [`float`]
        Signed z offsets ``(detector, camera, M2)``, in meters.
    blitz_utils, wf_task : `module`
        The blitz modules, from `_import_blitz`.
    noll_indices : `list` [`int`], optional
        Noll indices to fit. Defaults to `NOLL_INDICES`.
    binning : `int`, optional
        Stamp binning before fitting; 2 is blitz's config default. ``img`` and
        ``model_img`` come back on the binned grid, so anything measured on them
        in pixels is on a `binning` x 10 um pixel. Use 1 to keep them on the
        native detector pixel and directly comparable with the unbinned stamp.
    fwhm_max, fwhm_min : `float`, optional
        Bounds on the fitted blur FWHM, in arcsec. Reported with the result,
        because a fit that lands on the bound has not measured the blur.
        Ignored when `fwhm_fixed` is given.
    fwhm_fixed : `float`, optional
        Hold the blur FWHM at this value, in arcsec, instead of fitting it.
        Imposed as a degenerate bound rather than by repacking danish's
        parameter vector, so the forward model is untouched. Use when the seeing
        is known independently: a free blur can absorb real wavefront error, and
        on these donuts the intra-focal blur runs to its upper bound.
    spiders : `bool`, optional
        Model the spider shadows.
    rtp_deg : `float`, optional
        Rotator angle, in degrees, used only when `spiders` is True.
    max_nfev : `int`, optional
        Maximum least-squares function evaluations.

    Returns
    -------
    result : `dict`
        ``img`` and ``model_img`` (the binned stamp and its best-fit model, in
        electrons, on the same grid), ``zk_dev_um`` (Noll-indexed wavefront
        deviation, um), ``fwhm_arcsec``,
        ``fwhm_at_bound`` (`bool`), ``fwhm_was_fixed`` (`bool`), ``cost``,
        ``nfev``, ``success``, and the fitted ``flux_electrons``, ``dx_pix`` and
        ``dy_pix``.
    """
    import batoid
    import danish
    from scipy.optimize import least_squares
    from scipy.stats import median_abs_deviation

    noll_indices = list(noll_indices or NOLL_INDICES)
    instrument = blitz_utils._INSTRUMENT

    img = wf_task._bin_stamp_odd(np.asarray(stamp, dtype=float), binning)
    diff = (img[1:] - img[:-1]).ravel()
    bkg_std = median_abs_deviation(diff, scale='normal') / np.sqrt(2.0)

    wavelength_by_band = {b.value: w for b, w in instrument.wavelength.items()}
    wavelength_m = wavelength_by_band[BAND]

    telescope_dz = blitz_utils._telescope_for_offsets(tuple(offsets_m))
    # The annular-Zernike obscuration must be the one danish uses for the fitted
    # deviation, not the batoid model's nominal `pupilObscuration`.  v1000 keeps
    # the nominal at 0.612 while its traced inner edge is at 0.6204, so taking it
    # from the telescope put the reference wavefront and the fitted deviation on
    # *different* annular bases.  Z4 is the term most sensitive to annulus width
    # and absorbed the whole mismatch: its intra-minus-extra split read +4.11 um
    # of wavefront under v1000 against -0.22 um under v3.14, where the two bases
    # happen to agree.  Z11 moved only 0.002 um, which is why it survived.
    eps = instrument.obscuration
    nrad = 10
    zk_ref = batoid.zernikeTA(
        telescope_dz, thx_rad, thy_rad, wavelength_m,
        jmax=blitz_utils._ZK_JMAX, eps=eps,
        focal_length=instrument.focalLength, nrad=nrad,
        naz=int(2 * np.pi * nrad / (1 - eps)),
    ) * wavelength_m

    factory = danish.DonutFactory(
        R_outer=instrument.radius,
        R_inner=instrument.radius * instrument.obscuration,
        mask_params=instrument.maskParams,
        focal_length=instrument.focalLength,
        pixel_scale=instrument.pixelSize * binning,
        spider_angle=(rtp_deg if spiders else None),
    )

    dz_terms = [(1, j) for j in noll_indices]
    model = danish.DZMultiDonutModel(
        factory, z_refs=[zk_ref], dz_terms=dz_terms,
        field_radius=wf_task._DANISH_FIELD_RADIUS_RAD,
        thxs=[thx_rad], thys=[thy_rad], npix=img.shape[0], bkg_order=0,
    )

    # A fixed blur is a degenerate bound. least_squares rejects lb == ub, so the
    # interval is opened by a hair; the width is far below the 0.001 arcsec the
    # blur is reported to, so the value is fixed for every practical purpose.
    if fwhm_fixed is not None:
        fwhm_lo, fwhm_hi = fwhm_fixed - 1e-9, fwhm_fixed + 1e-9
        fwhm_start = float(fwhm_fixed)
    else:
        fwhm_lo, fwhm_hi = fwhm_min, fwhm_max
        fwhm_start = 1.0

    x0 = model.pack_params(
        fluxes=[float(np.clip(np.sum(img), 1e3, 1e9))], dxs=[0.0], dys=[0.0],
        fwhm=fwhm_start, bkgs=[[0.0] * model.nbkg],
        wavefront_params=[0.0] * len(dz_terms),
    )
    bounds = model.pack_params(
        fluxes=[[0.0, np.inf]], dxs=[[-np.inf, np.inf]], dys=[[-np.inf, np.inf]],
        fwhm=[fwhm_lo, fwhm_hi], bkgs=[[[-np.inf, np.inf]] * model.nbkg],
        wavefront_params=[[-np.inf, np.inf]] * len(dz_terms),
    )
    bounds = [list(b) for b in zip(*bounds)]
    x0 = np.clip(x0, bounds[0], bounds[1])

    # Solver settings copied from blitz's `lstsqKwargs` default, with danish's
    # analytic dense Jacobian, so convergence behaviour matches the pipeline's.
    # `tol` is exposed because blitz's 1e-3 default stops a giant-donut fit after
    # about 5 function evaluations -- fine for a 1.5 mm FAM donut, but these
    # stamps carry 100x the pixels and the fit is nowhere near converged there.
    fit = least_squares(
        model.chi, x0=x0, jac=model.jac, bounds=bounds,
        args=([img], [bkg_std ** 2]),
        xtol=tol, ftol=tol, gtol=tol, x_scale='jac', tr_solver='lsmr',
        max_nfev=max_nfev,
    )
    unpacked = model.unpack_params(fit.x)
    fwhm = float(np.atleast_1d(unpacked['fwhm'])[0])

    # The best-fit model image, on the same binned pixel grid as `img`.  Returned
    # so the fit can be profiled the same way the data is: a radial or azimuthal
    # profile of data minus model localises where the forward model fails, which
    # a single chi2/dof cannot.
    model_img = np.asarray(model.model(
        unpacked['fluxes'], unpacked['dxs'], unpacked['dys'],
        fwhm=unpacked['fwhm'], wavefront_params=unpacked['wavefront_params'],
        bkgs=unpacked['bkgs'], sky_levels=[0.0],
    )[0], dtype=float)

    zk_dev_um = {int(j): float(v * 1e6)
                 for j, v in zip(noll_indices, unpacked['wavefront_params'])}

    return {
        'img': img,
        'model_img': model_img,
        'zk_dev_um': zk_dev_um,
        'fwhm_arcsec': fwhm,
        'fwhm_bound_arcsec': (fwhm_lo, fwhm_hi),
        'fwhm_was_fixed': fwhm_fixed is not None,
        # 2 per cent of the bound range: a fit that lands this close has not
        # measured the blur, it has been stopped by the bound, and its wavefront
        # has absorbed whatever the blur could not.
        'fwhm_at_bound': (False if fwhm_fixed is not None else bool(
            min(abs(fwhm - fwhm_lo), abs(fwhm - fwhm_hi))
            < 0.02 * (fwhm_hi - fwhm_lo))),
        'flux_electrons': float(np.atleast_1d(unpacked['fluxes'])[0]),
        'dx_pix': float(np.atleast_1d(unpacked['dxs'])[0]),
        'dy_pix': float(np.atleast_1d(unpacked['dys'])[0]),
        'cost': float(fit.cost),
        'chi2_per_dof': float(2 * fit.cost / max(img.size - len(x0), 1)),
        'dof': int(img.size - len(x0)),
        'nfev': int(fit.nfev),
        'success': bool(fit.success),
        'npix_binned': int(img.shape[0]),
        'bkg_std_electrons': float(bkg_std),
    }


def main():
    ap = argparse.ArgumentParser(description='Fit giant donuts with danish.')
    ap.add_argument('--detector', default='R22_S10')
    ap.add_argument('--pupil-model', default='v3.14', choices=('v3.14', 'v1000'))
    ap.add_argument('--mask-params',
                    default='wfs/output/giant_donuts/maskParams_v1000.yaml')
    ap.add_argument('--fwhm-max', type=float, default=1.5,
                    help='upper bound on fitted blur FWHM, in arcsec')
    ap.add_argument('--fwhm-fixed', type=float, default=None,
                    help='hold the blur FWHM at this value, in arcsec, instead '
                         'of fitting it')
    ap.add_argument('--binning', type=int, default=2)
    ap.add_argument('--tol', type=float, default=1e-3,
                    help="least_squares xtol/ftol/gtol; blitz's default is 1e-3")
    ap.add_argument('--max-nfev', type=int, default=200)
    ap.add_argument('--spiders', action='store_true')
    ap.add_argument('--stamp-half', type=int, default=430)
    ap.add_argument('--blitz-python', default=BLITZ_PYTHON)
    args = ap.parse_args()

    import lsst.daf.butler as dafButler
    from lsst.obs.lsst import LsstCam

    blitz_utils, wf_task = _import_blitz(args.blitz_python)
    import donut_images as di

    installed = pupil_override(
        blitz_utils, args.pupil_model,
        mask_params_file=(args.mask_params if args.pupil_model == 'v1000' else None))
    print('pupil model installed:')
    for key, value in installed.items():
        print(f'  {key:24s} {value}')

    camera = LsstCam.getCamera()
    butler = dafButler.Butler(BUTLER_REPO)

    print(f'\nfitting {args.detector}, binning {args.binning}, '
          f'fwhm {f"fixed {args.fwhm_fixed}" if args.fwhm_fixed is not None else f"bound 0.1 to {args.fwhm_max}"} arcsec, '
          f'spiders {"on" if args.spiders else "off"}')
    results = {}
    for side, exposure in EXPOSURES.items():
        sign = +1.0 if side == 'extra' else -1.0
        offsets = tuple(sign * o for o in EXTRA_OFFSETS_M)

        image = di.run_isr(butler, exposure, args.detector, camera,
                           RAW_COLLECTION)
        x_pix, y_pix, diameter_pix, _ = di.find_donut(image)
        stamp, sky = di.cut_stamp(image, x_pix, y_pix, half=args.stamp_half)
        if stamp is None:
            print(f'  {side}: stamp falls off the detector, skipped')
            continue
        ax_deg, ay_deg = di.field_angle_deg(args.detector, x_pix, y_pix, camera)

        out = fit_stamp(
            stamp, np.deg2rad(ax_deg), np.deg2rad(ay_deg), offsets,
            blitz_utils, wf_task, binning=args.binning,
            fwhm_max=args.fwhm_max, fwhm_fixed=args.fwhm_fixed,
            spiders=args.spiders,
            tol=args.tol, max_nfev=args.max_nfev,
        )
        results[side] = out
        print(f'\n  {side}-focal, exposure {exposure}')
        print(f'    donut diameter        {diameter_pix} pixel '
              f'({diameter_pix * di.PIXEL_SIZE_M * 1e3:.2f} mm)')
        print(f'    field angle           ({ax_deg:+.3f}, {ay_deg:+.3f}) deg')
        print(f'    offsets (det,cam,M2)  '
              f'{tuple(round(o * 1e3, 1) for o in offsets)} mm')
        print(f'    fit success           {out["success"]}, '
              f'nfev {out["nfev"]}, chi2/dof {out["chi2_per_dof"]:.3f} '
              f'(dof {out["dof"]})')
        print(f'    blur FWHM             {out["fwhm_arcsec"]:.3f} arcsec'
              f'{"  AT BOUND" if out["fwhm_at_bound"] else ""}')
        print(f'    Z11 (spherical)       '
              f'{out["zk_dev_um"][NOLL_Z11]:+.4f} um of wavefront')

    if {'intra', 'extra'} <= set(results):
        split = (results['intra']['zk_dev_um'][NOLL_Z11]
                 - results['extra']['zk_dev_um'][NOLL_Z11])
        print(f'\n  intra minus extra Z11  {split:+.4f} um of wavefront')
        print('  For reference: the data show about 0.3 um of Z11 split, of '
              'which\n  diffraction explains about 0.10 um.')

        print(f"\n  {'Noll':>5}  {'extra':>10}  {'intra':>10}  {'intra-extra':>12}"
              '   [um of wavefront]')
        for j in NOLL_INDICES:
            e = results['extra']['zk_dev_um'][j]
            i = results['intra']['zk_dev_um'][j]
            print(f'  {j:>5}  {e:>+10.4f}  {i:>+10.4f}  {i - e:>+12.4f}')


if __name__ == '__main__':
    main()
