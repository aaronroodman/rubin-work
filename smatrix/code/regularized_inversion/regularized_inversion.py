"""Regularized inversions of the OFC sensitivity matrix, as alternatives to
truncated-SVD recovery of the optical state.

The Optical Feedback Control (OFC) open-loop recovery inverts a measured Double
Zernike (DZ) wavefront onto degrees of freedom (DOF) by a *truncated* singular value
decomposition (SVD): keep the leading ``n_keep`` singular modes, discard the rest.
Truncation is a blunt 0/1 regularizer. The per-DOF normalization weights
``w_j = r_j^0.5 f_j^-0.5`` put the allowed range ``r_j`` into the *metric* of the fit,
but never into the *feasible set*, so a recovered bending-mode amplitude is free to
exceed the actuator-force-limited range ``r_j`` that the mirror can physically reach.

This module provides three inversions over one design matrix, all returning DOF in
the mixed physical units of ``DOF_UNITS_50`` (µm for translations and bending-mode
amplitudes, arcsec for hexapod rotations):

``invert_truncated``
    Method 0, the current scheme. Truncated SVD at ``n_keep`` modes.
``invert_damped``
    Method A, damped SVD (Tikhonov in the normalized-DOF metric). Replaces the hard
    cut with a smooth roll-off ``s_i / (s_i^2 + lambda^2)``. One scalar knob, and it
    can retain full rank rather than choosing an arbitrary truncation index.
``invert_range_penalty``
    Method B, a per-DOF penalty that is small while ``|d_j| / r_j`` is below
    ``kappa`` and grows superlinearly past it, solved by iteratively reweighted least
    squares (IRLS). This is the method that targets the over-range amplitudes
    directly.

All three solve in the *normalized* DOF variable ``x = d / w``, which is the variable
the SVD is taken in, so the v-mode machinery is untouched. Method B's penalty is
expressed on the physical ``d = w * x``; penalizing a function of ``d`` while fitting
in ``x`` is a change of variables, not a new basis, so `aos_state` remains the sole
owner of the SVD (see ``svd-use-state-estimator``).

Notes
-----
The design matrix must be the **rank-``n_keep``** forward operator when comparing
against a truncated-SVD solution, not the full-rank sensitivity slab. Using the full
rank inflates a predicted wavefront by roughly a factor of 4 in µm of wavefront RMS
and destroys closure against the measurement, because the truncated solution by
construction carries no content in the discarded modes. `forward_operator` builds the
correct one.

The penalty in method B is a *regularizer*, not a likelihood: it encodes the prior
that the mirror cannot exceed its force-limited stroke. It therefore biases the
estimate toward zero, and a resulting amplitude is a constrained estimate rather than
an unbiased measurement of mirror figure.
"""
import numpy as np

__all__ = ['forward_operator', 'invert_truncated', 'invert_damped',
           'invert_range_penalty', 'achieved_residual', 'dof_range_vector']


def forward_operator(svd, rank=None):
    """Rank-limited forward operator mapping normalized DOF to a DZ wavefront.

    Parameters
    ----------
    svd : `lsst.ts.intrinsic.wavefront.ofc_svd.OFCSvd`
        From `build_ofc_svd`; supplies ``U_eff``, ``Sigma``, ``V``.
    rank : `int`, optional
        Singular modes retained in the operator. Default is the SVD's own
        ``n_keep_eff``, which makes the operator consistent with the truncated
        recovery the study compares against.

    Returns
    -------
    S : `numpy.ndarray`
        ``(n_kj, n_dof)``, in µm of wavefront per unit normalized DOF, such that
        ``dW = S @ (d / w)`` for physical DOF ``d`` and weights ``w``.

    Notes
    -----
    ``U_eff`` already holds only the kept columns, so a ``rank`` below
    ``n_keep_eff`` slices further into it; a ``rank`` above it cannot be honoured
    and raises.
    """
    n_kept = int(svd.U_eff.shape[1])
    r = n_kept if rank is None else int(rank)
    if r > n_kept:
        raise ValueError(
            f'forward_operator got rank={r} but the OFCSvd retains only {n_kept} '
            'modes in U_eff; rebuild build_ofc_svd with a larger n_keep')
    sig = np.asarray(svd.Sigma, float)[:r]
    return (np.asarray(svd.U_eff, float)[:, :r] * sig[None, :]) @ \
        np.asarray(svd.V, float)[:, :r].T


def dof_range_vector(svd, f_quadrature=None):
    """Allowed range ``r_j`` per DOF, in that DOF's own physical unit.

    Back-derived from the shipped normalization weights as ``r_j = w_j^2 * f_j``,
    with ``w_j = r_j^0.5 f_j^-0.5`` the weight and ``f_j`` the field-averaged PSF
    width response in arcsec per DOF unit. Back-deriving rather than recomputing the
    range literals is what guarantees the ``r_j`` returned here is the one the shipped
    weights were actually generated with.

    Parameters
    ----------
    svd : `lsst.ts.intrinsic.wavefront.ofc_svd.OFCSvd`
        Supplies ``normalization_weights`` and ``dof_idx``.
    f_quadrature : `numpy.ndarray`, optional
        ``f_j`` over the **full** 50-DOF set, arcsec per DOF unit, as returned by
        ``smatrix/code/normalization_weights.py:compute_f_quadrature``. Computed here
        if omitted, which needs `lsst.ts.ofc`.

    Returns
    -------
    r : `numpy.ndarray`
        ``(n_dof,)`` allowed range over the SVD's ``dof_idx``, µm for translations
        and bending-mode amplitudes, arcsec for hexapod rotations.

    Notes
    -----
    Two flavours of ``f_j`` circulate: the corner-point value that ts_ofc's
    ``compute_normalization_components`` returns, which is about a factor of sqrt(2)
    high, and the field-averaged quadrature value the shipped yaml was generated
    with. Only the latter reproduces the weights, so only the latter is used here.
    """
    w = np.asarray(svd.normalization_weights, float)
    if f_quadrature is None:
        import normalization_weights as NW
        from lsst.ts.ofc import OFCData
        sens = np.asarray(OFCData('lsst').sensitivity_matrix)
        f_quadrature = NW.compute_f_quadrature(sens, rings=5, spokes=6,
                                               znmin=4, znmax=22)
    f_full = np.asarray(f_quadrature, float)
    idx = list(svd.dof_idx) if svd.dof_idx else list(range(len(f_full)))
    return w ** 2 * f_full[idx]


def invert_truncated(dW, svd, rank=None):
    """Method 0 — the current truncated-SVD recovery.

    Parameters
    ----------
    dW : `numpy.ndarray`
        Measured DZ wavefront in µm of wavefront, over ``svd.kj_grid`` order.
    svd : `lsst.ts.intrinsic.wavefront.ofc_svd.OFCSvd`
        The sensitivity-matrix SVD.
    rank : `int`, optional
        Modes retained; default ``svd.n_keep_eff``.

    Returns
    -------
    d : `numpy.ndarray`
        Physical DOF, ``(n_dof,)``, µm and arcsec per ``DOF_UNITS_50``.

    Notes
    -----
    Identical in form to ``ofc_svd.recover_dof_per_visit``: ``A = U_eff^T dW``,
    ``x = V diag(1/s) A``, ``d = w * x``. Reimplemented here so that all three
    methods share one code path for the residual and FWHM accounting, and verified
    against `recover_dof_per_visit` in the study script.
    """
    dW = np.nan_to_num(np.asarray(dW, float))
    n_kept = int(svd.U_eff.shape[1])
    r = n_kept if rank is None else int(rank)
    U = np.asarray(svd.U_eff, float)[:, :r]
    s = np.asarray(svd.Sigma, float)[:r]
    V = np.asarray(svd.V, float)[:, :r]
    x = V @ ((U.T @ dW) / s)
    return np.asarray(svd.normalization_weights, float) * x


def invert_damped(dW, svd, lam, rank=None):
    """Method A — damped SVD (Tikhonov in the normalized-DOF metric).

    Minimizes ``||dW - S x||^2 + lambda^2 ||x||^2`` over normalized DOF ``x``, whose
    SVD solution rolls the mode gain off smoothly as ``s_i / (s_i^2 + lambda^2)``
    instead of cutting it at ``n_keep``.

    Parameters
    ----------
    dW : `numpy.ndarray`
        Measured DZ wavefront, µm of wavefront, over ``svd.kj_grid``.
    svd : `lsst.ts.intrinsic.wavefront.ofc_svd.OFCSvd`
        The sensitivity-matrix SVD.
    lam : `float`
        Damping ``lambda``, in the **units of a singular value** — µm of wavefront
        per unit normalized DOF. A mode with ``s_i = lambda`` is attenuated to half
        the gain the undamped inverse would give it, so ``lambda`` is read directly
        against the singular-value spectrum.
    rank : `int`, optional
        Modes available to the solution. Default ``svd.n_keep_eff``, which makes
        method A a strict softening of the current scheme; pass the full rank to let
        damping replace truncation entirely.

    Returns
    -------
    d : `numpy.ndarray`
        Physical DOF, ``(n_dof,)``, µm and arcsec per ``DOF_UNITS_50``.

    Notes
    -----
    Because ``x`` is the normalized DOF ``d / w`` and ``w_j = sqrt(r_j / f_j)``, the
    penalty ``||x||^2`` is ``sum_j (d_j / w_j)^2`` — already range-aware in the weak
    sense that a DOF with a small range is penalized harder per physical unit. It is
    still not a constraint: nothing stops ``|d_j|`` exceeding ``r_j``, which is the
    gap method B closes.
    """
    dW = np.nan_to_num(np.asarray(dW, float))
    n_kept = int(svd.U_eff.shape[1])
    r = n_kept if rank is None else int(rank)
    U = np.asarray(svd.U_eff, float)[:, :r]
    s = np.asarray(svd.Sigma, float)[:r]
    V = np.asarray(svd.V, float)[:, :r]
    lam = float(lam)
    gain = s / (s ** 2 + lam ** 2)
    x = V @ (gain * (U.T @ dW))
    return np.asarray(svd.normalization_weights, float) * x


def invert_range_penalty(dW, svd, ranges, *, kappa=0.5, power=2, lam0=0.0,
                         n_iter=200, tol=1e-10, rank=None, eps=1e-12,
                         relax=1.0, return_info=False):
    """Method B — a superlinear per-DOF range penalty, solved by IRLS.

    Minimizes, over normalized DOF ``x`` with physical ``d = w * x``::

        ||dW - S x||^2  +  lam0^2 ||x||^2
                        +  sum_j ( |d_j| / (kappa * r_j) ) ^ (2 * power)

    The penalty is far below unity while ``|d_j| / r_j`` stays under ``kappa`` and
    rises steeply as the ratio approaches and passes 1, which is the behaviour asked
    for: a DOF error that is small at half the range and grows substantially at the
    range and beyond.

    Parameters
    ----------
    dW : `numpy.ndarray`
        Measured DZ wavefront, µm of wavefront, over ``svd.kj_grid``.
    svd : `lsst.ts.intrinsic.wavefront.ofc_svd.OFCSvd`
        The sensitivity-matrix SVD.
    ranges : `numpy.ndarray`
        Allowed range ``r_j``, ``(n_dof,)``, in each DOF's own physical unit — from
        `dof_range_vector`. Must be strictly positive.
    kappa : `float`, optional
        Dimensionless knee position as a fraction of the range: the penalty reaches
        unity at ``|d_j| = kappa * r_j``. Default 0.5, i.e. the penalty starts to
        bite at half the allowed range.
    power : `int`, optional
        Penalty exponent ``p``; the penalty goes as ``|d_j| ^ (2 p)``, so ``p = 1``
        is plain Tikhonov on ``d_j / r_j`` and ``p = 2`` (default) is the quartic
        superlinear growth that leaves the sub-knee region nearly free.
    lam0 : `float`, optional
        Optional small Tikhonov floor on ``x``, in singular-value units, to keep the
        normal equations well posed if ``rank`` exposes near-null modes. Default 0.0.
    n_iter : `int`, optional
        Maximum IRLS iterations.
    tol : `float`, optional
        Convergence tolerance on the relative change of ``d`` between iterations,
        dimensionless.
    rank : `int`, optional
        Modes available to the solution. Default ``svd.n_keep_eff``.
    eps : `float`, optional
        Floor on ``|d_j|`` inside the reweighting, in DOF units, to avoid dividing
        by zero on a DOF that lands exactly at zero.
    relax : `float`, optional
        Initial IRLS step fraction in (0, 1]: the trial step goes ``relax`` of the way
        from the previous iterate to the freshly solved one, and is halved until the
        objective decreases. Default 1.0, a full trial step; backtracking makes the
        iteration monotone regardless, so this is a starting guess and not a knob that
        has to be tuned per ``power``.
    return_info : `bool`, optional
        If true, also return a diagnostics dict.

    Returns
    -------
    d : `numpy.ndarray`
        Physical DOF, ``(n_dof,)``, µm and arcsec per ``DOF_UNITS_50``.
    info : `dict`, optional
        Present when ``return_info``. Keys ``n_iter`` (`int`, iterations used),
        ``converged`` (`bool`), ``delta`` (`float`, final relative change,
        dimensionless), and ``max_ratio`` (`float`, largest ``|d_j| / r_j``,
        dimensionless).

    Notes
    -----
    IRLS linearizes the penalty as a quadratic ``sum_j q_j d_j^2`` whose weight
    ``q_j`` is recomputed from the previous iterate::

        q_j = (kappa r_j)^(-2p) * |d_j|^(2p - 2)

    so the normal equations stay linear and each iteration is one symmetric solve.
    The solve is done in the ``r`` retained mode coefficients ``b`` with ``x = V_r b``,
    not in ``x`` directly: the rank-``r`` operator leaves ``n_dof - r`` null
    directions in which a solve in ``x`` would return noise, whereas the truncated
    solution sets them to zero. Restricting to ``V_r`` reproduces that choice, and in
    that basis the data term is diagonal (``sig^2``), so each iteration costs one
    ``r``-by-``r`` solve.

    For ``power=1`` the weight is constant and the first iteration is exact. For
    ``power >= 2`` the objective is convex in ``b`` (a sum of even powers plus a
    quadratic), so the iteration has a unique minimum, but the reweighting is only a
    local model and a full step can overshoot: measured at ``power=3``, a fixed
    half-step enters a limit cycle that never converges. Each step is therefore
    backtracked until the objective decreases, which makes the iteration monotone and
    convergent for every ``power`` tested.
    """
    dW = np.nan_to_num(np.asarray(dW, float))
    r_j = np.asarray(ranges, float)
    if np.any(~np.isfinite(r_j)) or np.any(r_j <= 0):
        raise ValueError('invert_range_penalty needs strictly positive finite '
                         'ranges; got non-positive or non-finite entries')
    w = np.asarray(svd.normalization_weights, float)
    p = int(power)
    scale = float(kappa) * r_j                      # DOF units; penalty is 1 here

    # Solve in the retained mode coefficients b, with x = V_r b, rather than in x
    # itself. The forward operator has rank r < n_dof, so the normal equations in x
    # are singular in the discarded directions and a direct solve fills them with
    # noise; the truncated solution sets them to zero by construction. Restricting to
    # V_r reproduces that choice exactly and makes the system full rank, so the
    # penalty-off limit returns the truncated answer.
    n_kept = int(svd.U_eff.shape[1])
    r = n_kept if rank is None else int(rank)
    U = np.asarray(svd.U_eff, float)[:, :r]
    sig = np.asarray(svd.Sigma, float)[:r]
    V_r = np.asarray(svd.V, float)[:, :r]
    # S = U diag(sig) V_r^T, so in b: S V_r = U diag(sig), giving StS -> diag(sig^2)
    # and S^T dW -> sig * (U^T dW). Both are diagonal, which is why this is cheap.
    sig2 = sig ** 2
    rhs = sig * (U.T @ dW)
    WV = w[:, None] * V_r                            # d = WV @ b
    def _objective(dv):
        """The penalized least-squares objective, in µm of wavefront squared."""
        b_ = np.linalg.lstsq(WV, dv, rcond=None)[0]
        mis = dW - U @ (sig * b_)
        return (float(mis @ mis) + float(lam0) ** 2 * float(b_ @ b_)
                + float(np.sum((np.abs(dv) / scale) ** (2 * p))))

    d = invert_truncated(dW, svd, rank=r)
    obj = _objective(d)
    converged, used, rel = False, 0, np.inf
    for it in range(int(n_iter)):
        used = it + 1
        mag = np.maximum(np.abs(d), eps)
        q = scale ** (-2 * p) * mag ** (2 * p - 2)   # penalty curvature in d-units
        # Penalty sum_j q_j d_j^2 with d = WV b becomes b^T (WV^T diag(q) WV) b.
        A = np.diag(sig2 + float(lam0) ** 2) + WV.T @ (q[:, None] * WV)
        b = np.linalg.solve(A, rhs)
        d_full = WV @ b
        # Backtrack on the true objective. The weight q is evaluated at the previous
        # iterate, so for a steep penalty (power >= 3) a full or fixed-fraction step
        # overshoots into a limit cycle -- measured: at relax=0.5, power=3 the step
        # stalls at a relative change of 1.6e-2 forever and returns a max |d_j|/r_j
        # of 0.585 against the converged 0.537. Halving until the objective actually
        # decreases makes the iteration monotone and removes the tuning knob.
        step = float(relax)
        for _ in range(40):
            cand = d + step * (d_full - d)
            obj_c = _objective(cand)
            if obj_c <= obj:
                break
            step *= 0.5
        else:
            converged = True          # no downhill step remains: at a minimum
            break
        denom = max(float(np.max(np.abs(d))), eps)
        rel = float(np.max(np.abs(cand - d)) / denom)
        d, obj = cand, obj_c
        if rel < float(tol):
            converged = True
            break
    if return_info:
        return d, dict(n_iter=used, converged=converged, delta=rel,
                       max_ratio=float(np.max(np.abs(d) / r_j)))
    return d


def achieved_residual(dW, d, svd, rank=None):
    """Residual DZ wavefront left after applying a DOF correction.

    Parameters
    ----------
    dW : `numpy.ndarray`
        Measured DZ wavefront, µm of wavefront, over ``svd.kj_grid``.
    d : `numpy.ndarray`
        Physical DOF correction, ``(n_dof,)``, µm and arcsec per ``DOF_UNITS_50``.
    svd : `lsst.ts.intrinsic.wavefront.ofc_svd.OFCSvd`
        The sensitivity-matrix SVD.
    rank : `int`, optional
        Rank of the forward operator; default ``svd.n_keep_eff``.

    Returns
    -------
    res : `numpy.ndarray`
        ``dW - S @ (d / w)`` in µm of wavefront, over ``svd.kj_grid``.

    Notes
    -----
    This is the **achieved** residual, and it differs from
    ``aos_fwhm.residual_dW``, which returns the subspace projection
    ``(I - U_eff U_eff^T) dW``. The projection is what an *ideal* correction in the
    kept subspace would leave, and it is independent of the recovered amplitudes; the
    achieved residual depends on them, so it is the only one of the two that can see
    a regularizer trading wavefront for amplitude. For the unregularized truncated
    solution the two coincide, which is the check the study script runs.
    """
    S = forward_operator(svd, rank=rank)
    w = np.asarray(svd.normalization_weights, float)
    return np.nan_to_num(np.asarray(dW, float)) - S @ (np.asarray(d, float) / w)
