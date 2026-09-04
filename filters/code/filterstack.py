"""
Realistic filter-system model: bandpass coating on a fused-silica substrate
with a backside AR coat, evaluated vs angle of incidence and polarization, and
averaged over the Rubin annular f/1.23 beam.

Physics
-------
* Front bandpass coating (air -> glass) : coherent (tmm_fast).
* Thick fused-silica substrate          : incoherent (mm-scale; interference
                                          fringes washed out for broadband light).
* Backside AR coat (glass -> air)        : coherent.
* Substrate + two coated faces combined by incoherent (intensity) multiple-beam
  summation:
      T_sys = T1 a T2 / (1 - R1' R2' a^2)
  T1  = front-coating transmittance (air->glass)
  R1' = front-coating reflectance seen from the glass side
  T2  = AR transmittance (glass->air)
  R2' = AR reflectance seen from the glass side
  a   = substrate single-pass internal transmittance (~1 for fused silica)

Angle: incidence angle theta is defined in air. Inside the substrate the angle
is theta_g with sin(theta_g) = sin(theta)/n_glass (Snell).

Beam average: annular cone theta in [14, 23] deg, weight ~ sin th cos th
(uniformly illuminated annular pupil, Abbe sine condition), unpolarized (mean of
s and p).
"""
import numpy as np
import torch

import thinfilm as tf

SUBSTRATE = "SiO2"   # fused silica substrate index (from tmmax nk table)


def _coh_TR(seq_mats, d_inner_nm, wl_nm, theta_deg, pol, incident, exit, kdict):
    """Coherent T,R for one stack, arbitrary incident/exit media & angle."""
    from tmm_fast import coh_tmm as ctf
    N = tf.build_N(seq_mats, wl_nm, incident=incident, exit=exit, kdict=kdict)
    L = N.shape[1]
    d = torch.empty(1, L, dtype=torch.float64)
    d[0, 0] = np.inf
    d[0, -1] = np.inf
    if L > 2:
        d[0, 1:-1] = torch.as_tensor(np.asarray(d_inner_nm), dtype=torch.float64) * 1e-9
    lam = torch.tensor(np.asarray(wl_nm) * 1e-9, dtype=torch.float64)
    th = torch.tensor([np.deg2rad(theta_deg)], dtype=torch.float64)
    r = ctf(pol, N, d, th, lam)
    return np.asarray(r["T"]).reshape(-1), np.asarray(r["R"]).reshape(-1)


def system_T(front_mats, front_d_nm, wl_nm, theta_deg=0.0, pol="s",
             kdict=None, ar_mats=None, ar_d_nm=None, sub_internal=1.0):
    """
    Unpolarized-capable system transmittance for one (theta, pol).

    front_mats/front_d_nm : bandpass coating (inner layers), air -> substrate.
    ar_mats/ar_d_nm       : backside AR (inner layers), substrate -> air.
                            If None, bare substrate/air interface is used.
    kdict                 : material->k absorption (None = lossless).
    sub_internal          : substrate single-pass internal transmittance (<=1).
    """
    n_glass = np.real(tf.n_of(SUBSTRATE, wl_nm))
    theta_g = np.rad2deg(np.arcsin(np.sin(np.deg2rad(theta_deg)) / n_glass))  # per-lambda

    # Front coating: air -> glass (forward) and glass -> air seen from glass (R1')
    T1, _ = _coh_TR(front_mats, front_d_nm, wl_nm, theta_deg, pol,
                    "Air", SUBSTRATE, kdict)
    # reverse stack, incident = glass, at theta_g (theta_g varies with lambda ->
    # evaluate with mean angle; dispersion of theta_g is tiny across the band)
    tg = float(np.mean(theta_g))
    _, R1p = _coh_TR(list(reversed(front_mats)), list(reversed(front_d_nm)),
                     wl_nm, tg, pol, SUBSTRATE, "Air", kdict)

    # Backside AR: glass -> air
    if ar_mats is None:
        R2p = np.full_like(wl_nm, ((float(n_glass.mean()) - 1) /
                                   (float(n_glass.mean()) + 1)) ** 2)
        T2 = 1.0 - R2p
    else:
        T2, R2p = _coh_TR(ar_mats, ar_d_nm, wl_nm, tg, pol, SUBSTRATE, "Air", kdict)

    a = sub_internal
    T_sys = T1 * a * T2 / (1.0 - R1p * R2p * a ** 2)
    return T_sys


def system_T_unpol(front_mats, front_d_nm, wl_nm, theta_deg=0.0, **kw):
    """Unpolarized system T = mean of s and p."""
    Ts = system_T(front_mats, front_d_nm, wl_nm, theta_deg, pol="s", **kw)
    Tp = system_T(front_mats, front_d_nm, wl_nm, theta_deg, pol="p", **kw)
    return 0.5 * (Ts + Tp)


def beam_average(front_mats, front_d_nm, wl_nm,
                 theta_lo=14.0, theta_hi=23.0, n_theta=13, **kw):
    """
    Throughput averaged over the Rubin annular cone (14-23 deg), unpolarized.
    Weight w(theta) = sin(theta) cos(theta) (annular pupil, sine condition).
    Returns (T_beam, thetas, weights).
    """
    th = np.linspace(theta_lo, theta_hi, n_theta)
    w = np.sin(np.deg2rad(th)) * np.cos(np.deg2rad(th))
    w = w / w.sum()
    acc = np.zeros_like(wl_nm)
    for ti, wi in zip(th, w):
        acc += wi * system_T_unpol(front_mats, front_d_nm, wl_nm, ti, **kw)
    return acc, th, w
