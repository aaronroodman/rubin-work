"""
Thin-film multilayer engine + materials for Rubin filter-design studies.

Forward/gradient engine: tmm_fast (PyTorch, autograd on layer thicknesses),
validated against Steven Byrnes' `tmm` to ~1e-6 (see validate_engines.py).

Materials: refractive-index dispersion tables shipped with `tmmax`
(refractiveindex.info sources). We use the real n(lambda); k is optional
(set to 0 for an idealized lossless design, which is what these low-loss
dielectric coatings approach in the optical band).

Units: wavelengths and thicknesses are handled in nm at the API surface and
converted to meters for tmm_fast internally.
"""
import os
import numpy as np
import torch

_NK_DIR = os.path.join(
    os.path.dirname(__import__("tmmax").__file__), "nk_data", "numpy"
)


def load_nk(material):
    """Return (wl_nm, n, k) arrays for a tmmax material (e.g. 'SiO2')."""
    a = np.load(os.path.join(_NK_DIR, material + ".npy"))
    wl_nm = a[0] * 1e3  # microns -> nm
    return wl_nm, a[1], a[2]


# Realistic extinction coeff for high-quality ion-beam-sputtered optical
# coatings (the tmmax tabulated k is far too high -- lossy-film data). Anything
# not listed is treated as lossless (k=0), incl. the fused-silica substrate.
K_REAL = {"Ta2O5": 2e-4, "TiO2": 3e-4, "Nb2O5": 3e-4}


def n_of(material, wl_nm, kdict=None):
    """
    Interpolate refractive index n onto wl_nm (nm) and attach a constant
    extinction k from `kdict` (material -> k). Sign convention n + i*k
    (positive imaginary part = absorption), as required by tmm/tmm_fast.
    kdict=None -> lossless (k=0).
    """
    w, n, _ = load_nk(material)
    ni = np.interp(wl_nm, w, n)
    kk = 0.0 if kdict is None else float(kdict.get(material, 0.0))
    return ni + 1j * kk


def build_N(materials, wl_nm, incident="Air", exit="Air", kdict=None):
    """
    Assemble the [1 x L x W] complex-index tensor for a single stack.

    `materials` is the ordered list of the *inner* layer materials.
    Incident and exit semi-infinite media are prepended/appended.
    """
    seq = [incident] + list(materials) + [exit]
    W = len(wl_nm)
    N = np.ones((1, len(seq), W), dtype=complex)
    for i, m in enumerate(seq):
        N[0, i, :] = n_of(m, wl_nm, kdict=kdict)
    return torch.tensor(N, dtype=torch.complex128)


def transmission(N, d_inner_nm, wl_nm, theta=0.0, pol="both"):
    """
    Coherent transmission/reflection of one stack via tmm_fast.

    d_inner_nm : 1-D tensor/array of the L-2 inner-layer thicknesses (nm).
    Returns dict with numpy arrays T, R over wavelength (unpolarized = mean
    of s & p when pol='both').
    """
    from tmm_fast import coh_tmm as ctf

    if not torch.is_tensor(d_inner_nm):
        d_inner_nm = torch.tensor(np.asarray(d_inner_nm), dtype=torch.float64)
    L = N.shape[1]
    d = torch.empty(1, L, dtype=d_inner_nm.dtype)
    d[0, 0] = np.inf
    d[0, -1] = np.inf
    d[0, 1:-1] = d_inner_nm * 1e-9  # nm -> m
    lam = torch.tensor(np.asarray(wl_nm) * 1e-9, dtype=torch.float64)
    th = torch.tensor([float(theta)], dtype=torch.float64)

    out = {}
    pols = ["s", "p"] if pol == "both" else [pol]
    Ts, Rs = [], []
    for p in pols:
        r = ctf(p, N, d, th, lam)
        Ts.append(r["T"].reshape(-1))
        Rs.append(r["R"].reshape(-1))
    T = torch.stack(Ts).mean(0)
    R = torch.stack(Rs).mean(0)
    out["T"], out["R"] = T, R
    return out


def qwot_nm(material, lam0_nm, wl_ref=None):
    """Quarter-wave optical thickness (physical nm) at lam0 for a material."""
    n0 = float(np.real(n_of(material, np.array([lam0_nm]))[0]))
    return lam0_nm / (4.0 * n0)
