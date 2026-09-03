#!/usr/bin/env python
"""(a) Is k=1..6 sufficient to SPECIFY the 50 DOF / 34 v-modes?
   (b) Does the R subtraction actually remove the higher-k content?"""
import sys
import numpy as np

_PKG = "/Users/roodman/Astrophysics/Claude/packages"
sys.path.insert(0, _PKG + "/ts_ofc/python")
from lsst.ts.ofc import OFCData
import importlib.util as _iu
_spec = _iu.spec_from_file_location(
    "ofc_svd", _PKG + "/ts_intrinsic_wavefront/python/lsst/ts/intrinsic/wavefront/ofc_svd.py")
osv = _iu.module_from_spec(_spec); sys.modules["ofc_svd"] = osv
_spec.loader.exec_module(osv)

ZK = [z for z in range(4, 27) if z not in (20, 21)]
K_MIN, K_MAX, N_KEEP, K_ALL = 1, 6, 34, 30

S = np.nan_to_num(np.asarray(OFCData("lsst").sensitivity_matrix, float))[:, ZK, :]
N = osv.load_normalization_weights(None, osv.DEFAULT_NORM_YAML)

def hatS(k_hi):
    return S[K_MIN:k_hi + 1].reshape(-1, 50) @ np.diag(N)

S6, S31 = hatS(K_MAX), hatS(K_ALL)
print(f"S_hat  k<=6 : {S6.shape}     k<=30 : {S31.shape}")

# ---------- (a) does k<=6 determine the DOF / the v-modes? ----------
for lab, M in (("k<=6", S6), ("k<=30", S31)):
    sv = np.linalg.svd(M, compute_uv=False)
    print(f"  {lab:6s} rank {np.linalg.matrix_rank(M):2d}/50   sigma_1={sv[0]:.3g}  "
          f"sigma_34={sv[33]:.3g}  sigma_50={sv[49]:.3g}  cond(1/50)={sv[0]/sv[49]:.3g}"
          f"  cond(1/34)={sv[0]/sv[33]:.4g}")

V6 = np.linalg.svd(S6, full_matrices=False)[2].T
V31 = np.linalg.svd(S31, full_matrices=False)[2].T
for nk in (34, 50):
    ang = np.degrees(np.arccos(np.clip(
        np.linalg.svd(V6[:, :nk].T @ V31[:, :nk], compute_uv=False), -1, 1)))
    print(f"  top-{nk} v-mode subspace k<=6 vs k<=30: max principal angle "
          f"{ang.max():.2f} deg, median {np.median(ang):.2f} deg")
# per-mode pairing
dots = np.abs(np.diag(V6[:, :N_KEEP].T @ V31[:, :N_KEEP]))
print(f"  per-mode |v_m(k<=6) . v_m(k<=30)|: min {dots.min():.3f}, "
      f"median {np.median(dots):.3f}  ({int((dots > 0.99).sum())}/34 above 0.99)")

# ---------- (b) how much wavefront does the k<=6 reconstruction leave behind? ----
svd = osv.build_ofc_svd(ZK, K_MIN, K_MAX, N_KEEP)
V, Sig = svd.V, svd.Sigma
print("\nper-kept-mode: fraction of that mode's TOTAL wavefront power left unsubtracted"
      "\nbecause R reconstructs only k<=6:")
frac = []
for m in range(svd.n_keep_eff):
    r = np.tensordot(S, N * V[:, m], axes=([2], [0]))      # (31, 21), all field orders
    lo = np.sum(r[K_MIN:K_MAX + 1] ** 2)
    hi = np.sum(r[K_MAX + 1:] ** 2)
    frac.append(hi / (lo + hi))
frac = np.array(frac)
for m in range(0, 34, 1):
    if m < 8 or m >= 26:
        print(f"   u{m+1:<3d} unsubtracted power fraction {frac[m]:8.4f}"
              f"   amplitude {np.sqrt(frac[m]):6.3f}")
print(f"  median over 34 modes: power {np.median(frac):.4f} "
      f"(amplitude {np.sqrt(np.median(frac)):.3f})")
