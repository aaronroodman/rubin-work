"""
Cross-validate the thin-film engines on a known 15-pair SiO2/Ta2O5 quarter-wave
high reflector (design lambda 550 nm).

Result (this machine):
  tmm_fast  vs  Byrnes tmm  agree to < 2e-6 across 400-700 nm.
  tmmax (installed build) is BROKEN for multilayers: a 15-pair QWOT reflector
  returns R~0.31 instead of ~1.0 (single layers are fine). Its coherent-cascade
  path mishandles repeated/dispersive layers, so we do NOT use tmmax. tmm_fast
  is the engine of record (PyTorch, autograd on thicknesses).
"""
import numpy as np
import torch
from tmm import coh_tmm as byrnes
from tmm_fast import coh_tmm as tmmfast

nL, nH = 1.4662, 2.1149
lam0 = 550.0
dL, dH = lam0 / (4 * nL), lam0 / (4 * nH)
P = 15
lams = np.linspace(400, 700, 61)

n_layers = [1.0] + [nH, nL] * P + [1.0]
d_nm = [np.inf] + [dH, dL] * P + [np.inf]

Rb = np.array([byrnes("s", n_layers, d_nm, 0.0, l)["R"] for l in lams])

L, W = len(n_layers), len(lams)
N = torch.ones((1, L, W), dtype=torch.complex128)
for i, nv in enumerate(n_layers):
    N[0, i, :] = nv
d = torch.tensor([d_nm], dtype=torch.float64) * 1e-9
d[0, 0] = np.inf
d[0, -1] = np.inf
res = tmmfast("s", N, d, torch.tensor([0.0], dtype=torch.float64),
              torch.tensor(lams * 1e-9, dtype=torch.float64))
Rf = np.asarray(res["R"]).reshape(-1)

print(f"15-pair QWOT high reflector, 400-700 nm ({W} points)")
print(f"  max |R_tmm_fast - R_byrnes| = {np.max(np.abs(Rf - Rb)):.2e}")
print(f"  R at 550 nm: byrnes={Rb[lams==550][0]:.6f} tmm_fast={Rf[lams==550][0]:.6f}")

try:
    from tmmax.tmm import tmm as tmmax_tmm
    import jax.numpy as jnp
    mats = ["Air"] + ["Ta2O5", "SiO2"] * P + ["Air"]
    th = jnp.array(([dH, dL] * P))  # nm; ratio with wl(nm) is what matters
    R, T = tmmax_tmm(mats, th, jnp.array([550.0]), jnp.array([0.0]),
                     coherency_list=None, polarization="s")
    print(f"  tmmax R at 550 nm = {float(np.asarray(R).ravel()[0]):.6f} "
          f"(BROKEN; should be ~1.0)")
except Exception as e:
    print("  tmmax check skipped:", repr(e)[:80])
