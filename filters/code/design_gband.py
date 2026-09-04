"""
Design an LSST g-band-shaped dielectric bandpass with tmm_fast autograd, and
compare its throughput to the real Rubin g filter (baseline/filter_g.dat).

Architecture: SiO2 (low) / Ta2O5 (high) alternating stack, freestanding in air.
Init = long-wave-pass edge stack (blocks < ~402 nm) + short-wave-pass edge
stack (blocks > ~552 nm) => a ~150 nm-wide bandpass. All layer thicknesses are
then refined by Adam to flatten the in-band top and sharpen the edges.

Engine: tmm_fast (PyTorch). Gradients d(loss)/d(thickness) via autograd.
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import thinfilm as tf

HERE = os.path.dirname(os.path.abspath(__file__))
# scripts live in filters/code/; products go to filters/output/
OUT = os.path.join(os.path.dirname(HERE), "output")
THROUGHPUTS = "/Users/roodman/Astrophysics/LSST/Throughput/throughputs/baseline"

LO, HI = "SiO2", "Ta2O5"          # low / high index coating materials
BAND = (402.0, 552.0)             # g-band ~half-max edges (nm)
WL = np.linspace(350.0, 760.0, 420)


def load_filter_g():
    w, t = np.loadtxt(os.path.join(THROUGHPUTS, "filter_g.dat"), unpack=True)
    return np.interp(WL, w, t)


def initial_stack():
    """Two-edge QWOT construction -> materials list + init thicknesses (nm)."""
    # Stopband half-width fraction for this material pair sets where to center.
    nH = float(np.real(tf.n_of(HI, np.array([477.0]))[0]))
    nL = float(np.real(tf.n_of(LO, np.array([477.0]))[0]))
    frac = (2 / np.pi) * np.arcsin((nH - nL) / (nH + nL))  # ~0.116

    # long-wave-pass: stopband long-edge at blue band edge -> center below it
    lam_lwp = BAND[0] * (1 - frac)
    # short-wave-pass: stopband short-edge at red band edge -> center above it
    lam_swp = BAND[1] * (1 + frac)

    mats, dth = [], []
    for lam0, m in [(lam_lwp, 11), (lam_swp, 11)]:
        for _ in range(m):
            mats += [HI, LO]
            dth += [tf.qwot_nm(HI, lam0), tf.qwot_nm(LO, lam0)]
    return mats, np.array(dth)


def target_and_weights():
    """Top-hat target (1 in band, 0 outside) with a de-weighted edge zone."""
    tgt = ((WL >= BAND[0]) & (WL <= BAND[1])).astype(float)
    w = np.ones_like(WL)
    w[(WL >= BAND[0]) & (WL <= BAND[1])] = 3.0          # prioritise throughput
    edge = (np.abs(WL - BAND[0]) < 12) | (np.abs(WL - BAND[1]) < 12)
    w[edge] = 0.3                                        # don't fight finite slope
    return torch.tensor(tgt), torch.tensor(w)


def main():
    mats, d0 = initial_stack()
    N = tf.build_N(mats, WL, with_k=False)              # lossless dielectric
    tgt, wts = target_and_weights()

    # Parameterise thickness > 0 via softplus; init raw so softplus(raw)=d0.
    d0_t = torch.tensor(d0, dtype=torch.float64)
    raw = torch.log(torch.expm1(d0_t.clamp(min=1.0))).clone().requires_grad_(True)

    opt = torch.optim.Adam([raw], lr=2.0)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=250, gamma=0.5)

    def forward(raw):
        d = torch.nn.functional.softplus(raw)
        T = tf.transmission(N, d, WL, theta=0.0, pol="s")["T"]
        return T, d

    n_iter = 900
    for it in range(n_iter):
        opt.zero_grad()
        T, d = forward(raw)
        loss = (wts * (T - tgt) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
        if it % 100 == 0 or it == n_iter - 1:
            with torch.no_grad():
                Tn = T.numpy()
                inb = (WL >= BAND[0]) & (WL <= BAND[1])
                oob = (WL < BAND[0] - 15) | (WL > BAND[1] + 15)
                print(f"it{it:4d} loss={loss.item():.4e}  "
                      f"in-band<T>={Tn[inb].mean():.3f}  "
                      f"out-of-band max={Tn[oob].max():.3f}  "
                      f"nlayers={len(d)}  total={d.sum().item()/1e3:.2f}um")

    with torch.no_grad():
        Tdes = forward(raw)[0].numpy()
        dfin = torch.nn.functional.softplus(raw).numpy()
    Tg = load_filter_g()

    # ---- report ----
    inb = (WL >= BAND[0]) & (WL <= BAND[1])
    print("\n=== g-band throughput comparison (in-band = %.0f-%.0f nm) ==="
          % BAND)
    print(f"  designed  : peak={Tdes.max():.3f}  in-band<T>={Tdes[inb].mean():.3f}")
    print(f"  Rubin filter_g: peak={Tg.max():.3f}  in-band<T>={Tg[inb].mean():.3f}")
    print(f"  layers={len(dfin)}  total thickness={dfin.sum()/1e3:.2f} um")
    np.savez(os.path.join(OUT, "gband_design.npz"),
             WL=WL, T=Tdes, Tg=Tg, thickness_nm=dfin, materials=np.array(mats))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(WL, Tg, "k--", lw=1.8, label="Rubin g filter (filter_g.dat)")
    ax.plot(WL, Tdes, "C0-", lw=1.8,
            label=f"designed SiO2/Ta2O5 ({len(dfin)} layers, "
                  f"{dfin.sum()/1e3:.1f} µm)")
    ax.axvspan(*BAND, color="C2", alpha=0.06)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("transmission")
    ax.set_title("Designed dielectric bandpass vs Rubin g filter")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="center right", fontsize=9)
    ax.grid(alpha=0.3)
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    out = os.path.join(OUT, "figs", "gband_design.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
