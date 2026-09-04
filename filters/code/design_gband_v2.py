"""
Realistic g-band filter design + evaluation.

Steps
1. Optimize a backside broadband AR (glass->air) over the g passband.
2. Optimize the front bandpass coating for a flat top AND deep out-of-band
   blocking (target OD>3) using graduated longpass+shortpass blocker stacks and
   an OD-aware (log-barrier) loss. Design model = air|coating|fused-silica,
   normal incidence, lossless (thicknesses are what we solve for).
3. Evaluate the full system with realistic absorption (k>0), the fused-silica
   substrate, and the backside AR, at normal incidence AND averaged over the
   Rubin annular f/1.23 beam (14-23 deg, unpolarized). Compare to the real
   Rubin g filter (filter_g.dat).

Engine: tmm_fast (PyTorch autograd on thicknesses).
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import thinfilm as tf
import filterstack as fs

HERE = os.path.dirname(os.path.abspath(__file__))
# scripts live in filters/code/; products go to filters/output/
OUT = os.path.join(os.path.dirname(HERE), "output")
THRU = "/Users/roodman/Astrophysics/LSST/Throughput/throughputs/baseline"
LO, HI = "SiO2", "Ta2O5"
BAND = (402.0, 552.0)
WL = np.linspace(340.0, 820.0, 481)

# Realistic extinction for high-quality ion-beam-sputtered coatings in the
# optical (the tmmax tabulated k is far too high -- represents lossy films).
K_REAL = tf.K_REAL   # {"Ta2O5": 2e-4, ...}; SiO2/MgF2 lossless


def softplus_init(d_nm):
    d = torch.tensor(np.asarray(d_nm), dtype=torch.float64).clamp(min=1.0)
    return torch.log(torch.expm1(d)).clone().requires_grad_(True)


# ---------------------------------------------------------------- AR design
def design_ar():
    """4-layer broadband AR on fused silica (glass->air), minimise unpolarised
    R over the g band at normal incidence."""
    ar_mats = [HI, LO, HI, "MgF2"]
    lam0 = 477.0
    d0 = [0.5 * tf.qwot_nm(HI, lam0), 0.3 * tf.qwot_nm(LO, lam0),
          0.1 * tf.qwot_nm(HI, lam0), tf.qwot_nm("MgF2", lam0)]
    raw = softplus_init(d0)
    Ns = tf.build_N(ar_mats, WL, incident=fs.SUBSTRATE, exit="Air", kdict=None)
    band = (WL >= BAND[0]) & (WL <= BAND[1])
    opt = torch.optim.Adam([raw], lr=1.0)
    for it in range(500):
        opt.zero_grad()
        d = F.softplus(raw)
        Rs = tf.transmission(Ns, d, WL, theta=0.0, pol="s")["R"]
        loss = Rs[band].mean() + Rs[band].max()
        loss.backward()
        opt.step()
    d = F.softplus(raw).detach().numpy()
    with torch.no_grad():
        Rband = tf.transmission(Ns, torch.tensor(d), WL, 0.0, "s")["R"][band]
    print(f"[AR] 4-layer glass->air: <R>={float(Rband.mean()):.4f} "
          f"max R={float(Rband.max()):.4f} over band")
    return ar_mats, d


# ------------------------------------------------------------- blocker init
def blocker_init():
    """Graduated longpass (blue) + shortpass (red) QWOT blocker stacks."""
    nH = float(np.real(tf.n_of(HI, np.array([477.0]))[0]))
    nL = float(np.real(tf.n_of(LO, np.array([477.0]))[0]))
    frac = (2 / np.pi) * np.arcsin((nH - nL) / (nH + nL))
    mats, dth = [], []
    # blue longpass: stopband long-edge at 402 nm
    for lam0, npair in [(BAND[0] * (1 - frac), 15)]:
        for _ in range(npair):
            mats += [HI, LO]; dth += [tf.qwot_nm(HI, lam0), tf.qwot_nm(LO, lam0)]
    # red shortpass: two graduated stacks to widen the red block to ~800 nm
    for lam0, npair in [(BAND[1] * (1 + frac), 13), (700.0, 12)]:
        for _ in range(npair):
            mats += [LO, HI]; dth += [tf.qwot_nm(LO, lam0), tf.qwot_nm(HI, lam0)]
    return mats, np.array(dth)


# -------------------------------------------------------------- bandpass fit
def design_bandpass():
    mats, d0 = blocker_init()
    N = tf.build_N(mats, WL, incident="Air", exit=fs.SUBSTRATE, kdict=None)
    raw = softplus_init(d0)

    wl = torch.tensor(WL)
    inb = (wl >= BAND[0]) & (wl <= BAND[1])
    edge = (torch.abs(wl - BAND[0]) < 10) | (torch.abs(wl - BAND[1]) < 10)
    # out-of-band with guard bands around the edges
    oob = ((wl >= 355) & (wl <= BAND[0] - 12)) | ((wl >= BAND[1] + 12) & (wl <= 790))
    in_core = inb & ~edge

    opt = torch.optim.Adam([raw], lr=1.5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=400, gamma=0.5)
    n_iter = 1600
    for it in range(n_iter):
        opt.zero_grad()
        d = F.softplus(raw)
        T = tf.transmission(N, d, WL, theta=0.0, pol="s")["T"]
        # in-band: drive to 1 ; out-of-band: log-barrier pushing below OD3
        loss_in = ((T[in_core] - 1.0) ** 2).mean()
        logT = torch.log10(T[oob].clamp(min=1e-9))
        loss_oob = (F.relu(logT + 3.0) ** 2).mean()      # penalise T>1e-3
        loss = loss_in + 0.15 * loss_oob
        loss.backward()
        opt.step(); sched.step()
        if it % 200 == 0 or it == n_iter - 1:
            with torch.no_grad():
                Tn = T.numpy()
                print(f"  it{it:4d} loss={loss.item():.3e} "
                      f"in<T>={Tn[inb.numpy()].mean():.3f} "
                      f"oob max={Tn[oob.numpy()].max():.1e} "
                      f"oob<OD>={-np.log10(Tn[oob.numpy()].mean()+1e-12):.2f}")
    return mats, F.softplus(raw).detach().numpy()


def load_filter_g():
    w, t = np.loadtxt(os.path.join(THRU, "filter_g.dat"), unpack=True)
    return np.interp(WL, w, t)


def main():
    ar_mats, ar_d = design_ar()
    mats, d = design_bandpass()
    total_um = (d.sum() + ar_d.sum()) / 1e3
    print(f"\n[stack] bandpass {len(d)} layers + AR {len(ar_d)} layers, "
          f"total {total_um:.2f} um")

    kw = dict(kdict=K_REAL, ar_mats=ar_mats, ar_d_nm=ar_d, sub_internal=1.0)
    T_norm = fs.system_T_unpol(mats, d, WL, theta_deg=0.0, **kw)
    T_beam, thetas, w = fs.beam_average(mats, d, WL, n_theta=13, **kw)
    Tg = load_filter_g()

    inb = (WL >= BAND[0]) & (WL <= BAND[1])
    oob = ((WL >= 355) & (WL <= 390)) | ((WL >= 564) & (WL <= 790))
    print("\n=== g-band system throughput (realistic: k>0, substrate, AR) ===")
    for lbl, T in [("normal incidence", T_norm), ("beam-avg 14-23deg", T_beam)]:
        print(f"  {lbl:20s}: peak={T.max():.3f}  in-band<T>={T[inb].mean():.3f}  "
              f"out-of-band<OD>={-np.log10(T[oob].mean()+1e-12):.2f}  "
              f"max-leak={T[oob].max():.1e}")
    print(f"  {'Rubin filter_g':20s}: peak={Tg.max():.3f}  "
          f"in-band<T>={Tg[inb].mean():.3f}")

    np.savez(os.path.join(OUT, "gband_design_v2.npz"),
             WL=WL, T_norm=T_norm, T_beam=T_beam, Tg=Tg,
             bp_materials=np.array(mats), bp_thickness_nm=d,
             ar_materials=np.array(ar_mats), ar_thickness_nm=ar_d,
             thetas=thetas)

    # ---- plots: linear (throughput) + log (blocking) ----
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    a1.plot(WL, Tg, "k--", lw=1.6, label="Rubin g filter (filter_g.dat)")
    a1.plot(WL, T_norm, "C0-", lw=1.6,
            label=f"design, normal incidence  (in-band⟨T⟩={T_norm[inb].mean():.3f})")
    a1.plot(WL, T_beam, "C3-", lw=1.6,
            label=f"design, f/1.23 beam 14–23°  (⟨T⟩={T_beam[inb].mean():.3f})")
    a1.axvspan(*BAND, color="C2", alpha=0.06)
    a1.set_ylabel("system transmission"); a1.set_ylim(-0.02, 1.02)
    a1.legend(fontsize=8.5, loc="center right"); a1.grid(alpha=0.3)
    a1.set_title(f"Realistic g-band filter: {len(d)}+{len(ar_d)} layers, "
                 f"{total_um:.1f} µm, SiO₂/Ta₂O₅ on fused silica + AR")

    a2.semilogy(WL, np.clip(Tg, 1e-8, 1), "k--", lw=1.4, label="Rubin g")
    a2.semilogy(WL, np.clip(T_norm, 1e-8, 1), "C0-", lw=1.4, label="design normal")
    a2.semilogy(WL, np.clip(T_beam, 1e-8, 1), "C3-", lw=1.4, label="design beam")
    a2.axhline(1e-3, color="gray", ls=":", lw=1); a2.text(345, 1.3e-3, "OD3", fontsize=8)
    a2.axvspan(*BAND, color="C2", alpha=0.06)
    a2.set_xlabel("wavelength (nm)"); a2.set_ylabel("transmission (log)")
    a2.set_ylim(1e-6, 2); a2.legend(fontsize=8.5, loc="lower center"); a2.grid(alpha=0.3)

    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    out = os.path.join(OUT, "figs", "gband_design_v2.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
