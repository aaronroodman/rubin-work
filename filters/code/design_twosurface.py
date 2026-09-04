"""
Two-surface g-band filter with OD4 blocking across the full silicon range.

Architecture (as built for real Rubin filters):
  * FRONT surface  = long-wave-pass edge stack  -> blocks the blue (320-400 nm)
  * BACK  surface  = short-wave-pass edge stack -> blocks the red  (560-1080 nm)
  * The two coatings combine INCOHERENTLY through the mm fused-silica substrate
    (filterstack.system_T: T = T1 a T2 / (1 - R1' R2' a^2)).

Because the two edge filters live on opposite faces, the blue block OD comes
entirely from the front stack and the red block OD entirely from the back stack,
so each is designed independently to reach OD4 (T < 1e-4) over its stop region
while passing the 402-552 nm band at T~1.

High index = Nb2O5 (n~2.3, dispersion table to 2.5 um => physical NIR blocking);
low index = SiO2. Engine: tmm_fast autograd.
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
LO, HI = "SiO2", "Nb2O5"
BAND = (402.0, 552.0)
WL = np.linspace(320.0, 1120.0, 1001)      # full Si detector range
K_REAL = {"Nb2O5": 3e-4}                   # SiO2/substrate lossless
OD_TARGET = 4.0


def softplus_init(d_nm):
    d = torch.tensor(np.asarray(d_nm), dtype=torch.float64).clamp(min=1.0)
    return torch.log(torch.expm1(d)).clone().requires_grad_(True)


def qwot_stack(centers_pairs, air_side_low=True):
    """Concatenate QWOT stacks; each entry = (lam0_nm, n_pairs)."""
    mats, d = [], []
    for lam0, npair in centers_pairs:
        pair = ([LO, HI] if air_side_low else [HI, LO])
        for _ in range(npair):
            mats += pair
            d += [tf.qwot_nm(pair[0], lam0), tf.qwot_nm(pair[1], lam0)]
    return mats, np.array(d)


def optimize_surface(mats, d0, incident, exit, block_lo, block_hi,
                     n_iter=1600, lr=1.2, tag="", w_in=3.0):
    """Optimise one edge coating: T->1 in band, T->OD4 in [block_lo,block_hi]."""
    N = tf.build_N(mats, WL, incident=incident, exit=exit, kdict=None)
    raw = softplus_init(d0)
    wl = torch.tensor(WL)
    inband = (wl >= BAND[0] + 4) & (wl <= BAND[1] - 4)
    block = (wl >= block_lo) & (wl <= block_hi)
    opt = torch.optim.Adam([raw], lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=600, gamma=0.5)
    for it in range(n_iter):
        opt.zero_grad()
        T = tf.transmission(N, F.softplus(raw), WL, theta=0.0, pol="s")["T"]
        loss_in = ((T[inband] - 1.0) ** 2).mean()
        logT = torch.log10(T[block].clamp(min=1e-10))
        # push below OD4 with margin (target OD 4.5 so beam-shift still clears 4)
        loss_bl = (F.relu(logT + (OD_TARGET + 0.5)) ** 2).mean()
        loss = w_in * loss_in + 0.3 * loss_bl
        loss.backward(); opt.step(); sched.step()
        if it % 400 == 0 or it == n_iter - 1:
            with torch.no_grad():
                Tn = T.numpy()
                od = -np.log10(Tn[block.numpy()].max() + 1e-12)
                print(f"  [{tag}] it{it:4d} loss={loss.item():.2e} "
                      f"in<T>={Tn[inband.numpy()].mean():.3f} "
                      f"block-worst OD={od:.2f}")
    return F.softplus(raw).detach().numpy()


def load_filter_g():
    w, t = np.loadtxt(os.path.join(THRU, "filter_g.dat"), unpack=True)
    return np.interp(WL, w, t)


def main():
    # ---- FRONT: long-wave-pass, blocks 320-400 nm ----
    # stopband long-edge at 402 nm -> center ~ 402*0.855; 26 pairs for OD>4.
    f_mats, f_d0 = qwot_stack([(344.0, 26)], air_side_low=True)
    f_d = optimize_surface(f_mats, f_d0, "Air", fs.SUBSTRATE,
                           block_lo=322, block_hi=394, tag="LWP front")

    # ---- BACK: short-wave-pass, blocks 560-1120 nm ----
    # graduated stopbands, strong OVERLAP + top margin for the beam blue-shift:
    # 615->526-704, 775->663-887, 1015->868-1162 => continuous 526-1162 nm,
    # so the cone-shifted coverage (~-10 nm) still clears 1120 nm.
    b_mats, b_d0 = qwot_stack([(615.0, 17), (775.0, 17), (1015.0, 17)],
                              air_side_low=True)
    b_d = optimize_surface(b_mats, b_d0, fs.SUBSTRATE, "Air",
                           block_lo=560, block_hi=1120, n_iter=2600,
                           tag="SWP back")

    total_um = (f_d.sum() + b_d.sum()) / 1e3
    print(f"\n[stack] front LWP {len(f_d)} layers + back SWP {len(b_d)} layers, "
          f"total {total_um:.2f} um")

    # ---- system: front=LWP, back(ar slot)=SWP, incoherent combine ----
    kw = dict(kdict=K_REAL, ar_mats=b_mats, ar_d_nm=b_d, sub_internal=1.0)
    T_norm = fs.system_T_unpol(f_mats, f_d, WL, theta_deg=0.0, **kw)
    T_beam, thetas, w = fs.beam_average(f_mats, f_d, WL, n_theta=13, **kw)
    Tg = load_filter_g()

    inb = (WL >= BAND[0]) & (WL <= BAND[1])
    # OD metric excludes the ~10 nm edge-transition zones (not true leaks)
    oob = ((WL >= 322) & (WL <= 392)) | ((WL >= 566) & (WL <= 1120))
    print("\n=== two-surface g filter, OD4 target over 320-1120 nm ===")
    for lbl, T in [("normal incidence", T_norm), ("beam 14-23deg", T_beam)]:
        print(f"  {lbl:18s}: peak={T.max():.3f} in-band<T>={T[inb].mean():.3f} "
              f"worst-OD={-np.log10(T[oob].max()+1e-12):.2f} "
              f"median-OD={-np.log10(np.median(T[oob])+1e-12):.2f}")
    print(f"  {'Rubin filter_g':18s}: peak={Tg.max():.3f} "
          f"in-band<T>={Tg[inb].mean():.3f}")

    np.savez(os.path.join(OUT, "twosurface_design.npz"),
             WL=WL, T_norm=T_norm, T_beam=T_beam, Tg=Tg,
             front_materials=np.array(f_mats), front_thickness_nm=f_d,
             back_materials=np.array(b_mats), back_thickness_nm=b_d)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 8), sharex=True)
    a1.plot(WL, Tg, "k--", lw=1.4, label="Rubin g (filter_g.dat)")
    a1.plot(WL, T_norm, "C0-", lw=1.5,
            label=f"design normal (in-band⟨T⟩={T_norm[inb].mean():.3f})")
    a1.plot(WL, T_beam, "C3-", lw=1.5,
            label=f"design beam 14–23° (⟨T⟩={T_beam[inb].mean():.3f})")
    a1.axvspan(*BAND, color="C2", alpha=0.06)
    a1.set_ylabel("system transmission"); a1.set_ylim(-0.02, 1.02)
    a1.legend(fontsize=8.5); a1.grid(alpha=0.3)
    a1.set_title(f"Two-surface g filter (LWP front + SWP back), "
                 f"{len(f_d)}+{len(b_d)} layers, {total_um:.1f} µm, Nb₂O₅/SiO₂")

    for T, c, l in [(Tg, "k--", "Rubin g"), (T_norm, "C0-", "design normal"),
                    (T_beam, "C3-", "design beam")]:
        a2.semilogy(WL, np.clip(T, 1e-9, 1), c, lw=1.3, label=l)
    a2.axhline(1e-4, color="gray", ls=":", lw=1)
    a2.text(325, 1.4e-4, "OD4", fontsize=8)
    a2.axvspan(*BAND, color="C2", alpha=0.06)
    a2.set_xlabel("wavelength (nm)"); a2.set_ylabel("transmission (log)")
    a2.set_ylim(1e-7, 2); a2.legend(fontsize=8.5, loc="upper right")
    a2.grid(alpha=0.3, which="both")
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    out = os.path.join(OUT, "figs", "twosurface_design.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
