"""
Narrowest blue-edge band: 25 nm FWHM centered at 410 nm, with OD4 blocking
across the whole silicon range (320-1080 nm).

Architecture (two surfaces, combined incoherently through the substrate):
  * FRONT = 3-cavity Fabry-Perot @ 410 nm  -> defines the sharp ~25 nm band
            (it transmits at 410 but leaks outside its mirror stopband).
  * BACK  = wide blocker = long-wave-pass edge (~393 nm) + graduated short-
            wave-pass stacks (428 -> 1120 nm). It passes a ~35 nm window around
            410 and rejects everything else 320-1080 nm to OD4.
The front sets the band edges; the back supplies the OD4 rejection. Because the
two live on opposite faces they are optimised independently, then combined.

High index Nb2O5 (n~2.3, table to 2.5 um); low index SiO2. Engine tmm_fast.
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
LO, HI = "SiO2", "Nb2O5"
CENTER = 410.0
FWHM = 25.0
BAND = (CENTER - FWHM / 2, CENTER + FWHM / 2)     # 397.5 - 422.5
SI_CUTOFF = 1080.0
WL = np.linspace(320.0, 1120.0, 1101)
K_REAL = {"Nb2O5": 3e-4}
OD = 4.0


def sp_init(d_nm):
    d = torch.tensor(np.asarray(d_nm), dtype=torch.float64).clamp(min=1.0)
    return torch.log(torch.expm1(d)).clone().requires_grad_(True)


def fp_cavity(m=2, ncav=3, lam0=CENTER):
    dH, dL = tf.qwot_nm(HI, lam0), tf.qwot_nm(LO, lam0)
    mats, d = [], []
    for c in range(ncav):
        for _ in range(m):
            mats += [HI, LO]; d += [dH, dL]
        mats += [HI]; d += [2 * dH]
        for _ in range(m):
            mats += [LO, HI]; d += [dL, dH]
        if c < ncav - 1:
            mats += [LO]; d += [dL]
    return mats, np.array(d)


def graded(centers_pairs, air_side_low=True):
    mats, d = [], []
    for lam0, npair in centers_pairs:
        pair = [LO, HI] if air_side_low else [HI, LO]
        for _ in range(npair):
            mats += pair
            d += [tf.qwot_nm(pair[0], lam0), tf.qwot_nm(pair[1], lam0)]
    return mats, np.array(d)


def optimize(mats, d0, incident, exit, pass_mask, block_mask,
             n_iter=1800, lr=1.0, w_in=4.0, w_bl=0.3, tag=""):
    N = tf.build_N(mats, WL, incident=incident, exit=exit, kdict=None)
    raw = sp_init(d0)
    pm = torch.tensor(pass_mask)
    bm = None if block_mask is None else torch.tensor(block_mask)
    opt = torch.optim.Adam([raw], lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=700, gamma=0.5)
    for it in range(n_iter):
        opt.zero_grad()
        T = tf.transmission(N, F.softplus(raw), WL, theta=0.0, pol="s")["T"]
        loss = w_in * ((T[pm] - 1.0) ** 2).mean()
        if bm is not None:
            logT = torch.log10(T[bm].clamp(min=1e-10))
            loss = loss + w_bl * (F.relu(logT + (OD + 0.5)) ** 2).mean()
        loss.backward(); opt.step(); sched.step()
        if it % 400 == 0 or it == n_iter - 1:
            with torch.no_grad():
                Tn = T.numpy()
                print(f"  [{tag}] it{it:4d} pass<T>={Tn[pass_mask].mean():.3f} "
                      f"block-worst OD={-np.log10(Tn[block_mask].max()+1e-12):.2f}")
    return F.softplus(raw).detach().numpy()


def band_metrics(T):
    i = int(np.argmax(T)); pk = T[i]; h = 0.5 * pk
    lo = i
    while lo > 0 and T[lo] >= h:
        lo -= 1
    hi = i
    while hi < len(T) - 1 and T[hi] >= h:
        hi += 1
    reg = slice(lo, hi + 1)
    c = np.sum(WL[reg] * T[reg]) / np.sum(T[reg])
    return pk, c, WL[hi] - WL[lo]


def main():
    # ---- FRONT: 3-cavity FP @410; force ~25 nm width, no side block ----
    c_mats, c_d0 = fp_cavity()
    pass_core = (WL >= 399) & (WL <= 421)          # 22 nm core -> ~25 nm FWHM
    c_d = optimize(c_mats, c_d0, "Air", fs.SUBSTRATE, pass_core, None,
                   n_iter=1200, w_in=4.0, tag="FP front")

    # ---- BACK: LWP(~393) + graduated SWP(428..1120), deep (OD4) ----
    b_blue, bd_blue = graded([(336.0, 20)], air_side_low=True)   # block <393
    b_red, bd_red = graded([(500.0, 16), (660.0, 17), (870.0, 19),
                            (1120.0, 19)], air_side_low=True)     # block 428-1120
    b_mats = b_blue + b_red
    b_d0 = np.concatenate([bd_blue, bd_red])
    pass_win = (WL >= 398) & (WL <= 423)
    block_all = ((WL >= 322) & (WL <= 390)) | ((WL >= 430) & (WL <= 1120))
    b_d = optimize(b_mats, b_d0, fs.SUBSTRATE, "Air", pass_win, block_all,
                   n_iter=3200, w_in=2.0, w_bl=1.5, tag="blocker back")

    total_um = (c_d.sum() + b_d.sum()) / 1e3
    print(f"\n[stack] FP front {len(c_d)} layers + blocker back {len(b_d)} "
          f"layers, total {total_um:.2f} um")

    kw = dict(kdict=K_REAL, ar_mats=b_mats, ar_d_nm=b_d, sub_internal=1.0)
    T_norm = fs.system_T_unpol(c_mats, c_d, WL, theta_deg=0.0, **kw)
    T_beam, thetas, w = fs.beam_average(c_mats, c_d, WL, n_theta=15, **kw)
    T_14 = fs.system_T_unpol(c_mats, c_d, WL, theta_deg=14.0, **kw)
    T_23 = fs.system_T_unpol(c_mats, c_d, WL, theta_deg=23.0, **kw)

    oob = ((WL >= 322) & (WL <= 388)) | ((WL >= 432) & (WL <= SI_CUTOFF))
    print(f"\n=== 25 nm band @ 410 nm, OD4 to {SI_CUTOFF:.0f} nm ===")
    for lbl, T in [("normal", T_norm), ("theta=14", T_14),
                   ("theta=23", T_23), ("beam 14-23", T_beam)]:
        pk, c, fw = band_metrics(T)
        print(f"  {lbl:11s}: peak={pk:.3f} center={c:.1f} FWHM={fw:.1f}nm "
              f"| block worst-OD={-np.log10(T[oob].max()+1e-12):.2f}")
    pk0, c0, fw0 = band_metrics(T_norm); pkb, cb, fwb = band_metrics(T_beam)
    print(f"\n  beam vs collimated: center {c0:.1f}->{cb:.1f} nm "
          f"({cb-c0:+.1f}), FWHM {fw0:.1f}->{fwb:.1f} nm, "
          f"peak {pk0:.3f}->{pkb:.3f} ({(pkb/pk0-1)*100:+.0f}%)")

    np.savez(os.path.join(OUT, "narrowblock_design.npz"),
             WL=WL, T_norm=T_norm, T_beam=T_beam, T_14=T_14, T_23=T_23,
             front_materials=np.array(c_mats), front_thickness_nm=c_d,
             back_materials=np.array(b_mats), back_thickness_nm=b_d)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 8))
    a1.fill_between(WL, T_14, T_23, color="C1", alpha=0.2,
                    label="per-angle envelope 14°–23°")
    a1.plot(WL, T_norm, "C0-", lw=1.8,
            label=f"collimated: {fw0:.0f} nm @ {c0:.0f} nm, peak {pk0:.2f}")
    a1.plot(WL, T_beam, "C3-", lw=2.0,
            label=f"Rubin beam: {fwb:.0f} nm @ {cb:.0f} nm, peak {pkb:.2f}")
    a1.axvline(CENTER, color="gray", ls=":", lw=1)
    a1.set_xlim(380, 445); a1.set_ylim(-0.02, 1.02)
    a1.set_xlabel("wavelength (nm)"); a1.set_ylabel("system transmission")
    a1.set_title(f"25 nm band @ 410 nm in Rubin beam "
                 f"({len(c_d)}+{len(b_d)} layers, {total_um:.1f} µm)")
    a1.legend(fontsize=9); a1.grid(alpha=0.3)

    for T, c, l in [(T_norm, "C0-", "normal"), (T_beam, "C3-", "beam")]:
        a2.semilogy(WL, np.clip(T, 1e-9, 1), c, lw=1.2, label=l)
    a2.axhline(1e-4, color="gray", ls=":", lw=1); a2.text(325, 1.4e-4, "OD4", fontsize=8)
    a2.axvline(SI_CUTOFF, color="purple", ls="-.", lw=1)
    a2.set_xlabel("wavelength (nm)"); a2.set_ylabel("transmission (log)")
    a2.set_ylim(1e-7, 2); a2.legend(fontsize=8.5); a2.grid(alpha=0.3, which="both")
    a2.set_title("full-range blocking (320–1120 nm)", fontsize=10)
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    out = os.path.join(OUT, "figs", "narrowblock_design.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
