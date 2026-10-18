"""
Medium-band (30 nm) filter at 475 nm: design + Rubin-beam evaluation.

A 30 nm / 475 nm band (6.3% fractional) is too narrow for the edge-filter
approach used for g (a QWOT stopband is ~120 nm wide), so we use an all-
dielectric multi-cavity Fabry-Perot: coupled [(HL)^m 2H (LH)^m] cavities whose
resonance sits at 475 nm. Thicknesses are then refined by Adam (through
tmm_fast autograd) to a flat 30 nm top.

The point of the exercise: quantify how the Rubin annular f/1.23 beam
(incidence 14-23 deg) shifts, broadens, and attenuates a medium band relative
to a collimated (normal-incidence) beam -- the effect that decides narrow-band
feasibility.
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
LO, HI = "SiO2", "Ta2O5"
CENTER = 475.0
FWHM = 30.0
BAND = (CENTER - FWHM / 2, CENTER + FWHM / 2)   # 460-490
WL = np.linspace(400.0, 560.0, 481)
K_REAL = tf.K_REAL


def softplus_init(d_nm):
    d = torch.tensor(np.asarray(d_nm), dtype=torch.float64).clamp(min=1.0)
    return torch.log(torch.expm1(d)).clone().requires_grad_(True)


def fp_init(m=2, ncav=3, lam0=CENTER):
    """Coupled multi-cavity Fabry-Perot: [(HL)^m 2H (LH)^m] x ncav."""
    dH, dL = tf.qwot_nm(HI, lam0), tf.qwot_nm(LO, lam0)
    mats, d = [], []
    for c in range(ncav):
        for _ in range(m):
            mats += [HI, LO]; d += [dH, dL]
        mats += [HI]; d += [2 * dH]              # half-wave H spacer
        for _ in range(m):
            mats += [LO, HI]; d += [dL, dH]
        if c < ncav - 1:
            mats += [LO]; d += [dL]              # coupling layer
    return mats, np.array(d)


def design_ar(lam0=CENTER):
    """Backside AR optimised around the band center."""
    ar_mats = [HI, LO, HI, "MgF2"]
    d0 = [0.5 * tf.qwot_nm(HI, lam0), 0.3 * tf.qwot_nm(LO, lam0),
          0.1 * tf.qwot_nm(HI, lam0), tf.qwot_nm("MgF2", lam0)]
    raw = softplus_init(d0)
    Ns = tf.build_N(ar_mats, WL, incident=fs.SUBSTRATE, exit="Air", kdict=None)
    band = (WL >= BAND[0] - 20) & (WL <= BAND[1] + 20)
    opt = torch.optim.Adam([raw], lr=1.0)
    for _ in range(400):
        opt.zero_grad()
        R = tf.transmission(Ns, F.softplus(raw), WL, 0.0, "s")["R"]
        (R[band].mean() + R[band].max()).backward()
        opt.step()
    return ar_mats, F.softplus(raw).detach().numpy()


def design_bandpass():
    mats, d0 = fp_init()
    N = tf.build_N(mats, WL, incident="Air", exit=fs.SUBSTRATE, kdict=None)
    raw = softplus_init(d0)

    wl = torch.tensor(WL)
    inb = (wl >= BAND[0]) & (wl <= BAND[1])
    # demand T~1 across (almost) the whole 460-490 span so the optimizer must
    # widen the cavity to fill it (2 nm guard for finite edge slope).
    in_core = (wl >= BAND[0] + 2) & (wl <= BAND[1] - 2)
    oob = ((wl >= 405) & (wl <= BAND[0] - 6)) | ((wl >= BAND[1] + 6) & (wl <= 555))

    opt = torch.optim.Adam([raw], lr=1.0)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5)
    for it in range(2000):
        opt.zero_grad()
        T = tf.transmission(N, F.softplus(raw), WL, 0.0, "s")["T"]
        loss_in = ((T[in_core] - 1.0) ** 2).mean()
        loss_oob = (F.relu(torch.log10(T[oob].clamp(min=1e-9)) + 2.5) ** 2).mean()
        loss = loss_in + 0.1 * loss_oob
        loss.backward()
        opt.step(); sched.step()
        if it % 400 == 0 or it == 1999:
            with torch.no_grad():
                Tn = T.numpy()
                print(f"  it{it:4d} loss={loss.item():.3e} "
                      f"in<T>={Tn[inb.numpy()].mean():.3f} "
                      f"oob max={Tn[oob.numpy()].max():.1e}")
    return mats, F.softplus(raw).detach().numpy()


def band_metrics(T):
    """Peak, center, and *contiguous* FWHM around the global maximum."""
    i = int(np.argmax(T)); pk = T[i]; h = 0.5 * pk
    lo = i
    while lo > 0 and T[lo] >= h:
        lo -= 1
    hi = i
    while hi < len(T) - 1 and T[hi] >= h:
        hi += 1
    # centroid of the passband (robust center under asymmetry)
    reg = slice(lo, hi + 1)
    c = np.sum(WL[reg] * T[reg]) / np.sum(T[reg])
    return pk, c, (WL[lo], WL[hi]), WL[hi] - WL[lo]


def main():
    ar_mats, ar_d = design_ar()
    print(f"[AR] {len(ar_d)} layers")
    mats, d = design_bandpass()
    print(f"[stack] cavity {len(d)} layers + AR {len(ar_d)}, "
          f"total {(d.sum()+ar_d.sum())/1e3:.2f} um")

    kw = dict(kdict=K_REAL, ar_mats=ar_mats, ar_d_nm=ar_d, sub_internal=1.0)
    T_norm = fs.system_T_unpol(mats, d, WL, theta_deg=0.0, **kw)
    T_beam, thetas, w = fs.beam_average(mats, d, WL, n_theta=15, **kw)
    # also the two extreme angles, to show the spread that causes smearing
    T_14 = fs.system_T_unpol(mats, d, WL, theta_deg=14.0, **kw)
    T_23 = fs.system_T_unpol(mats, d, WL, theta_deg=23.0, **kw)

    print(f"\n=== {FWHM:.0f} nm band @ {CENTER:.0f} nm: collimated vs Rubin beam ===")
    rows = [("normal (collimated)", T_norm), ("theta=14 deg", T_14),
            ("theta=23 deg", T_23), ("beam-avg 14-23 deg", T_beam)]
    inb = (WL >= BAND[0]) & (WL <= BAND[1])
    for lbl, T in rows:
        pk, c, (lo, hi), fw = band_metrics(T)
        print(f"  {lbl:22s}: peak={pk:.3f} center={c:.1f}nm FWHM={fw:.1f}nm "
              f"in-band<T>={T[inb].mean():.3f}")
    pk0, c0, _, fw0 = band_metrics(T_norm)
    pkb, cb, _, fwb = band_metrics(T_beam)
    print(f"\n  beam vs collimated: center shift = {cb-c0:+.1f} nm, "
          f"FWHM {fw0:.1f}->{fwb:.1f} nm ({(fwb/fw0-1)*100:+.0f}%), "
          f"peak {pk0:.3f}->{pkb:.3f} ({(pkb/pk0-1)*100:+.0f}%)")
    # photon-throughput loss (integral over the band region, flat source)
    reg = (WL >= 440) & (WL <= 500)
    A0 = np.trapezoid(T_norm[reg], WL[reg]); Ab = np.trapezoid(T_beam[reg], WL[reg])
    print(f"  integrated throughput (440-500nm): {A0:.1f} -> {Ab:.1f} nm "
          f"({(Ab/A0-1)*100:+.0f}%)")

    np.savez(os.path.join(OUT, "narrow_design.npz"),
             WL=WL, T_norm=T_norm, T_beam=T_beam, T_14=T_14, T_23=T_23,
             materials=np.array(mats), thickness_nm=d,
             ar_materials=np.array(ar_mats), ar_thickness_nm=ar_d)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.fill_between(WL, T_14, T_23, color="C1", alpha=0.20,
                    label="per-angle envelope (14°–23°)")
    ax.plot(WL, T_norm, "C0-", lw=1.8,
            label=f"collimated: {fw0:.0f} nm @ {c0:.0f} nm, peak {pk0:.2f}")
    ax.plot(WL, T_beam, "C3-", lw=2.0,
            label=f"Rubin f/1.23 beam: {fwb:.0f} nm @ {cb:.0f} nm, peak {pkb:.2f}")
    ax.axvline(CENTER, color="gray", ls=":", lw=1)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("system transmission")
    ax.set_title(f"30 nm medium band at 475 nm in the Rubin annular beam "
                 f"({len(d)}+{len(ar_d)} layers)")
    ax.set_xlim(420, 530); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    out = os.path.join(OUT, "figs", "narrow_design.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
