"""
~10 nm narrow band at 655 nm (Halpha + [N II] 654.8/658.4 nm), OD4 blocking
across the silicon range, evaluated in the Rubin f/1.23 beam.

Same architecture as the 410 nm case (design_narrowblock_beam.py):
  * FRONT = 3-cavity Fabry-Perot (m=3 mirror pairs) -> ~10 nm band at 655 nm.
  * BACK  = beam-optimized wide blocker. Because the band now sits in the MIDDLE
            of the Si range, the back must block a wide BLUE side (320-632 nm,
            graded longpass stacks) AND a wide RED side (680-1080 nm, graded
            shortpass stacks), passing a window around 655 nm.
The blocker's out-of-band loss is the etendue-weighted BEAM average (14-23 deg,
s+p), so its OD4 rejection is robust to the cone (design-for-the-beam).

Pushing to 10 nm tests the narrowband feasibility limit: at 655 nm the cone
sweeps the band ~10 nm (14->23 deg), comparable to the bandwidth itself.
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
CENTER, FWHM = 655.0, 10.0
BAND = (650.0, 660.0)
SI_CUTOFF = 1080.0
WL = np.linspace(320.0, 1120.0, 1101)
K_REAL = {"Nb2O5": 3e-4}
OD = 4.0
NG = 1.46

TH_AIR = np.linspace(14.0, 23.0, 5)
W_BEAM = np.sin(np.deg2rad(TH_AIR)) * np.cos(np.deg2rad(TH_AIR))
W_BEAM = W_BEAM / W_BEAM.sum()
TH_GLASS = np.rad2deg(np.arcsin(np.sin(np.deg2rad(TH_AIR)) / NG))


def sp_init(d_nm):
    d = torch.tensor(np.asarray(d_nm), dtype=torch.float64).clamp(min=1.0)
    return torch.log(torch.expm1(d)).clone().requires_grad_(True)


def fp_cavity(m=3, ncav=3, lam0=CENTER):
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


def beam_T(N, d, thetas_glass, weights):
    acc = 0.0
    for tg, wt in zip(thetas_glass, weights):
        Ts = tf.transmission(N, d, WL, theta=tg, pol="s")["T"]
        Tp = tf.transmission(N, d, WL, theta=tg, pol="p")["T"]
        acc = acc + wt * 0.5 * (Ts + Tp)
    return acc


def optimize_back_beam(mats, d0, pass_mask, block_mask,
                       n_iter=1600, lr=0.3, w_in=30.0, w_bl=1.0, tag="beam-back"):
    N = tf.build_N(mats, WL, incident=fs.SUBSTRATE, exit="Air", kdict=None)
    raw = sp_init(d0)
    pm, bm = torch.tensor(pass_mask), torch.tensor(block_mask)
    opt = torch.optim.Adam([raw], lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=400, gamma=0.6)
    best = {"loss": np.inf, "d": None}
    for it in range(n_iter):
        opt.zero_grad()
        Tb = beam_T(N, F.softplus(raw), TH_GLASS, W_BEAM)
        loss = w_in * ((Tb[pm] - 1.0) ** 2).mean()
        logT = torch.log10(Tb[bm].clamp(min=1e-10))
        loss = loss + w_bl * (F.relu(logT + (OD + 0.5)) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], 1.0)
        opt.step(); sched.step()
        lv = loss.item()
        if lv < best["loss"]:
            best = {"loss": lv, "d": F.softplus(raw).detach().numpy().copy()}
        if it % 300 == 0 or it == n_iter - 1:
            with torch.no_grad():
                Tn = Tb.numpy()
                print(f"  [{tag}] it{it:4d} loss={lv:.3f} "
                      f"beam-pass<T>={Tn[pass_mask].mean():.3f} "
                      f"beam-block worst-OD={-np.log10(Tn[block_mask].max()+1e-12):.2f}")
    print(f"  [{tag}] best loss={best['loss']:.3f}")
    return best["d"]


def band_metrics(T, win=(600, 710)):
    m = (WL >= win[0]) & (WL <= win[1])
    idx = np.where(m)[0]
    i = idx[np.argmax(T[m])]; pk = T[i]; h = 0.5 * pk
    lo, hi = i, i
    while lo > 0 and T[lo] >= h:
        lo -= 1
    while hi < len(T) - 1 and T[hi] >= h:
        hi += 1
    c = np.sum(WL[slice(lo, hi + 1)] * T[slice(lo, hi + 1)]) / np.sum(T[slice(lo, hi + 1)])
    return pk, c, WL[hi] - WL[lo]


def main():
    c_mats, c_d = fp_cavity(m=3, ncav=3)
    print(f"cavity: {len(c_d)} layers")

    # BACK blocker: graded longpass (blue) + graded shortpass (red).
    # Stopbands kept clear of the pass window (blue top stopband ends 634 nm;
    # red bottom stopband starts 671 nm) so the blocker never fights the band.
    # Window spans the collimated band (650-660) AND its beam-shifted position
    # (~640-651); the cavity covers the 632-640 / 660-668 near-band shoulders.
    blue_mats, blue_d = graded([(554.0, 15), (408.0, 15), (304.0, 15)])
    red_mats, red_d = graded([(785.0, 16), (1050.0, 16)])
    b_mats = blue_mats + red_mats
    b_d0 = np.concatenate([blue_d, red_d])
    pass_win = (WL >= 636) & (WL <= 664)
    block_all = ((WL >= 322) & (WL <= 632)) | ((WL >= 668) & (WL <= 1080))
    print("beam-optimizing blocker over", np.round(TH_AIR, 1), "deg ...")
    b_d = optimize_back_beam(b_mats, b_d0, pass_win, block_all, n_iter=1600)

    total_um = (c_d.sum() + b_d.sum()) / 1e3
    print(f"\n[stack] FP front {len(c_d)} + beam-blocker back {len(b_d)} layers, "
          f"{total_um:.2f} um")

    kw = dict(kdict=K_REAL, ar_mats=b_mats, ar_d_nm=b_d, sub_internal=1.0)
    T_norm = fs.system_T_unpol(c_mats, c_d, WL, theta_deg=0.0, **kw)
    T_beam, _, _ = fs.beam_average(c_mats, c_d, WL, n_theta=21, **kw)
    T_14 = fs.system_T_unpol(c_mats, c_d, WL, theta_deg=14.0, **kw)
    T_23 = fs.system_T_unpol(c_mats, c_d, WL, theta_deg=23.0, **kw)

    oob = ((WL >= 322) & (WL <= 632)) | ((WL >= 668) & (WL <= SI_CUTOFF))
    print(f"\n=== ~10 nm band @ 655 nm, OD4 to {SI_CUTOFF:.0f} nm ===")
    for lbl, T in [("normal", T_norm), ("theta=14", T_14),
                   ("theta=23", T_23), ("beam 14-23", T_beam)]:
        pk, c, fw = band_metrics(T)
        print(f"  {lbl:11s}: peak={pk:.3f} center={c:.1f} FWHM={fw:.1f}nm "
              f"| block worst-OD={-np.log10(T[oob].max()+1e-12):.2f}")
    pk0, c0, fw0 = band_metrics(T_norm); pkb, cb, fwb = band_metrics(T_beam)
    # effective throughput: integral over the band region (flat source)
    reg = (WL >= 630) & (WL <= 685)
    A0 = np.trapezoid(T_norm[reg], WL[reg]); Ab = np.trapezoid(T_beam[reg], WL[reg])
    print(f"\n  beam vs collimated: center {c0:.1f}->{cb:.1f} nm ({cb-c0:+.1f}), "
          f"FWHM {fw0:.1f}->{fwb:.1f} nm, peak {pk0:.3f}->{pkb:.3f} "
          f"({(pkb/pk0-1)*100:+.0f}%)")
    print(f"  integrated in-band throughput (630-685nm): {A0:.1f}->{Ab:.1f} nm "
          f"({(Ab/A0-1)*100:+.0f}%)")

    np.savez(os.path.join(OUT, "halpha_design.npz"),
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
    for x in (654.8, 656.3, 658.4):
        a1.axvline(x, color="gray", ls=":", lw=0.8)
    a1.set_xlim(625, 685); a1.set_ylim(-0.02, 1.02)
    a1.set_xlabel("wavelength (nm)"); a1.set_ylabel("system transmission")
    a1.set_title(f"~10 nm band @ 655 nm (Hα+[N II]) in Rubin beam "
                 f"({len(c_d)}+{len(b_d)} layers, {total_um:.1f} µm)")
    a1.legend(fontsize=9); a1.grid(alpha=0.3)

    for T, c, l in [(T_norm, "C0-", "normal"), (T_beam, "C3-", "beam")]:
        a2.semilogy(WL, np.clip(T, 1e-9, 1), c, lw=1.2, label=l)
    a2.axhline(1e-4, color="k", ls=":", lw=1); a2.text(325, 1.4e-4, "OD4", fontsize=8)
    a2.axvline(SI_CUTOFF, color="purple", ls="-.", lw=1)
    a2.set_xlabel("wavelength (nm)"); a2.set_ylabel("transmission (log)")
    a2.set_ylim(1e-7, 2); a2.legend(fontsize=8.5); a2.grid(alpha=0.3, which="both")
    a2.set_title("full-range blocking (320–1120 nm), beam-optimized", fontsize=10)
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    out = os.path.join(OUT, "figs", "halpha_design.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
