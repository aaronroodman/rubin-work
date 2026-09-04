"""
Halpha filter, corrected for the Rubin beam: design the collimated band ~10 nm
to the RED (666 nm) so the f/1.23 cone shifts the beam-averaged passband onto
Halpha (656.3 nm). Objective: maximize beam transmission AT Halpha with the
thinnest useful band.

The finesse scan (scan_halpha.py) shows the operational width floors at ~9.4 nm
(the cone smear), and a 3-cavity FP with m=3 mirror pairs is the sweet spot:
collimated ~9.5 nm gives beam FWHM ~10 nm at T(Halpha) ~0.8; thinner cavities
don't narrow the beam band but throw away throughput. So m=3 @ 666 nm here.

Back = beam-optimized wide blocker (blue graded longpass + red graded shortpass)
for OD4 across the Si range, window shifted red to hold both the collimated
(666) and beam (656) band positions.
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
HALPHA = 656.3
LAM_DESIGN = 666.2                      # collimated center (from the scan)
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


def fp_cavity(m=3, ncav=3, lam0=LAM_DESIGN):
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


def graded(centers_pairs):
    mats, d = [], []
    for lam0, npair in centers_pairs:
        for _ in range(npair):
            mats += [LO, HI]
            d += [tf.qwot_nm(LO, lam0), tf.qwot_nm(HI, lam0)]
    return mats, np.array(d)


def beam_T(N, d):
    acc = 0.0
    for tg, wt in zip(TH_GLASS, W_BEAM):
        Ts = tf.transmission(N, d, WL, theta=tg, pol="s")["T"]
        Tp = tf.transmission(N, d, WL, theta=tg, pol="p")["T"]
        acc = acc + wt * 0.5 * (Ts + Tp)
    return acc


def optimize_back_beam(mats, d0, pass_mask, block_mask, n_iter=1600):
    N = tf.build_N(mats, WL, incident=fs.SUBSTRATE, exit="Air", kdict=None)
    raw = sp_init(d0)
    pm, bm = torch.tensor(pass_mask), torch.tensor(block_mask)
    opt = torch.optim.Adam([raw], lr=0.3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=400, gamma=0.6)
    best = {"loss": np.inf, "d": None}
    for it in range(n_iter):
        opt.zero_grad()
        Tb = beam_T(N, F.softplus(raw))
        loss = 30.0 * ((Tb[pm] - 1.0) ** 2).mean()
        logT = torch.log10(Tb[bm].clamp(min=1e-10))
        loss = loss + 1.0 * (F.relu(logT + (OD + 0.5)) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw], 1.0)
        opt.step(); sched.step()
        lv = loss.item()
        if lv < best["loss"]:
            best = {"loss": lv, "d": F.softplus(raw).detach().numpy().copy()}
        if it % 400 == 0 or it == n_iter - 1:
            with torch.no_grad():
                Tn = Tb.numpy()
                print(f"  it{it:4d} beam-pass<T>={Tn[pass_mask].mean():.3f} "
                      f"beam-block worst-OD={-np.log10(Tn[block_mask].max()+1e-12):.2f}")
    return best["d"]


def band_metrics(T, win=(600, 710)):
    m = (WL >= win[0]) & (WL <= win[1]); idx = np.where(m)[0]
    i = idx[np.argmax(T[m])]; pk = T[i]; h = 0.5 * pk
    lo, hi = i, i
    while lo > 0 and T[lo] >= h:
        lo -= 1
    while hi < len(T) - 1 and T[hi] >= h:
        hi += 1
    c = np.sum(WL[slice(lo, hi + 1)] * T[slice(lo, hi + 1)]) / np.sum(T[slice(lo, hi + 1)])
    return pk, c, WL[hi] - WL[lo]


def main():
    c_mats, c_d = fp_cavity(m=3, lam0=LAM_DESIGN)
    print(f"cavity m=3 @ {LAM_DESIGN} nm: {len(c_d)} layers")

    blue_mats, blue_d = graded([(562.0, 15), (414.0, 15), (308.0, 15)])
    red_mats, red_d = graded([(793.0, 16), (1058.0, 16)])
    b_mats = blue_mats + red_mats
    b_d0 = np.concatenate([blue_d, red_d])
    pass_win = (WL >= 648) & (WL <= 674)          # holds coll (666) + beam (656)
    block_all = ((WL >= 322) & (WL <= 644)) | ((WL >= 678) & (WL <= 1080))
    print("beam-optimizing blocker ...")
    b_d = optimize_back_beam(b_mats, b_d0, pass_win, block_all, n_iter=1600)

    total_um = (c_d.sum() + b_d.sum()) / 1e3
    print(f"[stack] {len(c_d)}+{len(b_d)} layers, {total_um:.2f} um")

    kw = dict(kdict=K_REAL, ar_mats=b_mats, ar_d_nm=b_d, sub_internal=1.0)
    T_norm = fs.system_T_unpol(c_mats, c_d, WL, theta_deg=0.0, **kw)
    T_beam, _, _ = fs.beam_average(c_mats, c_d, WL, n_theta=25, **kw)
    T_14 = fs.system_T_unpol(c_mats, c_d, WL, theta_deg=14.0, **kw)
    T_23 = fs.system_T_unpol(c_mats, c_d, WL, theta_deg=23.0, **kw)

    oob = ((WL >= 322) & (WL <= 644)) | ((WL >= 678) & (WL <= SI_CUTOFF))
    pk0, c0, fw0 = band_metrics(T_norm); pkb, cb, fwb = band_metrics(T_beam)
    tha = float(np.interp(HALPHA, WL, T_beam))
    print(f"\n=== Halpha filter (designed at {LAM_DESIGN} nm, m=3) ===")
    print(f"  collimated : {fw0:.1f} nm @ {c0:.1f} nm, peak {pk0:.3f}")
    print(f"  Rubin beam : {fwb:.1f} nm @ {cb:.1f} nm, peak {pkb:.3f}")
    print(f"  ** beam transmission AT Halpha (656.3 nm) = {tha:.3f} **")
    print(f"  block worst-OD (beam, 320-1080) = {-np.log10(T_beam[oob].max()+1e-12):.2f}")

    np.savez(os.path.join(OUT, "halpha_v2.npz"), WL=WL, T_norm=T_norm,
             T_beam=T_beam, T_14=T_14, T_23=T_23,
             front_thickness_nm=c_d, back_thickness_nm=b_d,
             front_materials=np.array(c_mats), back_materials=np.array(b_mats))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 8))
    a1.fill_between(WL, T_14, T_23, color="C1", alpha=0.2, label="per-angle 14°–23°")
    a1.plot(WL, T_norm, "C0-", lw=1.6,
            label=f"collimated (design): {fw0:.0f} nm @ {c0:.0f} nm")
    a1.plot(WL, T_beam, "C3-", lw=2.2,
            label=f"Rubin beam: {fwb:.0f} nm @ {cb:.0f} nm, T(Hα)={tha:.2f}")
    for x in (654.8, 656.3, 658.4):
        a1.axvline(x, color="k", ls=":", lw=0.8)
    a1.axvline(HALPHA, color="C3", ls="--", lw=0.8)
    a1.set_xlim(635, 685); a1.set_ylim(-0.02, 1.02)
    a1.set_xlabel("wavelength (nm)"); a1.set_ylabel("system transmission")
    a1.set_title(f"Hα filter pre-shifted for the beam: designed @ {LAM_DESIGN:.0f} nm "
                 f"→ beam-centered on Hα ({len(c_d)}+{len(b_d)} layers)")
    a1.legend(fontsize=9); a1.grid(alpha=0.3)

    for T, c, l in [(T_norm, "C0-", "normal"), (T_beam, "C3-", "beam")]:
        a2.semilogy(WL, np.clip(T, 1e-9, 1), c, lw=1.2, label=l)
    a2.axhline(1e-4, color="k", ls=":", lw=1); a2.text(325, 1.4e-4, "OD4", fontsize=8)
    a2.axvline(SI_CUTOFF, color="purple", ls="-.", lw=1)
    a2.set_xlabel("wavelength (nm)"); a2.set_ylabel("transmission (log)")
    a2.set_ylim(1e-7, 2); a2.legend(fontsize=8.5); a2.grid(alpha=0.3, which="both")
    a2.set_title("full-range blocking (beam)", fontsize=10)
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    out = os.path.join(OUT, "figs", "halpha_v2.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
