"""
Beam-optimized blocker for the 25 nm @ 410 nm narrowband.

Motivation: in design_narrowblock.py the back blocker was optimized at NORMAL
incidence, so its OD4 rejection collapsed to ~OD2.3 across the Rubin annular
cone (14-23 deg) -- every dielectric stopband shifts/weakens with angle. Here we
re-optimize the SAME back-blocker architecture but drive the ETENDUE-WEIGHTED
BEAM-AVERAGED transmission (14-23 deg, s+p) below OD4, i.e. design for the beam.

Front FP cavity (the 26 nm band) is reused unchanged from narrowblock_design.npz.
Compares normal-optimized vs beam-optimized blocking side by side.
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
CENTER, FWHM = 410.0, 25.0
BAND = (CENTER - FWHM / 2, CENTER + FWHM / 2)
SI_CUTOFF = 1080.0
WL = np.linspace(320.0, 1120.0, 801)
K_REAL = {"Nb2O5": 3e-4}
OD = 4.0
NG = 1.46                                   # mean fused-silica index for Snell

# beam angles (in air) + etendue weights, mirroring filterstack.beam_average
TH_AIR = np.linspace(14.0, 23.0, 5)
W_BEAM = np.sin(np.deg2rad(TH_AIR)) * np.cos(np.deg2rad(TH_AIR))
W_BEAM = W_BEAM / W_BEAM.sum()
TH_GLASS = np.rad2deg(np.arcsin(np.sin(np.deg2rad(TH_AIR)) / NG))


def sp_init(d_nm):
    d = torch.tensor(np.asarray(d_nm), dtype=torch.float64).clamp(min=1.0)
    return torch.log(torch.expm1(d)).clone().requires_grad_(True)


def graded(centers_pairs, air_side_low=True):
    mats, d = [], []
    for lam0, npair in centers_pairs:
        pair = [LO, HI] if air_side_low else [HI, LO]
        for _ in range(npair):
            mats += pair
            d += [tf.qwot_nm(pair[0], lam0), tf.qwot_nm(pair[1], lam0)]
    return mats, np.array(d)


def beam_T(N, d, thetas_glass, weights):
    """Differentiable etendue-weighted unpolarized T(WL) over the cone."""
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
        torch.nn.utils.clip_grad_norm_([raw], 1.0)      # tame the stiff barrier
        opt.step(); sched.step()
        # keep-best (guards against late Adam divergence)
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


def band_metrics(T):
    i = int(np.argmax(T)); pk = T[i]; h = 0.5 * pk
    lo, hi = i, i
    while lo > 0 and T[lo] >= h:
        lo -= 1
    while hi < len(T) - 1 and T[hi] >= h:
        hi += 1
    c = np.sum(WL[slice(lo, hi + 1)] * T[slice(lo, hi + 1)]) / np.sum(T[slice(lo, hi + 1)])
    return pk, c, WL[hi] - WL[lo]


def region_od(T):
    reg = [(322, 388, "blue"), (430, 470, "red-near"), (470, 700, "red-mid"),
           (700, 900, "red-hi"), (900, 1080, "red-deep")]
    return {lab: -np.log10(T[(WL >= a) & (WL <= b)].max() + 1e-12) for a, b, lab in reg}


def main():
    # ---- reuse the saved 26 nm FP cavity (front) ----
    saved = np.load(os.path.join(OUT, "narrowblock_design.npz"), allow_pickle=True)
    c_mats = [str(m) for m in saved["front_materials"]]
    c_d = saved["front_thickness_nm"]

    # ---- back blocker: same architecture, optimized over the BEAM ----
    b_blue, bd_blue = graded([(336.0, 20)], air_side_low=True)
    b_red, bd_red = graded([(500.0, 16), (660.0, 17), (870.0, 19), (1120.0, 19)],
                           air_side_low=True)
    b_mats = b_blue + b_red
    b_d0 = np.concatenate([bd_blue, bd_red])
    # pass core is up-weighted heavily; red block starts at 445 (24 nm guard from
    # the 421 pass edge) so the ~10 nm cone smear cannot close the passband. The
    # front cavity's mirror stopband covers the 421-445 near-band gap.
    pass_win = (WL >= 400) & (WL <= 421)
    block_all = ((WL >= 322) & (WL <= 388)) | ((WL >= 445) & (WL <= 1080))
    print("Optimizing back blocker over beam angles", np.round(TH_AIR, 1), "deg ...")
    b_d = optimize_back_beam(b_mats, b_d0, pass_win, block_all,
                             n_iter=1800, lr=0.3, w_in=30.0, w_bl=1.0)

    total_um = (c_d.sum() + b_d.sum()) / 1e3
    print(f"\n[stack] FP front {len(c_d)} + beam-blocker back {len(b_d)} layers, "
          f"{total_um:.2f} um")

    kw = dict(kdict=K_REAL, ar_mats=b_mats, ar_d_nm=b_d, sub_internal=1.0)
    T_norm = fs.system_T_unpol(c_mats, c_d, WL, theta_deg=0.0, **kw)
    T_beam, _, _ = fs.beam_average(c_mats, c_d, WL, n_theta=15, **kw)

    # normal-optimized reference (from the earlier run), re-evaluated on this grid
    ref = np.load(os.path.join(OUT, "narrowblock_design.npz"), allow_pickle=True)
    Tb_ref = np.interp(WL, ref["WL"], ref["T_beam"])

    oob = ((WL >= 322) & (WL <= 388)) | ((WL >= 432) & (WL <= SI_CUTOFF))
    print(f"\n=== beam-optimized blocker: 25 nm @ 410 nm, OD4 to {SI_CUTOFF:.0f} ===")
    for lbl, T in [("normal", T_norm), ("beam 14-23", T_beam)]:
        pk, c, fw = band_metrics(T)
        print(f"  {lbl:11s}: peak={pk:.3f} center={c:.1f} FWHM={fw:.1f}nm "
              f"| beam worst-OD={-np.log10(T[oob].max()+1e-12):.2f}")
    print("  per-region worst-OD (beam):")
    for k, v in region_od(T_beam).items():
        print(f"     {k:9s}: {v:.2f}")
    print(f"  [reference: normal-optimized blocker had beam worst-OD"
          f"={-np.log10(Tb_ref[oob].max()+1e-12):.2f}]")

    np.savez(os.path.join(OUT, "narrowblock_beam.npz"),
             WL=WL, T_norm=T_norm, T_beam=T_beam, Tb_ref=Tb_ref,
             back_thickness_nm=b_d, back_materials=np.array(b_mats))

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 8))
    pk0, c0, fw0 = band_metrics(T_norm); pkb, cb, fwb = band_metrics(T_beam)
    a1.plot(WL, T_norm, "C0-", lw=1.6, label=f"collimated: {fw0:.0f} nm @ {c0:.0f} nm")
    a1.plot(WL, T_beam, "C3-", lw=2.0, label=f"Rubin beam: {fwb:.0f} nm @ {cb:.0f} nm")
    a1.axvline(CENTER, color="gray", ls=":", lw=1)
    a1.set_xlim(385, 440); a1.set_ylim(-0.02, 1.02)
    a1.set_xlabel("wavelength (nm)"); a1.set_ylabel("system transmission")
    a1.set_title(f"Beam-optimized narrowband: {len(c_d)}+{len(b_d)} layers, {total_um:.1f} µm")
    a1.legend(fontsize=9); a1.grid(alpha=0.3)

    a2.semilogy(WL, np.clip(Tb_ref, 1e-9, 1), "0.6", lw=1.1,
                label="beam, normal-optimized blocker (old)")
    a2.semilogy(WL, np.clip(T_beam, 1e-9, 1), "C3-", lw=1.4,
                label="beam, beam-optimized blocker (new)")
    a2.axhline(1e-4, color="k", ls=":", lw=1); a2.text(325, 1.4e-4, "OD4", fontsize=8)
    a2.axvline(SI_CUTOFF, color="purple", ls="-.", lw=1)
    a2.set_xlabel("wavelength (nm)"); a2.set_ylabel("beam transmission (log)")
    a2.set_ylim(1e-7, 2); a2.legend(fontsize=8.5); a2.grid(alpha=0.3, which="both")
    a2.set_title("blocking under the f/1.23 beam: normal-opt vs beam-opt", fontsize=10)
    os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
    out = os.path.join(OUT, "figs", "narrowblock_beam.png")
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
