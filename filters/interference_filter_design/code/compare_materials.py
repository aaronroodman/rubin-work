"""
Does a higher-index high layer help an Halpha narrowband? Two effects:
  (1) higher contrast (nH-nL)/(nH+nL) -> sharper edges / narrower collimated band
      for a given number of layers;
  (2) higher effective index -> the f/1.23 cone shifts & smears the band LESS,
      lowering the ~9.4 nm operational floor (the real limit at Rubin).

For each high-index material (paired with SiO2), 3-cavity FP:
  - collimated FWHM at m=3 (edge/finesse proxy)
  - the CONE-SMEAR FLOOR: a very thin cavity (m=5) auto-centered on Halpha, whose
    beam-averaged width is essentially the pure cone smear
  - the useful m=3 beam band: FWHM and T(Halpha)
All bare-backside (relative comparison).
"""
import numpy as np
import torch
import thinfilm as tf
import filterstack as fs

LO = "SiO2"
HALPHA = 656.3
WL = np.linspace(610.0, 710.0, 1201)     # 0.083 nm
KDICT = {"Ta2O5": 6.5e-3}                 # only Ta2O5 has non-negligible k here


def fp_cavity(HI, m, ncav=3, lam0=666.0):
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


def fwhm(T):
    i = int(np.argmax(T)); pk = T[i]; h = 0.5 * pk
    lo, hi = i, i
    while lo > 0 and T[lo] >= h:
        lo -= 1
    while hi < len(T) - 1 and T[hi] >= h:
        hi += 1
    return pk, WL[i], WL[hi] - WL[lo]


def coll(mats, d):
    N = tf.build_N(mats, WL, incident="Air", exit=fs.SUBSTRATE, kdict=KDICT)
    return tf.transmission(N, torch.tensor(d), WL, 0.0, "s")["T"].numpy()


def beam(mats, d):
    T, _, _ = fs.beam_average(mats, d, WL, n_theta=25, kdict=KDICT,
                              ar_mats=None, ar_d_nm=None)
    return T


def autocenter(HI, m):
    lam = 666.0
    for _ in range(3):
        mats, d = fp_cavity(HI, m, lam0=lam)
        _, pk, _ = fwhm(beam(mats, d))
        lam += HALPHA - pk
    return lam


print(f"{'HI mat':8} {'nH@656':>6} {'contrast':>8} {'coll_FWHM(m3)':>13} "
      f"{'FLOOR(m5)':>10} {'shift(m5)':>9} {'beamFWHM(m3)':>12} {'T@Ha(m3)':>9}")
for HI in ["Ta2O5", "Nb2O5", "TiO2", "ZnSe"]:
    nH = float(np.interp(HALPHA, *tf.load_nk(HI)[:2]))
    nL = float(np.interp(HALPHA, *tf.load_nk(LO)[:2]))
    contrast = (nH - nL) / (nH + nL)

    lam3 = autocenter(HI, 3)
    mats3, d3 = fp_cavity(HI, 3, lam0=lam3)
    _, _, cfw3 = fwhm(coll(mats3, d3))
    Tb3 = beam(mats3, d3); bpk3, bwl3, bfw3 = fwhm(Tb3)
    tha3 = float(np.interp(HALPHA, WL, Tb3))

    lam5 = autocenter(HI, 5)
    mats5, d5 = fp_cavity(HI, 5, lam0=lam5)
    _, _, floor = fwhm(beam(mats5, d5))
    shift5 = lam5 - HALPHA

    print(f"{HI:8} {nH:6.3f} {contrast:8.3f} {cfw3:13.1f} {floor:10.2f} "
          f"{shift5:9.1f} {bfw3:12.1f} {tha3:9.3f}")

# --- overlay plot: beam-averaged Halpha band, m=3, per material -------------
import os, matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 5))
for HI, col in [("Ta2O5", "C0"), ("Nb2O5", "C1"), ("TiO2", "C2"), ("ZnSe", "C3")]:
    nH = float(np.interp(HALPHA, *tf.load_nk(HI)[:2]))
    lam3 = autocenter(HI, 3)
    mats3, d3 = fp_cavity(HI, 3, lam0=lam3)
    Tb = beam(mats3, d3); _, _, fw = fwhm(Tb)
    ax.plot(WL, Tb, col, lw=1.8,
            label=f"{HI} (n={nH:.2f}): beam FWHM {fw:.1f} nm")
for x in (654.8, 656.3, 658.4):
    ax.axvline(x, color="k", ls=":", lw=0.7)
ax.set_xlim(635, 685); ax.set_ylim(0, 1.0)
ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("beam-averaged transmission")
ax.set_title("Higher-index high layer → sharper Hα band in the Rubin beam "
             "(3-cavity FP, m=3, auto-centered)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "figs", "material_comparison.png")
fig.tight_layout(); fig.savefig(out, dpi=130)
print("wrote", out)
