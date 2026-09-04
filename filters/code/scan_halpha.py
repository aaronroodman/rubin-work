"""
Tradeoff scan for an Halpha (656.3 nm) filter in the Rubin f/1.23 beam.

Because the cone smears the cavity resonance over ~9 nm, the beam-averaged
passband has a WIDTH FLOOR: making the collimated cavity thinner (higher mirror
finesse) below that floor does not narrow the operational band -- it only lowers
the transmission at Halpha, because a narrow resonance overlaps Halpha for only a
thin slice of the angular cone.

For each finesse (mirror pairs m of a 3-cavity FP), we auto-center the collimated
design so the beam-averaged peak lands on Halpha, then report:
  - collimated FWHM (design)          - beam FWHM (operational, floored)
  - beam transmission AT 656.3 nm     - beam peak
(bare backside; relative comparison, blocker added later for the final design.)
"""
import numpy as np
import torch
import thinfilm as tf
import filterstack as fs

LO, HI = "SiO2", "Nb2O5"
HALPHA = 656.3
K_REAL = {"Nb2O5": 3e-4}
WL = np.linspace(600.0, 720.0, 1201)     # 0.1 nm, fine enough for sharp cavities


def fp_cavity(m, ncav=3, lam0=666.0):
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


def fwhm(T, wl):
    i = int(np.argmax(T)); pk = T[i]; h = 0.5 * pk
    lo, hi = i, i
    while lo > 0 and T[lo] >= h:
        lo -= 1
    while hi < len(T) - 1 and T[hi] >= h:
        hi += 1
    return pk, wl[i], wl[hi] - wl[lo]


def collimated_T(mats, d):
    N = tf.build_N(mats, WL, incident="Air", exit=fs.SUBSTRATE, kdict=None)
    return tf.transmission(N, torch.tensor(d), WL, 0.0, "s")["T"].numpy()


def beam_T(mats, d):
    T, _, _ = fs.beam_average(mats, d, WL, n_theta=25, kdict=K_REAL,
                              ar_mats=None, ar_d_nm=None)
    return T


print(f"{'m':>2} {'lam_design':>10} {'coll_FWHM':>10} {'beam_FWHM':>10} "
      f"{'T@Halpha':>9} {'beam_peak':>9}")
for m in [2, 3, 4, 5, 6]:
    lam_d = 666.0
    # auto-center: land the beam peak on Halpha (shift ~ const, 2 iterations)
    for _ in range(3):
        mats, d = fp_cavity(m, lam0=lam_d)
        Tb = beam_T(mats, d)
        _, pk_wl, _ = fwhm(Tb, WL)
        lam_d += (HALPHA - pk_wl)
    mats, d = fp_cavity(m, lam0=lam_d)
    Tc = collimated_T(mats, d); Tb = beam_T(mats, d)
    _, _, cfw = fwhm(Tc, WL)
    bpk, bwl, bfw = fwhm(Tb, WL)
    tha = float(np.interp(HALPHA, WL, Tb))
    print(f"{m:>2} {lam_d:>10.1f} {cfw:>10.1f} {bfw:>10.1f} {tha:>9.3f} {bpk:>9.3f}")
