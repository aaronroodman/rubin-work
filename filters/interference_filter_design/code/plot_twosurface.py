"""Re-plot the saved two-surface design against the physical Si range."""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# scripts live in filters/code/; products go to filters/output/
OUT = os.path.join(os.path.dirname(HERE), "output")
BAND = (402.0, 552.0)
SI_CUTOFF = 1080.0

d = np.load(os.path.join(OUT, "twosurface_design.npz"), allow_pickle=True)
WL, Tn, Tb, Tg = d["WL"], d["T_norm"], d["T_beam"], d["Tg"]
nf, nb = len(d["front_thickness_nm"]), len(d["back_thickness_nm"])
um = (d["front_thickness_nm"].sum() + d["back_thickness_nm"].sum()) / 1e3

inb = (WL >= BAND[0]) & (WL <= BAND[1])
oob = ((WL >= 322) & (WL <= 392)) | ((WL >= 566) & (WL <= SI_CUTOFF))
odn = -np.log10(Tn[oob].max()); odb = -np.log10(Tb[oob].max())

fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 8), sharex=True)
a1.plot(WL, Tg, "k--", lw=1.4, label="Rubin g (filter_g.dat)")
a1.plot(WL, Tn, "C0-", lw=1.5,
        label=f"design normal  (in-band⟨T⟩={Tn[inb].mean():.3f}, peak {Tn.max():.3f})")
a1.plot(WL, Tb, "C3-", lw=1.5,
        label=f"design beam 14–23°  (⟨T⟩={Tb[inb].mean():.3f})")
a1.axvspan(*BAND, color="C2", alpha=0.06)
a1.set_ylabel("system transmission"); a1.set_ylim(-0.02, 1.02)
a1.legend(fontsize=8.5); a1.grid(alpha=0.3)
a1.set_title(f"Two-surface g filter: LWP front + SWP back, {nf}+{nb} layers, "
             f"{um:.1f} µm, Nb₂O₅/SiO₂")

for T, c, l in [(Tg, "k--", "Rubin g"), (Tn, "C0-", "design normal"),
                (Tb, "C3-", "design beam")]:
    a2.semilogy(WL, np.clip(T, 1e-9, 1), c, lw=1.3, label=l)
a2.axhline(1e-4, color="gray", ls=":", lw=1); a2.text(325, 1.35e-4, "OD4", fontsize=8)
a2.axvline(SI_CUTOFF, color="purple", ls="-.", lw=1)
a2.text(SI_CUTOFF - 5, 3e-7, "Si cutoff", rotation=90, fontsize=8, va="bottom", ha="right")
a2.axvspan(*BAND, color="C2", alpha=0.06)
a2.set_xlabel("wavelength (nm)"); a2.set_ylabel("transmission (log)")
a2.set_ylim(1e-7, 2); a2.legend(fontsize=8.5, loc="upper right")
a2.grid(alpha=0.3, which="both")
a2.set_title(f"blocking: worst OD over 320–1080 nm  =  {odn:.2f} (normal), "
             f"{odb:.2f} (beam)", fontsize=10)

os.makedirs(os.path.join(OUT, "figs"), exist_ok=True)
out = os.path.join(OUT, "figs", "twosurface_design.png")
fig.tight_layout(); fig.savefig(out, dpi=130)
print("wrote", out, f"| normal OD={odn:.2f} beam OD={odb:.2f} "
      f"| in-band⟨T⟩ {Tn[inb].mean():.3f}/{Tb[inb].mean():.3f} "
      f"| peak {Tn.max():.3f}")
