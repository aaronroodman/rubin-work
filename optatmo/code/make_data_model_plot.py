"""Data-vs-model focal-plane comparison of the v-Mode fit.

Side-by-side (data | model) maps at the matched binned cell positions (OCS
frame, same moment convention on both sides): size, ellipticity whisker, coma
whisker, trefoil markers.  Shared scales per row so data and model are directly
comparable.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# moment vector order from Forward / moments_hsm
LAB = ['e0', 'e1', 'e2', 'M21', 'M12', 'M30', 'M03', 'M22', 'M31', 'M13', 'M40', 'M04']
FP_RADIUS = 1.75


def derived(mom):
    m = {k: mom[:, i] for i, k in enumerate(LAB)}
    e0 = m['e0']
    return dict(size=np.sqrt(np.clip(e0, 0, None) / 2) * 2.3548,   # FWHM-like (arcsec)
                e1=m['e1'] / e0, e2=m['e2'] / e0,
                coma1=m['M21'], coma2=m['M12'],
                tre1=m['M30'], tre2=m['M03'])


def panel(ax, x, y, title):
    ax.add_patch(Circle((0, 0), FP_RADIUS, fill=False, ls='--', color='r', lw=0.7))
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect('equal')
    ax.set_title(title, fontsize=9); ax.tick_params(labelsize=6)
    ax.set_xlabel('OCS x [deg]', fontsize=7); ax.set_ylabel('OCS y [deg]', fontsize=7)


def make(seq, nsub=250):
    d = np.load(f'data/vmodefit_{seq}.npz')
    x, y = d['thx'], d['thy']
    dd, mm = derived(d['data_mom']), derived(d['model_mom'])
    sub = (np.linspace(0, len(x) - 1, nsub).astype(int) if len(x) > nsub
           else np.arange(len(x)))
    x, y = x[sub], y[sub]

    fig, axes = plt.subplots(4, 2, figsize=(11, 20))
    # scales shared data/model
    smin, smax = np.percentile(np.r_[dd['size'], mm['size']], [3, 97])
    e_sc = 0.5 / max(np.percentile(np.hypot(dd['e1'], dd['e2']), 95), 1e-3)
    c_sc = 0.5 / max(np.percentile(np.hypot(dd['coma1'], dd['coma2']), 95), 1e-4)
    t_area = 0.5 / max(np.percentile(np.hypot(dd['tre1'], dd['tre2']), 95), 1e-4) * 300

    for col, (src, tag) in enumerate([(dd, 'DATA'), (mm, 'MODEL')]):
        panel(axes[0, col], x, y, f'{tag} size (FWHM-like) [arcsec]')
        sc = axes[0, col].scatter(x, y, c=src['size'][sub], s=14, cmap='viridis',
                                  vmin=smin, vmax=smax)
        fig.colorbar(sc, ax=axes[0, col], shrink=0.8)

        panel(axes[1, col], x, y, f'{tag} ellipticity (e1,e2)')
        a = 0.5 * np.arctan2(src['e2'][sub], src['e1'][sub])
        r = np.hypot(src['e1'][sub], src['e2'][sub])
        axes[1, col].quiver(x, y, r * np.cos(a), r * np.sin(a), angles='xy',
                            scale_units='xy', scale=1 / e_sc * 4, pivot='mid',
                            headlength=0, headaxislength=0, width=0.005)

        panel(axes[2, col], x, y, f'{tag} coma (3rd)')
        a = np.arctan2(src['coma2'][sub], src['coma1'][sub])
        r = np.hypot(src['coma1'][sub], src['coma2'][sub])
        axes[2, col].quiver(x, y, r * np.cos(a), r * np.sin(a), angles='xy',
                            scale_units='xy', scale=1 / c_sc * 4, pivot='mid',
                            headlength=0, headaxislength=0, width=0.005)

        panel(axes[3, col], x, y, f'{tag} trefoil (3rd)')
        tamp = np.hypot(src['tre1'][sub], src['tre2'][sub])
        tang = np.degrees(np.arctan2(src['tre2'][sub], src['tre1'][sub])) / 3
        for xi, yi, ai, si in zip(x, y, tang, tamp * t_area):
            axes[3, col].scatter(xi, yi, marker=(3, 0, 30 + ai), s=max(si, 3),
                                 color='k', lw=0.1)

    fig.suptitle(f'20260513 seq={seq}: data vs v-Mode model (OCS, {len(sub)} cells)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f'output/datamodel_{seq}.png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    # print rms residuals
    for k in ['size', 'e1', 'e2', 'coma1', 'coma2', 'tre1', 'tre2']:
        r = np.sqrt(np.nanmean((dd[k] - mm[k]) ** 2))
        s = np.nanstd(dd[k])
        print(f'  {k:6s} resid_rms={r:.4g}  data_std={s:.4g}  ratio={r/(s+1e-12):.2f}')
    print(f'saved output/datamodel_{seq}.png')


if __name__ == '__main__':
    for seq in [31, 34]:
        print(f'=== seq {seq} ===')
        make(seq)
