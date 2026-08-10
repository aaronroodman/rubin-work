"""Ellipticity-whisker frame validation vs RubinTV.

For a rotator~0 visit (seq 28, rot=-0.87 deg) and a large-rotator visit (seq 25,
rot=+59.16 deg) plot the PSF ellipticity whiskers in two frames:

  * DVCS  -- raw as measured: positions from cameraGeom FIELD_ANGLE (field_x,
             field_y) and e1/e2 straight off the pixel stamp.  Camera-fixed;
             related to the RubinTV Az/El display by a rotation (see Az/El
             below), NOT identical to it.
  * OCS    -- after frames.dvcs_to_ccs (x<->y swap + moment reflection) then the
             rotator rotation.  OCS is mirror-fixed, so the optical whisker
             PATTERN should look the SAME for both visits despite the 60 deg
             rotator difference -- that invariance is the real check that the
             DVCS->CCS->OCS chain is consistent.
  * Az/El  -- the RubinTV frame: Elevation on y, Azimuth on x.  Pinned from the
             RubinTV rosette (Aaron, 20260513 seq25): at rot~0
                 Az = +x_CCS = +field_y ,   El = -y_CCS = -field_x
             and as the rotator increases (rtp~+58 deg) the x_CCS axis rotates
             CLOCKWISE in the fixed Az/El frame.  Hence, with rho=rotator angle,
                 Az =  field_y*cos(rho) - field_x*sin(rho)
                 El = -field_y*sin(rho) - field_x*cos(rho)
             a PURE rotation of DVCS by theta = -(90deg + rho) (det = +1, no
             reflection -- exactly "rotate the printed DVCS page"), so the
             spin-2 whiskers pick up e -> e*exp(i*2*theta).  NB the sign here is
             fixed by the rosette, NOT by matching one exposure's whiskers to
             another (optics+atmosphere differ too much between exposures).
             Az/El differs from the fit's OCS frame by the CCS x<->y reflection.

Whisker: segment at each binned position, length ∝ |e|=sqrt(e1^2+e2^2),
orientation = 0.5*atan2(e2, e1).
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import frames

SEQS = {28: -0.865880, 25: 59.157623}   # seq -> rotator angle (deg), 20260513
DAY = 20260513


def visit_of(seq):
    return f'{DAY}{seq:05d}'


def bin_whiskers(thx, thy, e1, e2, cell=0.15):
    """Bin onto a regular grid; return per-cell mean position and (e1,e2)."""
    ix = np.round(thx / cell).astype(int)
    iy = np.round(thy / cell).astype(int)
    key = ix * 100000 + iy
    out = []
    for k in np.unique(key):
        m = key == k
        if m.sum() < 3:
            continue
        out.append((thx[m].mean(), thy[m].mean(), e1[m].mean(), e2[m].mean()))
    return np.array(out).T if out else np.empty((4, 0))


def draw_rosette(ax, rho, cx=1.45, cy=1.45, L=0.38):
    """Draw the Az/El/x_CCS/y_CCS rosette (like RubinTV) at rotator angle rho.

    In the Az(x)/El(y) plane the CCS axes sit at (from the pinned transform):
        x_CCS = ( cos rho, -sin rho)      y_CCS = (-sin rho, -cos rho)
    """
    def arrow(dx, dy, color, label):
        ax.annotate('', xy=(cx + L * dx, cy + L * dy), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=1.4))
        ax.text(cx + 1.25 * L * dx, cy + 1.25 * L * dy, label, color=color,
                ha='center', va='center', fontsize=7, weight='bold')
    arrow(1, 0, 'tab:red', 'Az')
    arrow(0, 1, 'tab:green', 'El')
    arrow(np.cos(rho), -np.sin(rho), 'k', r'$x_{CCS}$')
    arrow(-np.sin(rho), -np.cos(rho), 'dimgray', r'$y_{CCS}$')


def draw(ax, thx, thy, e1, e2, scale, title):
    e = np.hypot(e1, e2)
    ang = 0.5 * np.arctan2(e2, e1)
    dx = scale * e * np.cos(ang)
    dy = scale * e * np.sin(ang)
    ax.plot([thx - dx / 2, thx + dx / 2], [thy - dy / 2, thy + dy / 2],
            '-', color='k', lw=0.8)
    ax.set_aspect('equal')
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sign', type=int, default=1,
                    help='CCS->OCS rotation sign (fit convention)')
    ap.add_argument('--cell', type=float, default=0.15)
    ap.add_argument('--out', default='output/whiskers_frames.png')
    args = ap.parse_args()

    seqs = list(SEQS)
    fig, axes = plt.subplots(len(seqs), 3, figsize=(16, 5.4 * len(seqs)))
    axes = np.atleast_2d(axes)

    # common whisker scale from all data so panels are comparable
    scale = None
    for row, seq in enumerate(seqs):
        rot = SEQS[seq]
        df = pd.read_parquet(f'data/psfmoments_{visit_of(seq)}.parquet')
        # robust clip on e0/e1/e2
        keep = np.ones(len(df), bool)
        for k in ('e0', 'e1', 'e2'):
            v = df[k].to_numpy()
            med = np.nanmedian(v)
            mad = 1.4826 * np.nanmedian(np.abs(v - med)) + 1e-30
            keep &= np.abs(v - med) < 5 * mad
        df = df[keep]
        thx = df['thx_ccs_deg'].to_numpy()      # DVCS field_x
        thy = df['thy_ccs_deg'].to_numpy()      # DVCS field_y
        e1 = df['e1'].to_numpy()
        e2 = df['e2'].to_numpy()

        # DVCS (as measured; == RubinTV camera frame)
        bx, by, be1, be2 = bin_whiskers(thx, thy, e1, e2, args.cell)

        # OCS: swap x<->y + reflect moments, then rotate by rotator
        cx, cy = frames.dvcs_to_ccs_field(thx, thy)
        mom = np.zeros((len(df), 12))
        mom[:, 1] = e1
        mom[:, 2] = e2
        mom = frames.dvcs_to_ccs_moments(mom)
        a = np.deg2rad(rot)
        ce1, ce2 = mom[:, 1], mom[:, 2]
        ox, oy = frames.rotate_field(cx, cy, a, args.sign)
        # spin-2 rotation of (e1,e2)
        z = (ce1 + 1j * ce2) * np.exp(1j * 2 * args.sign * a)
        oe1, oe2 = z.real, z.imag
        ox2, oy2, oe1b, oe2b = bin_whiskers(ox, oy, oe1, oe2, args.cell)

        if scale is None:
            emax = np.nanpercentile(np.hypot(be1, be2), 95)
            scale = 0.25 / max(emax, 1e-3)   # arcmin-ish visual length per unit e

        draw(axes[row, 0], bx, by, be1, be2, scale,
             f'seq {seq}  rot={rot:+.1f}  DVCS (camera)')
        axes[row, 0].set_xlabel('field_x [deg]')
        axes[row, 0].set_ylabel('field_y [deg]')
        draw(axes[row, 1], ox2, oy2, oe1b, oe2b, scale,
             f'seq {seq}  rot={rot:+.1f}  OCS (mirror frame)')
        axes[row, 1].set_xlabel('thx_OCS [deg]')
        axes[row, 1].set_ylabel('thy_OCS [deg]')

        # Az/El (RubinTV), pinned from the rosette (see module docstring):
        #   Az =  field_y cos(rho) - field_x sin(rho)
        #   El = -field_y sin(rho) - field_x cos(rho)
        # a pure rotation of DVCS by theta = -(pi/2 + rho); whiskers: e*e^{i2theta}
        rho = a                                     # = deg2rad(rot)
        az = thy * np.cos(rho) - thx * np.sin(rho)
        el = -thy * np.sin(rho) - thx * np.cos(rho)
        theta = -(np.pi / 2 + rho)
        ze = (e1 + 1j * e2) * np.exp(1j * 2 * theta)
        exb, eyb, ee1, ee2 = bin_whiskers(az, el, ze.real, ze.imag, args.cell)
        draw(axes[row, 2], exb, eyb, ee1, ee2, scale,
             f'seq {seq}  rot={rot:+.1f}  Az/El (RubinTV)')
        axes[row, 2].set_xlabel('Azimuth [deg]')
        axes[row, 2].set_ylabel('Elevation [deg]')
        draw_rosette(axes[row, 2], rho)

    fig.suptitle('PSF ellipticity whiskers: DVCS (camera) | OCS (mirror, fit '
                 'frame) | Az/El (RubinTV: El-y, Az-x)   '
                 f'scale: |e| x {scale:.1f} deg/unit', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, dpi=130)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
