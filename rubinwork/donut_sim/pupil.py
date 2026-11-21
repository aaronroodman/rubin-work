"""Stop-plane comparison of a batoid model's vignetting with a Danish obscuration file.

Rays on a stop-plane grid are traced through the defocused batoid model; a ray is
batoid-clear if it reaches the detector. The same rays are tested against Danish's
mask: the outer pupil, each element's circle at the field angle (centre and radius
cubics in field radius, from the obscuration file) and the spider vanes. Each ray where
the two disagree is assigned to the Danish edge nearest to it, which gives each
element's mismatched area and, divided by the length of the edge that bounds the Danish
pupil, its mean edge offset.

Danish's circles are evaluated as ts_wep and danish do: centre displaced along the
field direction by ``polyval(center, theta_deg)``, radius ``polyval(radius, theta_deg)``,
an edge active for thetaMin <= theta <= thetaMax. The spider vanes use danish's own
projection.

Import as ``from rubinwork.donut_sim import pupil``.
"""
import numpy as np
import batoid

__all__ = ["stop_uv", "danish_edges", "compare"]


def stop_uv(optic, rv):
    """Ray positions on the stop plane (m)."""
    z0 = optic.stopSurface.coordSys.origin[2]
    t = (z0 - rv.z) / rv.vz
    return rv.x + t * rv.vx, rv.y + t * rv.vy


def danish_edges(mask_params, outer_radius_m, u, v, thx, thy, spider_angle=0.0):
    """Danish's mask at stop-plane points, one entry per edge.

    Parameters
    ----------
    mask_params : `dict`
        Danish obscuration parameters (``Instrument.maskParams``).
    outer_radius_m : `float`
        Pupil outer radius (m), ``Instrument.radius``.
    u, v : `numpy.ndarray`
        Stop-plane positions (m).
    thx, thy : `float`
        Field angle (rad), CCS.
    spider_angle : `float` or `None`
        Danish ``spider_angle`` (deg); `None` leaves the vanes out.

    Returns
    -------
    edges : `dict`
        ``{name: (blocks, distance)}``: `blocks` is where that edge masks the point,
        `distance` the absolute distance from the point to the edge (m). Names are
        ``Pupil``, ``<element>.<inner|outer>`` and ``Spider.<i>``.
    """
    from danish.factory import _project_spider_vane
    thr = np.hypot(thx, thy)
    thr_deg = np.rad2deg(thr)
    r0 = np.hypot(u, v)
    out = {"Pupil": (r0 > outer_radius_m, np.abs(r0 - outer_radius_m))}
    for item, val in mask_params.items():
        if item == "Spider_3D":
            if spider_angle is None:
                continue
            for i, vane in enumerate(val):
                p1x, p1y, s1, c1, p2x, p2y, s2, c2 = _project_spider_vane(
                    vane["r0"], vane["v0"], vane["width"], vane["length"],
                    vane["angle"] + spider_angle, thx, thy)
                d1 = (u - p1x) * s1 - (v - p1y) * c1
                d2 = (u - p2x) * s2 - (v - p2y) * c2
                pcx, pcy = 0.5 * (p1x + p2x), 0.5 * (p1y + p2y)
                along = np.abs((u - pcx) * c1 + (v - pcy) * s1)
                inside_len = along < 0.5 * vane["length"]
                blocks = (d1 * d2 < 0) & inside_len
                dist = np.where(inside_len, np.minimum(np.abs(d1), np.abs(d2)), np.inf)
                out[f"Spider.{i}"] = (blocks, dist)
            continue
        for edge_name, edge in val.items():
            if thr_deg < edge["thetaMin"] or thr_deg > edge["thetaMax"]:
                continue
            r = np.polyval(edge["radius"], thr_deg)
            c = np.polyval(edge["center"], thr_deg)
            cx = c * thx / thr if thr > 0 else 0.0
            cy = c * thy / thr if thr > 0 else 0.0
            d = np.hypot(u - cx, v - cy)
            inside = d < r
            out[f"{item}.{edge_name}"] = (~inside if edge["clear"] else inside, np.abs(d - r))
    return out


def compare(optic, mask_params, outer_radius_m, thx_deg, thy_deg, wavelength_m,
            nx=512, spider_angle=0.0):
    """Batoid vignetting against Danish's mask on a stop-plane grid, per Danish edge.

    Parameters
    ----------
    optic : `batoid.Optic`
        Defocused telescope.
    nx : `int`, optional
        Grid points across the pupil diameter.

    Returns
    -------
    total : `dict`
        ``area_batoid_m2``, ``area_danish_m2`` (clear areas, m²);
        ``area_batoid_only_m2`` (batoid clear, Danish masked) and
        ``area_danish_only_m2`` (Danish clear, batoid vignetted), m².
    per_edge : `list` [`dict`]
        Per Danish edge: ``edge``; ``batoid_only_m2``, ``danish_only_m2`` (mismatched
        area assigned to it, m²); ``length_m`` (length of the edge that bounds the
        Danish clear pupil, m); ``offset_mm`` = (batoid_only - danish_only) / length,
        positive where Danish masks more than batoid, mm.
    """
    th = np.deg2rad([thx_deg, thy_deg])
    rv = batoid.RayVector.asGrid(optic=optic, wavelength=wavelength_m, theta_x=th[0],
                                 theta_y=th[1], nx=nx)
    u, v = stop_uv(optic, rv)
    ok_batoid = ~optic.trace(rv.copy()).vignetted
    edges = danish_edges(mask_params, outer_radius_m, u, v, *th, spider_angle=spider_angle)
    names = list(edges)
    blocks = np.array([edges[n][0] for n in names])
    dist = np.array([edges[n][1] for n in names])
    ok_danish = ~blocks.any(0)
    spacing = optic.pupilSize / nx
    da = spacing ** 2
    nearest = np.argmin(dist, axis=0)

    # Boundary of the Danish clear pupil on the grid: clear points with a masked
    # 4-neighbour. Each boundary point stands for one spacing of edge length.
    g = ok_danish.reshape(nx, nx)
    bnd = np.zeros_like(g)
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        bnd |= g & ~np.roll(g, sh, axis=ax)
    bnd = bnd.ravel()

    b_only = ok_batoid & ~ok_danish
    d_only = ok_danish & ~ok_batoid
    per = []
    for i, n in enumerate(names):
        mine = nearest == i
        nb, nd, nl = int((b_only & mine).sum()), int((d_only & mine).sum()), int((bnd & mine).sum())
        if nb == nd == nl == 0:
            continue
        length = nl * spacing
        per.append(dict(edge=n, batoid_only_m2=nb * da, danish_only_m2=nd * da,
                        length_m=length,
                        offset_mm=1e3 * (nb - nd) * da / length if length > 0 else np.nan))
    total = dict(area_batoid_m2=float(ok_batoid.sum() * da),
                 area_danish_m2=float(ok_danish.sum() * da),
                 area_batoid_only_m2=float(b_only.sum() * da),
                 area_danish_only_m2=float(d_only.sum() * da))
    return total, per
