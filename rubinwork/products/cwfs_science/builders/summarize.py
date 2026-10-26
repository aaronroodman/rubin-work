"""Per-CCD and per-raft summaries of one visit's corner donuts.

Two selections:

- **quality**: our own, from the per-donut flags in ``donutBlitzCornerResults``;
- **mtaos**: the donuts the online pipeline averaged into its ``zernikes`` table, the
  values the Main Telescope AOS (MTAOS) uses. Read from that table's ``used`` flags and
  matched to the blitz donuts on detector and donut id, so both frames and the intrinsic
  part are available for them.

No Butler here, so the tests run on synthetic tables.
"""

import numpy as np

from rubinwork.common.utils import nmad

__all__ = ["READ_COLUMNS", "FRAMES", "quality_mask", "mtaos_selection", "summarize_visit"]

READ_COLUMNS = ("det_id", "det_name", "donut_id", "candidate", "group_fit_success",
                "group_fit_outcome", "snr", "group_fwhm", "defocal_offsets",
                "thx_ocs", "thy_ocs", "thx_ccs", "thy_ccs",
                "zk_deviation_ocs", "zk_intrinsic_ocs",
                "zk_deviation_ccs", "zk_intrinsic_ccs")
"""The ``donutBlitzCornerResults`` columns read; the per-donut images are left out."""

FRAMES = ("OCS", "CCS")


def _bare(tab, col):
    return np.asarray(getattr(tab[col], "value", tab[col]), float)


def _zk(tab, kind, frame, noll):
    """``(n_donut, n_noll)`` Zernikes in µm; the vectors are 0-indexed on Noll."""
    return _bare(tab, f"zk_{kind}_{frame.lower()}")[:, list(noll)].reshape(len(tab), len(noll))


def _median(x):
    return np.median(x, axis=0) if len(x) else np.full(x.shape[1:], np.nan)


def _mean(x):
    return np.mean(x, axis=0) if len(x) else np.full(x.shape[1:], np.nan)


def quality_mask(tab, noll):
    """Donuts the quality-cut summaries use.

    A donut is kept if it passed the pipeline's selection (``candidate``), its group fit
    succeeded with outcome ``ok``, and its SNR, blur and every deviation and intrinsic
    Zernike in `noll`, in both frames, are finite.

    Parameters
    ----------
    tab : `astropy.table.Table`
        One visit's corner table, with `READ_COLUMNS`.
    noll : `list` [`int`]
        Noll indices kept.

    Returns
    -------
    keep : `numpy.ndarray` [`bool`]
    """
    keep = (np.asarray(tab["candidate"], bool)
            & np.asarray(tab["group_fit_success"], bool)
            & (np.char.strip(np.asarray(tab["group_fit_outcome"]).astype(str)) == "ok")
            & np.isfinite(_bare(tab, "snr"))
            & np.isfinite(_bare(tab, "group_fwhm")))
    for frame in FRAMES:
        for kind in ("deviation", "intrinsic"):
            keep &= np.all(np.isfinite(_zk(tab, kind, frame, noll)), axis=1)
    return keep


def _side_rows(ztab):
    """``(det_name, donut_id, used)`` for each donut row of one ``zernikes`` table."""
    names = {s: (ztab.meta.get(s) or {}).get("det_name") for s in ("intra", "extra")}
    donuts = ztab[np.asarray(ztab["label"]).astype(str) != "average"]
    ids = {s: np.ma.filled(np.ma.asarray(donuts[f"{s}_donut_id"]).astype(str), "")
           for s in ("intra", "extra")}
    out = []
    for k, flag in enumerate(np.asarray(donuts["used"], bool)):
        side = "intra" if ids["intra"][k] not in ("", "--") else "extra"
        out.append((names[side], ids[side][k], bool(flag)))
    return out


def mtaos_selection(tab, zernikes, noll):
    """The MTAOS selection, on the blitz donuts.

    Parameters
    ----------
    tab : `astropy.table.Table`
        One visit's ``donutBlitzCornerResults``, with `READ_COLUMNS`.
    zernikes : `dict` [`int`, `astropy.table.Table`]
        The visit's ``zernikes`` tables, keyed by detector id: one per corner raft,
        holding both halves.
    noll : `list` [`int`]
        Noll indices.

    Returns
    -------
    used : `numpy.ndarray` [`bool`]
        Per blitz row, whether the online pipeline averaged it.
    rafts : `dict` [`str`, `dict`]
        Per raft (``R00`` ...): ``zernikes_detector_id``, ``n_mtaos_rows`` (donut rows
        in the table), ``n_mtaos_unmatched`` (rows with no blitz donut) and
        ``deviation_average``, the table's ``average`` row in µm, CCS.
    """
    key = {(d, i): k for k, (d, i) in enumerate(zip(
        np.asarray(tab["det_name"]).astype(str), np.asarray(tab["donut_id"]).astype(str)))}
    used = np.zeros(len(tab), bool)
    rafts = {}
    for det_id, ztab in sorted(zernikes.items()):
        rows = _side_rows(ztab)
        unmatched = 0
        for det, did, flag in rows:
            k = key.get((det, did))
            if k is None:
                unmatched += 1
            elif flag:
                used[k] = True
        avg = ztab[np.asarray(ztab["label"]).astype(str) == "average"]
        dev = np.array([float(avg[f"Z{j}_deviation"][0].to_value("um"))
                        if len(avg) and f"Z{j}_deviation" in avg.colnames else np.nan
                        for j in noll])
        raft = str((ztab.meta.get("extra") or ztab.meta.get("intra") or {})
                   .get("det_name", ""))[:3]
        rafts[raft] = dict(zernikes_detector_id=int(det_id), n_mtaos_rows=len(rows),
                           n_mtaos_unmatched=unmatched, deviation_average=dev)
    return used, rafts


def _zk_block(row, sub, noll, label, stat):
    """Total and intrinsic Zernikes of `sub` into `row`, both frames."""
    for frame in FRAMES:
        intr = _zk(sub, "intrinsic", frame, noll)
        tot = _zk(sub, "deviation", frame, noll) + intr
        t, i = stat(tot), stat(intr)
        for n, j in enumerate(noll):
            row[f"zk_{frame}_{label}_z{j}"] = float(t[n])
            row[f"zk_intrinsic_{frame}_{label}_z{j}"] = float(i[n])
        if label == "median":
            for n, j in enumerate(noll):
                row[f"zk_{frame}_nmad_z{j}"] = float(nmad(tot[:, n]))


def summarize_visit(tab, noll, zernikes=None, visit_info=None):
    """Summary rows of one visit: per corner CCD, and per corner raft.

    Parameters
    ----------
    tab : `astropy.table.Table`
        The visit's ``donutBlitzCornerResults``, with `READ_COLUMNS`.
    noll : `list` [`int`]
        Noll indices written.
    zernikes : `dict` [`int`, `astropy.table.Table`], optional
        The visit's ``zernikes`` tables by detector id; the mtaos columns are NaN
        without them.
    visit_info : `dict`, optional
        Columns copied into every row (``day_obs``, ``seq_num``, ``visit``, ...).

    Returns
    -------
    ccd_rows : `list` [`dict`]
        Per CCD: ``detector``, ``detector_id``, ``raft``, ``wfs_offset`` (-1
        intra-focal, 0 extra-focal), ``defocal_offset_mm``, ``n_donuts``,
        ``n_quality``, ``n_mtaos``. For each frame ``F`` and Noll ``j``, in µm of
        wavefront: ``zk_F_median_z{j}`` and ``zk_F_nmad_z{j}``, median and nMAD of the
        total Zernike (deviation plus intrinsic) over the quality cut;
        ``zk_F_mtaos_z{j}``, the mean over this CCD's MTAOS donuts; and
        ``zk_intrinsic_F_median_z{j}``, ``zk_intrinsic_F_mtaos_z{j}``. Blur (arcsec):
        ``blur_median``, ``blur_nmad``, ``blur_mtaos_median``. ``snr_median``
        (dimensionless); ``thx_F_median``, ``thy_F_median`` (deg).
    raft_rows : `list` [`dict`]
        Per raft, the MTAOS values: ``raft``, ``zernikes_detector_id``,
        ``n_mtaos_rows``, ``n_mtaos_unmatched``, ``n_mtaos``, ``n_mtaos_intra``,
        ``n_mtaos_extra``; ``zk_CCS_deviation_average_z{j}``, the table's own average
        (µm, CCS); ``zk_F_mtaos_z{j}`` and ``zk_intrinsic_F_mtaos_z{j}``, the mean over
        the raft's MTAOS donuts, both halves (µm); ``mtaos_check_um``, the largest
        difference between the table's average and the blitz CCS deviation mean over
        the same donuts (µm, zero when the match is exact); ``blur_mtaos_median``
        (arcsec).
    """
    visit_info = dict(visit_info or {})
    noll = [int(j) for j in noll]
    det = np.asarray(tab["det_name"]).astype(str)
    det_id = np.asarray(tab["det_id"]).astype(int)
    off = _bare(tab, "defocal_offsets").reshape(len(tab), -1)
    signed = off[np.arange(len(off)), np.argmax(np.abs(off), axis=1)]
    good = quality_mask(tab, noll)
    if zernikes:
        used, rafts = mtaos_selection(tab, zernikes, noll)
    else:
        used, rafts = np.zeros(len(tab), bool), {}
    blur_all = _bare(tab, "group_fwhm")

    ccd_rows = []
    for name in sorted(set(det)):
        on = det == name
        sub, mt = tab[on & good], tab[on & used]
        row = dict(visit_info)
        row.update(detector=name, detector_id=int(det_id[on][0]), raft=name[:3],
                   wfs_offset=-1 if signed[on][0] < 0 else 0,
                   defocal_offset_mm=float(signed[on][0]) * 1e3,
                   n_donuts=int(on.sum()), n_quality=len(sub), n_mtaos=len(mt))
        blur = _bare(sub, "group_fwhm")
        row.update(blur_median=float(_median(blur)), blur_nmad=float(nmad(blur)),
                   blur_mtaos_median=float(_median(blur_all[on & used])),
                   snr_median=float(_median(_bare(sub, "snr"))))
        _zk_block(row, sub, noll, "median", _median)
        _zk_block(row, mt, noll, "mtaos", _mean)
        for frame in FRAMES:
            for ax in ("thx", "thy"):
                row[f"{ax}_{frame}_median"] = float(_median(
                    np.degrees(_bare(sub, f"{ax}_{frame.lower()}"))))
        ccd_rows.append(row)

    raft_rows = []
    raft_of = np.char.ljust(np.asarray([d[:3] for d in det]), 3)
    for raft, info in sorted(rafts.items()):
        sel = (raft_of == raft) & used
        mt = tab[sel]
        row = dict(visit_info)
        row.update(raft=raft, zernikes_detector_id=info["zernikes_detector_id"],
                   n_mtaos_rows=info["n_mtaos_rows"],
                   n_mtaos_unmatched=info["n_mtaos_unmatched"], n_mtaos=int(sel.sum()),
                   n_mtaos_intra=int((sel & (signed < 0)).sum()),
                   n_mtaos_extra=int((sel & (signed >= 0)).sum()),
                   blur_mtaos_median=float(_median(blur_all[sel])))
        for n, j in enumerate(noll):
            row[f"zk_CCS_deviation_average_z{j}"] = float(info["deviation_average"][n])
        _zk_block(row, mt, noll, "mtaos", _mean)
        check = _mean(_zk(mt, "deviation", "CCS", noll)) - info["deviation_average"]
        row["mtaos_check_um"] = (float(np.nanmax(np.abs(check)))
                                 if np.isfinite(check).any() else np.nan)
        raft_rows.append(row)
    return ccd_rows, raft_rows
