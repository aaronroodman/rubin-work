"""Library functions for building intrinsic Zernike wavefront tables.

Extracts aggregate Zernike data from the LSST Butler, computes intrinsic
wavefront models, retrieves rotator angles and thermal data, and saves
the results as HDF5 files.

Usage from notebook:
    from intrinsics_lib import run_mktable, PARAM_SETS
    params = PARAM_SETS['fam_danish_triplets']
    aosTable, visit_info = await run_mktable(**params, coord_sys='OCS')

Usage from CLI:
    python run_mktable.py --param-set fam_danish_triplets
"""

import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from astropy.table import QTable, vstack
from astropy.time import Time, TimeDelta
from tqdm import tqdm

# LSST imports (available on RSP)
from lsst.daf.butler import Butler, DatasetNotFoundError
import lsst.afw.cameraGeom as cameraGeom
from lsst.obs.lsst import LsstCam
from lsst.ts.wep.task.estimateZernikesDanishTask import EstimateZernikesDanishTask
from lsst.ts.wep.task import CalcZernikesTask, CalcZernikesTaskConfig
from lsst.ts.wep.utils import getTaskInstrument, BandLabel
from lsst.summit.utils import ConsDbClient
from lsst.summit.utils.efdUtils import makeEfdClient

# Local utilities
from wcsutils import calc_rotator_from_visitinfo, pixel_to_focal

# Optional: M1M3 thermal analysis (may not be available in all environments)
try:
    from lsst.ts.m1m3.utils import ThermocoupleAnalysis
    HAS_M1M3_UTILS = True
except ImportError:
    HAS_M1M3_UTILS = False


# ============================================================
# Parameter Sets (loaded from param_sets.yaml)
# ============================================================

def load_param_sets(yaml_path=None):
    """Load parameter sets from param_sets.yaml.

    Parameters
    ----------
    yaml_path : str or Path, optional
        Path to param_sets.yaml. Default: aos/param_sets.yaml
        (auto-detected relative to this file or cwd).

    Returns
    -------
    param_sets : dict
        Mapping of name -> dict with butler_repo, fam_collections, etc.
    """
    import yaml
    if yaml_path is None:
        # Try relative to this file first, then cwd
        candidates = [
            Path(__file__).resolve().parent.parent / 'param_sets.yaml',
            Path('param_sets.yaml'),
        ]
        for c in candidates:
            if c.exists():
                yaml_path = c
                break
        else:
            raise FileNotFoundError(
                "param_sets.yaml not found. Expected in aos/ directory.")
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    # Strip 'description' key — not a pipeline parameter
    for name, cfg in data.items():
        cfg.pop('description', None)
    return data


# Load at import time for backward compatibility
try:
    PARAM_SETS = load_param_sets()
except FileNotFoundError:
    PARAM_SETS = {}


# ============================================================
# Defaults
# ============================================================

DEFAULT_ROTATOR_THRESHOLD = 90.0
DEFAULT_FP_RADIUS = 1.8
DEFAULT_FP_NSTEPS = 73
DEFAULT_MIN_VISITS_PER_DAY = 5
DEFAULT_TEMP_TIME_WINDOW_SEC = 0.2
DEFAULT_CONSDB_URL = "http://consdb-pq.consdb:8080/consdb"
DEFAULT_WEP_VER = 'wep_v16_8_0'
DEFAULT_DVIZ_VER = 'dviz_v3_5_0'


# ============================================================
# Zernike Utilities
# ============================================================

def infer_zernike_indices(nZk):
    """Infer the list of Noll Zernike indices from the number of terms."""
    if nZk == 25:
        return list(range(4, 29))
    elif nZk == 21:
        return list(range(4, 20)) + list(range(22, 27))
    elif nZk == 19:
        return list(range(4, 23))
    else:
        iZs = list(range(4, 4 + nZk))
        print(f"Warning: Unexpected number of Zernike terms ({nZk}), "
              f"assuming Z4-Z{3 + nZk}")
        return iZs


def derive_version_strings(collections):
    """Extract wep and dviz version strings from collection paths."""
    coll = collections[0]
    wep_match = re.search(r'(wep_v[\d_]+)', coll)
    dviz_match = re.search(r'(donut_viz_v[\d_]+)', coll)
    wep_ver = wep_match.group(1) if wep_match else DEFAULT_WEP_VER
    dviz_ver = (dviz_match.group(1).replace('donut_viz_', 'dviz_')
                if dviz_match else DEFAULT_DVIZ_VER)
    return wep_ver, dviz_ver


def parse_collection_info(collections, include_versions=False):
    """Parse collection name(s) to extract a filename phrase and date range.

    For a simple collection like 'aos_fam_danish_triplets':
        collection_phrase = 'aos_fam_danish_triplets', dates = (None, None)

    For 'u/brycek/aos_fam_danish_step2/wep_v16_5_0/donut_viz_v3_2_3/20251020_20251231':
        - Drop leading 'u/USERNAME/'
        - collection_phrase = 'aos_fam_danish_step2'
        - Optionally append version strings if include_versions=True
        - Parse trailing YYYYMMDD_YYYYMMDD as (day_obs_min, day_obs_max)

    Returns (collection_phrase, day_obs_min_or_None, day_obs_max_or_None).
    """
    coll = collections[0]
    parts = coll.split('/')

    # Check for u/USERNAME/... pattern
    if len(parts) >= 3 and parts[0] == 'u':
        # Drop u/USERNAME
        remaining = parts[2:]
        # First element is the collection phrase
        collection_phrase = remaining[0] if remaining else coll
        # Optionally include version strings
        if include_versions and len(remaining) > 1:
            version_parts = []
            for p in remaining[1:]:
                if re.match(r'wep_v[\d_]+', p) or re.match(r'donut_viz_v[\d_]+', p):
                    version_parts.append(p.replace('donut_viz_', 'dviz_'))
            if version_parts:
                collection_phrase += '_' + '_'.join(version_parts)
        # Check if last part is YYYYMMDD_YYYYMMDD or YYYYMMDD
        last = remaining[-1] if remaining else ''
        date_match = re.match(r'^(\d{8})_(\d{8})$', last)
        if date_match:
            return collection_phrase, int(date_match.group(1)), int(date_match.group(2))
        single_date = re.match(r'^(\d{8})$', last)
        if single_date:
            d = int(single_date.group(1))
            return collection_phrase, d, d
        return collection_phrase, None, None
    else:
        # Simple collection name
        return coll, None, None


# ============================================================
# Butler Data Extraction
# ============================================================

def get_aggregate_zernikes(butler, day_obs, seq_num, coord_sys, camera,
                           calc_focal_plane=False, calc_mean_zernike=False):
    """Get aggregate Zernike table for a single visit.

    Parameters
    ----------
    calc_focal_plane : bool
        If True, compute intra/extra focal plane coordinates (fpx, fpy).
    calc_mean_zernike : bool
        If True, compute per-visit mean Zernike and add as column.

    Returns (table, visit_meta_dict) or (None, None).
    """
    try:
        aosTable = butler.get('aggregateAOSVisitTableRaw',
                              day_obs=day_obs, seq_num=seq_num)
    except DatasetNotFoundError:
        print(f"DatasetNotFoundError: No data for day_obs={day_obs}, seq_num={seq_num}")
        return None, None
    except Exception as e:
        error_type = type(e).__name__
        if error_type == 'DimensionValueError':
            print(f"DimensionValueError for day_obs={day_obs}, seq_num={seq_num}")
        else:
            print(f"{error_type} for day_obs={day_obs}, seq_num={seq_num}: {e}")
        return None, None

    meta = aosTable.meta
    visit_meta = {
        'day_obs': day_obs,
        'seq_num': seq_num,
        'visit': meta.get('visit', None),
        'skyAngle': meta.get('rotAngle', np.nan),
        'ra': meta.get('ra', np.nan),
        'dec': meta.get('dec', np.nan),
        'az': meta.get('az', np.nan),
        'alt': meta.get('alt', np.nan),
        'band': meta.get('band', ''),
        'mjd': meta.get('mjd', np.nan),
        'nollIndices': meta.get('nollIndices', None),
    }

    # Extract blur (FWHM) from estimatorInfo metadata
    estimator_info = meta.get('estimatorInfo', {})
    fwhm = estimator_info.get('fwhm', None) if isinstance(estimator_info, dict) else None

    aosTable.meta = {}

    select = (aosTable['used'] == True)  # noqa: E712
    aosTable_sel = aosTable[select]

    if len(aosTable_sel) == 0:
        print(f"Warning: No 'used' donuts for day_obs={day_obs}, seq_num={seq_num}")
        return None, None

    aosTable_sel['seq_num'] = seq_num
    aosTable_sel['day_obs'] = day_obs

    # Add blur column
    if fwhm is not None:
        fwhm_arr = np.array(fwhm)
        if len(fwhm_arr) == len(aosTable):
            aosTable_sel['blur'] = fwhm_arr[select]
        elif len(fwhm_arr) == len(aosTable_sel):
            aosTable_sel['blur'] = fwhm_arr
        else:
            print(f"Warning: fwhm length ({len(fwhm_arr)}) doesn't match "
                  f"table ({len(aosTable)}) or selected ({len(aosTable_sel)})")
            aosTable_sel['blur'] = np.nan
    else:
        aosTable_sel['blur'] = np.nan

    # Focal plane coordinates (optional, expensive per-detector loop)
    if calc_focal_plane:
        nstars = len(aosTable_sel)
        intra_fpx = np.zeros(nstars)
        intra_fpy = np.zeros(nstars)
        extra_fpx = np.zeros(nstars)
        extra_fpy = np.zeros(nstars)

        for detector in camera:
            selone = (aosTable_sel['detector'] == detector.getName())
            if not np.any(selone):
                continue

            x_one = aosTable_sel[selone]['centroid_x_intra']
            y_one = aosTable_sel[selone]['centroid_y_intra']
            fpx_one, fpy_one = pixel_to_focal(x_one, y_one, detector)
            intra_fpx[selone] = fpx_one
            intra_fpy[selone] = fpy_one

            x_one = aosTable_sel[selone]['centroid_x_extra']
            y_one = aosTable_sel[selone]['centroid_y_extra']
            fpx_one, fpy_one = pixel_to_focal(x_one, y_one, detector)
            extra_fpx[selone] = fpx_one
            extra_fpy[selone] = fpy_one

        aosTable_sel['intra_fpx'] = intra_fpx
        aosTable_sel['intra_fpy'] = intra_fpy
        aosTable_sel['extra_fpx'] = extra_fpx
        aosTable_sel['extra_fpy'] = extra_fpy

    # Coordinate system column names
    zk_col = f'zk_{coord_sys}'
    thx_intra_col = f'thx_{coord_sys}_intra'
    thx_extra_col = f'thx_{coord_sys}_extra'
    thy_intra_col = f'thy_{coord_sys}_intra'
    thy_extra_col = f'thy_{coord_sys}_extra'

    thx_diff = np.abs(aosTable_sel[thx_intra_col] - aosTable_sel[thx_extra_col]) * 206265
    thy_diff = np.abs(aosTable_sel[thy_intra_col] - aosTable_sel[thy_extra_col]) * 206265
    matched_intra_extra = (thx_diff < 100.0) & (thy_diff < 100.0)
    aosTable_sel['matched_intra_extra'] = matched_intra_extra

    # Per-visit mean Zernike (optional)
    if calc_mean_zernike:
        zk_mean_col = f'zk_{coord_sys}_mean'
        values_array = np.stack(aosTable_sel[zk_col])
        mean_values = np.mean(values_array, axis=0)
        npts = len(aosTable_sel)
        aosTable_sel[zk_mean_col] = [mean_values for _ in range(npts)]

    return aosTable_sel, visit_meta


def get_zernikes_from_visits(visit_pairs, collections, butler_repo, coord_sys,
                             camera, calc_focal_plane=False,
                             calc_mean_zernike=False):
    """Get aggregate Zernikes for a list of (day_obs, seq_num) pairs.

    Returns (aosTable, visit_info_table) or (None, None).
    """
    print(f"Initializing Butler with repo={butler_repo}, collections: {collections}")
    butler = Butler(butler_repo, instrument='LSSTCam', collections=collections)

    agg_zernikes_list = []
    visit_meta_list = []
    success_count = 0
    error_count = 0
    ref_noll_indices = None
    noll_mismatch_count = 0

    print(f"\nExtracting Zernikes for {len(visit_pairs)} visits...")
    for day_obs_val, seq_num in tqdm(visit_pairs):
        agg_zern, visit_meta = get_aggregate_zernikes(
            butler, day_obs_val, seq_num, coord_sys, camera,
            calc_focal_plane=calc_focal_plane,
            calc_mean_zernike=calc_mean_zernike)
        if agg_zern is None:
            error_count += 1
            continue

        noll = visit_meta.get('nollIndices', None)
        if noll is not None:
            if ref_noll_indices is None:
                ref_noll_indices = list(noll)
            elif list(noll) != ref_noll_indices:
                print(f"WARNING: nollIndices mismatch for day_obs={day_obs_val}, "
                      f"seq_num={seq_num}: {list(noll)} != {ref_noll_indices} — skipping")
                noll_mismatch_count += 1
                error_count += 1
                continue

        agg_zernikes_list.append(agg_zern)
        visit_meta_list.append(visit_meta)
        success_count += 1

    print(f"\n{'='*60}")
    print(f"Extraction Summary:")
    print(f"  Total visits attempted: {len(visit_pairs)}")
    print(f"  Successful extractions: {success_count}")
    print(f"  Failed extractions:     {error_count}")
    if noll_mismatch_count > 0:
        print(f"  nollIndices mismatches: {noll_mismatch_count}")
    if ref_noll_indices is not None:
        print(f"  nollIndices: {ref_noll_indices}")

    if len(agg_zernikes_list) == 0:
        print(f"\nERROR: No Zernike data found for any visits!")
        print(f"Collections used: {collections}")
        return None, None

    agg_zernikes = vstack(agg_zernikes_list)
    print(f"  Total donut measurements: {len(agg_zernikes)}")
    print(f"{'='*60}\n")

    # Build visit-level info table
    visit_info = QTable()
    visit_info['day_obs'] = [m['day_obs'] for m in visit_meta_list]
    visit_info['seq_num'] = [m['seq_num'] for m in visit_meta_list]
    visit_info['visit'] = [m['visit'] for m in visit_meta_list]
    visit_info['skyAngle'] = [m['skyAngle'] for m in visit_meta_list]
    visit_info['ra'] = [m['ra'] for m in visit_meta_list]
    visit_info['dec'] = [m['dec'] for m in visit_meta_list]
    visit_info['az'] = [m['az'] for m in visit_meta_list]
    visit_info['alt'] = [m['alt'] for m in visit_meta_list]
    visit_info['band'] = [m['band'] for m in visit_meta_list]
    visit_info['mjd'] = [m['mjd'] for m in visit_meta_list]
    visit_info['nollIndices'] = [m['nollIndices'] for m in visit_meta_list]

    return agg_zernikes, visit_info


# ============================================================
# ConsDB Queries
# ============================================================

def get_visit_pairs_from_consdb(visits_df, programs, img_type='cwfs',
                                verify_pairing=True):
    """Extract (day_obs, seq_num) pairs from ConsDB dataframe."""
    program_mask = visits_df['science_program'].str.contains(programs[0], na=False)
    for prog in programs[1:]:
        program_mask |= visits_df['science_program'].str.contains(prog, na=False)

    filtered = visits_df[program_mask & (visits_df['img_type'] == img_type)].copy()

    if len(filtered) == 0:
        print("Warning: No matching visits found in ConsDB")
        return []

    filtered = filtered.sort_values(['day_obs', 'seq_num'])

    if not verify_pairing:
        visit_pairs = list(zip(filtered['day_obs'], filtered['seq_num']))
        print(f"\nFound {len(visit_pairs)} matching visits (no pairing verification)")
        return visit_pairs

    visit_pairs = []
    unpaired_count = 0

    for day_obs_val, group in filtered.groupby('day_obs'):
        seq_nums = sorted(group['seq_num'].values)
        i = 0
        while i < len(seq_nums) - 1:
            if seq_nums[i+1] == seq_nums[i] + 1:
                visit_pairs.append((day_obs_val, seq_nums[i+1]))
                i += 2
            else:
                if unpaired_count < 5:
                    print(f"Warning: Unpaired image at day_obs={day_obs_val}, "
                          f"seq_num={seq_nums[i]}")
                unpaired_count += 1
                i += 1
        if i == len(seq_nums) - 1:
            if unpaired_count < 5:
                print(f"Warning: Unpaired image at day_obs={day_obs_val}, "
                      f"seq_num={seq_nums[i]}")
            unpaired_count += 1

    if unpaired_count > 5:
        print(f"... and {unpaired_count - 5} more unpaired images")

    print(f"\nFound {len(visit_pairs)} FAM image pairs (keeping second of each pair)")
    if unpaired_count > 0:
        print(f"Note: {unpaired_count} unpaired images were skipped")

    return visit_pairs


def print_band_counts_by_day(df, block_names, img_type_value):
    """Print a table showing band counts by day_obs for filtered data."""
    if isinstance(block_names, str):
        block_names = [block_names]

    block_mask = df['science_program'].str.contains(block_names[0], na=False)
    for block_name in block_names[1:]:
        block_mask |= df['science_program'].str.contains(block_name, na=False)

    filtered = df[block_mask & (df['img_type'] == img_type_value)].copy()
    print(f"Total rows matching {block_names} and img_type='{img_type_value}': "
          f"{len(filtered)}")

    if len(filtered) == 0:
        print("No matching rows found.")
        return

    filtered['band_first'] = filtered['band'].str[0]
    band_table = pd.crosstab(filtered['day_obs'], filtered['band_first'])
    desired_order = ['u', 'g', 'r', 'i', 'z', 'y']
    band_table = band_table.reindex(columns=desired_order, fill_value=0)
    totals = band_table.sum(axis=0)
    band_table.loc['TOTAL'] = totals

    print("\nBand counts by day_obs:")
    print(band_table)


# ============================================================
# Rotator Angle Functions
# ============================================================

def get_rotator_angles(visits_df, visit_pairs):
    """Get physical_rotator_angle from ConsDB (visit1_quicklook)."""
    visits_indexed = visits_df.set_index(['day_obs', 'seq_num'])

    records = []
    for day_obs_val, seq_num in visit_pairs:
        rec = {'day_obs': day_obs_val, 'seq_num': seq_num,
               'physical_rotator_angle': np.nan}
        try:
            row = visits_indexed.loc[(day_obs_val, seq_num)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            phys_rot = row.get('physical_rotator_angle', np.nan)
            if pd.notna(phys_rot):
                rec['physical_rotator_angle'] = float(phys_rot)
        except KeyError:
            pass
        records.append(rec)

    rotator_df = pd.DataFrame(records)
    n_phys = rotator_df['physical_rotator_angle'].notna().sum()
    print(f"\nConsDB physical_rotator_angle: {n_phys}/{len(rotator_df)} visits")
    return rotator_df


def get_visitinfo_rotator_angles(butler, visit_pairs):
    """Get RotPA and parallactic angle from Butler raw.visitInfo."""
    records = []
    print(f"Querying Butler visitInfo for {len(visit_pairs)} visits...")
    for day_obs_val, seq_num in tqdm(visit_pairs):
        rec = {'day_obs': day_obs_val, 'seq_num': seq_num,
               'visitinfo_rotpa': np.nan, 'visitinfo_par_angle': np.nan,
               'visitinfo_rotator_angle': np.nan}
        try:
            visinfo = butler.get('raw.visitInfo', instrument='LSSTCam',
                                 detector=4, day_obs=day_obs_val, seq_num=seq_num)
            rotpa = visinfo.getBoresightRotAngle().asDegrees()
            par = visinfo.getBoresightParAngle().asDegrees()
            rec['visitinfo_rotpa'] = rotpa
            rec['visitinfo_par_angle'] = par
            rec['visitinfo_rotator_angle'] = calc_rotator_from_visitinfo(par, rotpa)
        except Exception:
            try:
                raw = butler.get('raw', instrument='LSSTCam',
                                 detector=4, day_obs=day_obs_val, seq_num=seq_num)
                visinfo = raw.visitInfo
                rotpa = visinfo.getBoresightRotAngle().asDegrees()
                par = visinfo.getBoresightParAngle().asDegrees()
                rec['visitinfo_rotpa'] = rotpa
                rec['visitinfo_par_angle'] = par
                rec['visitinfo_rotator_angle'] = calc_rotator_from_visitinfo(par, rotpa)
            except Exception:
                pass
        records.append(rec)

    df = pd.DataFrame(records)
    n_ok = df['visitinfo_rotpa'].notna().sum()
    print(f"  Got visitInfo for {n_ok}/{len(visit_pairs)} visits")
    return df


async def async_getEfdData_exprecord(client, topic, expRecord, columns=None,
                                     prePadding=0, postPadding=0):
    """Async EFD query using a Butler exposure record for time range."""
    begin = Time(expRecord.timespan.begin, scale="tai")
    end = Time(expRecord.timespan.end, scale="tai")
    if prePadding:
        begin -= TimeDelta(prePadding, format="sec")
    if postPadding:
        end += TimeDelta(postPadding, format="sec")

    data = await client.select_time_series(topic, columns or [], begin.utc, end.utc)
    if data is None or len(data) == 0:
        return None
    return data


async def get_efd_rotator_angles(efd_client, butler, visit_pairs):
    """Get actual rotator position from EFD MTRotator telemetry."""
    records = []
    print(f"Querying EFD rotator positions for {len(visit_pairs)} visits...")
    for day_obs_val, seq_num in tqdm(visit_pairs):
        rec = {'day_obs': day_obs_val, 'seq_num': seq_num,
               'efd_rotator_angle': np.nan}
        try:
            results = butler.registry.queryDimensionRecords(
                'exposure',
                where="instrument='LSSTCam' AND exposure.day_obs=:day_obs "
                      "AND exposure.seq_num=:seq_num",
                bind={'day_obs': day_obs_val, 'seq_num': seq_num}
            )
            exprec = next(iter(results), None)
            if exprec is None:
                records.append(rec)
                continue

            rotData = await async_getEfdData_exprecord(
                efd_client, "lsst.sal.MTRotator.rotation",
                expRecord=exprec, columns=["actualPosition"],
            )
            if rotData is not None and 'actualPosition' in rotData.columns:
                rec['efd_rotator_angle'] = float(
                    rotData['actualPosition'].values.mean())
        except Exception:
            pass
        records.append(rec)

    df = pd.DataFrame(records)
    n_ok = df['efd_rotator_angle'].notna().sum()
    print(f"  Got EFD rotator for {n_ok}/{len(visit_pairs)} visits")
    return df


async def get_rotator_data(visits_df, visit_pairs, butler_repo,
                           rotator_threshold=DEFAULT_ROTATOR_THRESHOLD):
    """Get rotator angles from all sources and compute best value.

    Returns rotator_df with columns: day_obs, seq_num, physical_rotator_angle,
    efd_rotator_angle, visitinfo_rotator_angle, rotator_angle, rotator_flagged.
    """
    rotator_df = get_rotator_angles(visits_df, visit_pairs)

    missing_mask = rotator_df['physical_rotator_angle'].isna()
    missing_pairs = rotator_df.loc[missing_mask, ['day_obs', 'seq_num']].values.tolist()
    n_missing = len(missing_pairs)
    print(f"ConsDB physical_rotator_angle: "
          f"{len(rotator_df) - n_missing} present, {n_missing} missing")

    if n_missing > 0:
        butler_rot = Butler(butler_repo, instrument='LSSTCam',
                            collections=['LSSTCam/raw/all'])
        visitinfo_df = get_visitinfo_rotator_angles(butler_rot, missing_pairs)
        rotator_df = rotator_df.merge(visitinfo_df, on=['day_obs', 'seq_num'], how='left')

        try:
            efd_client = makeEfdClient()
            efd_df = await get_efd_rotator_angles(efd_client, butler_rot, missing_pairs)
            rotator_df = rotator_df.merge(efd_df, on=['day_obs', 'seq_num'], how='left')
        except Exception as e:
            print(f"Warning: Could not access EFD: {e}")
            rotator_df['efd_rotator_angle'] = np.nan
    else:
        print("All visits have ConsDB physical_rotator_angle, "
              "skipping EFD/visitInfo queries")
        rotator_df['efd_rotator_angle'] = np.nan
        rotator_df['visitinfo_rotator_angle'] = np.nan
        rotator_df['visitinfo_rotpa'] = np.nan
        rotator_df['visitinfo_par_angle'] = np.nan

    # Best rotator: prefer ConsDB, then EFD, then visitInfo
    rotator_df['rotator_angle'] = (
        rotator_df['physical_rotator_angle']
        .fillna(rotator_df.get('efd_rotator_angle', np.nan))
        .fillna(rotator_df.get('visitinfo_rotator_angle', np.nan))
    )
    rotator_df['rotator_flagged'] = (
        rotator_df['rotator_angle'].abs() > rotator_threshold)

    n_flagged = rotator_df['rotator_flagged'].sum()
    n_phys = rotator_df['physical_rotator_angle'].notna().sum()
    n_efd = rotator_df['efd_rotator_angle'].notna().sum()
    n_vi = rotator_df['visitinfo_rotator_angle'].notna().sum()
    print(f"\nRotator angle sources: ConsDB={n_phys}, EFD={n_efd}, visitInfo={n_vi}")
    print(f"Flagged (|angle| > {rotator_threshold} deg): {n_flagged}")

    return rotator_df


# ============================================================
# Thermal Data Functions
# ============================================================

async def async_getEfdData_times(client, topic, obs_start, obs_end,
                                 columns=None, prePadding=0, postPadding=0,
                                 index=None):
    """Async EFD query using observation start/end times (TAI isot strings)."""
    begin = Time(obs_start, scale="tai")
    end = Time(obs_end, scale="tai")
    if prePadding:
        begin -= TimeDelta(prePadding, format="sec")
    if postPadding:
        end += TimeDelta(postPadding, format="sec")

    kwargs = {}
    if index is not None:
        kwargs["index"] = index
        kwargs["convert_influx_index"] = True

    data = await client.select_time_series(
        topic, columns or [], begin.utc, end.utc, **kwargs)
    if data is None or len(data) == 0:
        return None
    return data


async def get_ess_temperature(efd_client, obs_start, obs_end, index,
                              field="temperatureItem0",
                              post_padding=DEFAULT_TEMP_TIME_WINDOW_SEC):
    """Query a single ESS temperature sensor, return mean value."""
    data = await async_getEfdData_times(
        efd_client, "lsst.sal.ESS.temperature",
        obs_start, obs_end, columns=[field],
        postPadding=post_padding, index=index,
    )
    if data is not None and field in data.columns:
        return data[field].mean()
    return np.nan


async def get_m1m3_gradients(efd_client, visit_table):
    """Get M1M3 thermal gradients interpolated to observation times.

    Parameters
    ----------
    efd_client : EfdClient
    visit_table : DataFrame
        Must contain obs_start column (TAI isot strings).

    Returns
    -------
    DataFrame with x_gradient, y_gradient, z_gradient, radial_gradient.
    """
    if not HAS_M1M3_UTILS:
        print("Warning: lsst.ts.m1m3.utils not available, skipping M1M3 gradients")
        return pd.DataFrame({
            'x_gradient': np.nan, 'y_gradient': np.nan,
            'z_gradient': np.nan, 'radial_gradient': np.nan,
        }, index=visit_table.index)

    date_strings = Time(
        [str(x) for x in visit_table["obs_start"].values],
        format="isot", scale="tai"
    ).utc.isot
    data_times = pd.to_datetime(date_strings, format="ISO8601", utc=True)
    sorted_data_times = data_times.sort_values()
    start = Time(sorted_data_times[0])
    end = Time(sorted_data_times[-1])
    data_times_int = data_times.astype("int64")

    thermocouples = ThermocoupleAnalysis(efd_client)
    await thermocouples.load(start, end, time_bin=30)
    gradients = thermocouples.xyz_r_gradients
    grad_times = pd.to_datetime(
        gradients.index, format="ISO8601", utc=True
    ).astype("int64")
    t0 = grad_times[0]
    grad_times_norm = (grad_times - t0) / 1e9
    data_times_norm = (data_times_int - t0) / 1e9

    result = {}
    for name in ["x_gradient", "y_gradient", "z_gradient", "radial_gradient"]:
        values = gradients[name].values
        val_interpolated = pd.Series(values).interpolate().values
        result[name] = np.interp(data_times_norm, grad_times_norm, val_interpolated)

    return pd.DataFrame(result, index=visit_table.index)


async def get_thermal_data(consdb_client, efd_client, visit_info,
                           temp_time_window_sec=DEFAULT_TEMP_TIME_WINDOW_SEC):
    """Retrieve all thermal data for visits and return as a DataFrame.

    Queries ESS temperatures, M1M3 gradients, and TMA truss temperatures
    from the EFD, keyed on (day_obs, seq_num).

    Returns DataFrame with 13 thermal columns.
    """
    # Get obs_start/obs_end from ConsDB
    day_obs_list = sorted(set(np.array(visit_info['day_obs'])))
    day_obs_str = ", ".join(str(d) for d in day_obs_list)

    query = f"""
        SELECT e.day_obs, e.seq_num, e.obs_start, e.obs_end
        FROM cdb_lsstcam.exposure e
        WHERE e.day_obs IN ({day_obs_str})
        ORDER BY e.day_obs, e.seq_num
    """
    consdb_df = consdb_client.query(query).to_pandas()
    print(f"ConsDB returned {len(consdb_df)} exposure records for obs times")

    # Build working DataFrame from visit_info
    vi_df = pd.DataFrame({
        'day_obs': np.array(visit_info['day_obs']),
        'seq_num': np.array(visit_info['seq_num']),
    })
    vi_df = vi_df.merge(
        consdb_df[['day_obs', 'seq_num', 'obs_start', 'obs_end']],
        on=['day_obs', 'seq_num'], how='left',
    )
    n_matched = vi_df['obs_start'].notna().sum()
    print(f"Matched obs times for {n_matched}/{len(vi_df)} visits")

    # ESS temperatures
    ess_sensors = {
        "cam_air_temp": 111,
        "m2_air_temp": 112,
        "m1m3_air_temp": 113,
        "outside_temp": 301,
    }

    valid = vi_df.dropna(subset=['obs_start', 'obs_end']).copy()
    unique_visits = valid[['day_obs', 'seq_num', 'obs_start', 'obs_end']].drop_duplicates()
    print(f"Querying EFD temperatures for {len(unique_visits)} visits...")

    temp_records = []
    for _, row in tqdm(unique_visits.iterrows(), total=len(unique_visits)):
        record = {"day_obs": row["day_obs"], "seq_num": row["seq_num"]}
        for name, index in ess_sensors.items():
            record[name] = await get_ess_temperature(
                efd_client, row["obs_start"], row["obs_end"], index,
                post_padding=temp_time_window_sec,
            )
        temp_records.append(record)

    temp_df = pd.DataFrame(temp_records)

    # Delta-T quantities
    temp_df["m2_delta_t"] = temp_df["m2_air_temp"] - temp_df["m1m3_air_temp"]
    temp_df["cam_m1m3_delta_t"] = temp_df["cam_air_temp"] - temp_df["m1m3_air_temp"]
    temp_df["dome_delta_t"] = temp_df["outside_temp"] - temp_df["m1m3_air_temp"]

    for col in ess_sensors:
        n_valid = temp_df[col].notna().sum()
        print(f"  {col}: {n_valid}/{len(temp_df)} valid")

    # M1M3 thermal gradients (process per day_obs to avoid connection resets)
    gradient_parts = []
    for day_obs_val in sorted(unique_visits["day_obs"].unique()):
        day_visits = unique_visits[unique_visits["day_obs"] == day_obs_val].reset_index(drop=True)
        print(f"  M1M3 gradients day_obs {day_obs_val}: {len(day_visits)} visits")
        try:
            gdf = await get_m1m3_gradients(efd_client, day_visits)
            gdf["day_obs"] = day_visits["day_obs"].values
            gdf["seq_num"] = day_visits["seq_num"].values
            gradient_parts.append(gdf)
        except Exception as e:
            print(f"    WARNING: failed for {day_obs_val}: {e}")
            gdf = pd.DataFrame({
                "x_gradient": np.nan, "y_gradient": np.nan,
                "z_gradient": np.nan, "radial_gradient": np.nan,
                "day_obs": day_visits["day_obs"].values,
                "seq_num": day_visits["seq_num"].values,
            })
            gradient_parts.append(gdf)

    gradient_df = pd.concat(gradient_parts, ignore_index=True)

    # TMA truss temperatures
    truss_sensors = {
        "tma_truss_temp_pxpy": ("temperatureItem6", 122),
        "tma_truss_temp_mxmy": ("temperatureItem7", 122),
    }

    truss_records = []
    print(f"Querying TMA truss temperatures...")
    for _, row in tqdm(unique_visits.iterrows(), total=len(unique_visits)):
        record = {"day_obs": row["day_obs"], "seq_num": row["seq_num"]}
        for name, (field, index) in truss_sensors.items():
            record[name] = await get_ess_temperature(
                efd_client, row["obs_start"], row["obs_end"], index,
                field=field, post_padding=temp_time_window_sec,
            )
        truss_records.append(record)
    truss_df = pd.DataFrame(truss_records)

    # Merge all thermal data
    thermal_df = temp_df.merge(gradient_df, on=["day_obs", "seq_num"], how="left")
    thermal_df = thermal_df.merge(truss_df, on=["day_obs", "seq_num"], how="left")

    thermal_cols = (
        list(ess_sensors.keys())
        + ["m2_delta_t", "cam_m1m3_delta_t", "dome_delta_t"]
        + ["x_gradient", "y_gradient", "z_gradient", "radial_gradient"]
        + list(truss_sensors.keys())
    )
    print(f"\nThermal data: {len(thermal_cols)} columns for {len(thermal_df)} visits")

    return thermal_df


# ============================================================
# Intrinsic Wavefront Model
# ============================================================

def get_intrinsic_map(x, y, camera_id_map, band=BandLabel.LSST_I):
    """Get intrinsic wavefront Zernikes across a focal plane grid.

    Returns (X, Y, zkIntrinsics) where zkIntrinsics is [nZk x nPts] in meters.
    """
    config = CalcZernikesTaskConfig()
    config.estimateZernikes.retarget(EstimateZernikesDanishTask)
    config.donutStampSelector.maxSelect = 20
    config.donutStampSelector.maxFracBadPixels = 2.0e-4
    config.donutStampSelector.useCustomSnLimit = True
    config.donutStampSelector.minSignalToNoise = 100

    binFactor = 2
    config.estimateZernikes.binning = binFactor
    nollIndices = np.arange(4, 29)
    config.estimateZernikes.nollIndices = list(nollIndices)
    config.estimateZernikes.lstsqKwargs = {
        'ftol': 1.0e-3, 'xtol': 1.0e-3, 'gtol': 1.0e-3}
    config.estimateZernikes.saveHistory = False

    task = CalcZernikesTask(config=config)

    camName = 'LSSTCam'
    extra_detector_id = 195
    extra_detector_name = camera_id_map[extra_detector_id].getName()
    instrument = getTaskInstrument(
        camName, extra_detector_name,
        task.estimateZernikes.config.instConfigFile,
    )

    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    selpts = (R < 1.8)
    X = X[selpts].flatten()
    Y = Y[selpts].flatten()

    nZk = len(nollIndices)
    nPts = len(X)

    zkIntrinsics = np.zeros((nZk, nPts))
    for i in range(nPts):
        x_pt = float(X[i])
        y_pt = float(Y[i])
        zkIntrinsics[:, i] = instrument.getIntrinsicZernikes(
            xAngle=x_pt, yAngle=y_pt,
            defocalType=None,
            band=band, nollIndices=nollIndices,
        )

    return X, Y, zkIntrinsics


def create_intrinsic_interpolators(X, Y, zkIntrinsics):
    """Create interpolation functions for each Zernike term."""
    iZs = np.arange(4, 29)
    interpolators = {}
    points = np.column_stack([X, Y])
    for i, iZ in enumerate(iZs):
        values = zkIntrinsics[i, :]
        interpolators[iZ] = LinearNDInterpolator(points, values)
    return interpolators


def add_intrinsic_zernikes(aosTable, intrinsic_interpolators, coord_sys):
    """Add model intrinsic Zernikes to the data table and compute residuals."""
    zk_col = f'zk_{coord_sys}'
    thx_extra_col = f'thx_{coord_sys}_extra'
    thy_extra_col = f'thy_{coord_sys}_extra'

    thx_deg = np.rad2deg(aosTable[thx_extra_col])
    thy_deg = np.rad2deg(aosTable[thy_extra_col])

    zk_data = np.stack(aosTable[zk_col])
    npts, nZk_data = zk_data.shape
    iZs_data = infer_zernike_indices(nZk_data)

    print(f"Data has {nZk_data} Zernike terms per donut")
    print(f"Zernike indices used: {iZs_data}")

    zk_intrinsic = np.zeros((npts, nZk_data))
    print(f"Interpolating intrinsic Zernikes for {npts} measurements...")
    for i, iZ in enumerate(iZs_data):
        if iZ in intrinsic_interpolators:
            interp_func = intrinsic_interpolators[iZ]
            zk_intrinsic[:, i] = interp_func(thx_deg, thy_deg)
        else:
            print(f"Warning: No interpolator for Z{iZ}, setting to zero")

    aosTable['zk_intrinsic'] = list(zk_intrinsic)
    zk_residual = zk_data - zk_intrinsic
    aosTable['zk_residual'] = list(zk_residual)

    print("Added columns: 'zk_intrinsic', 'zk_residual'")
    return aosTable


# ============================================================
# Merge Helpers
# ============================================================

def merge_rotator_to_tables(rotator_df, aosTable, visit_info, rotator_threshold):
    """Add rotator_angle and rotator_flagged to aosTable and visit_info."""
    rot_dict = {}
    for _, r in rotator_df.iterrows():
        rot_dict[(r['day_obs'], r['seq_num'])] = r

    day_obs_arr = np.array(aosTable['day_obs'])
    seq_num_arr = np.array(aosTable['seq_num'])

    rot_arrays = {
        'rotator_angle': np.full(len(aosTable), np.nan),
        'rotator_flagged': np.zeros(len(aosTable), dtype=bool),
    }
    for (dobs, snum), rot_info in rot_dict.items():
        mask = (day_obs_arr == dobs) & (seq_num_arr == snum)
        for c in rot_arrays:
            rot_arrays[c][mask] = rot_info[c]

    for c in rot_arrays:
        aosTable[c] = rot_arrays[c]

    n_flagged = int(np.sum(rot_arrays['rotator_flagged']))
    print(f"Added rotator columns to aosTable. "
          f"Flagged: {n_flagged} (|angle| > {rotator_threshold} deg)")

    # Add rotator_angle to visit_info
    if visit_info is not None:
        rot_angle_col = np.full(len(visit_info), np.nan)
        vi_day_obs = np.array(visit_info['day_obs'])
        vi_seq_num = np.array(visit_info['seq_num'])
        for (dobs, snum), rot_info in rot_dict.items():
            mask = (vi_day_obs == dobs) & (vi_seq_num == snum)
            rot_angle_col[mask] = rot_info['rotator_angle']
        visit_info['rotator_angle'] = rot_angle_col
        print("Added rotator_angle to visit_info")

    return aosTable, visit_info


def merge_thermal_to_visit_info(thermal_df, visit_info):
    """Merge thermal columns into visit_info QTable."""
    vi_day_obs = np.array(visit_info['day_obs'])
    vi_seq_num = np.array(visit_info['seq_num'])

    thermal_indexed = thermal_df.set_index(['day_obs', 'seq_num'])
    thermal_cols = [c for c in thermal_df.columns if c not in ('day_obs', 'seq_num')]

    for col in thermal_cols:
        arr = np.full(len(visit_info), np.nan)
        for idx, row in thermal_indexed.iterrows():
            mask = (vi_day_obs == idx[0]) & (vi_seq_num == idx[1])
            arr[mask] = row[col]
        visit_info[col] = arr

    print(f"Added {len(thermal_cols)} thermal columns to visit_info")
    return visit_info


def drop_unwanted_columns(aosTable):
    """Drop columns with _W, _N, _NW, _ra_, _dec_ in the name."""
    drop_cols = [c for c in aosTable.colnames
                 if '_W' in c or '_N' in c or '_NW' in c
                 or '_ra_' in c or '_dec_' in c]
    if drop_cols:
        aosTable.remove_columns(drop_cols)
        print(f"Dropped {len(drop_cols)} columns: {drop_cols}")
    return aosTable


# ============================================================
# Pipeline
# ============================================================

async def run_mktable(
    butler_repo,
    fam_collections,
    day_obs_min=None,
    day_obs_max=None,
    fam_programs=None,
    collection_phrase=None,
    include_versions=False,
    coord_sys='OCS',
    output_dir='output',
    rotator_threshold=DEFAULT_ROTATOR_THRESHOLD,
    fp_radius=DEFAULT_FP_RADIUS,
    fp_nsteps=DEFAULT_FP_NSTEPS,
    intrinsic_band=None,
    min_visits_per_day=DEFAULT_MIN_VISITS_PER_DAY,
    include_thermal=True,
    calc_intrinsics=False,
    calc_mean_zernike=False,
    calc_focal_plane=False,
    temp_time_window_sec=DEFAULT_TEMP_TIME_WINDOW_SEC,
    consdb_url=DEFAULT_CONSDB_URL,
    # Legacy support
    prefix=None,
):
    """Run the full Zernike table-building pipeline.

    Parameters
    ----------
    collection_phrase : str, optional
        Override for output filename phrase. If None, auto-parsed from
        collection name.
    include_versions : bool
        If True, include wep/dviz version strings in the output filename.
    calc_intrinsics : bool
        If True, compute intrinsic Zernike model and residuals.
    calc_mean_zernike : bool
        If True, compute per-visit mean Zernike columns.
    calc_focal_plane : bool
        If True, compute focal plane coordinates (fpx, fpy).
    prefix : str, optional
        Deprecated; use collection_phrase instead.

    Returns (aosTable, visit_info) or (None, None).
    """
    # Setup
    os.environ.setdefault("no_proxy", "")
    if ".consdb" not in os.environ["no_proxy"]:
        os.environ["no_proxy"] += ",.consdb"

    camera = LsstCam.getCamera()
    camera_id_map = camera.getIdMap()
    if consdb_url is None:
        consdb_url = DEFAULT_CONSDB_URL
    # If using an external URL, embed token from ~/.lsst/consdb_token
    if "@" not in consdb_url and "consdb-pq.consdb" not in consdb_url:
        token_file = Path.home() / ".lsst" / "consdb_token"
        if token_file.exists():
            token = token_file.read_text().strip()
            consdb_url = consdb_url.replace("://", f"://user:{token}@", 1)
    consdb_client = ConsDbClient(consdb_url)

    if intrinsic_band is None:
        intrinsic_band = BandLabel.LSST_I

    # Parse collection info for output naming and default date range
    parsed_phrase, parsed_min, parsed_max = parse_collection_info(
        fam_collections, include_versions=include_versions)

    # Resolve collection_phrase: explicit > legacy prefix > auto-parsed
    if collection_phrase is None:
        collection_phrase = prefix if prefix is not None else parsed_phrase

    # Resolve day_obs range: explicit > auto-parsed from collection
    if day_obs_min is None:
        day_obs_min = parsed_min
    if day_obs_max is None:
        day_obs_max = parsed_max
    if day_obs_min is None or day_obs_max is None:
        raise ValueError(
            "day_obs_min and day_obs_max must be specified (either explicitly "
            "or parseable from collection name)")

    # Build output filename (single HDF5 with donuts + visits tables)
    os.makedirs(output_dir, exist_ok=True)
    output_file = (f'{output_dir}/{collection_phrase}_'
                   f'{day_obs_min}_{day_obs_max}.hdf5')

    print(f"Pipeline: {collection_phrase} {coord_sys} {day_obs_min}-{day_obs_max}")
    print(f"  Butler: {butler_repo}")
    print(f"  Collections: {fam_collections}")
    print(f"  Options: intrinsics={calc_intrinsics}, mean_zk={calc_mean_zernike}, "
          f"fp_coords={calc_focal_plane}, thermal={include_thermal}")
    print(f"  Output: {output_file}")

    # Query ConsDB for visits
    instrument = 'lsstcam'
    visits_query = f'''
        SELECT v1.*, ql.physical_rotator_angle
        FROM cdb_{instrument}.visit1 v1
        LEFT JOIN cdb_{instrument}.visit1_quicklook ql
        ON v1.visit_id = ql.visit_id
        WHERE v1.day_obs >= {day_obs_min} AND v1.day_obs <= {day_obs_max}
    '''
    visits = consdb_client.query(visits_query).to_pandas()
    print(f"Retrieved {len(visits)} visits from ConsDB")

    # Get visit pairs and filter sparse days
    print_band_counts_by_day(visits, fam_programs, 'cwfs')
    visit_pairs = get_visit_pairs_from_consdb(visits, fam_programs, img_type='cwfs')

    day_counts = Counter(d for d, s in visit_pairs)
    sparse_days = {d for d, n in day_counts.items() if n < min_visits_per_day}
    if sparse_days:
        n_before = len(visit_pairs)
        visit_pairs = [(d, s) for d, s in visit_pairs if d not in sparse_days]
        print(f"Removed {len(sparse_days)} day_obs with < {min_visits_per_day} "
              f"visit_pairs ({n_before - len(visit_pairs)} pairs dropped)")

    if len(visit_pairs) == 0:
        print("ERROR: No visit pairs remaining after filtering!")
        return None, None

    # Rotator angles
    rotator_df = await get_rotator_data(
        visits, visit_pairs, butler_repo, rotator_threshold)

    # Generate intrinsic wavefront model (optional)
    if calc_intrinsics:
        print("\nGenerating intrinsic wavefront model...")
        xbins = np.linspace(-fp_radius, fp_radius, fp_nsteps)
        ybins = np.linspace(-fp_radius, fp_radius, fp_nsteps)
        X_model, Y_model, zkIntrinsics_model = get_intrinsic_map(
            xbins, ybins, camera_id_map, band=intrinsic_band)
        print(f"Generated intrinsic model at {len(X_model)} points")

        intrinsic_interpolators = create_intrinsic_interpolators(
            X_model, Y_model, zkIntrinsics_model)
        print(f"Created interpolators for {len(intrinsic_interpolators)} Zernike terms")

    # Extract Zernikes from Butler
    aosTable, visit_info = get_zernikes_from_visits(
        visit_pairs, fam_collections, butler_repo, coord_sys, camera,
        calc_focal_plane=calc_focal_plane,
        calc_mean_zernike=calc_mean_zernike)
    if aosTable is None:
        return None, None

    # Add intrinsic model (optional)
    if calc_intrinsics:
        aosTable = add_intrinsic_zernikes(aosTable, intrinsic_interpolators, coord_sys)

    # Add rotator data
    aosTable, visit_info = merge_rotator_to_tables(
        rotator_df, aosTable, visit_info, rotator_threshold)

    # Thermal data
    if include_thermal:
        try:
            efd_client = makeEfdClient()
            thermal_df = await get_thermal_data(
                consdb_client, efd_client, visit_info,
                temp_time_window_sec=temp_time_window_sec,
            )
            visit_info = merge_thermal_to_visit_info(thermal_df, visit_info)
        except Exception as e:
            print(f"Warning: Could not retrieve thermal data: {e}")

    # Drop unwanted columns
    aosTable = drop_unwanted_columns(aosTable)

    # Save to single HDF5 file with donuts and visits tables
    aosTable.write(output_file, path='donuts', serialize_meta=True,
                   overwrite=True)
    visit_info.write(output_file, path='visits', serialize_meta=True,
                     append=True)
    print(f"\nSaved to {output_file}:")
    print(f"  donuts: {len(aosTable)} rows")
    print(f"  visits: {len(visit_info)} rows")

    print(f"\nFinal aosTable: {len(aosTable)} rows, {len(aosTable.columns)} columns")
    print(f"Columns: {aosTable.colnames}")

    return aosTable, visit_info
