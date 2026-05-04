"""Memory-friendly replacement for the fetch() in image_quality_trending.ipynb.

Changes vs. the original:
  1. Explicit INNER JOINs with visit1 as the driving table so the planner
     prunes by day_obs / science_program / physical_filter before fanning
     out by detector.
  2. Loops over day_obs in chunks (default 7 nights), concatenating the
     resulting frames. Peak server- and client-side memory scales with
     chunk_days, not the full date range.
  3. Per-chunk numeric coercion to float64 to kill object-dtype overhead
     from NULL/Decimal cells, plus gc.collect() between chunks.

Usage in the notebook (replaces both `fetch(...)` and the
`fetchA + fetchB + merge` workaround):

    from consdb_fetch import fetch_chunked
    ccdvisits = fetch_chunked(client, instrument, day_obs_min,
                              day_obs_max, SURVEY_PROGRAMS)
"""

import gc
import hashlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Bump this when changing the SELECT column list or query structure to
# invalidate any cached chunk parquet files written under the old schema.
CACHE_VERSION = "v1"

PIXEL_SCALE = 0.2
SIG2FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))

NUMERIC_COLS = (
    "psf_area", "psf_sigma", "psf_ixx", "psf_ixy", "psf_iyy",
    "coma_1", "coma_2", "trefoil_1", "trefoil_2",
    "e4_1", "e4_2", "kurtosis",
    "airmass", "altitude", "azimuth", "sky_rotation",
    "exp_midpt_mjd", "dimm_seeing", "s_ra", "s_dec",
    "psf_sigma_min", "psf_sigma_median",
    "aos_fwhm", "donut_blur_fwhm", "ringss_seeing",
    "seeing_zenith_500nm_min", "seeing_zenith_500nm_median",
    "physical_rotator_angle",
)

FILTERS = ("u_24", "g_6", "r_57", "i_39", "z_20", "y_10")
VIGNETTED_DETECTORS = (168, 188, 123, 27, 0, 20, 65, 161)
CORNER_WAVEFRONT_DETECTORS = (191, 192, 195, 196, 199, 200, 203, 204)


def _day_obs_to_date(d):
    s = str(int(d))
    return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))


def _date_to_day_obs(d):
    return d.year * 10000 + d.month * 100 + d.day


def _build_query(instrument, d_min, d_max, programs):
    programs_str = "', '".join(programs)
    filters_str = "', '".join(FILTERS)
    excluded = VIGNETTED_DETECTORS + CORNER_WAVEFRONT_DETECTORS
    excluded_str = ", ".join(str(x) for x in excluded)
    return f"""
    SELECT
        cvq.psf_area, cvq.psf_sigma, cvq.psf_ixx, cvq.psf_ixy, cvq.psf_iyy,
        cvq.coma_1, cvq.coma_2, cvq.trefoil_1, cvq.trefoil_2,
        cvq.e4_1, cvq.e4_2, cvq.kurtosis,
        cv.detector,
        v.visit_id, v.seq_num, v.band, v.physical_filter, v.day_obs,
        v.target_name, v.science_program, v.observation_reason,
        v.airmass, v.altitude, v.azimuth, v.sky_rotation,
        v.exp_midpt_mjd, v.dimm_seeing, v.s_ra, v.s_dec,
        vq.psf_sigma_min, vq.psf_sigma_median,
        vq.aos_fwhm, vq.donut_blur_fwhm, vq.ringss_seeing,
        vq.seeing_zenith_500nm_min, vq.seeing_zenith_500nm_median,
        vq.physical_rotator_angle
    FROM cdb_{instrument}.visit1 AS v
    INNER JOIN cdb_{instrument}.visit1_quicklook AS vq
        ON vq.visit_id = v.visit_id
    INNER JOIN cdb_{instrument}.ccdvisit1 AS cv
        ON cv.visit_id = v.visit_id
    INNER JOIN cdb_{instrument}.ccdvisit1_quicklook AS cvq
        ON cvq.ccdvisit_id = cv.ccdvisit_id
    WHERE v.day_obs >= {d_min} AND v.day_obs <= {d_max}
        AND v.airmass > 0
        AND v.science_program IN ('{programs_str}')
        AND v.physical_filter IN ('{filters_str}')
        AND cv.detector NOT IN ({excluded_str});
    """


def _coerce_numeric_inplace(df):
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def _chunk_ranges(day_obs_min, day_obs_max, chunk_days):
    start = _day_obs_to_date(day_obs_min)
    end = _day_obs_to_date(day_obs_max)
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
        yield _date_to_day_obs(cur), _date_to_day_obs(chunk_end)
        cur = chunk_end + timedelta(days=1)


def _programs_hash(programs):
    return hashlib.md5(",".join(sorted(programs)).encode()).hexdigest()[:8]


def _chunk_cache_path(cache_dir, instrument, programs, d_min, d_max):
    h = _programs_hash(programs)
    return cache_dir / (
        f"ccdvisits_chunk_{instrument}_{h}_{CACHE_VERSION}_{d_min}-{d_max}.pq"
    )


def fetch_chunked(client, instrument, day_obs_min, day_obs_max, programs,
                  chunk_days=7, cache_dir=None, progress=True):
    """Chunked fetch over [day_obs_min, day_obs_max] inclusive.

    If `cache_dir` is given, each 7-day (or chunk_days-sized) chunk is
    written to a parquet there after fetching, and read back from
    parquet on subsequent runs. Cache filename includes instrument, an
    8-char hash of the program list, and a schema version, so stale
    caches don't get reused after parameter or column changes.
    """
    ranges = list(_chunk_ranges(day_obs_min, day_obs_max, chunk_days))
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    chunks = []
    total_rows = 0
    cached_count = 0
    bar = tqdm(ranges, desc="ConsDB fetch", unit="chunk", disable=not progress)
    for d_min, d_max in bar:
        bar.set_postfix(
            chunk=f"{d_min}-{d_max}",
            rows=f"{total_rows:,}",
            cached=cached_count,
        )

        cache_path = (
            _chunk_cache_path(cache_dir, instrument, programs, d_min, d_max)
            if cache_dir is not None else None
        )

        if cache_path is not None and cache_path.exists():
            df = pd.read_parquet(cache_path)
            cached_count += 1
        else:
            query = _build_query(instrument, d_min, d_max, programs)
            df = client.query(query).to_pandas()
            _coerce_numeric_inplace(df)
            if cache_path is not None:
                df.to_parquet(cache_path)

        total_rows += len(df)
        bar.set_postfix(
            chunk=f"{d_min}-{d_max}",
            rows=f"{total_rows:,}",
            cached=cached_count,
        )

        chunks.append(df)
        del df
        gc.collect()

    ccdvisits = pd.concat(chunks, ignore_index=True, copy=False)
    del chunks
    gc.collect()

    # Derived columns (computed once on the full frame)
    ccdvisits["psf_fwhm"] = ccdvisits["psf_sigma"] * SIG2FWHM * PIXEL_SCALE
    ccdvisits["psf_fwhm_area"] = 0.663 * PIXEL_SCALE * np.sqrt(ccdvisits["psf_area"])

    sum_xx_yy = ccdvisits["psf_ixx"] + ccdvisits["psf_iyy"]
    ccdvisits["psf_e1"] = (ccdvisits["psf_ixx"] - ccdvisits["psf_iyy"]) / sum_xx_yy
    ccdvisits["psf_e2"] = (2.0 * ccdvisits["psf_ixy"]) / sum_xx_yy
    ccdvisits["psf_e"] = np.sqrt(ccdvisits["psf_e1"] ** 2 + ccdvisits["psf_e2"] ** 2)

    return ccdvisits
