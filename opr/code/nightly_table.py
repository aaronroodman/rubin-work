"""Nightly AOS data extraction — per-exposure table for one night.

Ported faithfully from the canonical generator
``lsst-sitcom/ts_aos_analysis`` notebook
``notebooks/nightly_report/nightly_report_ts_version.ipynb`` (branch
``tickets/DM-54406``), analysis-code cells only (plotting stripped).

This is the SUMMIT-only stage of the OPR pipeline: it queries ConsDB, the EFD,
and the embargo Butler to build one row per seq with scalar columns plus the
vector columns the OLR step consumes:

    dof_state (50), lut_state (50), zernikes_fwhm (25), vmodes (12),
    per-corner ConsDB Zernikes zernikes_191..zernikes_203, and the Butler
    per-corner dense Zernikes zk_{opd,intrinsic,deviation}_{R00,R04,R40,R44}.

Requires the LSST Science Pipelines / summit stack; runs on the Summit RSP.
"""

import os
import logging
import warnings

warnings.filterwarnings("ignore")

import galsim
import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

from astropy.time import Time, TimeDelta
from lsst.obs.lsst import LsstCam
from lsst.summit.utils import (
    ConsDbClient,
    getAirmassSeeingCorrection,
    getBandpassSeeingCorrection,
)
from lsst.summit.utils.efdUtils import (
    getMostRecentRowWithDataBefore,
    makeEfdClient,
)
import lsst.summit.utils.butlerUtils as butlerUtils
from lsst.ts.xml.tables.m1m3 import *  # noqa: F401,F403
from lsst.ts.m1m3.utils import *  # noqa: F401,F403  (ThermocoupleAnalysis)
from lsst.ts.ofc import OFCData, SensitivityMatrix
from lsst.ts.ofc.state_estimator import StateEstimator
from lsst.ts.wep.utils import makeDense
from tqdm import tqdm

# ConsDB lives on an internal host; bypass any HTTP proxy for it.
if "no_proxy" in os.environ:
    os.environ["no_proxy"] += ",.consdb"
else:
    os.environ["no_proxy"] = ".consdb"

__all__ = ["AOSDatabase", "build_nightly_table"]


# ----------------------------------------------------------------------------
# Zernike / PSF helpers
# ----------------------------------------------------------------------------
def getPsfGradPerZernike(
    diameter: float = 8.36,
    obscuration: float = 0.612,
    jmin: int = 4,
    jmax: int = 22,
) -> np.ndarray:
    """Get the gradient of the PSF FWHM with respect to each Zernike."""
    if jmin < 0:
        raise ValueError("jmin cannot be negative.")
    if jmax < jmin:
        raise ValueError("jmax must be greater than jmin.")
    conversion_factors = np.zeros(jmax + 1)
    for i in range(jmin, jmax + 1):
        coefs = [0] * i + [1]
        R_outer = diameter / 2
        R_inner = R_outer * obscuration
        Z = galsim.zernike.Zernike(coefs, R_outer=R_outer, R_inner=R_inner)
        rms_tilt = np.sqrt(np.sum(Z.gradX.coef**2 + Z.gradY.coef**2) / 2)
        rms_tilt = np.rad2deg(rms_tilt * 1e-6) * 3600
        fwhm_tilt = 2 * np.sqrt(2 * np.log(2)) * rms_tilt
        conversion_factors[i] = fwhm_tilt
    return conversion_factors[jmin:]


def convertZernikesToPsfWidth(
    zernikes: np.ndarray,
    diameter: float = 8.36,
    obscuration: float = 0.612,
    jmin: int = 4,
) -> np.ndarray:
    """Convert Zernike amplitudes to quadrature contribution to the PSF FWHM."""
    if jmin < 0:
        raise ValueError("jmin cannot be negative.")
    jmax = jmin + np.array(zernikes).shape[-1] - 1
    conversion_factors = getPsfGradPerZernike(
        jmin=jmin, jmax=jmax, diameter=diameter, obscuration=obscuration
    )
    dFWHM = conversion_factors * zernikes
    return dFWHM


# ----------------------------------------------------------------------------
# EFD-derived columns
# ----------------------------------------------------------------------------
async def find_faults(client, table):
    """Find faults during the night."""
    table_time = pd.to_datetime(table["time"], format="ISO8601", utc=True)
    topics = ["MTMount", "MTAOS", "MTHexapod", "MTCamera"]
    table["faults"] = [None for i in range(len(table))]
    start = Time(table["obs_start"].iloc[0], scale="tai").utc
    end = Time(table["obs_end"].iloc[-1], scale="tai").utc
    for topic in topics:
        efd_data = await client.select_time_series(
            f"lsst.sal.{topic}.logevent_summaryState",
            ["summaryState"],
            start,
            end,
        )
        if len(efd_data) == 0:
            continue
        faults = efd_data[efd_data["summaryState"] == 3]
        for i in range(len(faults)):
            fault_time = pd.to_datetime(faults.index[i], format="utc")
            closest_row = table.iloc[(table_time - fault_time).abs().argmin()]
            table.loc[table["seq"] == closest_row["seq"], "faults"] = topic
    return table


async def get_m1m3_gradients(client, data):
    """Get the M1M3 thermal gradients."""
    date_strings = Time(
        [str(x) for x in data["obs_start"].values], format="isot", scale="tai"
    ).utc.isot
    data_times = pd.to_datetime(date_strings, format="ISO8601", utc=True)
    sorted_data_times = data_times.sort_values()
    start = Time(sorted_data_times[0])
    end = Time(sorted_data_times[-1])
    data_times = data_times.astype("int64")
    thermocouples = ThermocoupleAnalysis(client)  # noqa: F405
    await thermocouples.load(start, end, time_bin=30)
    gradients = thermocouples.xyz_r_gradients
    grad_times = pd.to_datetime(
        gradients.index, format="ISO8601", utc=True
    ).astype("int64")
    t0 = grad_times[0]
    grad_times -= t0
    grad_times /= 1e9
    data_times -= t0
    data_times /= 1e9
    names = ["x_gradient", "y_gradient", "z_gradient", "radial_gradient"]
    for name in names:
        values = gradients[name].values
        val_series = pd.Series(values)
        val_interpolated = val_series.interpolate()
        data[name] = np.interp(data_times, grad_times, val_interpolated)
    return data


# ----------------------------------------------------------------------------
# Sensitivity-matrix SVD (for the zk_constrained columns)
# ----------------------------------------------------------------------------
def build_sensitivity_svd(ofc_data, dof_set_name="standard_22"):
    """Build sensitivity matrix SVD for the given DOF subset.

    Parameters
    ----------
    ofc_data : OFCData
        Configured OFC data object (with dof_idx set).
    dof_set_name : str
        DOF subset name ('standard_22', 'hexapod_10', etc.).

    Returns
    -------
    svd_result : dict
        Keys: U, s, V, dof_indices, norm_vector, A_sub, n_modes
    """
    sensor_name_list = ["R00_SW0", "R04_SW0", "R40_SW0", "R44_SW0"]

    # Zernike selection: z4-z19, z22-z26 (21 terms, skip z20, z21, z27, z28)
    zn = np.array(
        [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
         22, 23, 24, 25, 26]
    )
    zn_idx = zn - 4

    dof_sets = {
        "hexapod_10": list(range(0, 10)),
        "standard_22": list(range(0, 17)) + list(range(30, 35)),
    }

    field_angles = [ofc_data.sample_points[s] for s in sensor_name_list]
    dz_sensitivity_matrix = SensitivityMatrix(ofc_data)
    sens_3d = dz_sensitivity_matrix.evaluate(field_angles, 0.0)

    # Select Zernike subset, reshape to 2D (all 50 DOFs)
    sens_3d = sens_3d[:, zn_idx, :]
    A_full = sens_3d.reshape((-1, sens_3d.shape[2]))

    # Restrict to chosen DOF subset, then normalize
    dof_indices = dof_sets[dof_set_name]
    A_sub = A_full[:, dof_indices]
    norm_vector = ofc_data.normalization_weights[dof_indices]
    Atilde_sub = A_sub @ np.diag(norm_vector)

    # SVD
    U, s, Vh = np.linalg.svd(Atilde_sub, full_matrices=False)
    V = Vh.T

    return dict(
        U=U,
        s=s,
        V=V,
        dof_indices=dof_indices,
        norm_vector=norm_vector,
        A_sub=A_sub,
        n_modes=len(s),
    )


# ----------------------------------------------------------------------------
# AOSDatabase — ConsDB + EFD fetcher
# ----------------------------------------------------------------------------
class AOSDatabase:
    table: pd.DataFrame

    def __init__(
        self,
        day_obs: int = 20250415,
        seq_min: int = 1,
        seq_max: int = 9999,
        consdb_url: str = "http://consdb-pq.consdb:8080/consdb",
    ) -> None:
        """Create fetcher.

        Parameters
        ----------
        seq_max : int, optional
            Maximum sequence number to fetch. Default is 9999.
        consdb_url : str, optional
            URL to create ConsDB client.
            The default is "http://consdb-pq.consdb:8080/consdb".
        """
        self.log = logging.getLogger(__name__)

        self.efd_client = makeEfdClient()
        self.cdb_client = ConsDbClient(consdb_url)

        self.det_order = list([191, 195, 199, 203])
        camera = LsstCam().getCamera()
        self.detector_names = [
            camera.get(det_id).getName() for det_id in self.det_order
        ]

        self.day_obs = day_obs
        self.seq_max = seq_max
        self.seq_min = seq_min
        self.table = pd.DataFrame()

        self.time_window = TimeDelta(0.2, format="sec")
        self.temp_time_window = TimeDelta(0.2, format="sec")

    async def create(self, simplified=False):
        self.table = await self._fetch(
            self.day_obs, self.seq_min, self.seq_max, simplified
        )

    async def update(self, simplified: bool = False) -> None:
        """Update the database by grabbing more recent exposures.
        This will only grab new sequences, not re-fetch existing ones.
        """
        seq_min = self.table["seq"].max() + 1
        updated_table = await self._fetch(
            self.day_obs, seq_min, self.seq_max, simplified=simplified
        )
        self.table = pd.concat(
            [self.table, updated_table],
            ignore_index=True,
        )

    async def _fetch(
        self, day_obs: int, seq_min: int, seq_max: int, simplified: bool = False
    ) -> pd.DataFrame:
        query = f"""
            SELECT
            e.air_temp AS air_temp,
            e.airmass AS airmass,
            e.dimm_seeing AS dimm,
            e.altitude AS elevation,
            e.azimuth AS azimuth,
            e.exposure_id AS visit_id,
            e.physical_filter as band,
            e.day_obs AS day_obs,
            e.exp_midpt AS time,
            e.dimm_seeing AS seeing,
            e.seq_num AS seq,
            e.science_program AS block,
            ccdvisit1_quicklook.psf_sigma,
            ccdvisit1_quicklook.z4,
            ccdvisit1_quicklook.z5,
            ccdvisit1_quicklook.z6,
            ccdvisit1_quicklook.z7,
            ccdvisit1_quicklook.z8,
            ccdvisit1_quicklook.z9,
            ccdvisit1_quicklook.z10,
            ccdvisit1_quicklook.z11,
            ccdvisit1_quicklook.z12,
            ccdvisit1_quicklook.z13,
            ccdvisit1_quicklook.z14,
            ccdvisit1_quicklook.z15,
            ccdvisit1_quicklook.z16,
            ccdvisit1_quicklook.z17,
            ccdvisit1_quicklook.z18,
            ccdvisit1_quicklook.z19,
            ccdvisit1_quicklook.z20,
            ccdvisit1_quicklook.z21,
            ccdvisit1_quicklook.z22,
            ccdvisit1_quicklook.z23,
            ccdvisit1_quicklook.z24,
            ccdvisit1_quicklook.z25,
            ccdvisit1_quicklook.z26,
            ccdvisit1_quicklook.z27,
            ccdvisit1_quicklook.z28,
            ccdvisit1.detector as detector,
            q.psf_sigma_median AS psf_fwhm,
            q.psf_sigma_min AS psf_fwhm_min,
            q.psf_sigma_max AS psf_fwhm_max,
            q.psf_area_median AS psf_area,
            q.psf_area_min AS psf_area_min,
            q.psf_area_max AS psf_area_max,
            q.aos_fwhm AS aos_fwhm,
            q.donut_blur_fwhm AS donut_blur_fwhm,
            q.physical_rotator_angle AS rotation_angle,
            e.obs_end,
            e.obs_start
            FROM
            cdb_lsstcam.ccdvisit1_quicklook AS ccdvisit1_quicklook,
            cdb_lsstcam.ccdvisit1 AS ccdvisit1,
            cdb_lsstcam.visit1 AS visit1,
            cdb_lsstcam.visit1_quicklook AS q,
            cdb_lsstcam.exposure AS e
            WHERE
            ccdvisit1.detector IN (191, 192, 195, 196, 199, 200, 203, 204)
            AND ccdvisit1.ccdvisit_id = ccdvisit1_quicklook.ccdvisit_id
            AND ccdvisit1.visit_id = visit1.visit_id
            AND ccdvisit1.visit_id = q.visit_id
            AND ccdvisit1.visit_id = e.exposure_id
            AND (e.img_type = 'science' or e.img_type = 'acq' or e.img_type = 'engtest')
            AND e.day_obs = {day_obs}
            AND (e.seq_num BETWEEN {seq_min} AND {seq_max})
        """
        self.table = self.cdb_client.query(query).to_pandas()
        if len(self.table) == 0:
            return self.table

        # Correctly declare aos_fwhm and donut_blur_fwhm as float
        self.table["psf_fwhm"] = pd.to_numeric(self.table["psf_fwhm"])
        self.table["aos_fwhm"] = pd.to_numeric(self.table["aos_fwhm"])
        self.table["donut_blur_fwhm"] = pd.to_numeric(self.table["donut_blur_fwhm"])

        # Convert PSF sigma to FWHM
        sig2fwhm = 2 * np.sqrt(2 * np.log(2))
        pixel_scale = 0.2  # arcsec / pixel
        self.table["psf_fwhm"] = self.table["psf_fwhm"] * sig2fwhm * pixel_scale
        self.table["psf_fwhm_min"] = self.table["psf_fwhm_min"] * sig2fwhm * pixel_scale
        self.table["psf_fwhm_max"] = self.table["psf_fwhm_max"] * sig2fwhm * pixel_scale
        self.table["airmass"] = np.clip(self.table["airmass"], 1.0, 3.0)

        self.table["fwhm_zenith_500nm"] = [
            fwhm
            * getAirmassSeeingCorrection(airmass)
            * getBandpassSeeingCorrection(band)
            for fwhm, band, airmass in zip(
                self.table["psf_fwhm"], self.table["band"], self.table["airmass"]
            )
        ]

        zernike_columns = [f"z{i}" for i in range(4, 29)]
        self.table["zernikes"] = self.table[zernike_columns].apply(
            lambda row: np.array(row.fillna(0.0).values, dtype=float), axis=1
        )
        self.table["zernikes_fwhm"] = self.table["zernikes"].apply(
            convertZernikesToPsfWidth
        )

        # Get the data for the science CCDs for fwhm_05 and fwhm_95
        visits_query = f"""
        SELECT
        ccdvisit1_quicklook.psf_sigma,
        ccdvisit1_quicklook.psf_ixx,
        ccdvisit1_quicklook.psf_iyy,
        ccdvisit1_quicklook.psf_ixy,
        ccdvisit1.detector,
        visit1.visit_id,
        visit1.seq_num AS seq,
        visit1.day_obs,
        visit1.airmass
        FROM
        cdb_lsstcam.ccdvisit1_quicklook AS ccdvisit1_quicklook,
        cdb_lsstcam.ccdvisit1 AS ccdvisit1,
        cdb_lsstcam.visit1_quicklook AS visit1_quicklook,
        cdb_lsstcam.visit1 AS visit1
        WHERE
        ccdvisit1.ccdvisit_id = ccdvisit1_quicklook.ccdvisit_id
        AND ccdvisit1.visit_id = visit1.visit_id
        AND visit1.visit_id = visit1_quicklook.visit_id
        AND ccdvisit1.detector NOT IN (168, 188, 123, 27, 0, 20, 65, 161)
        AND visit1.airmass > 0
        AND visit1.day_obs = {self.day_obs}
        AND (visit1.seq_num BETWEEN {self.seq_min} AND {self.seq_max})
        AND (visit1.img_type = 'science' or visit1.img_type = 'acq' or visit1.img_type = 'engtest')
        """

        ccdvisits = self.cdb_client.query(visits_query).to_pandas()

        ccdvisits["psf_sigma"] = pd.to_numeric(ccdvisits["psf_sigma"], errors="coerce")
        ccdvisits["psf_ixx"] = pd.to_numeric(ccdvisits["psf_ixx"], errors="coerce")
        ccdvisits["psf_iyy"] = pd.to_numeric(ccdvisits["psf_iyy"], errors="coerce")
        ccdvisits["psf_ixy"] = pd.to_numeric(ccdvisits["psf_ixy"], errors="coerce")

        ccdvisits["psf_fwhm"] = ccdvisits["psf_sigma"] * sig2fwhm * pixel_scale

        denom = ccdvisits["psf_ixx"] + ccdvisits["psf_iyy"]
        denom = denom.replace(0, np.nan)

        e1 = (ccdvisits["psf_ixx"] - ccdvisits["psf_iyy"]) / denom
        e2 = (2 * ccdvisits["psf_ixy"]) / denom

        ccdvisits["ellipticity"] = np.sqrt(e1.to_numpy() ** 2 + e2.to_numpy() ** 2)

        # --- Group by visit ---------------------------------------------------
        clean = ccdvisits.dropna(subset=["psf_fwhm", "ellipticity"])
        groups = clean.groupby("visit_id")
        # For ellipticity, don't use the chips outside the good circle
        edge_chips = (
            0, 1, 2, 3, 18, 19, 20, 23, 64, 65, 68, 71, 155,
            158, 160, 161, 185, 186, 187, 188, 168, 169, 176, 165,
            123, 124, 120, 117, 30, 33, 27, 28,
        )
        ell_clean = clean[~(clean["detector"].isin(edge_chips))]
        ell_groups = ell_clean.groupby("visit_id")
        visits_summary = pd.DataFrame(
            {
                "day_obs": groups["day_obs"].first(),
                "seq": groups["seq"].median(),
                "psf_fwhm_05": groups["psf_fwhm"].quantile(0.05),
                "psf_fwhm_95": groups["psf_fwhm"].quantile(0.95),
                "psf_fwhm_sigma": groups["psf_fwhm"].std(),
                # ellipticity min/median/max (true across CCDs)
                "psf_ellipticity_median": ell_groups["ellipticity"].median(),
                "psf_ellipticity_sigma": ell_groups["ellipticity"].std(),
            }
        )
        visits_summary["psf_fwhm_95_05"] = np.sqrt(
            visits_summary["psf_fwhm_95"] ** 2 - visits_summary["psf_fwhm_05"] ** 2
        )
        self.table = pd.merge(
            self.table, visits_summary, how="left", on=["seq", "day_obs"]
        )

        self.table = await find_faults(self.efd_client, self.table)

        if not simplified:
            unique_day_seq = (
                self.table[["day_obs", "seq", "obs_end", "obs_start"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )
            (
                days,
                seqs,
                lut,
                cam_air_temp,
                states,
                m1m3_air_temp,
                outside_temp,
                m2_air_temp,
                correction_seq,
            ) = ([] for _ in range(9))
            for idx, row in tqdm(
                unique_day_seq.iterrows(), total=len(unique_day_seq), disable=True
            ):
                day_obs = int(row["day_obs"])
                seq = int(row["seq"])

                rec_end = row["obs_end"]
                rec_start = row["obs_start"]

                # ---------- State / LUT ----------
                # M2 and M1M3 LUT DOFs not available yet -> fill with nans.
                m2_dofs_lut = np.full(20, np.nan)
                m1m3_dofs_lut = np.full(20, np.nan)
                try:
                    cam_hexapod_data = getMostRecentRowWithDataBefore(
                        self.efd_client,
                        "lsst.sal.MTHexapod.logevent_compensationOffset",
                        Time(rec_end, scale="tai").utc,
                        maxSearchNMinutes=10,
                        where=lambda df: df["salIndex"] == 1,
                    )

                    m2_hexapod_data = getMostRecentRowWithDataBefore(
                        self.efd_client,
                        "lsst.sal.MTHexapod.logevent_compensationOffset",
                        Time(rec_end, scale="tai").utc,
                        maxSearchNMinutes=10,
                        where=lambda df: df["salIndex"] == 2,
                    )

                    hexapod_val = np.array(
                        [
                            m2_hexapod_data["z"],
                            m2_hexapod_data["x"],
                            m2_hexapod_data["y"],
                            m2_hexapod_data["u"],
                            m2_hexapod_data["v"],
                            cam_hexapod_data["z"],
                            cam_hexapod_data["x"],
                            cam_hexapod_data["y"],
                            cam_hexapod_data["u"],
                            cam_hexapod_data["v"],
                        ]
                    )
                    lut_val = np.concatenate([hexapod_val, m1m3_dofs_lut, m2_dofs_lut])
                except Exception:
                    lut_val = np.full(50, np.nan)

                event = getMostRecentRowWithDataBefore(
                    self.efd_client,
                    "lsst.sal.MTAOS.logevent_degreeOfFreedom",
                    timeToLookBefore=Time(rec_start, scale="tai").utc,
                )
                out = np.empty(50)
                for i in range(50):
                    out[i] = event[f"aggregatedDoF{i}"]
                states_val = out

                seq_num_corr = event["visitId"]
                seq_num_corr = np.where(
                    seq_num_corr < 10000,
                    seq_num_corr,
                    int(seq_num_corr - 1e5 * day_obs),
                )

                # Get outside temperature
                outside_temp_data = await self.efd_client.select_time_series(
                    "lsst.sal.ESS.temperature",
                    ["temperatureItem0"],
                    Time(rec_start, scale="tai").utc,
                    Time(rec_end, scale="tai").utc + self.temp_time_window,
                    index=301,
                    convert_influx_index=True,
                )
                if "temperatureItem0" in outside_temp_data:
                    outside_temp_val = outside_temp_data["temperatureItem0"].mean()
                else:
                    outside_temp_val = np.nan

                # Get M2 temperature
                m2_temp_data = await self.efd_client.select_time_series(
                    "lsst.sal.ESS.temperature",
                    ["temperatureItem0"],
                    Time(rec_start, scale="tai").utc,
                    Time(rec_end, scale="tai").utc + self.temp_time_window,
                    index=112,
                    convert_influx_index=True,
                )
                if "temperatureItem0" in m2_temp_data:
                    m2_air_temp_val = m2_temp_data["temperatureItem0"].mean()
                else:
                    m2_air_temp_val = np.nan

                # Get cam temperature
                cam_temp_data = await self.efd_client.select_time_series(
                    "lsst.sal.ESS.temperature",
                    ["temperatureItem0"],
                    Time(rec_start, scale="tai").utc,
                    Time(rec_end, scale="tai").utc + self.temp_time_window,
                    index=111,
                )
                if "temperatureItem0" in cam_temp_data:
                    cam_air_temp_val = cam_temp_data["temperatureItem0"].mean()
                else:
                    cam_air_temp_val = np.nan

                # Get temperature above m1m3
                m1m3_temp_data = await self.efd_client.select_time_series(
                    "lsst.sal.ESS.temperature",
                    ["temperatureItem0"],
                    Time(rec_start, scale="tai").utc,
                    Time(rec_end, scale="tai").utc + self.temp_time_window,
                    index=113,
                    convert_influx_index=True,
                )
                if "temperatureItem0" in m1m3_temp_data:
                    m1m3_air_temp_val = m1m3_temp_data["temperatureItem0"].mean()
                else:
                    m1m3_air_temp_val = np.nan

                lut.append(lut_val)
                m1m3_air_temp.append(m1m3_air_temp_val)
                cam_air_temp.append(cam_air_temp_val)
                m2_air_temp.append(m2_air_temp_val)
                outside_temp.append(outside_temp_val)
                states.append(states_val)
                days.append(day_obs)
                seqs.append(seq)
                correction_seq.append(seq_num_corr)

            efd_table = pd.DataFrame(
                {
                    "day_obs": days,
                    "seq": seqs,
                    "cam_air_temp": cam_air_temp,
                    "m2_air_temp": m2_air_temp,
                    "m1m3_air_temp": m1m3_air_temp,
                    "outside_temp": outside_temp,
                    "lut_state": lut,
                    "dof_state": states,
                    "seq_num_corr": correction_seq,
                }
            )

            self.table = pd.merge(
                self.table, efd_table, how="left", on=["seq", "day_obs"]
            )
            m1m3_gradient_table = await get_m1m3_gradients(
                self.efd_client, unique_day_seq
            )
            self.table = pd.merge(
                self.table, m1m3_gradient_table, how="left", on=["seq", "day_obs"]
            )
            self.table["m2_delta_t"] = (
                self.table["m2_air_temp"] - self.table["m1m3_air_temp"]
            )
            self.table["dome_delta_t"] = (
                self.table["outside_temp"] - self.table["m1m3_air_temp"]
            )
            self.table["cam_m1m3_delta_t"] = (
                self.table["cam_air_temp"] - self.table["m1m3_air_temp"]
            )

        return self.table


# ----------------------------------------------------------------------------
# build_nightly_table — assemble the one-row-per-seq output table
# ----------------------------------------------------------------------------
async def build_nightly_table(day_obs, seq_min=0, seq_max=9999):
    """Build the complete per-exposure AOS table for one night.

    Parameters
    ----------
    day_obs : int
        Observation day (e.g. 20260117).
    seq_min, seq_max : int
        Sequence number range.

    Returns
    -------
    result : pd.DataFrame
        One row per seq, with scalar columns plus vector columns:
        dof_state (50), lut_state (50), zernikes_fwhm (25), vmodes (12),
        per-corner Zernike columns zernikes_191 ... zernikes_203 (ConsDB),
        and zk_opd/intrinsic/deviation_R00/R04/R40/R44 (Butler).
    """
    # --- Fetch data via AOSDatabase ---
    db = AOSDatabase(day_obs=day_obs, seq_min=seq_min, seq_max=seq_max)
    await db.create(simplified=False)
    table = db.table

    if len(table) == 0:
        print(f"No data for {day_obs}, seq {seq_min}-{seq_max}")
        return pd.DataFrame()

    # --- raw_filtered_table: per-detector rows (4 WF CCDs per seq) ---
    raw_filtered_table = table[table["day_obs"] == day_obs].copy()

    # --- Fix obs_start/obs_end column duplicates from upstream merges ---
    for col in ["obs_start", "obs_end"]:
        if f"{col}_x" in raw_filtered_table.columns:
            raw_filtered_table[col] = raw_filtered_table[f"{col}_x"]
            raw_filtered_table.drop(columns=[f"{col}_x", f"{col}_y"], inplace=True)

    # --- Ensure dimm is numeric ---
    raw_filtered_table["dimm"] = pd.to_numeric(
        raw_filtered_table["dimm"], errors="coerce"
    )

    # --- filtered_table: one row per seq (numeric columns averaged) ---
    ft = raw_filtered_table.copy()
    ft["rotation_angle"] = ft["rotation_angle"].astype(float)
    ft["seq_num_corr"] = ft["seq_num_corr"].astype(int)
    ft_numeric = ft.select_dtypes(include="number").copy()

    # Re-add non-numeric columns we want to keep
    non_numeric_cols = ["band", "block", "faults", "time", "obs_start", "obs_end"]
    for col in non_numeric_cols:
        if col in ft.columns:
            ft_numeric[col] = ft[col]
    filtered_table = ft_numeric.groupby("seq").agg(
        {
            col: "first" if col in non_numeric_cols else "mean"
            for col in ft_numeric.columns
        }
    )

    # --- Extract state arrays (dof_state, lut_state, zernikes_fwhm) ---
    state_table = raw_filtered_table[
        ["seq", "dof_state", "zernikes_fwhm", "lut_state"]
    ]
    state_table = state_table.groupby("seq").mean()
    states_per_seq = state_table.dropna(
        subset=["dof_state", "zernikes_fwhm", "lut_state"]
    )
    dof_state = np.vstack(states_per_seq["dof_state"].values)
    zernikes_fwhm = np.vstack(states_per_seq["zernikes_fwhm"].values)
    lut_state = np.vstack(states_per_seq["lut_state"].values)

    # --- Compute vmodes from dof_state ---
    ofcData = OFCData()
    ofcData.configure_controller()
    await ofcData.configure_instrument("lsst")

    # Restrict to standard_22 DOFs (5+5+7+5 = 22)
    m2_hexapod = np.ones(5, dtype=bool)
    cam_hexapod = np.ones(5, dtype=bool)
    m1m3_bending = np.zeros(20, dtype=bool)
    m2_bending = np.zeros(20, dtype=bool)
    m1m3_bending[:7] = True
    m2_bending[:5] = True
    ofcData.comp_dof_idx = dict(
        m2HexPos=m2_hexapod,
        camHexPos=cam_hexapod,
        M1M3Bend=m1m3_bending,
        M2Bend=m2_bending,
    )

    se = StateEstimator(ofcData)
    vmodes = np.array([se.get_vmodes_from_dofs(d)[0:12] for d in dof_state])

    # --- Add vector columns to filtered_table ---
    seqs = states_per_seq.index.values
    vec_df = pd.DataFrame(index=seqs)
    vec_df["dof_state"] = list(dof_state)
    vec_df["lut_state"] = list(lut_state)
    vec_df["zernikes_fwhm"] = list(zernikes_fwhm)
    vec_df["vmodes"] = list(vmodes)

    # --- zk_constrained: project DOF state through 12-vmode SVD subspace ---
    svd = build_sensitivity_svd(ofcData, dof_set_name="standard_22")
    U, s, V = svd["U"], svd["s"], svd["V"]
    dof_indices = svd["dof_indices"]
    norm_vector = svd["norm_vector"]
    n_modes = 12
    corner_names = ["R00", "R04", "R40", "R44"]
    n_zk = 21  # per corner

    zk_con = {f"zk_constrained_{c}": [] for c in corner_names}
    for dof_vec in dof_state:
        dof_sub = dof_vec[dof_indices] / norm_vector
        v_proj = V[:, :n_modes].T @ dof_sub
        z_all = U[:, :n_modes] @ np.diag(s[:n_modes]) @ v_proj  # 84-vector
        z_corners = z_all.reshape(4, n_zk)
        for ic, c in enumerate(corner_names):
            zk_con[f"zk_constrained_{c}"].append(z_corners[ic])

    for col_name, arr_list in zk_con.items():
        vec_df[col_name] = arr_list

    filtered_table = filtered_table.join(vec_df, how="left")

    # --- Per-corner Zernikes from raw_filtered_table (ConsDB) ---
    wf_detectors = [191, 195, 199, 203]
    for det in wf_detectors:
        det_rows = raw_filtered_table[raw_filtered_table["detector"] == det][
            ["seq", "zernikes"]
        ]
        det_rows = det_rows.set_index("seq")
        det_rows = det_rows.rename(columns={"zernikes": f"zernikes_{det}"})
        filtered_table = filtered_table.join(det_rows, how="left")

    # --- Per-corner Zernikes from Butler (aggregateAOSVisitTableAvg) ---
    # Three Zernike types per corner: OPD, intrinsic, deviation.
    # Dense arrays (all terms from Z4 up, including unfitted).
    det_short = {
        "R00_SW0": "R00",
        "R04_SW0": "R04",
        "R40_SW0": "R40",
        "R44_SW0": "R44",
    }
    zk_types = ["opd", "intrinsic", "deviation"]
    col_map = {
        "opd": "zk_OCS",
        "intrinsic": "zk_intrinsic_OCS",
        "deviation": "zk_deviation_OCS",
    }

    butler = butlerUtils.makeDefaultButler("LSSTCam", embargo=True)
    refs = list(
        butler.query_datasets(
            "aggregateAOSVisitTableAvg",
            where=f"exposure.day_obs = {day_obs}",
        )
    )

    butler_rows = []
    for ref in tqdm(refs, desc="Butler", disable=True):
        seq = int(ref.dataId["visit"] - day_obs * 1e5)
        btable = butler.get(ref)
        noll_idx = btable.meta["nollIndices"]
        det_names = list(btable["detector"])

        row = {"seq": seq}
        for det_full, det_short_name in det_short.items():
            if det_full in det_names:
                det_row = det_names.index(det_full)
                for zk_type in zk_types:
                    col = col_map[zk_type]
                    dense = makeDense(np.array(btable[col][det_row]), noll_idx)
                    row[f"zk_{zk_type}_{det_short_name}"] = dense
        butler_rows.append(row)

    if butler_rows:
        butler_df = pd.DataFrame(butler_rows).set_index("seq")
        filtered_table = filtered_table.join(butler_df, how="left")
        n_butler = len(butler_df)
    else:
        n_butler = 0

    n_consdb = len(filtered_table)
    if n_butler < n_consdb:
        print(f"  WARNING: Butler has {n_butler} seqs vs ConsDB {n_consdb}")

    # --- Drop redundant columns ---
    drop_cols = [
        c for c in ["psf_sigma", "seeing", "zernikes"] if c in filtered_table.columns
    ]
    if drop_cols:
        filtered_table.drop(columns=drop_cols, inplace=True)

    print(f"Built table for {day_obs}: {len(filtered_table)} exposures")
    print(f"  dof_state: ({len(dof_state)}, {dof_state.shape[1]})")
    print(f"  lut_state: ({len(lut_state)}, {lut_state.shape[1]})")
    print(f"  zernikes_fwhm: ({len(zernikes_fwhm)}, {zernikes_fwhm.shape[1]})")
    print(f"  vmodes: ({len(vmodes)}, {vmodes.shape[1]})")
    print(f"  Per-corner Zernike columns (ConsDB): {[f'zernikes_{d}' for d in wf_detectors]}")
    butler_cols = [f"zk_{t}_{d}" for t in zk_types for d in det_short.values()]
    print(f"  Per-corner Zernike columns (Butler): {butler_cols} ({n_butler} seqs)")
    if n_butler > 0:
        sample = filtered_table["zk_opd_R00"].dropna()
        if len(sample) > 0:
            print(f"  Dense Zernike length: {len(sample.iloc[0])}")
    print(f"  StateEstimator restricted to standard_22 ({len(ofcData.dof_idx)} active DOFs)")
    zk_con_cols = [f"zk_constrained_{c}" for c in corner_names]
    print(f"  zk_constrained columns: {zk_con_cols} ({n_zk} Zernikes/corner, microns WF)")

    return filtered_table
