import os

import numpy as np
import pandas as pd
from astropy.table import Table
from lsst.summit.utils import ConsDbClient

__all__ = ["PSFMomentsTable"]


class PSFMomentsTable:
    """Query PSF moments per CCD from the Rubin ConsDB.

    Returns an astropy Table with one row per visit and PSF moment
    columns (psf_sigma, psf_ixx, psf_iyy, psf_ixy) for each requested
    detector, plus day_obs and seq_num.

    Parameters
    ----------
    day_obs : int
        Observation day in YYYYMMDD format.
    detectors : list of int, optional
        Detector IDs to query. Default is the 4 corner raft CCDs
        [191, 195, 199, 203].
    seq_min : int, optional
        Minimum sequence number. Default 1.
    seq_max : int, optional
        Maximum sequence number. Default 9999.
    consdb_url : str, optional
        ConsDB endpoint URL.
    """

    def __init__(
        self,
        day_obs: int,
        detectors: list[int] | None = None,
        seq_min: int = 1,
        seq_max: int = 9999,
        consdb_url: str = "http://consdb-pq.consdb:8080/consdb",
    ) -> None:
        if "no_proxy" in os.environ:
            if ".consdb" not in os.environ["no_proxy"]:
                os.environ["no_proxy"] += ",.consdb"
        else:
            os.environ["no_proxy"] = ".consdb"

        self.day_obs = day_obs
        self.detectors = detectors if detectors is not None else [191, 195, 199, 203]
        self.seq_min = seq_min
        self.seq_max = seq_max
        self.client = ConsDbClient(consdb_url)

    def fetch(self) -> Table:
        """Query ConsDB and return an astropy Table of PSF moments.

        Returns
        -------
        astropy.table.Table
            One row per visit with columns: visit_id, day_obs, seq_num,
            and psf_sigma_<det>, psf_ixx_<det>, psf_iyy_<det>,
            psf_ixy_<det> for each detector.
        """
        detector_list = ", ".join(str(d) for d in self.detectors)

        query = f"""
            SELECT
                ccdvisit1_quicklook.psf_sigma,
                ccdvisit1_quicklook.psf_ixx,
                ccdvisit1_quicklook.psf_iyy,
                ccdvisit1_quicklook.psf_ixy,
                ccdvisit1.detector,
                visit1.visit_id,
                visit1.seq_num,
                visit1.day_obs
            FROM
                cdb_lsstcam.ccdvisit1_quicklook AS ccdvisit1_quicklook,
                cdb_lsstcam.ccdvisit1 AS ccdvisit1,
                cdb_lsstcam.visit1 AS visit1
            WHERE
                ccdvisit1.detector IN ({detector_list})
                AND ccdvisit1.ccdvisit_id = ccdvisit1_quicklook.ccdvisit_id
                AND ccdvisit1.visit_id = visit1.visit_id
                AND visit1.day_obs = {self.day_obs}
                AND (visit1.seq_num BETWEEN {self.seq_min} AND {self.seq_max})
                AND (visit1.img_type = 'science'
                     OR visit1.img_type = 'acq'
                     OR visit1.img_type = 'engtest')
        """

        df = self.client.query(query).to_pandas()
        if df.empty:
            return Table()

        moment_cols = ["psf_sigma", "psf_ixx", "psf_iyy", "psf_ixy"]
        for col in moment_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        pivoted = df.pivot_table(
            index=["visit_id", "day_obs", "seq_num"],
            columns="detector",
            values=moment_cols,
        )

        # Flatten MultiIndex columns: ("psf_sigma", 191) -> "psf_sigma_191"
        pivoted.columns = [f"{moment}_{int(det)}" for moment, det in pivoted.columns]
        pivoted = pivoted.reset_index()

        return Table.from_pandas(pivoted)
