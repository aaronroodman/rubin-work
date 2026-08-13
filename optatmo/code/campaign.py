"""Output layout for optatmo fitting campaigns.

Separates INPUTS (expensive, shared across campaigns) from OUTPUTS (per campaign)
so a night's psfmoments/cwfs are extracted once and reused, while every fitting
run's products live under its own labelled directory:

    data/                                  # INPUTS (shared, campaign-independent)
      psfmoments_<visit>.parquet
      cwfs_<visit>.parquet
      visitmeta_<visit>.parquet
      svd/ofc_svd_*.npz                    # v-mode SVD basis

    output/runs/<campaign>/                # OUTPUTS (one dir per campaign)
      manifest.json                        # options/provenance for every run
      <day_obs>/
        config_snapshot.yaml
        fits/     vmodefit_<seq>.npz  fitprog_<seq>.npz
        reports/  fit_<seq>.pdf  fitmon_<seq>.png
        ensemble/ ensemble_vmodes.pdf/.csv  ensemble_corners.pdf/.csv

output/runs is under output/ so it inherits the USDF output/ symlink to the big
filesystem.  A "campaign" is a named run with a fixed set of options; different
options or different day_obs get their own directory.
"""
import json
import os
import shutil

DATA = 'data'                 # shared per-visit inputs + SVD basis
RUNS_ROOT = 'output/runs'     # per-campaign outputs (inherits output/ symlink)


# ---- shared inputs (campaign-independent) -------------------------------
def psfmoments_path(visit):
    return f'{DATA}/psfmoments_{visit}.parquet'


def cwfs_path(visit):
    return f'{DATA}/cwfs_{visit}.parquet'


def visitmeta_path(visit):
    return f'{DATA}/visitmeta_{visit}.parquet'


class Campaign:
    """Paths for one campaign/day_obs.  `label` names the campaign."""

    def __init__(self, label, day):
        if not label:
            raise ValueError('a campaign label is required (--campaign)')
        self.label = label
        self.day = int(day)
        self.dir = f'{RUNS_ROOT}/{label}'
        self.day_dir = f'{self.dir}/{self.day}'
        self.fits = f'{self.day_dir}/fits'
        self.reports = f'{self.day_dir}/reports'
        self.ensemble = f'{self.day_dir}/ensemble'

    def ensure(self):
        for d in (self.fits, self.reports, self.ensemble):
            os.makedirs(d, exist_ok=True)
        return self

    # per-visit outputs
    def fit_npz(self, seq):
        return f'{self.fits}/vmodefit_{seq}.npz'

    def fitprog_npz(self, seq):
        return f'{self.fits}/fitprog_{seq}.npz'

    def report_pdf(self, seq):
        return f'{self.reports}/fit_{seq}.pdf'

    def fitmon_png(self, seq):
        return f'{self.reports}/fitmon_{seq}.png'

    # provenance
    def snapshot_config(self, config_path='config.yaml'):
        if os.path.exists(config_path):
            os.makedirs(self.day_dir, exist_ok=True)
            shutil.copy(config_path, f'{self.day_dir}/config_snapshot.yaml')

    def write_manifest(self, opts):
        """Append this run's options to output/runs/<label>/manifest.json."""
        os.makedirs(self.dir, exist_ok=True)
        p = f'{self.dir}/manifest.json'
        man = {'label': self.label, 'runs': []}
        if os.path.exists(p):
            try:
                man = json.load(open(p))
            except Exception:
                pass
        man.setdefault('runs', []).append(opts)
        with open(p, 'w') as f:
            json.dump(man, f, indent=2, default=str)
