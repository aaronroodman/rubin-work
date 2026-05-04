# PSF — Point Spread Function

PSF simulation, measurement, and analysis for the Rubin Observatory.

## Notebooks

| Notebook | Description | Created | Last Modified |
|----------|-------------|---------|---------------|
| `zernike_psf_moments.ipynb` | Simulate the Rubin PSF from annular Zernike wavefront coefficients (Z4–Z22) using GalSim, measure FWHM and ellipticity via HSM adaptive moments, and explore the coefficient space with an interactive widget. | 2026-02-24 | 2026-03-19 |
| `psf_moments_scatter.ipynb` | Scatter plots of PSF moments (FWHM, ixx, iyy, ixy, e1, e2) vs sequence number for a given night. Fetches per-CCD data from ConsDB via `PSFMomentsTable`. | 2026-03-18 | 2026-03-18 |
| `wavefront_to_psf.ipynb` | Generate focal-plane PSF mosaics from per-donut Zernike measurements. Computes median Zernikes per detector per visit, draws PSFs with GalSim, and produces one PNG per visit showing PSFs at raft positions with FWHM color-coding and ellipticity whiskers. | 2026-03-19 | 2026-03-19 |

## Data dependencies

- **zernike_psf_moments**: Standalone (GalSim, optionally `lsst.ts.wep`)
- **psf_moments_scatter**: Requires ConsDB access via `common/psf_moments_consdb.py` (run on RSP)
- **wavefront_to_psf**: Requires parquet output from `aos/intrinsics_mktable.ipynb`, plus `lsst.obs.lsst` camera geometry
