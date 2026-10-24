# Viewers: danish masks and Batoid model geometry

Two self-contained HTML pages in `../../figures/`, built on the laptop with MacPorts
Python and Batoid 0.8.1 (2026-10-23).

| page | what it shows | built by |
|---|---|---|
| `pupil_mask_viewer.html` | The danish v4.02g and v4.02s obscuration masks in the stop plane (z 0.4765 m, field along +x) against field angle, compared with a Batoid `LSST_r_v4.02` trace at the giant configurations (spider removed from both): a pupil slice, the inner edge along −x against field angle, and a 3D stack of the mask circles. | `build_data.py`, `build_patch.py`, `pupil_mask_viewer_template.html` |
| `rubin_model_geometry.html` | Every surface and obscuration in `Rubin_v1000_r` and `LSST_r_v4.02` in the telescope frame, in selectable groups, with meridional ray fans at 0 and 1.7 deg through the nominal models. Shows what v4.02's 165 `CameraSilhouette` slices and v1000's `CameraBody` disk are. | `build_models.py`, `model_geometry_template.html` |

## Inputs

- `LSST_r_v4.02.yaml`: the rubin-optical-model repo (`aaronroodman/rubin-optical-model`, commit
  e2a1e54). Laptop `~/Astrophysics/Claude/rubin-optical-model/`, S3DF
  `~/notebooks/rubin-optical-model/`.
- `Rubin_v1000_r.yaml` (here): copied from Batoid 0.9.0's `batoid/data/LSST/` in the
  `w_2026_42` stack. The laptop's Batoid 0.8.1 does not ship it.
- The masks: `output/20261019_pupil2/masks/` (v4.02g) and `output/20261023_pupil6/masks/` (v4.02s).

## Rebuilding

```bash
cd /tmp/viewer_build && /opt/local/bin/python3 <study>/code/viewers/build_data.py viewer_data.json && /opt/local/bin/python3 <study>/code/viewers/build_patch.py viewer_data.json
```

Then substitute the JSON for `__DATA__` in `pupil_mask_viewer_template.html`.

For the model page, run `build_models.py` beside `Rubin_v1000_r.yaml`. Then substitute its
`model_data.json` for `__DATA__` in `model_geometry_template.html`, and the camera-body
table rows for `__CBTABLE__`.

The mask paths in `build_data.py` point at the laptop's mirror of the study output.

## Checks made

- The mask circle center lies along the field direction: sign +1 against Batoid. With the
  other sign, the disagreement grows from 0.0010 to 0.19 of the lit pupil at 1.7 deg.
- On the whole pupil at 1.70 deg, v4.02g disagrees with Batoid on 0.56 % of the lit area
  and v4.02s on 0.10 % (extra-focal giant configuration).
