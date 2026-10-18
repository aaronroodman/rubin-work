---
study: camera_rotator_survey
topic: blocks
status: done
question: "Which source gives the camera rotator angle per visit, and how complete is it in ConsDB?"
answer: "Use ConsDB physical_rotator_angle, not visitInfo boresightRotAngle, which is the sky position angle of the boresight. Refined in aos/docs/frame_conventions.md."
uses: {}  # no rubin-work products; ConsDB, EFD and the Butler
started: 2026-09-02
---

# `camera_rotator_survey`: which source gives the camera rotator angle

> **Notebooks:** `notebooks/camera_rotator_consdb_survey.ipynb` (ConsDB coverage, wide and fast), `notebooks/camera_rotator_angle_check.ipynb` (per-night ConsDB against EFD against Butler `visitInfo`)

## Question

Which source gives the camera rotator angle per visit, and how complete is it in ConsDB?

## Current answer

Use ConsDB `physical_rotator_angle`, not `visitInfo.boresightRotAngle`, which is the sky position angle of the boresight (69.27 deg against a rotator 9.84 deg away on day_obs 20260420, seq_num 89). The rotator follows from `visitInfo` as rotTelPos = parallacticAngle − boresightRotAngle − 90 deg, which agrees with `physical_rotator_angle` to a mean of 0.2 deg with 0.26 deg scatter over a 90 deg sweep on day_obs 20260709. The current rule on which source to read is in `aos/docs/frame_conventions.md`.

## Log

### 2026-09-02 — backfilled from docs and history

Backfilled 2026-10-18 from the notebook headers, `aos/docs/frame_conventions.md` and the commits `76e9dac` to `9a6e272`. Done in the phase 5 triage. The survey also lists the science visits with no rotator angle from either ConsDB source (`rotator_unrecoverable_science.csv`, old tree `blocks/output/`). Both notebooks moved here and now write to this study's `output/`.
