---
name: dp2-cloud-rsp-only
description: "Early DP2 catalogs are on the cloud RSP (data.lsst.cloud), not USDF"
metadata: 
  node_type: memory
  type: project
  originSessionId: 571e7a62-df89-49d3-bff8-492f55e0d31c
  modified: 2026-07-28T21:33:13.890Z
---

Early DP2 was released 2026-07-27 on the **cloud RSP at data.lsst.cloud** (portal, notebook, API) — it is **not** on the USDF RSP as of the early release. Querying `DiaSource` (or other DP2 catalogs) via the USDF TAP service fails with "Table [dp2.DiaSource] is not found in TapSchema". Run DP2 catalog work on data.lsst.cloud.

Early release = complete DP2 campaign **catalogs** + deep coadds; difference/visit **images** are deferred to full DP2 (Oct–Dec 2026). So the `DiaSource` catalog (~1e9 rows) is queryable now even though diff images aren't.

The exact TAP schema string was not confirmed from docs ("dp2" is a guess) — discover it at runtime via `SELECT schema_name FROM tap_schema.schemas` and `... FROM tap_schema.tables WHERE table_name LIKE '%DiaSource%'`.

Cloud-RSP work lives in its own repo **rubin-cloud** (github.com/aaronroodman/rubin-cloud, private), kept separate from rubin-work so USDF/slacrd-only code (Butler /repo/main, ConsDB, EFD, ts_ofc) is not dragged onto the cloud RSP. common/ helpers are duplicated there, not shared. The DP2 alert-focal-plane notebook is rubin-cloud/alerts/alerts_focal_plane_positions_v1.ipynb (was briefly in rubin-work/alerts, now removed). See [[usdf-access-slacrd]], [[transformed-efd-consdb]].
