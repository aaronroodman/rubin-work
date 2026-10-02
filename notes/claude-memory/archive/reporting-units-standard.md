---
name: reporting-units-standard
description: "Aaron's standing requirement: every numerical quantity I cite must carry its units and a named quantity — no bare numbers"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 25f21140-87f5-47d9-b0ac-9fc8c2e9c154
  modified: 2026-09-03T23:55:11.850Z
---

**Every numerical value I report must carry (a) the NAME of the quantity and
(b) its units, or an explicit statement that it is dimensionless with the ratio
spelled out.** No bare numbers, ever — not in prose, tables, captions, or commit
messages.

**Why:** Aaron flagged (2026-09-03) two of my failures in one message —
"(Z5 ≈ 1.7e-3, Z16 ≈ 3e-4)" where he had to guess microns, and "the drift control
comes out at 0.93 — adjacent-pair inflation 7.9 vs block-mean 8.5" where he had
"no idea what units or quantity is being referenced". A number he cannot decode is
worse than no number: it silently invites a wrong physical interpretation, and in
this work the same symbol can be µm of wavefront, a dimensionless ratio, or a
correlation coefficient.

**How to apply — by category:**

| kind | required form | example |
|---|---|---|
| physical | value + unit | `Z5 coefficient sigma = 1.7e-3 µm` |
| ratio of like things | value + "dimensionless" + numerator/denominator named | `sigma_emp/sigma_RLM = 8.5 (dimensionless; empirical over formal coefficient error)` |
| correlation | statistic + both variables with units + n | `Pearson r = +0.57 (dimensionless) between \|\|r_null\|\| [µm] and \|a_31\| [µm], n=1126 visits` |
| chi2 | always as chi2/dof, with dof stated | `chi2_50/dof = 4.2 (dimensionless), dof = 76` |
| R^2 / variance explained | say which model, target, and CV scheme | `CV R2 = +0.10 (dimensionless), target \|\|r_null\|\| [µm], leave-night-out` |
| counts | "N of M" | `39 of 2838 effective independent donuts` |
| fraction / percent | of WHAT | `54% of (1-P)w POWER (not amplitude)` |
| angles / field | deg, and frame (OCS/CCS) | `field radius 1.75 deg, OCS` |

**Extra rules that bit me here:**
- Say **power vs amplitude** explicitly whenever quoting a fraction of a residual;
  they differ by a square and I have conflated them before.
- Tables: units go in the header row, once, not repeated per cell.
- A dimensionless ratio still needs its two operands named — "inflation 7.9" is
  useless, "sigma_adjacent/sigma_RLM = 7.9 (dimensionless)" is not.
- Units belong in code output too (print statements, axis labels, column names),
  not just in prose, so the artifact is self-describing.

Related: [[miw-coadd-retrieval-bias]] (where most of these numbers live).
