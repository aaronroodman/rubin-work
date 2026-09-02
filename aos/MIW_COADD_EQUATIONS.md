# MIW nomenclature and the coadd-vs-MIW residual — equation level

**Purpose.** Fix notation for the Measured Intrinsic Wavefront (MIW) construction and
then *derive* what the per-block coadd-minus-MIW difference actually equals, because
that derivation determines which quantity is the correct regressor in the
coadd-vs-MIW correlation study (`aos/code/recompute_coadd_metrics.py`).

**Relation to existing docs.** The iterative MIW method is *already* written at
equation level in [`notes/aos-measured-intrinsics/note.md`](../notes/aos-measured-intrinsics/note.md)
§"MIW method" (steps 1–6). That note is the primary reference and is not repeated
here beyond the notation table. `ts_intrinsic_wavefront`'s `README.rst`,
`doc/index.rst`, `calibration/README.md` and `pipelines/README.md` describe the
pipeline and config but carry no equations. What is **new here** is §3–§5: the
fixed-point form of the iteration, the gauge freedom it contains, and the
consequence that the *removed* u-modes are the wrong first-order regressor.

All numbers below are for `pathA_50_34_i_5rot` (i-band, 50 DOF / 34 v-modes) as
configured in `ts_intrinsic_wavefront/pipelines/mi_config.yaml`.

---

## 1. Notation

| symbol | meaning | value / shape here |
|---|---|---|
| $\vec\theta$ | field angle, OCS | $\hat\theta=\vec\theta/\theta_{\rm fp}$, $\theta_{\rm fp}=1.8^\circ$ |
| $j$ | pupil (annular) Zernike Noll index | $j\in\{4..26\}\setminus\{20,21\}$, $n_j=21$ |
| $k$ | focal-plane (circular) Zernike index | $k=1..6$ (`k_min`/`k_max`), $n_k=6$ |
| $Z_j^{\rm meas}(\vec\theta)$ | Danish per-donut annular Zernike | per intra/extra FAM pair |
| $\mathcal{Z}_k(\hat\theta)$ | focal-plane Zernike basis | piston, tip/tilt, focus, 2 astig |
| $w_{kj}$ | Double-Zernike (DZ) coefficient | flattened $\mathbf{w}$, $n_{kj}=n_k n_j=126$ |
| $\mathbf{S}$ | OFC sensitivity, $S_{(kj),i}=\partial w_{kj}/\partial\mathrm{DoF}_i$ | $126\times 50$ |
| $\mathbf{N}$ | diagonal geom-mean DOF normalization | $50\times 50$ |
| $\hat{\mathbf S}=\mathbf{S}\mathbf{N}$ | normalized sensitivity | $126\times 50$ |
| $\mathbf{U},\boldsymbol\Sigma,\mathbf{V}$ | SVD $\hat{\mathbf S}=\mathbf U\boldsymbol\Sigma\mathbf V^\top$ | $\mathbf U$ is $126\times 50$ |
| $\mathbf{U}_{\rm eff}$ | kept left-singular vectors | $126\times 34$ (`n_keep=34`) |
| $\mathbf{P}=\mathbf U_{\rm eff}\mathbf U_{\rm eff}^\top$ | projector onto the corrected subspace | rank 34 of 126 |
| $a_m$ | **u-mode amplitude**, $a_m=\mathbf u_m^\top\mathbf w$ | µm of wavefront, $m=1..34$ |
| $c_m=a_m/\sigma_m$ | **v-mode amplitude** (normalized DOF coords) | dimensionless |
| $I(\vec\theta)$ | intrinsic-wavefront estimate (the MIW) | $73\times73$ grid, $\pm1.8^\circ$ |

Code correspondence: $\mathbf U_{\rm eff}$ = `OFCSvd.U_eff`; $a_m$ =
`OFCSvd.project_amplitudes(W)` $=\mathbf{W}\mathbf U_{\rm eff}$; $c_m$ =
`OFCSvd.vmodes`; row order of $\mathbf w$ is $(k-k_{\min})n_j+j_{\rm idx}$
(`OFCSvd.kj_grid`, $k$ outer / $j$ inner).

**Note on "u-mode" vs "v-mode".** $\mathbf u_m$ (left-singular, wavefront side)
and $\mathbf v_m$ (right-singular, DOF side) are paired by $\sigma_m$. The stored
per-block quantity is $a_m$ — the wavefront-side amplitude in µm, which keeps all
34 modes on one physical scale. $c_m=a_m/\sigma_m$ is the DOF-space amplitude and
is what the mode-truncation ordering refers to. `block_umodes()` stores $a_m$.

---

## 2. Subspace bookkeeping (matters for §4)

$\hat{\mathbf S}$ is $126\times 50$, so $\mathbf U$ has only **50** columns and the
DZ space splits three ways:

| subspace | dim | meaning | fate in the MIW |
|---|---|---|---|
| $\mathrm{span}(\mathbf U_{\rm eff})$ | 34 | reachable **and** kept | **removed** |
| $\mathrm{span}(\mathbf u_{35..50})$ | 16 | reachable but discarded (ill-conditioned v-modes) | **survives** |
| $\mathrm{null}(\hat{\mathbf S}^\top)$ | 76 | not reachable by any of the 50 DOF | **survives** |

So $\mathbf 1-\mathbf P$ is 92-dimensional, of which 16 dimensions are *physically
achievable optical states that the 34-mode truncation chooses not to correct*.
Those 16 are the AOS-relevant part of the surviving wavefront.

---

## 3. The iteration as a fixed point

Per the note's steps 4–6 (`n_iter=3`, `build_measured_intrinsic_uconstrained`),
for a visit set $\mathcal V$ and iteration $n$:

$$
\mathbf w_v^{(n)}=\mathcal F\!\left[Z_v^{\rm meas}-I^{(n-1)}\right],\qquad
I^{(n)}(\vec\theta)=\Big\langle\, Z_v^{\rm meas}(\vec\theta)-\textstyle\sum_k(\mathbf P\mathbf w_v^{(n)})_{kj}\,\mathcal Z_k(\hat\theta)\,\Big\rangle^{\rm bin}_{v\in\mathcal V}
$$

where $\mathcal F$ is the per-visit least-squares fit onto the $\mathcal Z_k$ basis
and $\langle\cdot\rangle^{\rm bin}$ is the per-cell **median** over all good donuts
of all good visits. $I^{(0)}=$ batoid design intrinsic. Note the **projected**
$\mathbf P\mathbf w$ is subtracted while the **raw** $\mathbf w$ is what gets stored
(`fit_rows_raw`), so $a_m$ is measured from the raw fit.

At the fixed point $I^{(n)}\to I_{\mathcal V}$, drop the iteration index:

$$
\boxed{\;I_{\mathcal V}=\big\langle Z_v^{\rm meas}-(\mathbf P\mathbf w_v)\!\cdot\!\mathcal Z\big\rangle_{\mathcal V},\qquad
\mathbf w_v=\mathcal F\big[Z_v^{\rm meas}-I_{\mathcal V}\big]\;}
$$

### 3.1 What the fixed point contains

Model the truth as an intrinsic plus a DZ-representable optical state,
$Z_v^{\rm meas}=I_{\rm true}+\mathbf W_v\!\cdot\!\mathcal Z+n_v$, with $\mathbf W_v$
the true state's DZ coefficients and $n_v$ retrieval noise. Since $\mathcal F$ is
the identity on DZ-representable fields, write
$\boldsymbol\Delta\equiv\mathcal F[I_{\rm true}-I_{\mathcal V}]$ and assume
$\mathcal F[n_v]$ averages away. Then $\mathbf w_v=\boldsymbol\Delta+\mathbf W_v$ and

$$
I_{\mathcal V}=I_{\rm true}+(\mathbf 1-\mathbf P)\langle\mathbf W\rangle_{\mathcal V}\!\cdot\!\mathcal Z-\mathbf P\boldsymbol\Delta\!\cdot\!\mathcal Z .
$$

Applying $\mathcal F$ to this expression gives $(\mathbf 1-\mathbf P)\boldsymbol\Delta=-(\mathbf 1-\mathbf P)\langle\mathbf W\rangle_{\mathcal V}$,
which fixes the $(\mathbf 1-\mathbf P)$ part of $\boldsymbol\Delta$ but leaves
$\mathbf P\boldsymbol\Delta$ **undetermined**.

> **Gauge freedom.** Any $\mathbf P$-representable field pattern can be moved
> between "intrinsic" and "optical state" without changing the data. The iteration
> resolves it only through its starting point $I^{(0)}=$ batoid. Two builds started
> from the same batoid intrinsic land in the same gauge, so $\mathbf P\boldsymbol\Delta$
> cancels in differences — but a build started elsewhere is **not** comparable.
> This is also why the MIW lacks an absolute $Z_{1..3}$ (piston/tilt) reference.

---

## 4. The coadd-vs-MIW residual — the key result

`run_coadd_blocks_miw.py` runs the *same* fixed point per contiguous block $b$,
giving $I_b$; the reference is the pooled build $I_{\mathcal B}$ (the 16 build
blocks). Subtracting two fixed points in the same gauge:

$$
\boxed{\;I_b-I_{\mathcal B}\;\simeq\;(\mathbf 1-\mathbf P)\big[\langle\mathbf W\rangle_b-\langle\mathbf W\rangle_{\mathcal B}\big]\!\cdot\!\mathcal Z\;}
$$

**The residual is driven by the $(\mathbf 1-\mathbf P)$ — discarded — part of the
state difference, not the $\mathbf P$ part.** The $\mathbf P$ part is exactly what
both builds subtract, so it cancels at first order.

This **corrects** the heuristic I used when first setting up Tier 1
($I_b-I_{\mathcal B}=\sum_i\varepsilon_i(a_{b,i}-\langle a_i\rangle_{\mathcal B})M_i$):
that expression is not the first-order term, it is one of the second-order ones.

### 4.1 Consequences for the regressors

The stored u-modes $a_m=\mathbf u_m^\top\mathbf w$ span *precisely the subspace that
cancels*. They can therefore only enter through second-order channels:

1. **Sensitivity-matrix error.** If the true response is $\mathbf S+\delta\mathbf S$,
   then $\mathbf P$ is the wrong projector and a term
   $\sim(\mathbf 1-\mathbf P)\,\delta\mathbf S\,\mathbf N\,\mathbf V\boldsymbol\Sigma^{-1}\Delta\mathbf a$
   leaks in — first order in $\delta\mathbf S$, linear in $\Delta a_m$. This is the
   only channel that makes $\Delta a_m$ a legitimate (and interesting) regressor: a
   significant coefficient measures a **gain error in mode $m$'s removal**.
2. **Nonlinearity** of the true DOF→wavefront response (the DZ fit is linear).
3. **Correlation** between the $\mathbf P$ and $(\mathbf 1-\mathbf P)$ parts of the
   state across blocks — physically likely, since one thermal state drives both.
   This makes $\Delta a_m$ a *proxy* rather than a cause, and is why the partial
   correlation controlling for `z_gradient` must always be reported alongside.

Empirically (rebin 3, $n=221$, 16 build blocks) this is what the data show:
u-mode-only ML $R^2=-0.33$ and env+u-mode $R^2=+0.64$ vs environmental-only
$+0.62$ — i.e. $\Delta a_m$ adds **essentially nothing** beyond the environmental
telemetry, exactly as the boxed equation predicts.

### 4.2 The quantity that *is* first order — and is not currently saved

$$
\mathbf r_v\;\equiv\;(\mathbf 1-\mathbf P)\,\mathbf w_v
$$

the **discarded** DZ coefficients (the note's $w^{\rm resid}$). Per block,
$\langle\mathbf r\rangle_b$ is the first-order driver of $I_b-I_{\mathcal B}$.
It is 92-dimensional; useful reductions:

- $\|\langle\mathbf r\rangle_b-\langle\mathbf r\rangle_{\mathcal B}\|$ — a scalar distance;
- amplitudes on $\mathbf u_{35..50}$ — the **16 reachable-but-discarded** modes, the
  physically meaningful and best-motivated set (§2);
- amplitudes on the leading directions of $\mathrm{null}(\hat{\mathbf S}^\top)$.

`run_coadd_blocks_miw.py` currently stores only $a_m$ (34 kept) in
`block_grids.npz`. Adding $\mathbf r$ — or equivalently the full raw
$\mathbf w_v$ (126 per visit), from which both $\mathbf P\mathbf w$ and
$(\mathbf 1-\mathbf P)\mathbf w$ follow — is a one-line addition to
`block_umodes()`/the npz save and requires an RSP rerun. **This is the single
highest-value change to the coadd study.**

---

## 5. The reference state $\langle a_i\rangle_{\mathcal B}$

For the (second-order) gain-error test, $\Delta a_{b,i}=a_{b,i}-\langle a_i\rangle_{\mathcal B}$
needs the build-set reference. Two sources:

- **exact** — project the build's own per-visit raw DZ fits
  (`pathA_50_34_i_5rot/fits.parquet`, 1126 rows of `z1toz6_*` columns) onto the same
  $\mathbf U_{\rm eff}$, over exactly the build-selected visits, with the build's own
  central statistic (median; strictly the MIW cell is a per-donut median, so the
  effective centre is donut-weighted and can drift slightly cell to cell — second
  order, ignored). Pass via `--umode-ref`.
- **approximate** (laptop default) — `n_visits`-weighted mean of $a_i$ over the 16
  `build_used` blocks. A median-of-medians; fine for a first look.

**No test lives at the centre.** $\langle\Delta a\rangle_{\mathcal B}=0$ identically
by construction, so build blocks straddle zero as a definition, not a check. The
meaningful scales are the build envelope $\sigma_{{\rm build},i}$ (per-block spread)
and the reference's own resolution floor $\mathrm{SEM}_i=\sigma_{{\rm build},i}/\sqrt{16}$
($=0.25\,\sigma$), below which a displacement is indistinguishable from the
calibration state. Individual build blocks scatter by $\sigma$, **not** by SEM — so
$|\Delta a|/\mathrm{SEM}\sim 4\times$ the z-score and reaching ~13 is expected, not a
failure.

---

## 6. Summary of what changes in the analysis

1. First-order driver is $(\mathbf 1-\mathbf P)\mathbf w$, **not** the u-modes → save
   it (RSP rerun) and make it the primary regressor.
2. Within it, prioritize the **16 reachable-but-discarded** modes $\mathbf u_{35..50}$.
3. Keep $\Delta a_m$ as the **sensitivity-gain-error** test, second order, always with
   the `z_gradient` partial correlation.
4. Report $\Delta a$ in units of $\sigma_{\rm build}$ with the SEM floor marked.
5. Both builds must start from the same $I^{(0)}$ (batoid) or the $\mathbf P$ gauge
   does not cancel.
