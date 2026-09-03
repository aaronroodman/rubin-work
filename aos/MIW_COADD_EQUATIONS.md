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
| $\mathbf W_v$ | **true optical-state DZ coefficients** of visit $v$ — the physical misalignment/figure state expressed in the same DZ basis. The thing $\mathbf P$ is meant to remove. | 126, µm |
| $\mathcal R$ | reconstruction, coefficients $\to$ field function: $\mathcal R[\mathbf w](\vec\theta)=\sum_k w_{kj}\mathcal Z_k(\hat\theta)$ | linear |
| $\mathcal F$ | per-visit least-squares fit, field function $\to$ coefficients | linear |
| $\mathbf{S}$ | OFC sensitivity, $S_{(kj),i}=\partial w_{kj}/\partial\mathrm{DoF}_i$ | $126\times 50$ |
| $\mathbf{N}$ | diagonal geom-mean DOF normalization | $50\times 50$ |
| $\hat{\mathbf S}=\mathbf{S}\mathbf{N}$ | normalized sensitivity | $126\times 50$ |
| $\mathbf{U},\boldsymbol\Sigma,\mathbf{V}$ | SVD $\hat{\mathbf S}=\mathbf U\boldsymbol\Sigma\mathbf V^\top$ | $\mathbf U$ is $126\times 50$ |
| $\mathbf{U}_{\rm eff}$ | kept left-singular vectors | $126\times 34$ (`n_keep=34`) |
| $\mathbf{P}=\mathbf U_{\rm eff}\mathbf U_{\rm eff}^\top$ | projector onto the corrected subspace | rank 34 of 126 |
| $a_m$ | **u-mode amplitude**, $a_m=\mathbf u_m^\top\mathbf w$ | µm of wavefront, $m=1..34$ |
| $c_m=a_m/\sigma_m$ | **v-mode amplitude** (normalized DOF coords) | dimensionless |
| $I_{\mathcal V}(\vec\theta)$ | **the MIW** — the converged intrinsic-wavefront estimate built from visit set $\mathcal V$ | $73\times73$ grid, $\pm1.8^\circ$ |
| $I_{\mathcal B}$, $I_b$ | the **calibration MIW** (16 build blocks $\mathcal B$) and one block's **coadd** ($b$) — both are $I_{\mathcal V}$ for different $\mathcal V$ | same grid |
| $I_{\rm true}$ | the true intrinsic wavefront of the ideally-aligned system | — |

Code correspondence: $\mathbf U_{\rm eff}$ = `OFCSvd.U_eff`; $a_m$ =
`OFCSvd.project_amplitudes(W)` $=\mathbf{W}\mathbf U_{\rm eff}$; $c_m$ =
`OFCSvd.vmodes`; row order of $\mathbf w$ is $(k-k_{\min})n_j+j_{\rm idx}$
(`OFCSvd.kj_grid`, $k$ outer / $j$ inner).

### 1.1 Properties of $\mathcal F$ and $\mathcal R$ (used throughout §3)

Both are linear. $\mathcal R$ maps a coefficient vector to a field-dependent
wavefront; $\mathcal F$ fits a field-dependent wavefront back to coefficients. The
two properties the derivation needs:

$$
\mathcal F\mathcal R=\mathbb 1 \quad\text{(on coefficient space)},\qquad
\mathcal R\mathcal F=\Pi \quad\text{(projection onto DZ-representable functions)}.
$$

$\mathcal F\mathcal R=\mathbb 1$ is the statement that fitting a field that *is*
exactly a DZ sum returns its coefficients. $\mathcal R\mathcal F=\Pi\neq\mathbb 1$:
a general field — $I_{\rm true}$ and $I_{\mathcal V}$ included — is **not**
DZ-representable, and $\mathcal F$ applied to it returns its best-fit projection,
discarding the rest. Nothing below assumes otherwise.

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

So $\mathbb 1-\mathbf P$ is 92-dimensional, of which 16 dimensions are *physically
achievable optical states that the 34-mode truncation chooses not to correct*.
Those 16 are the AOS-relevant part of the surviving wavefront.

---

## 3. The iteration as a fixed point

Per the note's steps 4–6 (`n_iter=3`, `build_measured_intrinsic_uconstrained`),
for a visit set $\mathcal V$ and iteration $n$:

$$
\mathbf w_v^{(n)}=\mathcal F\!\left[Z_v^{\rm meas}-I^{(n-1)}\right],\qquad
I^{(n)}=\Big\langle\, Z_v^{\rm meas}-\mathcal R\big[\mathbf P\,\mathbf w_v^{(n)}\big]\,\Big\rangle^{\rm bin}_{v\in\mathcal V}
$$

where $\langle\cdot\rangle^{\rm bin}$ is the per-cell **median** over all good donuts
of all good visits, on the $73\times73$ grid. $I^{(0)}=$ batoid design intrinsic.
Note the **projected** $\mathbf P\mathbf w$ is subtracted while the **raw**
$\mathbf w$ is what gets stored (`fit_rows_raw`), so $a_m$ is measured from the raw
fit.

At the fixed point $I^{(n)}\to I_{\mathcal V}$, drop the iteration index:

$$
\boxed{\;I_{\mathcal V}=\Big\langle Z_v^{\rm meas}-\mathcal R\big[\mathbf P\,\mathbf w_v\big]\Big\rangle_{\mathcal V},\qquad
\mathbf w_v=\mathcal F\big[Z_v^{\rm meas}-I_{\mathcal V}\big]\;}
$$

These two equations are coupled: the fit is made against the very intrinsic
estimate that the fit's own output defines. §3.1 unpacks what that self-consistency
forces.

### 3.1 What the fixed point contains

> **Why this section exists.** It establishes exactly one thing needed downstream:
> that the coadd-vs-MIW difference is driven by the $(\mathbb 1-\mathbf P)$ part of
> the state and **not** by the u-modes (§4). Steps 1–3 are the algebra for that;
> Step 4 disposes of the starting-point/gauge question, which turns out **not** to
> affect the comparison. A reader who accepts §4's boxed result can skip to it.

**Assumptions.** (i) The data model is
$$Z_v^{\rm meas}=I_{\rm true}+\mathcal R[\mathbf W_v]+n_v,$$
i.e. the true *optical state* is DZ-representable in the **fitted** basis, while
$I_{\rm true}$ need not be. This is only approximate: the state acts through
$\mathbf w$ by construction of the sensitivity matrix, but ts_ofc's DZ matrix
carries 31 field orders whereas the build fits only $k=1..6$, so the state's
field content above $k=6$ is *not* representable and behaves like additional
$I_{\rm true}$. (ii) The per-cell **median** is replaced by a **mean** so that
$\langle\cdot\rangle$ is linear. (iii) Retrieval noise averages away,
$\langle\mathcal F[n_v]\rangle\approx0$.

All three are approximations. (i) and (ii) are the ones worth revisiting — (i)
because the $k>6$ state leakage is a genuine physical term, not just algebra.

**Step 1 — what each visit's fit returns.** Substitute the data model into
$\mathbf w_v=\mathcal F[Z_v^{\rm meas}-I_{\mathcal V}]$ and use linearity:

$$
\mathbf w_v=\underbrace{\mathcal F\big[I_{\rm true}-I_{\mathcal V}\big]}_{\textstyle\equiv\,\boldsymbol\Delta}
+\underbrace{\mathcal F\mathcal R[\mathbf W_v]}_{=\;\mathbf W_v}
+\;\mathcal F[n_v]
\;\;\Longrightarrow\;\;
\langle\mathbf w\rangle_{\mathcal V}=\boldsymbol\Delta+\langle\mathbf W\rangle_{\mathcal V}.
$$

Two things to be clear about here. $\boldsymbol\Delta$ is a **definition**, not a
claim: it is simply the coefficient vector $\mathcal F$ returns when handed the
field $I_{\rm true}-I_{\mathcal V}$, and requires no representability assumption —
this is where the earlier draft misplaced its justification. The property
$\mathcal F\mathcal R=\mathbb 1$ is needed only for the **middle** term, the
optical state, which *is* DZ-representable by assumption (i).

$\boldsymbol\Delta$ carries no $v$ label, but strictly $\mathcal F$ is per-visit —
each visit has its own donut positions, so $\mathcal F_v$ and hence
$\boldsymbol\Delta_v$ differ slightly. Treating $\boldsymbol\Delta$ as common to the
set assumes equivalent field sampling across visits; the residual
sampling-dependence is one of the second-order terms collected in §4.1.

**Step 2 — what the MIW then equals.** Put Step 1 into the fixed-point definition
$I_{\mathcal V}=\langle Z_v^{\rm meas}\rangle-\mathcal R[\mathbf P\langle\mathbf w\rangle]$:

$$
I_{\mathcal V}
= I_{\rm true}+\mathcal R\big[\langle\mathbf W\rangle_{\mathcal V}\big]
-\mathcal R\big[\mathbf P(\boldsymbol\Delta+\langle\mathbf W\rangle_{\mathcal V})\big]
= I_{\rm true}+\mathcal R\big[(\mathbb 1-\mathbf P)\langle\mathbf W\rangle_{\mathcal V}-\mathbf P\boldsymbol\Delta\big].
\tag{$\ast$}
$$

Already the main message is visible: the MIW differs from the truth by the
**uncorrected** part of the mean state, $(\mathbb 1-\mathbf P)\langle\mathbf W\rangle$,
plus a term $\mathbf P\boldsymbol\Delta$ still to be pinned down.

**Step 3 — close the loop on $\boldsymbol\Delta$.** Rearrange $(\ast)$ to
$I_{\rm true}-I_{\mathcal V}=-\mathcal R\big[(\mathbb 1-\mathbf P)\langle\mathbf W\rangle_{\mathcal V}-\mathbf P\boldsymbol\Delta\big]$
and apply $\mathcal F$ to both sides. The left side is $\boldsymbol\Delta$ by
definition; on the right $\mathcal F\mathcal R=\mathbb 1$ applies because the
argument is now a DZ *coefficient* vector:

$$
\boldsymbol\Delta=-(\mathbb 1-\mathbf P)\langle\mathbf W\rangle_{\mathcal V}+\mathbf P\boldsymbol\Delta .
$$

Split this by projecting with $(\mathbb 1-\mathbf P)$ and with $\mathbf P$, using
$(\mathbb 1-\mathbf P)\mathbf P=0$ and $\mathbf P^2=\mathbf P$:

$$
(\mathbb 1-\mathbf P)\boldsymbol\Delta=-(\mathbb 1-\mathbf P)\langle\mathbf W\rangle_{\mathcal V},
\qquad
\mathbf P\boldsymbol\Delta=\mathbf P\boldsymbol\Delta .
$$

The first fixes the uncorrectable half of $\boldsymbol\Delta$. The second is a
tautology: **$\mathbf P\boldsymbol\Delta$ is undetermined by the fixed-point
condition.**

> **Gauge freedom.** Any $\mathbf P$-representable field pattern can be moved
> between "intrinsic" and "optical state" without changing the data at all, so the
> fixed-point equations cannot separate them. This is the same degeneracy that
> leaves the MIW without an absolute $Z_{1..3}$ (piston/tilt) reference.

**Step 4 — which fixed point? Convergence does not answer this.** Run the
iteration explicitly from $I^{(0)}=I_{\rm batoid}$. Iteration 1 gives
$\mathbf w_v^{(1)}=\mathcal F[I_{\rm true}-I_{\rm batoid}]+\mathbf W_v+\mathcal F[n_v]$, so

$$
I^{(1)}=I_{\rm true}+\mathcal R\big[(\mathbb 1-\mathbf P)\langle\mathbf W\rangle_{\mathcal V}\big]-\mathcal R[\mathbf G],
\qquad
\mathbf G\equiv\mathbf P\,\mathcal F\big[I_{\rm true}-I_{\rm batoid}\big].
$$

Feeding $I^{(1)}$ back through Step 1 gives
$\mathbf w_v^{(2)}=\mathbf W_v-(\mathbb 1-\mathbf P)\langle\mathbf W\rangle_{\mathcal V}+\mathbf G+\mathcal F[n_v]$,
and since $\mathbf P(\mathbb 1-\mathbf P)=0$ and $\mathbf P\mathbf G=\mathbf G$,

$$
I^{(2)}=I_{\rm true}+\mathcal R\big[(\mathbb 1-\mathbf P)\langle\mathbf W\rangle_{\mathcal V}\big]-\mathcal R[\mathbf G]=I^{(1)}.
$$

So in the linearized model the iteration reaches a fixed point after **one** step
and then sits there — but the point it sits at **retains $\mathbf G$**, i.e. it
depends on $I^{(0)}$. Step 3 already showed why: $\mathbf P\boldsymbol\Delta$ is
unconstrained, so the fixed points form a *manifold* over the 34-dimensional
$\mathbf P$ subspace, and $I^{(0)}$ selects which point on it you land on.
Consistency check: starting from $I^{(0)}=I_{\mathcal V}$ reproduces
$I_{\mathcal V}$ exactly, as a fixed point must.

> **This is why iteration $2\approx3$ is not evidence of start-independence.**
> Agreement between successive iterations shows you have *landed* on the manifold
> — which is exactly what validates using the fixed-point equations above — but
> says nothing about *which* point. A contractive map toward a unique answer and a
> map that lands immediately on a fixed-point manifold both produce
> "iteration 3 = iteration 2".

**How much of the MIW is actually gauge?** Only the part of
$I_{\rm batoid}-I_{\rm true}$ that is simultaneously (a) inside the fitted basis
$k\le6$ (field radial order $\le2$) and (b) in $\mathrm{span}(\mathbf U_{\rm eff})$.
The MIW's headline excess over the batoid design — $\sim0.1\,\mu$m of astig/coma at
**field radial order $n=3$–5** ([smatrix/MIW_astig_coma_investigation.md](../smatrix/MIW_astig_coma_investigation.md))
— is *outside* the $k\le6$ fit basis entirely, so it is **not** gauge and cannot be
an artifact of the batoid start. The gauge worry is confined to low field order.

**Consequence for this analysis: none.** Both $I_b$ and $I_{\mathcal B}$ start from
the same $I_{\rm batoid}$, so $\mathbf G$ is common and cancels in the difference.
The gauge matters only if one wants to interpret $I_{\mathcal V}$ *absolutely*, or
to compare builds started differently.

**Decisive test (cheap).** Rerun one build with $I^{(0)}=I_{\rm batoid}+\mathcal R[\delta]$
for a deliberate $\mathbf P$-representable $\delta$ (say a $0.1\,\mu$m v-mode
pattern at $k\le6$). The algebra above predicts the converged MIW shifts by
$-\mathcal R[\mathbf P\delta]$; genuine start-independence predicts no shift. One
build rerun settles it.

---

## 4. The coadd-vs-MIW residual — the key result

`run_coadd_blocks_miw.py` runs the *same* fixed point per contiguous block $b$,
giving $I_b$; the reference is the pooled build $I_{\mathcal B}$ (the 16 build
blocks). Differencing $(\ast)$ for the two sets, with $\mathbf G$ shared (Step 4):

$$
\boxed{\;I_b-I_{\mathcal B}\;\simeq\;\mathcal R\Big[(\mathbb 1-\mathbf P)\big(\langle\mathbf W\rangle_b-\langle\mathbf W\rangle_{\mathcal B}\big)\Big]\;}
$$

Both $I_{\rm true}$ and the gauge term $\mathcal R[\mathbf G]$ drop out; what
remains is the uncorrected part of the state difference.

**The residual is driven by the $(\mathbb 1-\mathbf P)$ — discarded — part of the
state difference, not the $\mathbf P$ part.** The $\mathbf P$ part is exactly what
both builds subtract, so it cancels at first order.

This **corrects** the heuristic I used when first setting up Tier 1
($I_b-I_{\mathcal B}=\sum_i\varepsilon_i(a_{b,i}-\langle a_i\rangle_{\mathcal B})M_i$):
that expression is not the first-order term, it is one of the second-order ones.

### 4.1 Consequences for the regressors

The stored u-modes $a_m=\mathbf u_m^\top\mathbf w$ span *precisely the subspace that
cancels*. They can therefore only enter through second-order channels:

1. **Field-order truncation of the fit basis — the dominant channel.** The build
   fits only $k=1..6$, but ts_ofc's DZ sensitivity carries **31** field orders. A
   DOF direction excited during a block produces wavefront at *all* field orders;
   only its $k\le6$ projection is fitted, and therefore only that part is
   subtracted. The $k>6$ part is **invisible to the fit and lands in the MIW**.
   Writing the state's field contribution as $g_v$ and $\Pi=\mathcal R\mathcal F$,

   $$
   I_{\mathcal V}=I_{\rm true}+\big\langle(\mathbb 1-\Pi)g_v\big\rangle_{\mathcal V}
   +\mathcal R\big[(\mathbb 1-\mathbf P)\langle\mathbf W\rangle_{\mathcal V}\big]-\mathcal R[\mathbf G].
   $$

   Because $(\mathbb 1-\Pi)g_v$ and $\mathbf W_v=\mathcal F[g_v]$ are driven by the
   **same** DOF amplitude, $a_m$ is a *linear proxy* for the un-removed $k>6$
   content of mode $m$ — with a computable per-mode weight (§4.3). This is a
   software/basis effect: it is present no matter how accurate $\mathbf S$ is.
2. **Nonlinearity** of the true DOF→wavefront response (the DZ fit is linear).
3. **Correlation** between the $\mathbf P$ and $(\mathbb 1-\mathbf P)$ parts of the
   state across blocks — physically likely, since one thermal state drives both.
   This makes $\Delta a_m$ a *proxy* rather than a cause, and is why the partial
   correlation controlling for `z_gradient` must always be reported alongside.
4. **Assumption violations from §3.1**: median-vs-mean (ii) and per-visit sampling
   differences in $\mathcal F_v$ — each makes $\mathbf G$ only approximately
   build-independent, leaving a residual that need not be orthogonal to $\Delta a_m$.

> **Not a channel: sensitivity-matrix error.** An earlier draft claimed a
> $\delta\mathbf S$ "gain error in mode $m$'s removal". That is **wrong**. The
> removal is done entirely **in software**: the build subtracts
> $\mathcal R[\mathbf P\mathbf w_v]$, exactly the quantity it just fitted, from the
> measured wavefront. No hardware applies anything, so there is no gain to be in
> error. The boxed result of §4 holds for *whatever* $\mathbf P$ is — an inaccurate
> $\mathbf S$ changes *which* subspace is removed, not whether the removal is
> faithfully executed. $\delta\mathbf S$ matters for applying corrections to the
> telescope; it does not enter this comparison.

### 4.3a The $k\le6$ truncation is an inconsistency with how the OFC corrects

`ts_ofc`'s `SensitivityMatrix.evaluate()` evaluates galsim DoubleZernikes at the
requested field angles, summing **all 31 field terms**. So a closed-loop correction
computed from the corner WFS and applied to hardware removes the DOF change's
*entire* field dependence. The MIW build does not: `_apply_uconstraint` returns
coefficients on `kj_grid` ($k\le6$ only) and `_dz_contrib_from_params` reconstructs
from just those, so **only the $k\le6$ part of the estimated state is subtracted.**

Two checks (`code/check_k_truncation.py`):

- **$k\le6$ *is* sufficient to specify the state.** $\hat{\mathbf S}_{k\le6}$
  (126×50) has full rank 50, condition number 1.12e4 vs 0.998e4 for $k\le30$; the
  top-50 v-mode subspace is *identical* (max principal angle 0.00°) and the top-34
  agrees to 4.9° (28/34 modes align >0.99; the rest are an ordering swap at the
  near-degenerate $\sigma_{34}/\sigma_{35}$ boundary).
- **but the $k>6$ response of that state is never subtracted.** Fraction of each
  mode's *total* wavefront power left behind: modes 1–7 ~0.0000, modes 8–27
  0.0007–0.001, **modes 28–34 0.086–0.179**. Median 0.0013.

**Because the first holds, the second is fixable.** The DOF are fully determined by
the $k\le6$ fit, so the principled reconstruction is: recover
$\mathbf d=\mathbf N\mathbf V\boldsymbol\Sigma^{-1}\mathbf U_{\rm eff}^\top\mathbf w$,
then subtract $\mathbf S_{\rm full}\mathbf d$ over all 31 field orders instead of
re-expanding only the fitted $k\le6$ coefficients. (Identity worth noting:
$\mathbf S_{k\le6}\mathbf d=\mathbf P\mathbf w$ exactly, so the present code is
precisely the $k\le6$ half of the right answer.)

**How much would that change the MIW?** For the build's own mean state the
un-subtracted part is $\|w\|_{k>6}=0.022\,\mu$m over all 21 pupil Zernikes — per
Zernike, **11.5% / 10.1%** of the MIW's Z5/Z6 amplitude and **1.5% / 2.9%** of
Z7/Z8. Real, worth fixing, not transformative.

**But it does not explain the MIW astigmatism** (`code/analyze_miw_field_order.py`,
`output/miw_k_gt6_leakage.pdf`). Map-to-map correlation of the un-subtracted $k>6$
term against the MIW: Z5 **+0.19**, Z6 **−0.02** — right order of magnitude, wrong
shape. Against coma it is the reverse: Z7 **+0.80**, Z8 **+0.64** at only ~2%
amplitude. (The $k\le6$ part correlates −0.15…+0.08, as expected — that is what got
subtracted.)

### 4.3b Field-order content of the MIW itself

Fitting the MIW maps to focal Zernikes $k=1..30$ — fraction of power above $k=6$:

| | Z5 | Z6 | Z7 | Z8 | Z11 |
|---|---|---|---|---|---|
| power at $k>6$ | **0.957** | 0.729 | **0.929** | 0.881 | 0.064 |
| leading $k>6$ terms | 23, 15, 25 | 22, 24, 11 | **7**, 17, 9 | **8**, 16, 10 | 11, 22 |

So Z5–Z8 are overwhelmingly high-field-order, and spherical (Z11) is not.
**Caveat: this is partly by construction** — the build removes $k\le6$ content at
every iteration, so the remainder is necessarily depleted there. The informative
part is *where* the $k>6$ power sits: coma peaks at $k=7,8$ (field radial order
$n=3$, immediately above the cut), astigmatism at $k=15,22,23,25$ ($n=4$–6, far
above it). That is exactly why the leakage model — whose $k>6$ content is dominated
by the lowest un-fitted orders — matches the coma shape and misses the astigmatism.

### 4.3c Per-mode field-order leakage — which $\Delta a_m$ can matter

Computed from the ts_ofc DZ matrix (31 field orders) by mapping each kept mode's
DOF direction $\mathbf N\mathbf v_m$ through the full sensitivity and comparing the
fitted ($k\le6$) and un-fitted ($k>6$) wavefront norms:

| mode $m$ | leakage $\|k{>}6\|/\|k{\le}6\|$ |
|---|---|
| 1–7 | 0.000 – 0.002 |
| 8–15 | 0.005 – 0.033 |
| 16–27 | 0.03 – 0.18 |
| 28–34 | **0.13 – 0.47** |

Median 0.036 as a ratio; as a **fraction of total power** (the fairer statement) this
is 0.0013 median, reaching 0.086-0.179 for modes 28-34.

**Consequence.** For the leading modes the removal is essentially exact — their
wavefront lives entirely inside $k\le6$, so $\Delta a_{1..10}$ **cannot** cause MIW
residual and any correlation there is proxy or coincidence. Only the high-index
modes leak enough to be causal. This predicts that the *tail* of the u-mode
spectrum should carry the correlation, and it does: `du_norm_tail` gives
$\rho=-0.49$ vs spatial $r$ against `du_norm_lead` $-0.21$, and $u_{30},u_{31},u_{32}$
appear among both the highest-leakage and the strongest-correlating modes.

The recommended scalar is therefore the **leakage-weighted displacement**
$\;\|\mathrm{diag}(\rho_m)\,\Delta\mathbf a\|\;$ (`du_leak`), which is
physically motivated where `du_norm`/`du_maha` are not. Caveat: $u_{13},u_{14}$
correlate strongly ($+0.43,+0.48$) despite low leakage ($0.011,0.033$), so
channel 3 (thermal driving both subspaces) is certainly also operating —
`du_leak` must still be reported with the `z_gradient` partial correlation.

Empirically (rebin 3, $n=221$, 16 build blocks) this is what the data show:
u-mode-only ML $R^2=-0.33$ and env+u-mode $R^2=+0.64$ vs environmental-only
$+0.62$ — i.e. $\Delta a_m$ adds **essentially nothing** beyond the environmental
telemetry, exactly as the boxed equation predicts.

### 4.2 The quantity that *is* first order — and is not currently saved

$$
\mathbf r_v\;\equiv\;(\mathbb 1-\mathbf P)\,\mathbf w_v
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
$(\mathbb 1-\mathbf P)\mathbf w$ follow — is a one-line addition to
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

1. First-order driver is $(\mathbb 1-\mathbf P)\mathbf w$, **not** the u-modes → save
   it (RSP rerun) and make it the primary regressor.
2. Within it, prioritize the **16 reachable-but-discarded** modes $\mathbf u_{35..50}$.
3. Keep $\Delta a_m$ as the **sensitivity-gain-error** test, second order, always with
   the `z_gradient` partial correlation.
4. Report $\Delta a$ in units of $\sigma_{\rm build}$ with the SEM floor marked.
5. Both builds must start from the same $I^{(0)}$ (batoid) or the $\mathbf P$ gauge
   does not cancel.
