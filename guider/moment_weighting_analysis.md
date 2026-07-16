# Weighted vs. Unweighted Moments in the Guider Ellipticity Decomposition

**Author:** Aaron Roodman
**Date:** 2026-07-15
**Context:** Guider star ellipticity study (`guider_star_ellipticity.ipynb`). Analytic
justification for how centroids and second moments should be measured when
decomposing guide-star shape into a static (optics-bound) part and a
time-variable (turbulence tip/tilt) part.

---

## 1. Goal and measurement scheme

We characterize guide-star shape entirely in **unnormalized second moments**
(pixel², reported in arcsec² via the pixel scale), never the normalized
ellipticity $e_i = Q_i/T$. Define

$$M_{xx},\; M_{yy},\; M_{xy} \quad\Rightarrow\quad
Q_1 \equiv M_{xx}-M_{yy},\quad Q_2 \equiv 2M_{xy},\quad T \equiv M_{xx}+M_{yy}.$$

The decomposition rests on the additivity of second moments for a **mean** coadd
of stamps in a common pixel frame (law of total variance). The practical question
addressed here: **which moments should be weighted, and which unweighted?**

The proposed scheme:

1. **Centroid of each stamp** — use a **weighted** moment (Gaussian weight) for a
   lower-noise centroid, even though the additivity formalism is written with the
   unweighted centroid.
2. **Moment of the centroid distribution** (image motion) — **unweighted** (the
   plain sample covariance of the per-stamp centroids).
3. **Second moment of each stamp** (shape) — **unweighted**, but taken about the
   weighted centroid from step 1.

The conclusion of the analysis below: this scheme is sound and, with one
consistency requirement, exact to the order that matters. The dominant systematic
is not the weighting bias but **centroid measurement noise**, which is exactly why
the weighted centroid is preferred and which forces an explicit noise-floor
subtraction.

---

## 2. Notation

For stamp $i$ (of $N$), flux-normalized so $\int f_i\,d\mathbf r = 1$:

| Symbol | Meaning |
|---|---|
| $f_i(\mathbf r)$ | background-subtracted, flux-normalized stamp intensity |
| $\bar f = \frac1N\sum_i f_i$ | the **mean** coadd |
| $c_i = \int \mathbf r\,f_i\,d\mathbf r$ | **true (unweighted) flux centroid** |
| $\tilde c_i = c_i + \boldsymbol\delta_i$ | **weighted centroid**; $\boldsymbol\delta_i$ = weighting bias |
| $M_i = \int(\mathbf r-c_i)(\mathbf r-c_i)^{\!\top} f_i$ | per-stamp moment about the true centroid |
| $S_i = \int(\mathbf r-\tilde c_i)(\mathbf r-\tilde c_i)^{\!\top} f_i$ | per-stamp moment about the weighted centroid |
| $\bar c = \frac1N\sum_i c_i,\;\; \bar{\tilde c} = \frac1N\sum_i \tilde c_i$ | mean centroids |
| $\langle\,\cdot\,\rangle$ | average over stamps |
| $\mathrm{Cov}(x_i)_{ab} = \frac1N\sum_i (x_{i,a}-\bar x_a)(x_{i,b}-\bar x_b)$ | sample covariance |
| $(\cdot)_{\rm sym}$ | symmetric part, $A_{\rm sym}=\tfrac12(A+A^{\!\top})$ |

All moments are $2\times2$ symmetric matrices with components $ab \in \{xx, xy, yy\}$.

---

## 3. The exact identity (all unweighted, mean coadd)

Center the coadd moment on $\bar c$ and expand each stamp about its own true
centroid $c_i$ (parallel-axis theorem):

$$M^{\rm co} \equiv \int(\mathbf r-\bar c)(\mathbf r-\bar c)^{\!\top}\bar f\,d\mathbf r
= \frac1N\sum_i \int(\mathbf r-\bar c)(\mathbf r-\bar c)^{\!\top} f_i.$$

Writing $\mathbf r-\bar c = (\mathbf r-c_i) + (c_i-\bar c)$ and using the defining
property $\int(\mathbf r - c_i)f_i = 0$ (the **cross term vanishes**), this collapses to

$$\boxed{\,M^{\rm co} = \langle M_i\rangle + \mathrm{Cov}(c_i)\,}$$

i.e. component-by-component $T^{\rm co}=\langle T\rangle + T^{\rm mot}$ and
$Q_i^{\rm co}=\langle Q_i\rangle + Q_i^{\rm mot}$, with the image-motion terms
$Q_1^{\rm mot}=\mathrm{Var}(x_c)-\mathrm{Var}(y_c)$, $Q_2^{\rm mot}=2\,\mathrm{Cov}(x_c,y_c)$,
$T^{\rm mot}=\mathrm{Var}(x_c)+\mathrm{Var}(y_c)$.

**Exactness conditions.** This identity is exact iff:

- (a) the coadd is the **mean** (arithmetic) of the $f_i$ — *not* a median;
- (b) the per-stamp moments are about the **true unweighted centroid** $c_i$;
- (c) the coadd moment is about $\bar c = \frac1N\sum_i c_i$;
- (d) the integration domain is all space (no aperture truncation).

The cross-term-vanishing step in (b) is the crux: it holds **only** for the
unweighted centroid. Everything below quantifies what happens when we substitute
the weighted centroid.

---

## 4. Result 1 — weighting perturbs only through asymmetry

Substitute the weighted centroid $\tilde c_i = c_i + \boldsymbol\delta_i$. Two
parallel-axis expansions (both exact, infinite aperture):

**Per-stamp shape about the weighted centroid.** With
$\mathbf r-\tilde c_i = (\mathbf r-c_i)-\boldsymbol\delta_i$ and $\int(\mathbf r-c_i)f_i=0$:

$$S_i = M_i + \boldsymbol\delta_i\boldsymbol\delta_i^{\!\top}.$$

Moving the reference point off the true centroid **always increases** the second
moment by the rank-1, positive-semidefinite outer product $\boldsymbol\delta_i\boldsymbol\delta_i^{\!\top}$.

**Motion covariance of the weighted centroids.** With
$\tilde c_i-\bar{\tilde c} = (c_i-\bar c)+(\boldsymbol\delta_i-\bar\delta)$:

$$C \equiv \mathrm{Cov}(\tilde c_i)
= \mathrm{Cov}(c_i) + 2\,\mathrm{Cov}(c_i,\boldsymbol\delta_i)_{\rm sym} + \mathrm{Cov}(\boldsymbol\delta_i).$$

**Key property of $\boldsymbol\delta_i$.** For a point-symmetric profile
$f_i(c_i+\mathbf u)=f_i(c_i-\mathbf u)$, any symmetric weight centered at $c_i$ yields
$\boldsymbol\delta_i = 0$. The weighting bias is sourced **only** by odd-order
asymmetry — coma, asymmetric wings, neighbor contamination — coupled to the weight
compactness. For a clean guide star $\boldsymbol\delta_i$ is a second-order small
quantity.

---

## 5. Result 2 — additivity survives if the weighted centroid is used consistently

Use the *same* weighted centroid everywhere: center the per-stamp shape on
$\tilde c_i$, form the motion covariance from the $\tilde c_i$, and center the coadd
moment on $\bar{\tilde c}=\frac1N\sum_i\tilde c_i$. Then:

$$S^{\rm co} \equiv \int(\mathbf r-\bar{\tilde c})(\mathbf r-\bar{\tilde c})^{\!\top}\bar f
= \langle M_i\rangle + \mathrm{Cov}(c_i) + \bar\delta\bar\delta^{\!\top}, \tag{I}$$

$$\langle S_i\rangle + C
= \langle M_i\rangle + \mathrm{Cov}(c_i) + \bar\delta\bar\delta^{\!\top}
  + 2\,\mathrm{Cov}(c_i,\boldsymbol\delta_i)_{\rm sym} + 2\,\mathrm{Cov}(\boldsymbol\delta_i). \tag{II}$$

Subtracting, the shared $\bar\delta\bar\delta^{\!\top}$ terms cancel and

$$\boxed{\;\big[\langle S_i\rangle + C\big] - S^{\rm co}
= 2\,\mathrm{Cov}(\boldsymbol\delta_i,\;\tilde c_i)_{\rm sym}\;}$$

Two consequences:

1. **Static weighting bias cancels identically.** A constant asymmetry
   ($\boldsymbol\delta_i = \bar\delta$, e.g. static optical coma) gives
   $\mathrm{Cov}(\boldsymbol\delta_i)=0$ and $\mathrm{Cov}(c_i,\boldsymbol\delta_i)=0$, so the
   residual is **zero** — regardless of the magnitude of $\bar\delta$. The additivity
   check is immune to static asymmetry as long as centering is consistent.
2. **Only the fluctuating bias correlated with motion survives.** The residual
   $2\,\mathrm{Cov}(\boldsymbol\delta_i,\tilde c_i)$ is a product of two small fluctuating
   quantities (turbulence-induced asymmetry $\times$ tilt-induced motion): second
   order, negligible for bright near-symmetric stars.

**Requirement.** Center the coadd moment on $\bar{\tilde c}$ = the mean of the
per-stamp weighted centroids, *not* on a fresh weighted centroid of the coadd image
(weighting is nonlinear under averaging; the difference is a further small term).

---

## 6. Result 3 — noise is the dominant effect, and it forces a subtraction

Add zero-mean centroid measurement noise $n_i$ with $\langle n_i n_i^{\!\top}\rangle = N_i$,
so the *measured* weighted centroid is $\hat c_i = \tilde c_i + n_i$. Parallel axis with
noise inflates **both** channels:

$$\langle \hat S_i\rangle = \langle S_i\rangle + \langle N_i\rangle, \qquad
\hat C = C + \langle N_i\rangle,$$

whereas the coadd — centroided once on the high-SNR mean image — has a noise floor
$\approx \langle N_i\rangle / N$, negligible. Hence with noise the additivity check is
broken by

$$\big[\langle \hat S_i\rangle + \hat C\big] - \hat S^{\rm co} \;\approx\; 2\,\langle N_i\rangle,$$

**unless the noise floor is subtracted from both the mean per-stamp shape and the
motion covariance.**

### 6.1 Why the weighted centroid: variance scaling

The centroid variance depends strongly on the estimator:

| Estimator | Per-axis centroid variance | $R=4\sigma,\ \mathrm{SNR}=10,\ \sigma=2\,\mathrm{px}$ |
|---|---|---|
| Unweighted, aperture radius $R$ | $\propto R^4\,\sigma_{\rm bg}^2 / F^2$ | $N \sim 0.4\text{–}4\ \mathrm{px}^2$ |
| Weighted (matched Gaussian) | $\approx \sigma_{\rm PSF}^2 / \mathrm{SNR}^2$ | $N \approx 0.04\ \mathrm{px}^2$ |

The unweighted centroid gives full lever-arm weight $r$ to the noisy wings, so its
variance grows as $R^4$. The weighted (matched-filter) centroid downweights the
wings, reducing the variance by a factor $\sim R^4/\sigma^4 \sim \mathcal O(10^2)$.

For a realistic motion moment — 0.1 px RMS jitter → $\mathrm{Var}=0.01\ \mathrm{px}^2$ —
the unweighted-centroid noise floor (0.4–4 px²) **swamps** the signal by 1–2 orders
of magnitude; even the weighted floor (0.04 px²) exceeds it. Therefore:

- Use the **weighted centroid** to minimize the motion noise floor — the single
  largest systematic for sub-pixel image motion.
- **Still subtract the residual floor.** The tracker already provides per-stamp
  `xerr`, `yerr` (from `calcGalsimError`), giving

$$M^{\rm mot}_{aa} \to \mathrm{Var}(\tilde c_a) - \langle \mathrm{err}_a^2\rangle, \qquad
\langle M^{\rm shape}\rangle_{aa} \to \langle S_{i,aa}\rangle - \langle \mathrm{err}_a^2\rangle.$$

### 6.2 Why center the shape at the weighted centroid

Centering the per-stamp shape at the noisy *unweighted* centroid would inject the
large unweighted $\langle N\rangle$ into the shape moment. Centering at the weighted
centroid trades that for the tiny static $\boldsymbol\delta\boldsymbol\delta^{\!\top}$: a 0.02 px
bias gives $4\times10^{-4}\ \mathrm{px}^2$ versus a shape moment $\sim\sigma^2\sim4\ \mathrm{px}^2$,
i.e. $\sim 10^{-4}$ fractional. The weighted-centroid centering is strictly better.

---

## 7. Verdict on the three choices

| Choice | Verdict |
|---|---|
| **1. Weighted centroid per stamp** | ✅ Correct. Minimizes the noise floor that otherwise dominates the motion moment. Bias $\boldsymbol\delta$ is asymmetry-only, mostly static, and cancels (§5). |
| **2. Unweighted covariance of the centroids** | ✅ Correct and unambiguous — it is the sample covariance of the centroid points (flux-weight the stamps to match the mean-coadd definition). Subtract the noise floor $\langle\mathrm{err}^2\rangle$. |
| **3. Unweighted shape about the weighted centroid** | ✅ Reasonable. Adds only $\boldsymbol\delta_i\boldsymbol\delta_i^{\!\top}$ (parallel axis); the static part cancels in the check and the magnitude is $\mathcal O(|\delta|^2)$. Far better than centering at the noisy unweighted centroid. |

---

## 8. Recommended recipe

1. **Fixed Gaussian weight**, width matched to the exposure's median FWHM — used to
   measure the centroid on **every stamp** and to center the coadd. Avoid a fully
   *adaptive* per-stamp width: it makes $\boldsymbol\delta_i$ fluctuate with the
   instantaneous shape/noise, reintroducing the $\mathrm{Cov}(\boldsymbol\delta_i,\tilde c_i)$
   residual of §5. (HSM adaptive is acceptable but adds a small fluctuating term.)
2. **Coadd** = mean (`np.nanmean`) stack in the raw ROI frame (no re-registration),
   centered on $\bar{\tilde c}$ = mean of the per-stamp weighted centroids.
3. **Per-stamp shape and coadd shape** = unweighted second moments within a fixed
   aperture (radius $\approx 4\sigma$ from the HSM FWHM), about the weighted centroid.
4. **Image motion** = flux-weighted covariance of the per-stamp weighted centroids.
5. **Subtract noise floors** $\langle\mathrm{err}_a^2\rangle$ from both the mean per-stamp
   shape and the motion covariance before comparing to the coadd.

### Residual systematics to watch

- **Background pedestal.** An unweighted second moment has a large lever arm
  ($\propto r^2$), so a small residual background biases $T=M_{xx}+M_{yy}$ far more
  than the shape $Q_i$. Check coadd-vs-stamp background consistency; the annulus
  subtraction mitigates but does not eliminate this.
- **Aperture truncation.** A finite aperture makes the moments smaller than the true
  (infinite-domain) values, but the same truncation applies to stamps and coadd, so
  the identity holds for the *truncated* moments provided the aperture is fixed in
  size and centered consistently. Because the aperture rides with the (small) motion,
  a further sub-dominant term enters at order (motion/aperture)².
- **Median vs mean coadd.** `getStampArrayCoadd` returns a *median* stack; the
  identity requires the mean. Build a mean coadd for the additivity test.

---

## 9. One-line summary

Weight the **centroid** (noise); leave the **moments** unweighted (additivity);
center everything on the **same** weighted centroid (static bias cancels); and
**subtract the centroid-noise floor** from the shape and motion terms (the only
first-order thing that otherwise breaks the check).
