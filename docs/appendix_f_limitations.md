# Appendix F: Methodological, Statistical, and Architectural Limitations of the Pilot Study ($N=152$)
---

## 1. Summary of Pilot Constraints

This document details the statistical and architectural limitations of the current pilot study ($N = 152$ survey responses, $M = 54$ prototype embedding vectors).

While the dual-paradigm framework, combining **Paradigm A (Post-Processing Decision Boundary Alignment via SCRUF-D)** and **Paradigm B (In-Processing Hyperspherical Two-Tower Neural Networks on $\mathbb{S}^{D-1}$)**, demonstrates working proofs of concept, we must interpret claims within the data constraints documented below.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                PILOT LIMITATIONS MATRIX                                │
├───────────────────────────────┬───────────────────────────────┬────────────────────────┤
│ Area                          │ Pilot Data State              │ Methodological Impact  │
├───────────────────────────────┼───────────────────────────────┼────────────────────────┤
│ A. Statistical Power          │ N = 152 (n=76 per test arm)   │ Low power for small    │
│                               │                               │ subgroup differences   │
│ B. Rating Distribution        │ Median = 6.0, Mean = 5.83     │ Left-skewed data       │
│                               │                               │ requires non-parametric│
│                               │                               │ rank tests             │
│ C. Geographic Scope           │ Gauteng concentration (> 50%) │ Under-represents rural │
│                               │                               │ connectivity patterns  │
│ D. Categorical Tables         │ 91.7% contingency cells < 5   │ Chi-square tests       │
│                               │                               │ unstable at high dof   │
│ E. Manifold Geometry          │ N_total = 54 vectors          │ t-SNE perplexity must  │
│                               │                               │ be clamped or use PCA  │
└───────────────────────────────┴───────────────────────────────┴────────────────────────┘
```

---

## 2. Statistical Audit

### A. Sample Size and Power Constraints ($N=152$)

#### 1. Sample-to-Feature Ratio in Supervised Machine Learning
When we encode user demographic states, platform choices, genre preferences, and discovery paths, the feature dimension $p$ reaches $p \approx 35-48$.

With sample size $N = 152$, the ratio is:
$$\frac{N}{p} \approx \frac{152}{42} \approx 3.62$$

Supervised non-parametric models (such as Random Forests or Gradient Boosted Trees) require $\frac{N}{p} \gg 20$ to prevent overfitting to respondent-level noise:
$$\mathbb{E}_{\mathcal{D}}\left[ \left( f(X; \mathcal{D}) - f^*(X) \right)^2 \right] = \operatorname{Bias}^2(f) + \operatorname{Var}_{\mathcal{D}}(f)$$
In this pilot sample, supervised predictive models risk memorizing idiosyncratic responses rather than learning true population patterns.

#### 2. Minimum Detectable Effect (MDE) Limits
From the power calculations in `notebooks/power_analysis_script.py`, comparing two equal arms ($n_1 = n_2 = 76$) at standard thresholds ($\alpha = 0.05$, power $1 - \beta = 0.80$):

- **Continuous Scores (Cohen's $d$):**
  $$d_{\text{MDE}} = \frac{|\mu_1 - \mu_2|}{\sigma} = \frac{z_{1 - \alpha/2} + z_{1 - \beta}}{\sqrt{n / 2}} = \frac{1.96 + 0.8416}{\sqrt{38}} \approx 0.457$$
  The pilot can reliably detect medium-to-large effect sizes ($d \ge 0.457$), but is underpowered to confirm smaller shifts ($d < 0.40$).

- **Binary Proportions (Baseline $p_0 = 0.50$):**
  $$\Delta p_{\text{MDE}} \approx 22.8\text{ percentage points}$$
  The pilot cannot resolve single-digit percentage variations ($\Delta p \le 5\%$) across demographic sub-groups.

---

### B. Rating Distribution Skewness

The survey satisfaction rating `rec_system_rating` shows a left-skewed, bounded distribution:

$$\text{Scale} \in \{1, 2, 3, 4, 5, 6, 7\}, \quad \text{Mean} = 5.828, \quad \text{Median} = 6.000, \quad \text{Std} = 1.171, \quad \text{Min} = 1.0, \quad \text{Max} = 7.0$$

```
   Relative Frequency (%)
     50% ┼                                                ██
     40% ┼                                                ██
     30% ┼                                          ██    ██
     20% ┼                                    ██    ██    ██
     10% ┼                              ██    ██    ██    ██
      0% ┼────██──────────██────────────██────██────██────██────
             Rating 1    Rating 2      Rating 4 Rating 5 Rating 6 Rating 7
                                       [Left Skew / Ceiling Effect]
```

#### Invalidation of Parametric Assumptions
Parametric models (such as ANOVA and two-sample t-tests) assume normally distributed residuals with equal variance across groups:
$$Y_{ij} = \mu + \alpha_i + \epsilon_{ij}, \quad \epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)$$

Because ratings cap at 7.0 and cluster at 6.0 and 7.0, the residual variance is heteroscedastic ($\sigma_7^2 \ll \sigma_3^2$). This justifies our use of rank-based non-parametric tests in `src/kulture/analysis/pipeline.py`:
- **Mann-Whitney U Test:** Tests whether one distribution stochastically dominates another without assuming normality.
- **Kruskal-Wallis H Test:** Evaluates differences in mean ranks across multiple categorical groups ($k \ge 3$).

---

### C. Geographic Distribution (Gauteng Focus)

Geographic responses (`province`) reflect an urban concentration:

$$\text{Gauteng} = 76 / 152 \quad (50.0\%), \quad \text{Metropolitan Total (Gauteng + Western Cape + eThekwini)} \approx 75.7\%$$

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        GEOGRAPHIC DISTRIBUTION BREAKDOWN (N=152)                       │
├───────────────────────────────┬───────────────────────────────┬────────────────────────┤
│ Province                      │ Respondent Count (n)          │ Sample Share (%)       │
├───────────────────────────────┼───────────────────────────────┼────────────────────────┤
│ Gauteng (Urban Hub)           │ 76                            │ 50.0%                  │
│ KwaZulu-Natal                 │ 23                            │ 15.1%                  │
│ Western Cape                  │ 16                            │ 10.5%                  │
│ Limpopo (Regional / Rural)    │ 10                            │ 6.6%                   │
│ Eastern Cape                  │ 9                             │ 5.9%                   │
│ Other / Unspecified           │ 18                            │ 11.9%                  │
└───────────────────────────────┴───────────────────────────────┴────────────────────────┘
```

#### Rural Context
The pilot represents urban listeners with broadband, home fiber, or workplace Wi-Fi. It fails to capture rural listening habits where:
1. **Data Costs:** In rural areas, mobile data costs (R85-R120/GB) discourage unmetered high-bitrate streaming.
2. **Offline Sharing:** Subgenres like Lekompo, traditional Maskandi, and Xitsonga Electro often circulate offline via USB flash drives at local taxi ranks, Bluetooth transfers, and community events.

---

### D. Contingency Table Sparsity

In `notebooks/categorical_analysis.py`, high-cardinality cross-tabulations trigger cell count warnings.

#### Cochran's Criterion
For **`province` vs. `discovery_method`** ($13 \times 7$ table, $\text{dof} = 55$):

$$\text{Total Cells} = 91, \quad \text{Cells with Expected Count } E_{ij} < 5 = 91.67\%$$

Cochran's standard rule for chi-square validity requires that no cell has expected frequency $E_{ij} < 1.0$ and at most $20\%$ of cells have $E_{ij} < 5.0$. 

Because $91.67\%$ of cells have expected counts below 5, the nominal test statistic ($\chi^2 = 80.258, p = 0.0148$) reflects table sparsity rather than demographic divergence. We do not recommend treating provincial discovery differences in this pilot as conclusive population estimates.

---

### E. Latent Space Dimensionality and t-SNE Bounds

In `representation-alignment/src/plot_embeddings.py`, we visualize embeddings using:
$$N_{\text{total}} = N_{\text{items}} + N_{\text{prototypes}} = 50 + 4 = 54 \text{ vectors}$$

#### t-SNE Perplexity Clamping
Standard t-SNE (van der Maaten & Hinton, 2008) uses perplexities $\text{Perp} \in [30, 50]$, requiring $N_{\text{samples}} \ge 3 \times \text{Perp}$.

For $N = 54$, default perplexity ($\text{Perp} = 30$) exceeds the sample limit ($3 \times 30 = 90 > 54$). In `plot_embeddings.py`, we clamp perplexity to $\text{Perp} = 15$ and provide an SVD-based Spherical PCA fallback:
$$\mathbf{X}_{\text{centred}} = \mathbf{V} - \frac{1}{N}\mathbf{1}\mathbf{1}^\top \mathbf{V}, \quad \mathbf{X}_{\text{centred}} = \mathbf{U} \mathbf{\Sigma} \mathbf{W}^\top$$

Linear PCA provides numerical stability, though projecting from $\mathbb{S}^{D-1} \subset \mathbb{R}^{16}$ to $\mathbb{R}^2$ flattens higher-order spherical curvature.

---

## 3. Research Expansion Roadmap ($N = 5,000$)

To scale `for the kulture` into a production-grade system, we outline a four-part research expansion plan:

```
                            [SCALED EXPANSION ROADMAP: N = 5,000]
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
     [1. Causal Learning]          [2. Representation Scale]       [3. Socio-Spatial Equity]
   Multi-Cause Deconfounding      1,000,000+ Item Scale          All 9 SA Provinces
   Continuous Latent Confounder   Overparameterized Two-Tower    Rural / Taxi-Rank Ingestion
   True Potential Outlier Space   Distributed JAX TPU Clusters   Zero-Data Tariff Testing
```

### 1. Multi-Cause Latent Deconfounding (MCLD)
With $N = 5,000$ longitudinal listener histories, we can apply the Multi-Cause Deconfounder (Wang & Blei, 2019):
1. **Factor Model on Exposure Causes:** Train a deep Poisson Factorization or Variational Autoencoder (VAE) on the exposure matrix $\mathbf{C} \in \{0, 1\}^{N \times M}$ to infer the latent unobserved confounder $\hat{Z}_u \in \mathbb{R}^d$:
   $$P(\mathbf{C} \mid \hat{Z}) = \prod_{i=1}^M P(C_i \mid \hat{Z})$$
2. **Outcome Modelling:** Estimate user engagement conditioning on the latent confounder:
   $$\mathbb{E}[B_{ui} \mid \text{do}(R_{ui} = 1), X_u] = \mathbb{E}_{\hat{Z}_u}\left[ \mathbb{E}[B_{ui} \mid R_{ui} = 1, X_u, \hat{Z}_u] \right]$$

### 2. Full Catalog Scaling ($M \ge 1,000,000$)
1. **Continuous Manifold Resolution:** With $N = 5,000$ and $M = 10^6$, t-SNE and UMAP can operate at standard density settings ($\text{Perp} = 50$, $k_{\text{neighbours}} = 30$), resolving fine subgenre variations (such as Sgija, Private School Amapiano, Bacardi, and Quantum).
2. **Distributed Hyperspherical Training:** Scaling Flax Two-Tower models across multi-host Google Cloud TPU v5e pods using `jax.pmap` and `optax.sharded_gradient`:
   $$\mathcal{L}_{\text{distributed}} = \frac{1}{|\mathcal{B}|}\sum_{u, i \in \mathcal{B}} \ell_{\text{recon}} + \lambda_{\text{proto}} \mathcal{L}_{\text{proto}}(\mathbb{S}^{D-1}) + \lambda_{\text{gini}} \mathcal{G}_\epsilon(\mathbf{e}_{\text{batch}})$$

### 3. Geographic Stratification Across All Nine Provinces
1. **Census Stratification:** Sample respondents proportional to Statistics South Africa census distributions (Gauteng: $26\%$, KZN: $19\%$, Western Cape: $12\%$, Eastern Cape: $11\%$, Limpopo: $10\%$, Mpumalanga: $8\%$, North West: $7\%$, Free State: $5\%$, Northern Cape: $2\%$).
2. **Offline Interaction Logging:** Deploy a lightweight client to log local offline playback and Bluetooth sharing, bridging the digital-divide data gap.

### 4. Churn and Platform-Hopping Prediction
With $N = 5,000$, the sample-to-feature ratio expands to $\frac{N}{p} \approx \frac{5000}{42} \approx 119 \gg 20$, allowing:
1. **Survival Modelling:** Train Cox Proportional Hazards models to estimate how high Context Tax ($\tau_c(u) > 0.50$) impacts platform retention.
2. **Platform-Hopping Quantifications:** Quantify listener churn when recommendation engines fail to serve local subgenre demand.

---

## 4. Pilot vs. Scaled Comparison

| Domain | Pilot Study ($N=152$) | Scaled Goal ($N=5,000$) |
| :--- | :--- | :--- |
| **Statistical Power** | Underpowered for ML ($\frac{N}{p} \approx 3.6$); MDE $d \ge 0.457$ | Robust ML ($\frac{N}{p} > 100$); MDE detects $\le 2.5\%$ shifts ($d \le 0.08$) |
| **Hypothesis Testing** | Rank Non-Parametrics (Mann-Whitney / Kruskal) | Semi-Parametric Structural Equation Modelling |
| **Geographic Scope** | Primarily Gauteng urban broadband ($>50\%$) | Stratified across all 9 South African provinces |
| **Contingency Tables** | $91.7\%$ cells violate Cochran's rule ($E < 5$) | Cell frequencies $E_{ij} \gg 50$, exact $\chi^2$ tests |
| **Manifold Projection**| SVD Spherical PCA fallback ($N=54$) | Continuous high-resolution UMAP / t-SNE manifolds |
| **Causal Modelling** | Post-processing heuristic adjustments (SCRUF-D CAFL) | Multi-Cause Latent Deconfounders (MCLD) with VAEs |

---

## References

1. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
2. **van der Maaten, L., & Hinton, G.** (2008). *Visualizing Data using t-SNE*. Journal of Machine Learning Research, 9(86), 2579-2605.
3. **Wang, Y., & Blei, D. M.** (2019). *The Blessings of Multiple Causes*. Journal of the American Statistical Association, 114(528), 1574-1596.
4. **Cochran, W. G.** (1954). *Some Methods for Strengthening the Common $\chi^2$ Tests*. Biometrics, 10(4), 417-451.
5. **Statistics South Africa.** (2023). *Census 2022: Statistical Release*. Stats SA Report No. 03-01-22.
