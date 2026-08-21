# Appendix B: Formalizing the Inversion Problem and Causal Deconfounding in Preference Elicitation

**Author:** Shailoh (Independent Socio-Technologist & ML Practitioner)  
**Target Repository:** `for the kulture`  

---

## 1. Structural Causal Model: Directed Acyclic Graph (DAG)

In recommender systems, user streaming logs do not provide an unfiltered readout of user taste. Observed behaviour is filtered through platform interface choices, position bias, and catalog availability.

### 1.1 Structural Equations
Let the data-generating process be $\mathcal{M} = \langle \mathbf{V}, \mathbf{U}, \mathcal{F}, P(\mathbf{U}) \rangle$, where:
- $\mathbf{U} = \{U_u, U_i, U_\epsilon\}$ represents exogenous background variables (user background, raw acoustic features, session noise).
- $\mathbf{V} = \{X, M, C, R, B\}$ represents observable and latent variables for user $u \in \mathcal{U}$ and item $i \in \mathcal{I}$:
  1. $X \in \mathcal{X}$: **User & Context Covariates** (network bandwidth, subscription tier, listening history).
  2. $M \in \mathcal{M}_{\text{latent}}$: **Latent Preference** ($M_{ui} \in [0, 1]$ represents unconstrained preference).
  3. $C \in \mathcal{C}$: **Platform Exposure Constraints** ($C_{ui} \in \{0, 1\}$ represents playlist placement, UI prominence, and search indexing).
  4. $R \in \mathcal{R}$: **Recommendation Policy** ($R_{ui} \in \{0, 1\}$ indicates whether the system serves item $i$ to user $u$).
  5. $B \in \{0, 1\}$: **Observed Behaviour** ($B_{ui} = 1$ denotes a completed stream or favourite).

```
                            ┌────────────────────────┐
                            │  User / Context        │
                            │  Covariates (X)        │
                            └───────────┬────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              ┌─────────────────────┐       ┌─────────────────────┐
              │ Latent Preference   │       │ Platform Exposure   │
              │ (M)                 │       │ Constraints (C)     │
              └──────────┬──────────┘       └──────────┬──────────┘
                         │                             │
                         │      ┌───────────────┐      │
                         └─────►│ Observed      │◄─────┘
                                │ Behaviour (B)  │
                                └───────▲───────┘
                                        │
                                ┌───────┴───────┐
                                │ Recommendation│
                                │ Policy (R)    │
                                └───────────────┘
```

The structural equations $\mathcal{F} = \{f_M, f_C, f_R, f_B\}$ are:
$$M_{ui} := f_M(X_u, U_u, U_i)$$
$$C_{ui} := f_C(X_u, \text{Popularity}(i), U_\epsilon)$$
$$R_{ui} := f_R(X_u, C_{ui}, \hat{M}_{ui})$$
$$B_{ui} := f_B(M_{ui}, C_{ui}, R_{ui}, U_\epsilon) = M_{ui} \cdot \mathbb{I}(R_{ui} = 1) \cdot \kappa(C_{ui}) + \xi_{ui}$$

where $\kappa(C_{ui}) \in (0, 1]$ accounts for position and interface friction, and $\xi_{ui}$ is noise.

---

## 2. The Inversion Problem: Proof of Observational Bias

The **Inversion Problem** occurs when a recommendation engine treats observed streams $B$ as direct measurements of latent preference $M$:
$$\mathbb{P}(M \mid B = 1) \stackrel{?}{\propto} \mathbb{P}(B = 1 \mid M)$$

Platform delivery constraints $C$ break this equivalence.

---

### Theorem 1 (The Inversion Non-Equivalence)
*Let $M \in \{0, 1\}$ denote latent preference and $B \in \{0, 1\}$ denote observed streaming. Under non-uniform exposure constraints $C \in \{0, 1\}$ with $\mathbb{P}(C = 1 \mid M) \neq 1$, observed behaviour is confounded:*
$$\mathbb{P}(B = 1 \mid M, C) \neq \mathbb{P}(B = 1 \mid M)$$
*Naive posterior estimates $\hat{\mathbb{P}}(M = 1 \mid B = 1)$ overestimate mainstream popularity and underestimate demand for unpromoted local subgenres.*

---

### Proof:

#### Step 1: Observational Likelihood Decomposition
By the law of total probability:
$$\mathbb{P}(B = 1 \mid M) = \sum_{c \in \{0, 1\}} \mathbb{P}(B = 1 \mid M, C = c) \cdot \mathbb{P}(C = c \mid M)$$

Without exposure ($C = 0$), observation is impossible:
$$\mathbb{P}(B = 1 \mid M, C = 0) = 0$$

Therefore:
$$\mathbb{P}(B = 1 \mid M) = \mathbb{P}(B = 1 \mid M, C = 1) \cdot \mathbb{P}(C = 1 \mid M)$$

Since $\mathbb{P}(C = 1 \mid M) < 1$ for items without universal placement:
$$\mathbb{P}(B = 1 \mid M, C = 1) = \frac{\mathbb{P}(B = 1 \mid M)}{\mathbb{P}(C = 1 \mid M)} \neq \mathbb{P}(B = 1 \mid M)$$

#### Step 2: Posterior Distortion
By Bayes' rule, unconfounded preference given behaviour is:
$$\mathbb{P}(M = 1 \mid B = 1) = \frac{\mathbb{P}(B = 1 \mid M = 1) \mathbb{P}(M = 1)}{\mathbb{P}(B = 1)}$$

Standard collaborative filtering fits observational logs where $B=1$ implies $C=1$:
$$\mathbb{P}_{\text{obs}}(M = 1 \mid B = 1) = \mathbb{P}(M = 1 \mid B = 1, C = 1) = \frac{\mathbb{P}(B = 1 \mid M = 1, C = 1) \mathbb{P}(M = 1 \mid C = 1)}{\mathbb{P}(B = 1 \mid C = 1)}$$

Let $d \in \mathcal{I}_d$ be a mainstream item with high promotional exposure:
$$\mathbb{P}(C = 1 \mid d) = 1.0 - \delta_d, \quad \delta_d \approx 0$$
Let $p \in \mathcal{I}_p$ be an unpromoted local subgenre item (such as Lekompo or Maskandi):
$$\mathbb{P}(C = 1 \mid p) = \epsilon_p, \quad \epsilon_p \ll 1$$

For equal latent preference $\mathbb{P}(M = 1 \mid d) = \mathbb{P}(M = 1 \mid p) = \theta$:
$$\mathbb{P}(B = 1 \mid d) = \theta \cdot (1 - \delta_d) \approx \theta$$
$$\mathbb{P}(B = 1 \mid p) = \theta \cdot \epsilon_p \ll \theta$$

The observed interaction ratio is:
$$\frac{\mathbb{P}(B = 1 \mid d)}{\mathbb{P}(B = 1 \mid p)} = \frac{1 - \delta_d}{\epsilon_p} \gg 1$$

Even when true latent demand $\theta$ is identical, observational logs record fewer interactions for $p$. An unconstrained recommender treating $B=1$ as $M=1$ inverts the causal relationship, incorrectly treating platform-level delivery deficits as user-level disinterest.
$$\tag*{$\blacksquare$}$$

---

## 3. Causal Adjustment for Feedback Loops (CAFL)

To estimate unconfounded preferences, we evaluate the interventional distribution using Pearl's $\text{do}(\cdot)$ operator.

### 3.1 Interventional Target Distribution
We seek the probability of user engagement when an item is deliberately presented, severing the backdoor path $R \leftarrow C \leftarrow X \to M$:
$$\mathbb{P}\left(B = 1 \mid \text{do}(R = i), u\right)$$

Conditioning on user and session covariates $X$ satisfies the backdoor criterion:

```
                            ┌────────────────────────┐
                            │  User / Context        │
                            │  Covariates (X)        │
                            └───────────┬────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              ┌─────────────────────┐       ┌─────────────────────┐
              │ Latent Preference   │       │ Platform Exposure   │
              │ (M)                 │       │ Propensity (C)      │
              └──────────┬──────────┘       └──────────┬──────────┘
                         │                             │
                         │      ┌───────────────┐      │
                         └─────►│ Observed      │◄─────┘
                                │ Behaviour (B)  │
                                └───────▲───────┘
                                        │ (do)
                                ┌───────┴───────┐
                                │ [Intervention]│
                                │   do(R = i)   │
                                └───────────────┘
```

By backdoor adjustment:
$$\mathbb{P}(B = 1 \mid \text{do}(R = i), u) = \int_{\mathcal{X}} \mathbb{P}(B = 1 \mid R = i, X = x, u) \, d\mathbb{P}(X = x \mid u)$$

---

### 3.2 IPS and Doubly Robust Estimators

Let the propensity score $e(u, i)$ be:
$$e(u, i) \triangleq \mathbb{P}(C_{ui} = 1 \mid X_u, i) = \sigma\left(\mathbf{w}^\top \mathbf{x}_{ui}\right)$$

#### 1. Inverse Propensity Scoring (IPS):
$$\mathcal{L}_{\text{IPS}}(\Theta) = \frac{1}{|\mathcal{U}| |\mathcal{I}|} \sum_{u \in \mathcal{U}} \sum_{i \in \mathcal{I}} \frac{C_{ui} \cdot \ell\left(\hat{y}_{ui}(\Theta), Y_{ui}\right)}{e(u, i)}$$

When $e(u, i) > \epsilon > 0$, $\mathcal{L}_{\text{IPS}}$ is an unbiased estimator of the full-information interventional risk $\mathcal{L}_{\text{ideal}}(\Theta)$.

#### 2. Doubly Robust (DR) Estimator:
With outcome model $\hat{r}_{ui} = g(X_u, i)$:
$$\mathcal{L}_{\text{DR}}(\Theta) = \frac{1}{|\mathcal{U}||\mathcal{I}|} \sum_{u, i} \left[ \hat{r}_{ui} + \frac{C_{ui} \left( \ell(\hat{y}_{ui}, Y_{ui}) - \hat{r}_{ui} \right)}{e(u, i)} \right]$$
The DR estimator remains consistent if either the propensity model $e(u, i)$ or the outcome model $\hat{r}_{ui}$ is correctly specified.

---

## 4. Systems Integration: The Causal Arbiter in SCRUF-D

In `src/kulture/postprocessing/scruf_d.py`, this correction runs at the ranking stage via the **Causal Arbiter Agent ($A_{\text{arbiter}}$)**.

### 4.1 Context Tax $\tau_c(u)$
We use Context Tax $\tau_c(u)$ as an empirical signal of platform exposure bias:
$$\tau_c(u) = S_{\text{global}}(u) - D_{\text{local}}(u)$$
A positive $\tau_c(u) > 0$ indicates that the user streams heavily overall ($S_{\text{global}} \to 1$) but must leave the app's algorithm to discover local music ($D_{\text{local}} \to 0$).

### 4.2 Causal Arbiter Scoring
When ranking candidate item $i$ for user $u$:
$$\text{Score}_{\text{arbiter}}(u, i) = \hat{y}_{ui} - \beta \cdot \max(0, \tau_c(u)) \cdot \Phi(i)$$
where:
- $\hat{y}_{ui} = \langle \mathbf{u}_u, \mathbf{v}_i \rangle$ is the base model score.
- $\beta \ge 0$ controls penalty strength.
- $\Phi(i) = \frac{\text{Popularity}(i)}{\max_j \text{Popularity}(j)}$ acts as a proxy for platform exposure propensity.

```
                                  [SCRUF-D Social Choice Committee]
                                                  │
                 ┌────────────────────────────────┼────────────────────────────────┐
                 ▼                                ▼                                ▼
    ┌───────────────────────────┐   ┌───────────────────────────┐    ┌───────────────────────────┐
    │ 1. Myopic Exploiter       │   │ 2. Causal Arbiter (CAFL)  │    │ 3. Adversarial Preserver  │
    │                           │   │                           │    │                           │
    │ S_exploit = ŷ_ui          │   │ S_arbiter =               │    │ S_preserve =              │
    │                           │   │   ŷ_ui - β·τ_c(u)·Φ(i)    │    │   Q_g · 𝟙(genre(i) ∈ G_p) │
    └─────────────┬─────────────┘   └─────────────┬─────────────┘    └─────────────┬─────────────┘
                  │                               │                                │
                  └───────────────────────┬───────┴────────────────────────────────┘
                                          │
                                          ▼
                                [Negotiated Social Utility]
                         U(u, i) = w_e·S_e + w_a·S_a + w_p·S_p
                                          │
                                          ▼
                             [Quota-Enforced Final Slate]
```

### 4.3 Social Choice Aggregation
The committee calculates composite scores:
$$U(u, i) = w_{\text{exploit}} \cdot S_{\text{exploit}}(u, i) + w_{\text{arbiter}} \cdot S_{\text{arbiter}}(u, i) + w_{\text{preserve}} \cdot S_{\text{preserve}}(i)$$
subject to a minimum local genre quota per slate:
$$\sum_{i \in \mathcal{S}_u^K} \mathbb{I}\left(\text{genre}(i) \in \mathcal{G}_{\text{local}}\right) \ge Q_g$$
where $\mathcal{G}_{\text{local}} = \{\text{Gqom}, \text{Lekompo}, \text{Maskandi}, \text{SA House}\}$.

---

## 5. Comparison of Frameworks

| Component | Standard Collaborative Filtering | Causal Arbiter & CAFL (This Architecture) |
| :--- | :--- | :--- |
| **Target Distribution** | $\mathbb{P}(B = 1 \mid M, C)$ (Observational) | $\mathbb{P}(B = 1 \mid \text{do}(R=i), u)$ (Interventional) |
| **Feedback Loop Disparity** | Superlinear escalation ($\alpha > 1$) | Neutral bounded dynamics ($\alpha \le 1.0$) |
| **Peripheral Content Exposure** | Decays toward zero ($\lim a_T^{(p)} = 0$) | Protected by integer floor ($\sum \mathbb{I}(g \in \mathcal{G}_p) \ge Q_g$) |
| **Inversion Problem** | Equates streams with unconstrained preference | Uses Context Tax $\tau_c(u)$ to discount platform exposure bias |

---

## References

1. **Pearl, J.** (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.
2. **Truong, Q. T., Salah, A., & Lauw, H. W.** (2025). *Mitigating Feedback Loops in Latent Representation Recommenders*. IEEE Transactions on Knowledge and Data Engineering (TKDE).
3. **Krauth, K., Dean, S., Zhao, A., Jiang, W., & Jordan, M. I.** (2022). *Do Offline Metrics Predict Online Performance in Recommender Systems?* Proceedings of the 36th Conference on Neural Information Processing Systems (NeurIPS 2022).
4. **Schnabel, T., Swaminathan, A., Singh, A., Chandak, N., & Joachims, T.** (2016). *Recommendations as Treatments: Debiasing Learning and Evaluation*. In International Conference on Machine Learning (ICML '16), pp. 1670–1679.
5. **Burke, R., Sonboli, N., & Ordonez-Gauger, A.** (2018). *Balanced Neighbourhoods for Multi-Sided Fairness in Recommendation*. In Conference on Fairness, Accountability and Transparency (FAT*), pp. 202–214.
