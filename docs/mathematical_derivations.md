# Appendix A: Mathematical Derivation of Superlinear Bias Escalation in Closed-Loop Recommendation Dynamics

**Author:** Shailoh (Independent Socio-Technologist & ML Practitioner)  
**Target Repository:** `for the kulture`  

---

## 1. Dynamical System Setup: Closed-Loop Collaborative Filtering

Let $\mathcal{U} = \{u_1, \dots, u_N\}$ denote $N$ users, and $\mathcal{I} = \mathcal{I}_d \cup \mathcal{I}_p$ denote $M$ items split into two cohorts:
1. **Dominant Cohort ($\mathcal{I}_d$):** High-resource, chart-topping items with dense prior interaction logs (such as commercial pop and mainstream Amapiano singles).
2. **Peripheral Cohort ($\mathcal{I}_p$):** Long-tail, local subgenre items with sparse historical logs (such as regional Gqom, Lekompo, and Maskandi).

Let $T \in \mathbb{N}_0$ index discrete retraining epochs. At epoch $T$, the interaction matrix is $\mathbf{Y}_T \in \{0, 1\}^{N \times M}$, where $Y_{T, ui} = 1$ if user $u$ interacted with item $i$ before epoch $T$.

### 1.1 Recommendation Pipeline
At each epoch $T$:
1. **Representation Learning:**
   The model maps users and items to vectors $\mathbf{u}_{T, u}, \mathbf{v}_{T, i} \in \mathbb{R}^D$ (or on the unit sphere $\mathbb{S}^{D-1}$) by minimizing regularized loss over historical interactions $\mathcal{D}_T = \{(u, i) \mid Y_{T, ui} = 1\}$:
   $$\Theta_T^* = \arg\min_\Theta \sum_{(u, i) \in \mathcal{D}_T} \ell\left(\hat{y}_{T, ui}(\Theta), Y_{T, ui}\right) + \frac{\lambda_{\text{reg}}}{2} \|\Theta\|_2^2$$
   where predicted score is:
   $$\hat{y}_{T, ui} = \langle \mathbf{u}_{T, u}, \mathbf{v}_{T, i} \rangle$$

2. **Slate Exposure Policy:**
   For each user $u$, the platform selects items according to a softmax policy with temperature $\tau > 0$:
   $$\mathbb{P}_T(\text{expose } i \mid u) = \pi_T(i \mid u) = \frac{\exp(\hat{y}_{T, ui} / \tau)}{\sum_{j \in \mathcal{I}} \exp(\hat{y}_{T, uj} / \tau)}$$

### 1.2 Feedback Generation and Closed-Loop Updates
Let user $u$'s aesthetic affinity for item $i$ be $A_{ui} \in [0, 1]$, and platform context (position, UI placement) be $C_{ui} \in [0, 1]$.
The probability of interaction $B_{T, ui} \in \{0, 1\}$ upon exposure is:
$$\mathbb{P}(B_{T, ui} = 1 \mid \text{expose } i, u) = A_{ui} \cdot \kappa(C_{ui})$$
where $\kappa(\cdot)$ represents position bias.

Expected exposure $a_T(i)$ and interaction accumulation $\Delta Y_{T, ui}$ during epoch $T$ are:
$$a_T(i) = \sum_{u \in \mathcal{U}} \pi_T(i \mid u)$$
$$Y_{T+1, ui} = Y_{T, ui} + B_{T, ui}, \quad \text{where } B_{T, ui} \sim \text{Bernoulli}\left(\pi_T(i \mid u) \cdot A_{ui} \cdot \kappa(C_{ui})\right)$$

The training set updates recursively:
$$\mathcal{D}_{T+1} = \mathcal{D}_T \cup \{(u, i) \mid B_{T, ui} = 1\}$$

---

## 2. Evolution of Exposure Disparity

Let $a_T^{(d)}$ and $a_T^{(p)}$ denote the average exposure probabilities for representative dominant and peripheral items at epoch $T$:
$$a_T^{(d)} = \frac{1}{N |\mathcal{I}_d|} \sum_{u \in \mathcal{U}} \sum_{d \in \mathcal{I}_d} \pi_T(d \mid u), \quad a_T^{(p)} = \frac{1}{N |\mathcal{I}_p|} \sum_{u \in \mathcal{U}} \sum_{p \in \mathcal{I}_p} \pi_T(p \mid u)$$

Define the **Cohort Disparity Ratio** at epoch $T$:
$$\rho_T \triangleq \frac{a_T^{(d)}}{a_T^{(p)}}$$

### 2.1 The Recurrence Map
Under policy $\pi_T$, the expected ratio at epoch $T+1$ is:
$$\rho_{T+1} = \frac{a_{T+1}^{(d)}}{a_{T+1}^{(p)}} = \frac{\frac{1}{|\mathcal{I}_d|} \sum_{d \in \mathcal{I}_d} \sum_{u \in \mathcal{U}} \exp\left(\hat{y}_{T+1, ud} / \tau\right)}{\frac{1}{|\mathcal{I}_p|} \sum_{p \in \mathcal{I}_p} \sum_{u \in \mathcal{U}} \exp\left(\hat{y}_{T+1, up} / \tau\right)}$$

Averaged across users:
$$\rho_{T+1} \approx \exp\left( \frac{\bar{y}_{T+1}^{(d)} - \bar{y}_{T+1}^{(p)}}{\tau} \right)$$
where $\bar{y}_{T+1}^{(d)} = \mathbb{E}_{u}[\hat{y}_{T+1, ud}]$ and $\bar{y}_{T+1}^{(p)} = \mathbb{E}_{u}[\hat{y}_{T+1, up}]$.

---

## 3. Proof of Superlinear Escalation ($\alpha > 1$)

We show that disparity evolves according to a power law:
$$\rho_{T+1} \propto \left( \rho_T \right)^\alpha \quad \text{with} \quad \alpha > 1$$

---

### Theorem 1 (Superlinear Exponent Under Latent Embedding Updates)
*In unconstrained collaborative filtering models trained on closed-loop feedback $\mathcal{D}_T \to \mathcal{D}_{T+1}$, the recurrence mapping $\rho_T \mapsto \rho_{T+1}$ has an elasticity exponent:*
$$\alpha = 1 + \eta \cdot \Gamma > 1$$
*where $\eta > 0$ is the learning rate, and $\Gamma > 0$ reflects latent space curvature from popularity bias.*

---

### Proof:

#### Step 1: Gradient Updates for Item Embeddings
Under gradient descent, the update for item embedding $\mathbf{v}_i$ between epochs $T$ and $T+1$ is:
$$\mathbf{v}_{T+1, i} = \mathbf{v}_{T, i} + \eta \sum_{u \in \mathcal{U}} Y_{T, ui} \mathbf{u}_{T, u} - \eta \lambda_{\text{reg}} \mathbf{v}_{T, i}$$

Taking the expectation conditioned on policy $\pi_T$:
$$\mathbb{E}[Y_{T, ui} \mid \pi_T] \propto a_T(i) \cdot \bar{A}_i$$
where $\bar{A}_i = \mathbb{E}_u[A_{ui} \kappa(C_{ui})]$.

The expected embedding updates for dominant vs. peripheral items are:
$$\mathbf{v}_{T+1, d} = (1 - \eta \lambda_{\text{reg}})\mathbf{v}_{T, d} + \eta N a_T^{(d)} \bar{A}_d \bar{\mathbf{u}}$$
$$\mathbf{v}_{T+1, p} = (1 - \eta \lambda_{\text{reg}})\mathbf{v}_{T, p} + \eta N a_T^{(p)} \bar{A}_p \bar{\mathbf{u}}$$
where $\bar{\mathbf{u}} = \frac{1}{N}\sum_u \mathbf{u}_u$ is the mean user preference vector.

#### Step 2: Projections on the Mean User Vector
Dominant items $d \in \mathcal{I}_d$ receive $a_T^{(d)} \gg a_T^{(p)}$ gradient updates along $\bar{\mathbf{u}}$:
$$\langle \mathbf{v}_{T+1, d}, \bar{\mathbf{u}} \rangle = (1 - \eta \lambda_{\text{reg}})\langle \mathbf{v}_{T, d}, \bar{\mathbf{u}} \rangle + \eta N a_T^{(d)} \bar{A}_d \|\bar{\mathbf{u}}\|_2^2$$
$$\langle \mathbf{v}_{T+1, p}, \bar{\mathbf{u}} \rangle = (1 - \eta \lambda_{\text{reg}})\langle \mathbf{v}_{T, p}, \bar{\mathbf{u}} \rangle + \eta N a_T^{(p)} \bar{A}_p \|\bar{\mathbf{u}}\|_2^2$$

Subtracting the two equations gives the difference $\Delta \bar{y}_{T+1} = \bar{y}_{T+1}^{(d)} - \bar{y}_{T+1}^{(p)}$:
$$\Delta \bar{y}_{T+1} = (1 - \eta \lambda_{\text{reg}}) \Delta \bar{y}_T + \eta N \|\bar{\mathbf{u}}\|_2^2 \left( a_T^{(d)} \bar{A}_d - a_T^{(p)} \bar{A}_p \right)$$

Assuming similar intrinsic quality $\bar{A}_d \approx \bar{A}_p = \bar{A}$:
$$\Delta \bar{y}_{T+1} = (1 - \eta \lambda_{\text{reg}}) \Delta \bar{y}_T + \eta N \bar{A} \|\bar{\mathbf{u}}\|_2^2 \cdot a_T^{(p)} (\rho_T - 1)$$

#### Step 3: Logarithmic Derivative
Taking logarithms on both sides of $\rho_{T+1}$:
$$\ln \rho_{T+1} = \frac{\Delta \bar{y}_{T+1}}{\tau} = \frac{1 - \eta \lambda_{\text{reg}}}{\tau} \Delta \bar{y}_T + \frac{\eta N \bar{A} \|\bar{\mathbf{u}}\|_2^2}{\tau} a_T^{(p)} (\rho_T - 1)$$

Substituting $\Delta \bar{y}_T = \tau \ln \rho_T$:
$$\ln \rho_{T+1} = (1 - \eta \lambda_{\text{reg}}) \ln \rho_T + \left(\frac{\eta N \bar{A} \|\bar{\mathbf{u}}\|_2^2 a_T^{(p)}}{\tau}\right) (\rho_T - 1)$$

Differentiating $\ln \rho_{T+1}$ with respect to $\ln \rho_T$:
$$\alpha \triangleq \frac{\partial \ln \rho_{T+1}}{\partial \ln \rho_T} = (1 - \eta \lambda_{\text{reg}}) + \left(\frac{\eta N \bar{A} \|\bar{\mathbf{u}}\|_2^2 a_T^{(p)}}{\tau}\right) \cdot \rho_T$$

For learning rate $\eta > 0$, population $N \ge 1$, and non-zero user mean $\|\bar{\mathbf{u}}\|_2 > 0$:
$$\Gamma \triangleq \frac{N \bar{A} \|\bar{\mathbf{u}}\|_2^2 a_T^{(p)}}{\tau} \cdot \rho_T - \lambda_{\text{reg}} > 0$$

Thus:
$$\alpha = 1 + \eta \cdot \Gamma > 1 \quad \forall \; T \ge 1$$
$$\tag*{$\blacksquare$}$$

---

## 4. Disparity Growth and Asymptotic Exposure Limits

### 4.1 Non-Linear Difference Solution
From the relation $\rho_{T+1} = c_0 \cdot \left(\rho_T\right)^\alpha$ with $\alpha > 1$:
$$\ln \rho_T = \alpha^T \left( \ln \rho_0 + \frac{\ln c_0}{\alpha - 1} \right) - \frac{\ln c_0}{\alpha - 1}$$

Exponentiating:
$$\rho_T = \mathcal{O}\left( \exp\left( C \cdot \alpha^T \right) \right)$$

The exposure ratio $\rho_T = \frac{a_T^{(d)}}{a_T^{(p)}}$ grows doubly exponentially in epoch $T$.

---

### Theorem 2 (Asymptotic Exposure of Peripheral Subgenres)
*In an unconstrained closed-loop recommender system ($\alpha > 1$), exposure probability for peripheral items converges to zero as $T \to \infty$:*
$$\lim_{T \to \infty} a_T^{(p)} = 0$$

### Proof:
Under the softmax normalization:
$$|\mathcal{I}_d| a_T^{(d)} + |\mathcal{I}_p| a_T^{(p)} = 1 \implies a_T^{(p)} \left( |\mathcal{I}_p| + |\mathcal{I}_d| \rho_T \right) = 1$$

Solving for $a_T^{(p)}$:
$$a_T^{(p)} = \frac{1}{|\mathcal{I}_p| + |\mathcal{I}_d| \exp\left( C \cdot \alpha^T \right)}$$

Taking the limit $T \to \infty$ with $\alpha > 1$ and $C > 0$:
$$\lim_{T \to \infty} a_T^{(p)} = 0$$
$$\tag*{$\blacksquare$}$$

---

## 5. Representation Collapse

Without spherical constraints and exposure regularization, unconstrained optimisation collapses latent representations toward the dominant popularity direction:

```
       [Unconstrained Closed-Loop]                     [Spherical Prototype Alignment]
       
            v_d1    v_d2 (Dominant)                              p_Amapiano   p_Gqom
              \    /                                                 *          *
               \  /                                                 / \        / \
                ▼                                                  /   \      /   \
          ───────●───────► u_centroid                         ────●─────●────●─────●──── 𝕊^(D-1)
                ▲                                                 \    /      \   /
               / \                                                 \  /        \ /
              /   \                                                 *          *
            v_p1   v_p2 (Peripheral)                             p_Lekompo   p_Maskandi
       
       (All vectors collapse into 1D                 (Prototypes p_k span 𝕊^(D-1); Gini
        dominant popularity subspace)                 penalty bounds exposure variance)
```

1. Dominant items receive regular gradient reinforcement along the principal eigenvector of $\mathbf{Y}_T^\top \mathbf{Y}_T$, inflating $\|\mathbf{v}_d\|_2$.
2. Peripheral items receive few interaction gradients, so their norms shrink under weight decay $\lambda_{\text{reg}}$:
   $$\|\mathbf{v}_{T, p}\|_2 \approx (1 - \eta \lambda_{\text{reg}})^T \|\mathbf{v}_{0, p}\|_2 \to 0$$
3. As $\|\mathbf{v}_{T, p}\|_2 \to 0$, cosine similarity directions become noisy, collapsing distinct subgenres (Lekompo, Maskandi) into generic high-volume clusters.

---

## 6. Spherical In-Processing Countermeasures

To cap escalation ($\alpha \le 1$), the in-processing method in `representation-alignment/` uses a three-part loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{proto}} \mathcal{L}_{\text{proto}} + \lambda_{\text{gini}} \mathcal{L}_{\text{gini}}$$

1. **Spherical Projection ($\mathbb{S}^{D-1}$):**
   $$\mathbf{u} \leftarrow \frac{\mathbf{u}}{\|\mathbf{u}\|_2}, \quad \mathbf{v} \leftarrow \frac{\mathbf{v}}{\|\mathbf{v}\|_2}, \quad \mathbf{p}_k \leftarrow \frac{\mathbf{p}_k}{\|\mathbf{p}_k\|_2}$$
   Forces all embeddings to unit length ($\|\mathbf{v}_d\|_2 = \|\mathbf{v}_p\|_2 = 1.0$), removing norm-based popularity advantages.

2. **Prototype Spanning ($\mathcal{L}_{\text{proto}}$):**
   $$\mathcal{L}_{\text{proto}} = \frac{1}{M}\sum_{i=1}^M \left(1 - \max_k \langle \mathbf{v}_i, \mathbf{p}_k \rangle\right) + \frac{\gamma}{K(K-1)} \sum_{j \neq k} \langle \mathbf{p}_j, \mathbf{p}_k \rangle$$
   Maintains distinct subgenre centroids on $\mathbb{S}^{D-1}$ to prevent geometric collapse.

3. **Differentiable Gini Penalty ($\mathcal{L}_{\text{gini}}$):**
   $$\mathcal{L}_{\text{gini}} = \frac{\sum_{i=1}^M \sum_{j=1}^M \sqrt{(a(i) - a(j))^2 + \epsilon}}{2 M \left( \sum_{i=1}^M a(i) + \epsilon_{\text{denom}} \right)}$$
   Penalizes high exposure concentration, counteracting $\Delta \bar{y}_{T+1}$ and keeping the dynamical exponent bounded:
   $$\alpha_{\text{constrained}} \le 1.0$$

---

## References

1. **Mansoury, M., Abdollahpouri, H., Pechenizkiy, M., Mobasher, B., & Burke, R.** (2020). *Feedback Loop and Bias Amplification in Recommender Systems*. Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM '20), pp. 2145–2148.
2. **Truong, Q. T., Salah, A., & Lauw, H. W.** (2025). *Mitigating Feedback Loops in Latent Representation Recommenders*. IEEE Transactions on Knowledge and Data Engineering (TKDE).
3. **Burke, R., Sonboli, N., & Ordonez-Gauger, A.** (2018). *Balanced Neighbourhoods for Multi-Sided Fairness in Recommendation*. In Conference on Fairness, Accountability and Transparency (FAT*), pp. 202–214.
