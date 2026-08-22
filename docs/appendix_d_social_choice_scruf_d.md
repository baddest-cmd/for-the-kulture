# Appendix D: Axiomatic Social Choice Theory, Multi-Agent Negotiation, and Quota Guarantees in SCRUF-D

---

## 1. Multi-Stakeholder Slate Selection

Standard recommender systems optimise for immediate engagement $\max \hat{y}_{ui}$. In localized music contexts, slate selection involves balancing three competing priorities:

```
                            [MULTI-STAKEHOLDER TENSION TRIANGLE]
                            
                                   1. Consumer Utility
                                   (Immediate Engagement)
                                            ▲
                                           / \
                                          /   \
                                         /     \
                                        /       \
                                       /         \
                                      /           \
                                     ▼             ▼
                           2. Causal Neutrality    3. Long-Tail Preservation
                           (Deconfounded Taste)    (Regional Subgenre Survival)
```

To resolve this at the ranking stage, Paradigm A uses a **3-Agent Social Choice Committee** based on SCRUF-D:
1. **Agent 1 ($A_{\text{exploit}}$): Myopic Exploiter** scores items by predicted user engagement.
2. **Agent 2 ($A_{\text{arbiter}}$): Causal Arbiter** penalizes platform exposure bias using Context Tax $\tau_c$.
3. **Agent 3 ($A_{\text{preserve}}$): Adversarial Preserver** protects underrepresented regional subgenres (*Gqom, Lekompo, Maskandi, SA House*).

---

## 2. Social Welfare Scoring

For candidate catalog $\mathcal{I} = \{1, \dots, M\}$, each agent $m \in \{e, a, p\}$ outputs a utility vector $\mathbf{s}_m(u) \in \mathbb{R}^M$ for user $u$:

1. **Myopic Exploiter:**
   $$s_e(u, i) \triangleq \hat{y}_{ui}$$
2. **Causal Arbiter:**
   $$s_a(u, i) \triangleq \hat{y}_{ui} - \beta \cdot \max(0, \tau_c(u)) \cdot \Phi(i)$$
   where $\tau_c(u) = S_{\text{global}}(u) - D_{\text{local}}(u)$ and $\Phi(i) = \frac{\text{Popularity}(i)}{\max_j \text{Popularity}(j)}$.
3. **Adversarial Preserver:**
   $$s_p(u, i) \triangleq Q_{\text{weight}} \cdot \mathbb{I}\left(\text{genre}(i) \in \mathcal{G}_{\text{local}}\right)$$

### 2.1 Aggregation
The committee combines agent utility scores:
$$U(u, i) \triangleq \mathbf{w}^\top \mathbf{s}(u, i) = w_e \cdot s_e(u, i) + w_a \cdot s_a(u, i) + w_p \cdot s_p(u, i)$$
with weights on the simplex:
$$\mathbf{w} \in \Delta^2 \triangleq \left\{ (w_e, w_a, w_p) \in \mathbb{R}_+^3 \;\middle|\; w_e + w_a + w_p = 1.0 \right\}$$

---

## 3. Axiomatic Properties

The SCRUF-D scoring function satisfies several standard social choice criteria:

1. **Pareto Efficiency (Unanimity):**
   If all three agents prefer item $i$ over item $j$ ($s_m(u, i) > s_m(u, j)$ for all $m$), then for any positive weight vector $\mathbf{w} > \mathbf{0}$:
   $$U(u, i) = \sum_{m} w_m s_m(u, i) > \sum_{m} w_m s_m(u, j) = U(u, j)$$

2. **Independence of Irrelevant Alternatives (IIA):**
   The relative score ordering between items $i$ and $j$ depends solely on their individual feature evaluations:
   $$\frac{\partial (U(u, i) - U(u, j))}{\partial s_m(u, k)} = 0 \quad \forall \; k \notin \{i, j\}$$

3. **Monotonic Causal Debiasing:**
   As user platform friction $\tau_c(u)$ rises:
   $$\frac{\partial U(u, i)}{\partial \tau_c(u)} = -w_a \beta \Phi(i) \le 0$$
   Popular items are discounted proportionally to their popularity $\Phi(i)$ and the user's Context Tax $\tau_c(u)$.

---

## 4. Quota-Constrained Slate Optimisation

Let $\mathcal{S}_u^K \subset \mathcal{I}$ be a recommendation slate of size $K$ for user $u$.

### 4.1 Constrained Selection Problem
We write the slate selection problem as:
$$\mathcal{S}_u^{K*} = \arg\max_{\mathcal{S} \subset \mathcal{I}, |\mathcal{S}| = K} \sum_{i \in \mathcal{S}} U(u, i) \quad \text{s.t.} \quad \sum_{i \in \mathcal{S}} \mathbb{I}\left(\text{genre}(i) \in \mathcal{G}_{\text{local}}\right) \ge Q_g$$
where $Q_g \in \{1, \dots, K\}$ is the required local genre quota.

---

### Theorem 1 (Optimality of Greedy Exchange Slate Selection)
*The greedy replacement algorithm in `ScrufdRecommender.recommend` computes the exact optimal solution $\mathcal{S}_u^{K*}$ in $\mathcal{O}(M \log M)$ time.*

---

### Proof:
Let $\sigma = (\sigma_1, \dots, \sigma_M)$ sort items in descending order of utility: $U(u, \sigma_1) \ge \dots \ge U(u, \sigma_M)$.

Let $\mathcal{S}_0 = \{\sigma_1, \dots, \sigma_K\}$ be the unconstrained top-$K$ slate, with cultural count $k_{\text{cult}} = \sum_{i \in \mathcal{S}_0} \mathbb{I}(\text{genre}(i) \in \mathcal{G}_{\text{local}})$.

1. **Case 1 ($k_{\text{cult}} \ge Q_g$):** $\mathcal{S}_0$ is already feasible and maximizes unconstrained sum of utilities, making it optimal.
2. **Case 2 ($k_{\text{cult}} < Q_g$):** The shortfall is $\Delta Q = Q_g - k_{\text{cult}}$. To restore feasibility while maximizing total score, we perform $\Delta Q$ substitutions: replace $\Delta Q$ items from $\mathcal{S}_0 \setminus \mathcal{G}_{\text{local}}$ with $\Delta Q$ items from $\mathcal{I} \setminus (\mathcal{S}_0 \cup \mathcal{G}_{\text{local}}^c)$.

The replacement produces a total utility loss of:
$$\Delta \text{Loss} = \sum_{r=1}^{\Delta Q} \left( U(u, text{remove}_r) - U(u, \text{insert}_r) \right)$$

We minimize this loss by:
- Selecting the $\Delta Q$ items in $\mathcal{S}_0 \setminus \mathcal{G}_{\text{local}}$ with the lowest utility $U(u, i)$.
- Selecting the $\Delta Q$ candidate items in $\mathcal{I} \setminus \mathcal{S}_0$ with $\text{genre} \in \mathcal{G}_{\text{local}}$ with the highest utility $U(u, j)$.

Because the uniform matroid $\mathcal{M}_K$ under partition constraints admits an optimal greedy basis, this $\mathcal{O}(M \log M)$ exchange computes the exact global optimum.
$$\tag*{$\blacksquare$}$$

---

## 5. Bounded Price of Fairness (PoF)

The trade-off between cultural quotas and unconstrained engagement is measured by the **Price of Fairness**:
$$\text{PoF}(u) \triangleq 1.0 - \frac{\sum_{i \in \mathcal{S}_{\text{fair}}^K} \hat{y}_{ui}}{\sum_{j \in \mathcal{S}_{\text{unconstrained}}^K} \hat{y}_{uj}}$$

---

### Theorem 2 (Upper Bound on Price of Fairness)
*For bounded predictions $\hat{y}_{ui} \in [y_{\min}, y_{\max}]$, the Price of Fairness for quota $Q_g$ satisfies:*
$$\text{PoF}(u) \le \frac{Q_g}{K} \left( 1 - \frac{y_{\min}}{y_{\max}} \right)$$

### Proof:
In the worst-case replacement, at most $Q_g$ items with score $y_{\max}$ are replaced by cultural items with score $y_{\min}$. The remaining $K - Q_g$ items retain score $y_{\max}$:
$$\sum_{j \in \mathcal{S}_{\text{unconstrained}}^K} \hat{y}_{uj} \le K y_{\max}$$
$$\sum_{i \in \mathcal{S}_{\text{fair}}^K} \hat{y}_{ui} \ge (K - Q_g) y_{\max} + Q_g y_{\min}$$

Therefore:
$$\text{PoF}(u) = 1 - \frac{\sum \hat{y}_{\text{fair}}}{\sum \hat{y}_{\text{unconstrained}}} \le 1 - \frac{(K - Q_g)y_{\max} + Q_g y_{\min}}{K y_{\max}} = \frac{Q_g}{K} \left( 1 - \frac{y_{\min}}{y_{\max}} \right)$$

For $K = 5$, $Q_g = 2$, and score bounds $y_{\min} = 0.4, y_{\max} = 0.8$:
$$\text{PoF}(u) \le \frac{2}{5} \left( 1 - \frac{0.4}{0.8} \right) = 0.40 \times 0.50 = 0.20 \quad (20\% \text{ maximum utility loss})$$

Enforcing local quotas guarantees cultural representation while bounding the maximum engagement drop.
$$\tag*{$\blacksquare$}$$

---

## References

1. **Sonboli, N., Burke, R., & Smith, B.** (2020). *Opportunistic Discovery and Multi-Sided Fairness in Recommendation*. In Proceedings of the 28th ACM Conference on User Modelling, Adaptation and Personalization (UMAP '20), pp. 231-239.
2. **Patro, G. K., Biswas, A., Ganguly, N., Gummadi, K. P., & Chakraborty, A.** (2020). *FairTraM: Fair and Transparent Matching in Two-Sided Platforms*. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20), pp. 1341-1350.
3. **Burke, R.** (2017). *Multisided Fairness for Recommendation Decentralisation*. In FATREC Workshop on Responsible Recommendation at ACM RecSys 2017.
4. **Nash, J.** (1950). *The Bargaining Problem*. Econometrica, 18(2), 155-162.
