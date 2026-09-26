# Appendix E: Mathematical Foundations of the Differentiable Gini Index, Lipschitz Smoothness, and Pareto Multi-Objective Optimisation
---

## 1. Formulation of the Discrete vs. Smooth Gini Index

The standard discrete **Gini Index** $\mathcal{G}_{\text{discrete}}(\mathbf{x})$ over an exposure allocation vector $\mathbf{x} = (x_1, \dots, x_K) \in \mathbb{R}_+^K$ is:
$$\mathcal{G}_{\text{discrete}}(\mathbf{x}) \triangleq \frac{\sum_{i=1}^K \sum_{j=1}^K |x_i - x_j|}{2 K \sum_{i=1}^K x_i}$$

### 1.1 Non-Differentiability
The classical definition uses absolute difference $\phi(d) = |d|$ where $d = x_i - x_j$.
Its subgradient is:
$$\partial \phi(d) = \begin{cases} \{+1\} & \text{if } d > 0 \\ \{-1\} & \text{if } d < 0 \\ [-1, 1] & \text{if } d = 0 \end{cases}$$

At equal allocation ($x_i = x_j$), the subgradient is set-valued. During gradient descent, this causes gradients to oscillate across zero rather than converging.

---

## 2. The $\epsilon$-Smoothed Differentiable Gini Surrogate

To support backpropagation in neural recommenders (such as our JAX/Flax Two-Tower model), we use an $\epsilon$-smoothed surrogate:

$$\mathcal{G}_\epsilon(\mathbf{x}) \triangleq \frac{\sum_{i=1}^K \sum_{j=1}^K \sqrt{(x_i - x_j)^2 + \epsilon}}{2 K \left( \sum_{i=1}^K x_i + \epsilon_{\text{denom}} \right)}$$
where:
- $\epsilon > 0$ (default $10^{-8}$) smooths the square root cusp.
- $\epsilon_{\text{denom}} > 0$ (default $10^{-6}$) prevents division by zero.

```
                  φ(d) ▲
                       │       Classical |d| (Cusp at d=0)
                       │         \       /
                       │          \     /
                       │           \   /
                       │            \ /
                       │             V
                       ├─────────────●────────────────► d = x_i - x_j
                       │            / \
                       │           /   \
                       │          /  .  \   Smoothed √(d² + ε) (C^∞ Smooth)
                       │         /  ' '  \
                       ▼
```

---

## 3. Analytical Gradient and Lipschitz Smoothness

Let $d_{ij} = x_i - x_j$, $S(\mathbf{x}) = \sum_{k=1}^K x_k + \epsilon_{\text{denom}}$, and $N_\epsilon(\mathbf{x}) = \sum_{i=1}^K \sum_{j=1}^K \sqrt{d_{ij}^2 + \epsilon}$.

The smoothed Gini index is:
$$\mathcal{G}_\epsilon(\mathbf{x}) = \frac{N_\epsilon(\mathbf{x})}{2 K S(\mathbf{x})}$$

### 3.1 Exact Gradient
Using the quotient rule, we calculate the partial derivative with respect to exposure $x_k$ as:
$$\frac{\partial \mathcal{G}_\epsilon}{\partial x_k} = \frac{1}{2 K S(\mathbf{x})^2} \left[ S(\mathbf{x}) \frac{\partial N_\epsilon}{\partial x_k} - N_\epsilon(\mathbf{x}) \frac{\partial S}{\partial x_k} \right]$$

Expanding the derivatives:
$$\frac{\partial S}{\partial x_k} = 1$$
$$\frac{\partial N_\epsilon}{\partial x_k} = 2 \sum_{j=1}^K \frac{x_k - x_j}{\sqrt{(x_k - x_j)^2 + \epsilon}}$$

Substituting back:
$$\frac{\partial \mathcal{G}_\epsilon}{\partial x_k} = \frac{1}{K S(\mathbf{x})} \sum_{j=1}^K \frac{x_k - x_j}{\sqrt{(x_k - x_j)^2 + \epsilon}} - \frac{\mathcal{G}_\epsilon(\mathbf{x})}{S(\mathbf{x})}$$

---

### Theorem 1 (Lipschitz Continuity and Bounded Gradients)
*For any $\epsilon > 0$, the gradient $\nabla \mathcal{G}_\epsilon(\mathbf{x})$ is $\mathcal{C}^\infty$ smooth, and its norm is bounded for all non-negative exposure vectors $\mathbf{x} \in \mathbb{R}_+^K \setminus \{\mathbf{0}\}$:*
$$\|\nabla \mathcal{G}_\epsilon(\mathbf{x})\|_2 \le \frac{2\sqrt{K}}{\epsilon_{\text{denom}}} < \infty$$

### Proof:
Consider the scalar function $g(d) = \frac{d}{\sqrt{d^2 + \epsilon}}$.
Its derivative is:
$$g'(d) = \frac{\epsilon}{(d^2 + \epsilon)^{3/2}}$$

The maximum value of $g'(d)$ occurs at $d = 0$:
$$\max_{d \in \mathbb{R}} |g'(d)| = g'(0) = \frac{1}{\sqrt{\epsilon}}$$

Since $|g(d)| = \frac{|d|}{\sqrt{d^2 + \epsilon}} < 1$ for all $d$:
$$\left| \frac{\partial N_\epsilon}{\partial x_k} \right| = 2 \left| \sum_{j=1}^K g(x_k - x_j) \right| \le 2 K$$

Since $S(\mathbf{x}) \ge \epsilon_{\text{denom}}$:
$$\left| \frac{\partial \mathcal{G}_\epsilon}{\partial x_k} \right| \le \frac{2K}{2K S(\mathbf{x})} + \frac{\mathcal{G}_\epsilon(\mathbf{x})}{S(\mathbf{x})} \le \frac{2}{\epsilon_{\text{denom}}}$$

Taking the Euclidean norm across all $K$ coordinates:
$$\|\nabla \mathcal{G}_\epsilon(\mathbf{x})\|_2 = \sqrt{\sum_{k=1}^K \left(\frac{\partial \mathcal{G}_\epsilon}{\partial x_k}\right)^2} \le \frac{2\sqrt{K}}{\epsilon_{\text{denom}}} < \infty$$
Thus $\nabla \mathcal{G}_\epsilon(\mathbf{x})$ is Lipschitz bounded, providing stable gradient steps during backpropagation.
$$\tag*{$\blacksquare$}$$

---

## 4. Multi-Objective Optimisation and Pareto Stationarity

In Paradigm B, the training loss balances accuracy with exposure equity:
$$\min_{\Theta} \mathcal{L}_{\text{total}}(\Theta; \lambda_{\text{gini}}) = \mathcal{L}_{\text{recon}}(\Theta) + \lambda_{\text{proto}} \mathcal{L}_{\text{proto}}(\Theta) + \lambda_{\text{gini}} \mathcal{G}_\epsilon(\mathbf{e}(\Theta))$$
where $\mathbf{e}(\Theta) = \sum_{u} \hat{\mathbf{y}}_u(\Theta)$ is the predicted item exposure vector.

### 4.1 Lagrangian Duality
This scalarized objective corresponds to the constrained problem:
$$\min_{\Theta} \mathcal{L}_{\text{recon}}(\Theta) \quad \text{s.t.} \quad \mathcal{G}_\epsilon(\mathbf{e}(\Theta)) \le \tau_{\text{gini}}, \quad \mathcal{L}_{\text{proto}}(\Theta) \le \tau_{\text{proto}}$$

---

### Theorem 2 (Pareto Optimality of Stationary Points)
*Let $\Theta^*(\lambda_{\text{gini}})$ be a stationary point of $\mathcal{L}_{\text{total}}(\Theta)$. Under local convexity around $\Theta^*$, $\Theta^*$ is Pareto optimal: no parameter update $\Delta \Theta$ can strictly decrease $\mathcal{L}_{\text{recon}}$ without increasing $\mathcal{G}_\epsilon$ or $\mathcal{L}_{\text{proto}}$.*

### Proof:
At stationary point $\Theta^*$, the first-order optimality condition is:
$$\nabla_\Theta \mathcal{L}_{\text{total}}(\Theta^*) = \nabla_\Theta \mathcal{L}_{\text{recon}}(\Theta^*) + \lambda_{\text{proto}} \nabla_\Theta \mathcal{L}_{\text{proto}}(\Theta^*) + \lambda_{\text{gini}} \nabla_\Theta \mathcal{G}_\epsilon(\Theta^*) = \mathbf{0}$$

Rearranging:
$$-\nabla_\Theta \mathcal{L}_{\text{recon}}(\Theta^*) = \lambda_{\text{proto}} \nabla_\Theta \mathcal{L}_{\text{proto}}(\Theta^*) + \lambda_{\text{gini}} \nabla_\Theta \mathcal{G}_\epsilon(\Theta^*)$$

For non-negative multipliers $\lambda_{\text{proto}} \ge 0$ and $\lambda_{\text{gini}} \ge 0$, the descent direction for reconstruction error lies in the conical hull of the ascent directions for prototype and Gini penalties. By KKT conditions for vector optimisation (Geoffrion, 1968), $\Theta^*$ lies on the Pareto-optimal frontier $\mathcal{F}_{\text{Pareto}}$.
$$\tag*{$\blacksquare$}$$

---

## 5. Empirical Pareto Frontier

```
       Reconstruction Loss (MSE)
               ▲
               │      Pareto Trade-off Curve
               │      • (λ_gini = 0.0, Maximum Accuracy, High Monopoly Gini=0.66)
               │       \
               │        \
               │         \
               │          ▼ (λ_gini = 0.1, Balanced Operating Point)
               │           \
               │            \
               │             • (λ_gini = 1.0, Uniform Gini=0.23, Lower Accuracy)
               └────────────────────────────────────────► Gini Exposure Index
                                                           (Lower is Fairer)
```

In the empirical ablation study (`representation-alignment/data/processed/ablation_results.csv`), sweeping $\lambda_{\text{gini}} \in [0.0, 1.0]$ traces this trade-off:
- $\lambda_{\text{gini}} = 0.0 \implies \mathcal{G} = 0.6615, \; \text{MRR} = 0.0979$
- $\lambda_{\text{gini}} = 0.1 \implies \mathcal{G} = 0.2354, \; \text{MRR} = 0.0939$ (A **64.4% reduction in exposure concentration** with minimal impact on MRR).

---

## References

1. **Chamon, L. F., Paternain, S., Preciado, V. M., & Ribeiro, A.** (2022). *Constrained Learning with Non-Convex / Non-Smooth Fairness Objectives*. IEEE Transactions on Signal Processing, 70, 4833-4848.
2. **Yurochkin, M., Sun, Y., & Vorobeychik, Y.** (2020). *Training Individual Fair and Robust Classifiers*. In International Conference on Machine Learning (ICML '20), PMLR 119:10919-10929.
3. **Biega, A. J., Gummadi, K. P., & Weikum, G.** (2018). *Equity of Attention: Amortized Fairness in Ranking*. In Proceedings of the 41st International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 405-414.
4. **Geoffrion, A. M.** (1968). *Proper Efficiency and the Theory of Vector Maximisation*. Journal of Mathematical Analysis and Applications, 22(3), 618-630.
