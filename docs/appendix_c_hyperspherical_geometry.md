# Appendix C: Hyperspherical Representation Geometry and Prototype Alignment on $\mathbb{S}^{D-1}$
---

## 1. Geometric Motivation: Why Euclidean $\mathbb{R}^D$ Fails in Cultural Curation

Standard Euclidean collaborative filtering (such as matrix factorization or unconstrained Two-Tower networks) combines two properties into a single embedding vector $\mathbf{z} \in \mathbb{R}^D$:
1. **Magnitude ($\|\mathbf{z}\|_2$):** Scales with overall item interaction frequency and platform popularity.
2. **Direction ($\mathbf{z} / \|\mathbf{z}\|_2$):** Encodes the latent acoustic and cultural characteristics of the music.

### 1.1 Popularity Norm Dominance
When models score items using unconstrained dot products $\hat{y}_{ui} = \langle \mathbf{u}_u, \mathbf{v}_i \rangle$:
$$\hat{y}_{ui} = \|\mathbf{u}_u\|_2 \|\mathbf{v}_i\|_2 \cos \angle(\mathbf{u}_u, \mathbf{v}_i)$$

For chart-topping mainstream tracks $d \in \mathcal{I}_d$, repeated gradient updates inflate the vector norm:
$$\|\mathbf{v}_d\|_2 \gg \|\mathbf{v}_p\|_2 \quad \forall \; p \in \mathcal{I}_p$$

As a result, a dominant track $d$ receives high predicted affinity $\hat{y}_{ud}$ across nearly all users $u$, even when the angular match is poor ($\cos \angle(\mathbf{u}_u, \mathbf{v}_d) \ll 1$). The model suppresses local subgenres (*Gqom, Lekompo, Maskandi*) with strong angular alignment simply because their embedding norms are small ($\|\mathbf{v}_p\|_2 \approx 0$).

---

## 2. Formulation of the Unit Hypersphere $\mathbb{S}^{D-1}$

To separate cultural style from total playback volume, we project user embeddings, item embeddings, and subgenre prototype centroids onto the unit sphere in $\mathbb{R}^D$:
$$\mathbb{S}^{D-1} \triangleq \left\{ \mathbf{x} \in \mathbb{R}^D \;\middle|\; \|\mathbf{x}\|_2 = \sqrt{\sum_{k=1}^D x_k^2} = 1.0 \right\}$$

### 2.1 Spherical Projection
For any non-zero vector $\tilde{\mathbf{z}} \in \mathbb{R}^D \setminus \{\mathbf{0}\}$:
$$\Pi_{\mathbb{S}^{D-1}}(\tilde{\mathbf{z}}) \triangleq \frac{\tilde{\mathbf{z}}}{\|\tilde{\mathbf{z}}\|_2 + \epsilon_{\text{norm}}}$$
where $\epsilon_{\text{norm}} = 10^{-8}$ prevents division by zero.

```
                                  [The Unit Sphere 𝕊^(D-1)]
                                              
                                              +z
                                              ▲
                                              │  p_Amapiano
                                        . ─── * ─── .
                                     .        │        .
                                   /          │          \
                                  /           │           \
                   -x ◄──────────●────────────┼────────────●──────────► +x
                                 │ \          │           /│
                                 │   \        │ p_Gqom  /  │
                                 │     . ──── * ──── .     │
                                 │            │            │
                                 │            │ p_Maskandi │
                                  \           *           /
                                   \          │          /
                                     .        │        .
                                        . ─── * ─── .
                                              │  p_Lekompo
                                              ▼
                                              -z
```

---

## 3. Metric Properties and Euclidean-Cosine Equivalence

On the sphere $(\mathbb{S}^{D-1}, g_{\mathbb{S}})$, geodesic distance $d_{\text{geo}}(\mathbf{x}, \mathbf{y})$ between points $\mathbf{x}, \mathbf{y} \in \mathbb{S}^{D-1}$ is:
$$d_{\text{geo}}(\mathbf{x}, \mathbf{y}) = \arccos\langle \mathbf{x}, \mathbf{y} \rangle = \theta \in [0, \pi]$$

---

### Theorem 1 (Equivalence of Chordal Euclidean Distance and Cosine Similarity)
*For unit vectors $\mathbf{x}, \mathbf{y} \in \mathbb{S}^{D-1}$, minimizing Euclidean distance $\|\mathbf{x} - \mathbf{y}\|_2$ is equivalent to maximizing cosine similarity $\langle \mathbf{x}, \mathbf{y} \rangle$.*

### Proof:
Expanding the squared Euclidean norm:
$$\|\mathbf{x} - \mathbf{y}\|_2^2 = \langle \mathbf{x} - \mathbf{y}, \mathbf{x} - \mathbf{y} \rangle = \|\mathbf{x}\|_2^2 + \|\mathbf{y}\|_2^2 - 2\langle \mathbf{x}, \mathbf{y} \rangle$$
Since $\mathbf{x}, \mathbf{y} \in \mathbb{S}^{D-1}$, $\|\mathbf{x}\|_2^2 = 1$ and $\|\mathbf{y}\|_2^2 = 1$:
$$\|\mathbf{x} - \mathbf{y}\|_2^2 = 2 - 2\langle \mathbf{x}, \mathbf{y} \rangle = 2\left(1 - \cos \angle(\mathbf{x}, \mathbf{y})\right)$$

Taking the square root:
$$\|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{2\left(1 - \langle \mathbf{x}, \mathbf{y} \rangle\right)}$$

Because $f(t) = \sqrt{2(1 - t)}$ is strictly decreasing for $t \in [-1, 1]$:
$$\arg\min_{\mathbf{y} \in \mathbb{S}^{D-1}} \|\mathbf{x} - \mathbf{y}\|_2 \equiv \arg\max_{\mathbf{y} \in \mathbb{S}^{D-1}} \langle \mathbf{x}, \mathbf{y} \rangle$$
$$\tag*{$\blacksquare$}$$

---

## 4. Multi-Prototype Clustering on $\mathbb{S}^{D-1}$

Let $\{\mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_K\} \subset \mathbb{S}^{D-1}$ be $K$ learnable prototype vectors representing distinct South African subgenres ($K=4$ for *Amapiano, Gqom, Lekompo, Maskandi*).

### 4.1 Prototype Loss $\mathcal{L}_{\text{proto}}$
The prototype loss balances cluster tightness with prototype separation:
$$\mathcal{L}_{\text{proto}} = \mathcal{L}_{\text{cluster}} + \frac{\gamma}{2} \mathcal{L}_{\text{sep}}$$

1. **Compactness Loss ($\mathcal{L}_{\text{cluster}}$):**
   Pulls item embedding $\mathbf{v}_i \in \mathbb{S}^{D-1}$ toward its nearest prototype centroid:
   $$\mathcal{L}_{\text{cluster}} = \frac{1}{M}\sum_{i=1}^M \left(1 - \max_{k \in \{1,\dots,K\}} \langle \mathbf{v}_i, \mathbf{p}_k \rangle\right)$$

2. **Separation Loss ($\mathcal{L}_{\text{sep}}$):**
   Pushes prototype centroids apart to prevent them from collapsing into a single dominant genre:
   $$\mathcal{L}_{\text{sep}} = \frac{1}{K(K-1)} \sum_{j=1}^K \sum_{k \neq j}^K \langle \mathbf{p}_j, \mathbf{p}_k \rangle = \frac{1}{K(K-1)} \left( \left\| \sum_{k=1}^K \mathbf{p}_k \right\|_2^2 - K \right)$$

---

## 5. Optimal Spanning: Equiangular Tight Frames (ETF)

We can identify the global minimum of the prototype separation loss $\mathcal{L}_{\text{sep}}$.

---

### Theorem 2 (Equiangular Tight Frame Prototype Spanning)
*Let $\mathcal{P} = \{\mathbf{p}_1, \dots, \mathbf{p}_K\} \subset \mathbb{S}^{D-1}$ with $K \le D + 1$. The global minimum of $\mathcal{L}_{\text{sep}}(\mathcal{P})$ occurs when $\mathcal{P}$ forms a regular simplex (Equiangular Tight Frame) centred at the origin, with pairwise cosine similarity:*
$$\langle \mathbf{p}_j, \mathbf{p}_k \rangle = -\frac{1}{K - 1} \quad \forall \; j \neq k$$
*The minimum angular separation between any two subgenre prototypes is:*
$$\theta_{\min} = \arccos\left(-\frac{1}{K - 1}\right) > \frac{\pi}{2}$$

---

### Proof:
From the separation formula:
$$\sum_{j=1}^K \sum_{k \neq j}^K \langle \mathbf{p}_j, \mathbf{p}_k \rangle = \left\| \sum_{k=1}^K \mathbf{p}_k \right\|_2^2 - K$$

Because $\|\sum \mathbf{p}_k\|_2^2 \ge 0$:
$$\sum_{j \neq k} \langle \mathbf{p}_j, \mathbf{p}_k \rangle \ge -K$$

Equality holds when the sum of centroids is zero:
$$\sum_{k=1}^K \mathbf{p}_k = \mathbf{0}$$

Assuming symmetric pairwise similarity ($\langle \mathbf{p}_j, \mathbf{p}_k \rangle = c$ for all $j \neq k$):
$$K(K-1) c = -K \implies c = -\frac{1}{K - 1}$$

For $K = 4$ subgenres (*Amapiano, Gqom, Lekompo, Maskandi*):
$$\langle \mathbf{p}_j, \mathbf{p}_k \rangle = -\frac{1}{3} \approx -0.3333$$
$$\theta_{\min} = \arccos\left(-\frac{1}{3}\right) \approx 109.47^\circ > 90^\circ$$

Minimizing $\mathcal{L}_{\text{sep}}$ places the four prototype vectors at the vertices of a regular tetrahedron on $\mathbb{S}^{D-1}$, maintaining angular separation between subgenres and preventing representation collapse.
$$\tag*{$\blacksquare$}$$

---

## 6. Contrastive Properties: Alignment vs. Uniformity

Following Wang & Isola (2020), we measure representation quality on the sphere using alignment and uniformity:

1. **Alignment:**
   $$\mathcal{L}_{\text{align}}(f) \triangleq \mathbb{E}_{(x, x^+) \sim p_{\text{pos}}}\left[ \|f(x) - f(x^+)\|_2^2 \right]$$
2. **Uniformity:**
   $$\mathcal{L}_{\text{uniform}}(f) \triangleq \ln \mathbb{E}_{x, y \stackrel{i.i.d.}{\sim} p_{\text{data}}}\left[ \exp\left(-2\|f(x) - f(y)\|_2^2\right) \right]$$

In our network:
- $\mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{cluster}}$ promotes **Alignment** by grouping tracks around their subgenre prototype.
- $\mathcal{L}_{\text{sep}} + \mathcal{L}_{\text{gini}}$ promotes **Uniformity** by distributing item embeddings across the sphere, preventing items from collapsing into a single popularity axis.

---

## References

1. **Wang, T., & Isola, P.** (2020). *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*. Proceedings of the 37th International Conference on Machine Learning (ICML '20), PMLR 119:9929-9939.
2. **Mettes, P., van der Pol, E., & Snoek, C. G.** (2019). *Hyperspherical Prototype Networks*. Advances in Neural Information Processing Systems (NeurIPS 2019), 32.
3. **Snell, J., Swersky, K., & Zemel, R.** (2017). *Prototypical Networks for Few-shot Learning*. Advances in Neural Information Processing Systems (NeurIPS 2017), 30.
4. **Cohn, H., & Kumar, A.** (2007). *Universally optimal distribution of points on spheres*. Journal of the American Mathematical Society, 20(1), 99-148.
