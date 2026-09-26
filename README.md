```markdown
# Socio-Technical Alignment and Algorithmic Flattening in Local Music Curation

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![JAX/Flax](https://img.shields.io/badge/JAX-Flax-red.svg)](https://github.com/google/jax)

This repository contains the official implementation, empirical dataset, and dual-alignment algorithms for the technical report **"Socio-Technical Alignment and Algorithmic Flattening: A Mixed-Methods Empirical Study of Local Music Curation in South Africa"** ($N=152$).

---

## Abstract

Commercial collaborative filtering recommenders systematically flatten localized South African music subgenres (*Motswako, Gqom, Lekompo, Mbaqanga, Bacardi*) into broad regional categories. This study investigates two primary empirical failure modes:

1. **The Inversion Problem:** Delivery constraints $C$ bias observed consumer behaviour $B$, where $P(B \mid M, C) \neq P(B \mid M)$.
2. **Metric Decoupling:** Respondent satisfaction and local discovery success were not significantly associated ($p=0.086$), though the pilot is underpowered to detect small effects ([Appendix F](./docs/appendix_f_limitations.md)). Reliance on in-app recommendation algorithms drops by **7.05 percentage points** (41.45% to 34.21%) when users search specifically for localized subgenres.

To mitigate algorithmic flattening, we evaluate two counter-alignment paradigms:

* **Paradigm A (Post-Processing Decision Boundary Alignment):** A 3-agent social choice committee using SCRUF-D (Myopic Exploiter, Causal Arbiter via Context Tax $\tau_c$, and Adversarial Preserver).
* **Paradigm B (In-Processing Representation Alignment):** A JAX/Flax Two-Tower neural network mapping embeddings onto the unit hypersphere $\mathbb{S}^{D-1}$ optimized over joint Reconstruction MSE, Prototype Simplex Spanning ($\mathcal{L}_{\text{proto}}$), and an $\mathcal{O}(K^2)$ Smooth Differentiable Gini Exposure Loss ($\mathcal{L}_{\text{gini}}$).

**Interpretive stance:** Pilot findings are directional, not confirmatory; Appendix F documents power, sparsity, and geographic limits.

---

## Repository Structure

```text
├── .agents/                    # Custom MLOps scripts
├── data/
│   ├── raw/                    # Immutable raw survey dataset (N=152)
│   └── processed/              # Feature-engineered arrays
├── docs/                       # Appendices (LaTeX proofs & causal DAGs)
├── notebooks/                  # Statistical verification & exploratory analysis
├── reports/                    # Generated empirical reports
├── representation-alignment/   # Paradigm B: JAX/Flax hyperspherical neural network
├── src/kulture/                # Core Python package (Paradigm A, metrics, pipelines)
├── tests/                      # Unit tests & property verification suite
└── pyproject.toml              # Build configuration & dependencies

```

## Dual Alignment Paradigms

### Paradigm A: Post-Processing Committee (SCRUF-D)

Combines agent preference scores into a composite ranking utility:

$$U(u, i) = w_e s_e(u,i) + w_a s_a(u,i) + w_p s_p(u,i)$$

* **Myopic Exploiter ($s_e$):** Ranks items by predicted engagement $\hat{y}_{ui}$.
* **Causal Arbiter ($s_a$):** Penalizes mainstream bias via Context Tax $\tau_c(u)$:

$$s_a(u,i) = \hat{y}_{ui} - \beta \cdot \max(0, \tau_c(u)) \cdot \Phi(i)$$

* **Adversarial Preserver ($s_p$):** Enforces local subgenre quotas $Q_g$ using an $\mathcal{O}(M \log M)$ greedy matroid exchange.

### Paradigm B: Spherical Two-Tower Model (JAX/Flax)

Embeddings are projected onto $\mathbb{S}^{D-1}$ to prevent popularity-driven embedding magnitude expansion:

$$\min_{\Theta} \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_{\text{proto}} \mathcal{L}_{\text{proto}} + \lambda_{\text{gini}} \mathcal{L}_{\text{gini}}$$

$$\mathcal{L}_{\text{gini}} = \frac{\sum_{i=1}^M \sum_{j=1}^M \sqrt{(e_i - e_j)^2 + \epsilon}}{2 M \left( \sum_{i=1}^M e_i + \epsilon_{\text{denom}} \right)}$$

---

## Key Empirical Findings

| Metric / Phenomenon | Quantitative Result | Empirical Implication |
| --- | --- | --- |
| **Local Discovery Shift** | $-7.05\%$ ($41.45\% \to 34.21\%$) | Recommendation reliance drops when searching for local subgenres. |
| **Metric Decoupling** | $p = 0.086$ (Mann-Whitney $U$) | Satisfaction and discovery lack significant association, though pilot is underpowered. |
| **Platform-Hopping** | $100\%$ Rule Confidence ($\text{Lift} = 1.15$) | Weak association; secondary-account use co-occurs with reported bias. |

---

## Installation & Usage

### Prerequisites

* Python 3.10+
* JAX / Flax

### Setup

```bash
# Clone repository
git clone https://github.com/baddest-cmd/for-the-kulture.git
cd for-the-kulture

# Create virtual environment and install package in editable mode
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev,jax]"

```

## Reproducibility & Testing

```bash
# Run unit test suite & property assertions
pytest tests/

# Run Paradigm A (SCRUF-D post-processing)
python -m kulture.pipelines.run_scruf

# Train Paradigm B (Hyperspherical Two-Tower Model)
python representation-alignment/train.py

```

## Technical Documentation & Appendices

Detailed mathematical proofs, causal diagrams, and experimental methodologies are located in the `docs/` directory:

* [Appendix A: Closed-Loop Dynamics & Proof of $\alpha > 1$](./docs/mathematical_derivations.md)
* [Appendix B: The Inversion Problem & CAFL](./docs/appendix_b_causal_deconfounding.md)
* [Appendix C: Hyperspherical Geometry](./docs/appendix_c_hyperspherical_geometry.md)
* [Appendix D: SCRUF-D Axiomatic Social Choice](./docs/appendix_d_social_choice_scruf_d.md)
* [Appendix E: Differentiable Gini & Pareto Optimization](./docs/appendix_e_smooth_gini_optimization.md)
* [Appendix F: Methodological Audit & Limitations](./docs/appendix_f_limitations.md)

## Citation

```bibtex
@article{peterson2026sociotechnical,
  title={Socio-Technical Alignment and Algorithmic Flattening: A Mixed-Methods Empirical Study of Local Music Curation in South Africa},
  author={Peterson, Shailoh},
  journal={Technical Report},
  year={2026}
}

```
