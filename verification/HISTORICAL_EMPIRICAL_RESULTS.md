# Historical empirical results

Archived from `docs/DOCUMENTATION.md`. These values are historical and
**were not rerun or independently validated by the current CI**. Do not
treat the correlations or causal-language conclusions below as current
verification evidence.

## Empirical Validation Results

### 1. Static Validation Suite

Six controlled empirical protocols executed across canonical architecture families:

#### Protocol 1: Architectural Discriminability

| Architecture | Nodes ($\|V\|$) | Edges ($\|E\|$) | Density ($D$) | Coherence ($C$) | Bottleneck ($B$) | Resilience ($R$) |
|---|---|---|---|---|---|---|
| Sequential MLP | 9 | 8 | 0.1111 | 0.9970 | 1.1251 | 0.3600 |
| Feedforward CNN | 9 | 8 | 0.1111 | 0.9970 | 1.1251 | 0.3600 |
| ResNet Block | 8 | 9 | 0.1607 | 0.9956 | 1.5000 | 0.3333 |
| DenseNet Block | 12 | 17 | 0.1287 | 0.9951 | 1.7653 | 0.2608 |
| U-Net Toy | 14 | 14 | 0.0769 | 0.9985 | 1.5000 | 0.3333 |
| Self-Attention Toy | 10 | 12 | 0.1333 | 0.9951 | 1.6673 | 0.2941 |

Sequential MLP and CNN produce identical topological metrics due to identical sequential dependency structures. DenseNet exhibits maximum connectivity dispersion ($B=1.7653$), while Self-Attention exhibits high concentration ($B=1.6673$).

#### Protocol 2: Controlled Synthetic Confounder Isolation

Fixed-size graphs with identical order $|V|=10$ and size $|E|=9$:

| Topology | Density | Coherence | Bottleneck | Resilience |
|---|---|---|---|---|
| Linear Chain | 0.1000 | 0.9979 | 1.1115 | 0.3571 |
| Star (Single Hub) | 0.1000 | 0.9210 | 5.0025 | 0.3448 |
| Binary Tree | 0.1000 | 0.9896 | 1.6673 | 0.3571 |
| Bottleneck Funnel | 0.1000 | 0.9814 | 2.2231 | 0.3571 |

Under identical $|V|$ and $|E|$, the Star topology yields maximum bottleneck ($5.0025$) and lowest coherence ($0.9210$), proving metrics measure connectivity geometry rather than counts.

#### Protocol 3: Width-Scale Invariance

| Configuration | Parameters | $\|V\|$ | $\|E\|$ | $B$ | $C$ |
|---|---|---|---|---|---|
| MLP width=64 | 13,130 | 9 | 8 | 1.1251 | 0.9970 |
| MLP width=512 | 793,098 | 9 | 8 | 1.1251 | 0.9970 |

A $\sim 60\times$ increase in parameter volume yields bit-for-bit identical topological metrics, proving complete decoupling of topology from parameter scale.

#### Protocol 4: Metric Correlation & Non-Collinearity

| | Density | Coherence | Bottleneck | Resilience |
|---|---|---|---|---|
| **Density** | $1.000$ | $-0.936$ | $0.328$ | $-0.389$ |
| **Coherence** | $-0.936$ | $1.000$ | $-0.559$ | $0.666$ |
| **Bottleneck** | $0.328$ | $-0.559$ | $1.000$ | $-0.928$ |
| **Resilience** | $-0.389$ | $0.666$ | $-0.928$ | $1.000$ |

#### Protocol 5: Perturbation Predictive Validity

| Architecture | Spearman $\rho$ | $p$-value | Pearson $r$ | $p$-value |
|---|---|---|---|---|
| ResNet Block | 0.7555 | 0.0495 | 0.7783 | 0.0392 |
| U-Net Toy | 0.4513 | 0.1216 | 0.3284 | 0.2732 |

#### Protocol 6: Asymptotic Order Scaling ($N \in \{10, 25, 50, 100\}$)

- **Linear Chain:** $B \to 1.0$ asymptotically ($1.1115 \to 1.0150$).
- **Star Graph:** $B$ scales linearly with $N/2$ ($5.0025 \to 50.2512$).

---

### 2. Dynamic Flow Benchmark Suite

Seven dynamic empirical protocols across canonical models:

#### Architecture-Level Dynamic Summary

| Architecture | Density | Coherence | Bottleneck | Resilience | Act RMS | Grad RMS | $G_{\max}$ | $\rho(A,G)$ | Mean CV |
|---|---|---|---|---|---|---|---|---|---|
| Sequential MLP | 0.1111 | 0.9970 | 1.1251 | 0.3600 | 0.3011 | 0.0036 | 3.5114 | $-1.0000$ | 0.0409 |
| ResNet Block | 0.1428 | 0.9956 | 1.5000 | 0.3333 | 0.6126 | 0.0045 | 2.8582 | $-0.2182$ | 0.0243 |
| DenseBlock | 0.1287 | 0.9951 | 1.7653 | 0.2608 | 0.5014 | 0.0034 | 3.6713 | $-0.3825$ | 0.0220 |
| U-Net Toy | 0.0769 | 0.9985 | 1.5000 | 0.3333 | 0.2512 | 0.0031 | 4.0313 | $-0.6784$ | 0.0534 |
| Self-Attention | 0.1333 | 0.9951 | 1.6673 | 0.2941 | 0.5895 | 0.0039 | 3.2122 | $+0.2997$ | 0.0365 |

#### Research Findings (Q1–Q5)

**Q1 — Structural vs. Functional Importance:**
Across 53 nodes, Spearman $\rho = -0.0583$ ($p = 0.678$) between static structural centrality and dynamic gradient magnitude. High structural connectivity does not guarantee high gradient flux. Type II (high topology, low gradient) and Type III (low topology, high gradient) nodes confirm structural-functional divergence.

**Q2 — Structural vs. Gradient Bottleneck:**
Static topological bottleneck $B$ identifies structural routing choke points (skip merges), while dynamic $G_{\max}$ identifies computational signal concentration points (final projections, output layers).

**Q3 — Gradient Attenuation Across Paths:**
In ResNet Block, the residual shortcut (`x → add`) exhibits backward ratio $0.9652$ (log ratio $-0.0354$, minimal attenuation). The sequential convolutional path (`x → l1 → r1 → l2 → add`) experiences substantial gradient attenuation ($\Delta = 0.38 \to 1.76$).

**Q4 — Forward-Backward Alignment:**
Activation scale is an unreliable proxy for gradient flux:

| Architecture | $\rho(A, G)$ |
|---|---|
| Sequential MLP | $-1.0000$ |
| U-Net Toy | $-0.6784$ |
| DenseBlock | $-0.3825$ |
| ResNet Block | $-0.2182$ |
| Self-Attention | $+0.2997$ |

**Q5 — Static Perturbation vs. Dynamic Stability:**
Static perturbation resilience correlates positively with reachability retention under ablation ($\rho = 0.1699$), verifying that topological fragility scores predict structural disconnection under node removal.
