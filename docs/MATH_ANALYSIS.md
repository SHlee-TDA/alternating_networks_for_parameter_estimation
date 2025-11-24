# Mathematical Analysis: Distributional Support and Sim-to-Real Gap

## 1. Preliminaries: Hierarchical Dynamical Systems

We consider a dynamical system governed by physical parameters $\Theta \in \Omega \subset \mathbb{R}^d$, which follows a population distribution $\Theta \sim \pi(\theta)$. The state of the system at time $t$, denoted by $X_t \in \mathcal{X} \subset \mathbb{R}^n$ (e.g., Glucose and Insulin), evolves according to the dynamics dependent on $\Theta$.

We define the **Marginal Distribution** of the state at time $t$, denoted as $\mu_t$, as the measure on $\mathcal{X}$ obtained by integrating the conditional transition dynamics over the parameter space $\Omega$.

## 2. Measure-Theoretic Formulation of Distributions

### 2.1. Deterministic Simulation Distribution ($\mu_t^{det}$)
In the deterministic setting, the dynamics are given by an ODE $\dot{x} = f(x, \theta)$. Let $\Phi_t(\cdot; \theta): \mathcal{X} \to \mathcal{X}$ be the flow map induced by the ODE. The distribution at time $t$ is the **Push-forward Measure** of the parameter prior $\pi$:

$$
\mu_t^{det}(A) = \int_{\Omega} \mathbb{1}_A(\Phi_t(x_0; \theta)) \pi(\theta) d\theta, \quad \forall A \subset \mathcal{X}
$$

### 2.2. Stochastic Simulation Distribution ($\mu_t^{sde}$)
In the SDE setting, we introduce a diffusion term $\sigma(t)$. The dynamics follow $dX_t = f(X_t, \theta)dt + \sigma(t)dW_t$. The transition probability density $p(x, t | x_0, \theta)$ satisfies the Fokker-Planck equation. The marginal distribution is defined as:

$$
\mu_t^{sde}(x) = \int_{\Omega} p(x, t | x_0, \theta) \pi(\theta) d\theta
$$

### 2.3. Real Data Distribution ($\mu_t^{real}$)
The real data is observed as a finite set of samples $\mathcal{D} = \{x_t^{(i)}\}_{i=1}^N$. We treat this as an **Empirical Measure**:

$$
\mu_t^{real} = \frac{1}{N} \sum_{i=1}^N \delta_{x_t^{(i)}}
$$

## 3. Theoretical Analysis of Support Mismatch

We rigorously analyze why $\mu_t^{det}$ fails to cover $\mu_t^{real}$ and how $\mu_t^{sde}$ resolves this issue.

### **Proposition 1 (Dimensional Collapse of Deterministic Models)**
*Assumption:* The dimension of the parameter space $d$ is smaller than the dimension of the augmented state trajectory space or the variability of real data. The mapping $\theta \mapsto \Phi_t(x_0; \theta)$ is smooth.

*Statement:* The support of the deterministic measure, $\text{supp}(\mu_t^{det})$, is contained in a smooth manifold $\mathcal{M}_t$ of dimension at most $d$. If the real data includes noise $\epsilon$ such that $x_{real} \notin \mathcal{M}_t$ almost surely, then the supports are disjoint:
$$
\text{supp}(\mu_t^{det}) \cap \text{supp}(\mu_t^{real}) = \emptyset \quad (\text{a.s.})
$$
*Consequence:* $\mu_t^{det}$ is singular with respect to the Lebesgue measure on $\mathcal{X}$. Standard likelihood-based methods ($p_{det}(x_{real})$) diverge or are undefined.

### **Proposition 2 (Stochastic Smoothing via Mollification)**
*Assumption:* The SDE noise coefficient $\sigma(t)$ is uniformly elliptic (non-degenerate) over the time interval.

*Statement:* The SDE distribution $\mu_t^{sde}$ can be represented as a convolution of the deterministic measure with a noise kernel (approximately Gaussian for small $\sigma$):
$$
\mu_t^{sde} \approx \mu_t^{det} * \mathcal{N}(0, \Sigma_t)
$$
*Proof Sketch:* The solution to the SDE can be locally approximated as $X_t^{sde} \approx \Phi_t(\theta) + \xi_t$, where $\xi_t$ accumulates the diffusion term. The integration over $\xi_t$ acts as a **mollifier**, transforming the singular measure $\mu_t^{det}$ into an absolutely continuous measure $\mu_t^{sde}$ with full topological support in the neighborhood of the manifold.

*Consequence:* $\text{supp}(\mu_t^{real}) \subset \text{supp}(\mu_t^{sde})$ holds with high probability if $\sigma$ is calibrated correctly. The probability density $p_{sde}(x_{real})$ is well-defined and non-zero.

## 4. Metric for Evaluation: Wasserstein Distance

Since $\mu_t^{det}$ and $\mu_t^{real}$ may have disjoint supports, KL-Divergence is inappropriate ($D_{KL} \to \infty$). We employ the **Wasserstein-1 Distance**:

$$
W_1(\mu, \nu) = \inf_{\gamma \in \Gamma(\mu, \nu)} \int_{\mathcal{X} \times \mathcal{X}} \|x - y\| d\gamma(x, y)
$$

**Hypothesis:** Due to the support coverage (Proposition 2), the SDE-augmented distribution is geometrically closer to the real distribution:
$$
W_1(\mu_t^{real}, \mu_t^{sde}) < W_1(\mu_t^{real}, \mu_t^{det})
$$