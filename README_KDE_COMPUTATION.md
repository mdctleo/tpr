# KDE Computation — BayesianGaussianMixtureGPU

Offline RKHS vector computation for temporal graph edges using a
GPU-accelerated Variational Bayesian Gaussian Mixture Model (DPGMM).

## Design Overview

### What it does

`kde_computation.py` reads pre-extracted temporal graph data
(`.TemporalData.simple` files produced by the feat_inference pipeline),
groups events by edge `(src, dst, edge_type)`, fits a density model per
edge, and saves fixed-size RKHS feature vectors to disk.  Downstream
consumers (`RKHSVectorLoader`, `reduce_graphs_kde.py`, etc.) load these
vectors for O(1) lookup during training.

### Two modes

| Mode | Config flag | Data fed to GMM | Example config |
|------|-------------|-----------------|----------------|
| **kde_ts** | `use_timestamp_diffs: false` (or omitted) | Raw timestamp values | `kairos_cicids_kde_ts` |
| **kde_diff** | `use_timestamp_diffs: true` | Absolute inter-arrival times (`|diff(sort(t))|`) | `kairos_cicids_kde_diff` |

### Core model: `BayesianGaussianMixtureGPU`

A fully GPU-native variational Bayesian Gaussian mixture fitted via
**Coordinate Ascent Variational Inference (CAVI)** with closed-form
Normal–Inverse-Gamma conjugate updates and a stick-breaking Dirichlet
Process weight prior.

Key properties:

* **Batched**: processes hundreds of edges simultaneously in a single
  `(B, N_max)` padded tensor — no per-edge Python loops.
* **K-means initialisation**: deterministic k-means++ seeding on GPU
  followed by Lloyd refinement (default `init_method: kmeans`).
* **MaxAbsScaler**: each edge's data is divided by its max absolute
  value so the result lies in `[-1, 1]`, keeping the CAVI priors
  well-scaled.
* **CIC-IDS noon anchor**: for CIC-IDS-2017 in `kde_ts` mode, each
  edge's mean timestamp is shifted to 12:30 PM (45 000 s past midnight)
  so that all five weekday files share a common reference point.
* **Automatic GPU/CPU fallback**: uses CUDA when available; falls back
  to a scipy `gaussian_kde` path on CPU.

### Variational Objective (ELBO)

#### Generative model

$$
v_k \sim \operatorname{Beta}(1,\, \gamma), \qquad
\pi_k = v_k \prod_{j < k}(1 - v_j)
\qquad \text{(stick-breaking)}
$$

$$
\mu_k \sim \mathcal{N}(\mu_0,\, \lambda_0^{-1}),\qquad
\sigma^2_k \sim \operatorname{IG}(\alpha_0,\, \beta_0)
$$

$$
z_n \sim \operatorname{Cat}(\boldsymbol{\pi}),\qquad
x_n \mid z_n = k \;\sim\; \mathcal{N}(\mu_k,\, \sigma^2_k)
$$

Fixed priors (MaxAbsScaled data lies in $[-1,1]$):

| Symbol | Value | Meaning |
|--------|-------|---------|
| $\gamma$ | 5.0 | DP concentration — expected $\approx$5 active components |
| $\mu_0$ | 0 | Prior mean at origin |
| $\lambda_0$ | 1 | Prior precision on means |
| $\alpha_0$ | 3 | IG shape — prior mode at $\beta_0/(\alpha_0+1) = 0.125$ |
| $\beta_0$ | 0.5 | IG rate |

#### Mean-field variational family

$$
q(\boldsymbol{v},\, \boldsymbol{\mu},\, \boldsymbol{\sigma}^2,\, \mathbf{Z})
=
\prod_{k=1}^K
  \operatorname{Beta}(\tilde\alpha_k, \tilde\beta_k)\;
  \mathcal{N}(m_k,\, \tilde\lambda_k^{-1})\;
  \operatorname{IG}(\tilde{a}_k, \tilde{b}_k)
\;\prod_{n=1}^N
  \operatorname{Cat}(\mathbf{r}_n)
$$

#### Full ELBO

$$
\mathcal{L}
=
\underbrace{
  \sum_{n=1}^{N} \log \sum_{k=1}^{K}
  \exp\!\Bigl[\mathbb{E}[\log \pi_k]
              + \mathbb{E}[\log p(x_n \mid k)]\Bigr]
}_{\text{(a) data likelihood + assignment entropy}}
\;-\;
\underbrace{
  \sum_{k=1}^{K}
  \mathrm{KL}\!\left[q(v_k)\,\|\,p(v_k)\right]
}_{\text{(b) stick-breaking KL}}
\;-\;
\underbrace{
  \sum_{k=1}^{K}
  \mathrm{KL}\!\left[q(\mu_k)\,\|\,p(\mu_k)\right]
}_{\text{(c) mean KL}}
\;-\;
\underbrace{
  \sum_{k=1}^{K}
  \mathrm{KL}\!\left[q(\sigma^2_k)\,\|\,p(\sigma^2_k)\right]
}_{\text{(d) variance KL}}
$$

Normalised per data point for convergence comparison:
$\hat{\mathcal{L}} = \mathcal{L} / N_\text{eff}$ (`elbo / N_eff` in code).

---

##### (a) Data likelihood + assignment entropy — `ell_assign`

The expected log-likelihood of $x_n$ under component $k$, averaged over
the variational posteriors $q(\mu_k)$ and $q(\sigma^2_k)$, is:

$$
\mathbb{E}[\log p(x_n \mid k)]
= -\tfrac{1}{2}\log(2\pi)
  -\tfrac{1}{2}\underbrace{\mathbb{E}[\log \sigma^2_k]}_{\log\tilde{b}_k - \psi(\tilde{a}_k)}
  -\tfrac{1}{2}
   \underbrace{\mathbb{E}[\sigma^{-2}_k]}_{\tilde{a}_k/\tilde{b}_k}
   \!\left[(x_n - m_k)^2 + \tilde\lambda_k^{-1}\right]
$$

The $\tilde\lambda_k^{-1}$ term inside the bracket is the posterior
variance of $\mu_k$; omitting it would under-count the uncertainty in
the mean.

The expected log mixing weight under the stick-breaking prior is:

$$
\mathbb{E}[\log \pi_k]
= \psi(\tilde\alpha_k) - \psi(\tilde\alpha_k + \tilde\beta_k)
  + \sum_{j < k}\!\bigl[\psi(\tilde\beta_j)
                        - \psi(\tilde\alpha_j + \tilde\beta_j)\bigr]
$$

Term (a) combines the expected complete-data log-likelihood with the
entropy of $q(\mathbf{Z})$ via the logsumexp identity:

$$
\sum_{n}\Bigl[\sum_k r_{nk}\,\ell_{nk} + H(q(\mathbf{z}_n))\Bigr]
= \sum_{n} \log\sum_k \exp(\ell_{nk}),
\qquad \ell_{nk} = \mathbb{E}[\log \pi_k] + \mathbb{E}[\log p(x_n \mid k)]
$$

where the equality holds because $r_{nk} = \exp(\ell_{nk})/\sum_j\exp(\ell_{nj})$ is
the softmax (optimal $q(\mathbf{z}_n)$).

---

##### (b) Stick-breaking KL — `kl_v`

$$
\mathrm{KL}\!\left[\operatorname{Beta}(\tilde\alpha_k, \tilde\beta_k)
             \,\|\,
             \operatorname{Beta}(1, \gamma)\right]
= \underbrace{\log B(1,\gamma)}_{-\log\gamma}
  - \log B(\tilde\alpha_k, \tilde\beta_k)
  + (\tilde\alpha_k - 1)\,\psi(\tilde\alpha_k)
  + (\tilde\beta_k - \gamma)\,\psi(\tilde\beta_k)
  + (1 + \gamma - \tilde\alpha_k - \tilde\beta_k)\,\psi(\tilde\alpha_k + \tilde\beta_k)
$$

where
$\log B(1,\gamma) = -\log\gamma$
(since $B(1,\gamma) = \Gamma(1)\Gamma(\gamma)/\Gamma(1+\gamma) = 1/\gamma$),
and
$\log B(\tilde\alpha_k, \tilde\beta_k)
= \log\Gamma(\tilde\alpha_k) + \log\Gamma(\tilde\beta_k)
- \log\Gamma(\tilde\alpha_k + \tilde\beta_k)$.

---

##### (c) Mean KL — `kl_mu`

$$
\mathrm{KL}\!\left[\mathcal{N}(m_k, \tilde\lambda_k^{-1})
             \,\|\,
             \mathcal{N}(\mu_0, \lambda_0^{-1})\right]
= \tfrac{1}{2}\!\left[
  \log\!\frac{\tilde\lambda_k}{\lambda_0}
  + \frac{\lambda_0}{\tilde\lambda_k}
  + \lambda_0\,(m_k - \mu_0)^2
  - 1
\right]
$$

---

##### (d) Variance KL — `kl_sigma`

$$
\mathrm{KL}\!\left[\operatorname{IG}(\tilde{a}_k, \tilde{b}_k)
             \,\|\,
             \operatorname{IG}(\alpha_0, \beta_0)\right]
= \alpha_0\!\left(\log\tilde{b}_k - \log\beta_0\right)
  + \log\Gamma(\alpha_0) - \log\Gamma(\tilde{a}_k)
  + (\tilde{a}_k - \alpha_0)\,\psi(\tilde{a}_k)
  + \frac{\beta_0\,\tilde{a}_k}{\tilde{b}_k}
  - \tilde{a}_k
$$

Derivation: using $\mathbb{E}_{IG(a,b)}[\log x] = \log b - \psi(a)$ and
$\mathbb{E}_{IG(a,b)}[x^{-1}] = a/b$,
$\mathrm{KL}[IG(a,b)\|IG(c,d)]$
$= c(\log b - \log d) + \log\Gamma(c) - \log\Gamma(a) + (a-c)\psi(a) + da/b - a$.

---

#### M-step: CAVI closed-form updates

At each iteration the variational parameters are updated in closed form
(coordinate ascent step that increases $\mathcal{L}$ for each block):

$$
\tilde\alpha_k = 1 + N_k, \qquad
\tilde\beta_k  = \gamma + \sum_{j > k} N_j
$$

$$
\tilde\lambda_k = \lambda_0 + N_k, \qquad
m_k = \frac{\lambda_0\,\mu_0 + \sum_n r_{nk}\,x_n}{\tilde\lambda_k}
$$

$$
\tilde{a}_k = \alpha_0 + \tfrac{N_k}{2}, \qquad
\tilde{b}_k = \beta_0
  + \frac{S_k}{2}
  + \frac{\lambda_0\,N_k\,(\bar{x}_k - \mu_0)^2}{2\,\tilde\lambda_k}
$$

where
$N_k = \sum_n r_{nk}$,
$\bar{x}_k = \sum_n r_{nk}\,x_n / N_k$,
and $S_k = \sum_n r_{nk}(x_n - \bar{x}_k)^2$ (weighted sum of squares).

The third term in $\tilde{b}_k$ is the contribution from the
Normal–InvGamma conjugacy: the prior pulls the mean towards $\mu_0$, and
any residual disagreement between $\bar{x}_k$ and $\mu_0$ increases the
posterior variance estimate.

#### Convergence

CAVI is run for at most `max_iter` iterations.  The loop halts early
(per edge in the batch) when $|\hat{\mathcal{L}}^{(t)} - \hat{\mathcal{L}}^{(t-1)}| < \text{tol}$
for `patience` consecutive steps.  In practice the ELBO improves
monotonically at every step (verified empirically: 0 violations in 42
steps on a held-out 4-edge, 60-point batch).

---

### Pipeline stages

```
extract_edge_timestamps()        # parallel file loading + vectorised grouping
        |
filter_merged_edges()            # keep edges with >= min_occurrences
        |                        #   kde_ts  -> raw timestamps
        |                        #   kde_diff -> |diff(sort(t))|
        |
_anchor_cicids_noon()            # (CIC-IDS + kde_ts only)
        |
scale_merged_edges()             # MaxAbsScaler per edge -> [-1, 1]
        |
preprocess_long_merged_edges()   # subsample edges > max_n_per_edge
        |
merged_edges_to_matrix()         # pad into (B, N_max) + mask
        |
BayesianGaussianMixtureGPU       # fit batched CAVI
  .fit_batch()
        |
  .score_samples_grid()           # evaluate density on rkhs_dim-point grid
        |
save_results()                   # {DATASET}_kde_vectors.pt + stats JSON
```

### Output format

```python
torch.load("{output_dir}/{DATASET}_kde_vectors.pt")
# -> {'edge_vectors': {(src, dst, edge_type): Tensor(rkhs_dim), ...},
#     'metadata':     {<params, counts, timestamp>}}
```

This is consumed by:
* `pidsmaker/utils/kde_vector_loader.py` — `RKHSVectorLoader` for O(1) lookup
* `scripts/reduce_graphs_kde.py` — determines which edges to collapse

---

## Configuration

All BGMM hyper-parameters live under `kde_params:` in the YAML config.
The four CIC-IDS configs are:

| Config file | Mode |
|-------------|------|
| `config/kairos_cicids_kde_ts.yml` | Raw timestamps |
| `config/kairos_cicids_kde_diff.yml` | Timestamp diffs |
| `config/orthrus_cicids_kde_ts.yml` | Raw timestamps |
| `config/orthrus_cicids_kde_diff.yml` | Timestamp diffs |

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rkhs_dim` | 20 | Dimension of output feature vector |
| `min_occurrences` | 10 | Minimum timestamps per edge to compute a vector |
| `n_components` | 100 | DP truncation level K (max mixture components) |
| `gamma` | 5.0 | DP concentration (larger → more components) |
| `max_iter` | 300 | Max CAVI iterations |
| `tol` | 1e-4 | Convergence tolerance |
| `patience` | 20 | Early-stop patience |
| `n_init` | 1 | Independent restarts (best kept) |
| `init_method` | `kmeans` | `kmeans` or `linear` |
| `truncate_threshold` | 0.99 | Cumulative weight cutoff for pruning |
| `max_n_per_edge` | 50000 | Subsample limit per edge |
| `chunk_size` | 256 | Edges per GPU batch |

---

## Steps to Run

### Prerequisites

1. **feat_inference artifacts** must already exist at
   `artifacts/feat_inference/{DATASET}/feat_inference/{hash}/edge_embeds/{train,val}/`.
   These are produced by the main PIDSMaker pipeline.

2. Python environment with: `torch`, `numpy`, `scipy`, `scikit-learn`,
   `pyyaml`, `tqdm`.

### Step 1: Compute RKHS vectors

```bash
cd /scratch/asawan15/PIDSMaker

# Raw timestamps mode (CIC-IDS Monday, auto-detect latest feat_hash)
python kde_computation.py kairos_cicids_kde_ts CICIDS_MONDAY

# Timestamp differences mode
python kde_computation.py kairos_cicids_kde_diff CICIDS_MONDAY

# With explicit feat_hash and more IO threads
python kde_computation.py kairos_cicids_kde_ts CICIDS_MONDAY \
    --feat_hash abc123 --n_workers 4

# Orthrus configs
python kde_computation.py orthrus_cicids_kde_ts CICIDS_MONDAY
python kde_computation.py orthrus_cicids_kde_diff CICIDS_MONDAY

# Override output directory
python kde_computation.py kairos_cicids_kde_ts CICIDS_MONDAY \
    --output_dir my_kde_vectors
```

### Step 2: Verify output

```bash
# Check the generated files
ls -lh kde_vectors/CICIDS_MONDAY_kde_vectors.pt
cat kde_vectors/CICIDS_MONDAY_kde_stats.json | python -m json.tool
```

### Step 3: (Optional) Reduce graphs

```bash
python scripts/reduce_graphs_kde.py CICIDS_MONDAY \
    --kde_vectors_dir kde_vectors
```

### Step 4: Train with KDE vectors

The training pipeline (`pidsmaker`) automatically loads precomputed
vectors via `RKHSVectorLoader` when `use_precomputed: true` is set in
the config.

---

## Differences from Previous Implementation

| Aspect | Old (Pyro SVI / CAVI) | New (BayesianGaussianMixtureGPU) |
|--------|----------------------|----------------------------------|
| Model | `_PyroDPGMM` (Pyro SVI) or inline CAVI | `BayesianGaussianMixtureGPU` class |
| Dependencies | `pyro-ppl`, `sklearn` | `sklearn` only (no Pyro) |
| Scaler | Z-score (mean/std) | MaxAbsScaler (divide by max abs) |
| Initialisation | Linear (evenly spaced) or sklearn K-Means | GPU batched k-means++ |
| CIC-IDS handling | None | 12:30 PM noon anchor per edge |
| CPU fallback | `multiprocessing.Pool` of Pyro SVI workers | scipy `gaussian_kde` |
| GPU path | Inline CAVI tensor ops | `BayesianGaussianMixtureGPU.fit_batch()` |
| Output format | Same `{edge_vectors, metadata}` dict | Same (fully compatible) |
