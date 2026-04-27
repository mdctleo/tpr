import math
from typing import Optional, Tuple, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


class BayesianGaussianMixtureGPU:
    """GPU-accelerated variational Bayesian Gaussian mixture for batched 1D data.

    This follows the update equations of sklearn's `BayesianGaussianMixture`
    closely, but is intentionally restricted to 1D data and batched fitting.

    Supported input layouts:
    - `(n_samples,)`
    - `(n_samples, 1)` sklearn-style single dataset
    - `(batch_size, n_samples)` batched 1D datasets
    - `(batch_size, n_samples, 1)`

    For variable-length batches, pass `mask` or pad with `NaN`.
    """

    def __init__(
        self,
        *,
        n_components: int = 1,
        covariance_type: str = "diag",
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        init_params: str = "kmeans",
        weight_concentration_prior_type: str = "dirichlet_process",
        weight_concentration_prior: Optional[float] = None,
        mean_precision_prior: Optional[float] = None,
        mean_prior: Optional[ArrayLike] = None,
        degrees_of_freedom_prior: Optional[float] = None,
        covariance_prior: Optional[ArrayLike] = None,
        random_state: Optional[int] = None,
        warm_start: bool = False,
        verbose: int = 0,
        verbose_interval: int = 10,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.weight_concentration_prior_type = weight_concentration_prior_type
        self.weight_concentration_prior = weight_concentration_prior
        self.mean_precision_prior = mean_precision_prior
        self.mean_prior = mean_prior
        self.degrees_of_freedom_prior = degrees_of_freedom_prior
        self.covariance_prior = covariance_prior
        self.random_state = random_state
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype

        self._validate_constructor()

    def fit(
        self,
        X: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
        mask: Optional[ArrayLike] = None,
    ):
        x, valid_mask, sample_weight, input_was_batched = self._prepare_input(
            X, sample_weight=sample_weight, mask=mask
        )
        self._input_was_batched = input_was_batched
        self._batch_size = x.shape[0]

        self._check_parameters(x, valid_mask)

        do_init = not (
            self.warm_start
            and hasattr(self, "_means")
            and self._means.shape[1] == self._n_components_fit
        )
        n_init = self.n_init if do_init else 1

        best_lower_bound = None
        best_params = None
        best_converged = None
        best_n_iter = None
        best_history = None

        for init in range(n_init):
            if do_init:
                self._initialize_parameters(x, valid_mask, sample_weight, init)

            lower_bound = torch.full(
                (self._batch_size,),
                -torch.inf,
                dtype=self.dtype,
                device=self.device,
            )
            converged = torch.zeros(self._batch_size, dtype=torch.bool, device=self.device)
            current_history = torch.full(
                (self.max_iter, self._batch_size),
                torch.nan,
                dtype=self.dtype,
                device=self.device,
            )

            if self.max_iter == 0:
                current_n_iter = torch.zeros(
                    self._batch_size, dtype=torch.int64, device=self.device
                )
            else:
                current_n_iter = torch.full(
                    (self._batch_size,),
                    self.max_iter,
                    dtype=torch.int64,
                    device=self.device,
                )
                for n_iter in range(1, self.max_iter + 1):
                    prev_lower_bound = lower_bound
                    _, log_resp = self._e_step(x, valid_mask)
                    self._m_step(x, log_resp, valid_mask, sample_weight)
                    lower_bound = self._compute_lower_bound(log_resp, valid_mask, sample_weight)
                    current_history[n_iter - 1] = lower_bound

                    change = lower_bound - prev_lower_bound
                    newly_converged = (~converged) & (torch.abs(change) < self.tol)
                    current_n_iter[newly_converged] = n_iter
                    converged = converged | newly_converged

                    if self.verbose and (n_iter % self.verbose_interval == 0 or n_iter == 1):
                        min_lb = float(lower_bound.min().detach().cpu())
                        max_lb = float(lower_bound.max().detach().cpu())
                        print(
                            f"[BayesianGaussianMixtureGPU] init={init + 1}/{n_init} "
                            f"iter={n_iter} lower_bound_range=({min_lb:.6f}, {max_lb:.6f})",
                            flush=True,
                        )

                    if torch.all(converged):
                        break

            current_params = self._get_parameters()
            if best_lower_bound is None:
                best_lower_bound = lower_bound.clone()
                best_params = self._clone_parameters(current_params)
                best_converged = converged.clone()
                best_n_iter = current_n_iter.clone()
                best_history = current_history.clone()
            else:
                improved = lower_bound > best_lower_bound
                best_lower_bound = torch.where(improved, lower_bound, best_lower_bound)
                best_params = self._select_parameters(best_params, current_params, improved)
                best_converged = torch.where(improved, converged, best_converged)
                best_n_iter = torch.where(improved, current_n_iter, best_n_iter)
                best_history = torch.where(improved.unsqueeze(0), current_history, best_history)

        self._set_parameters(best_params)
        self._sync_public_attributes()

        self.converged_ = self._maybe_squeeze(best_converged.detach().cpu().numpy())
        self.n_iter_ = self._maybe_squeeze(best_n_iter.detach().cpu().numpy())
        self.lower_bound_ = self._maybe_squeeze(best_lower_bound.detach().cpu().numpy())
        self.lower_bounds_ = self._history_to_numpy(best_history, best_n_iter)
        return self


    def _validate_constructor(self):
        if self.n_components < 1:
            raise ValueError("n_components must be >= 1.")
        if self.covariance_type not in {"diag", "spherical"}:
            raise ValueError("Only 'diag' and 'spherical' covariance types are supported for 1D data.")
        if self.init_params not in {"kmeans", "random", "random_from_data", "k-means++"}:
            raise ValueError("Unsupported init_params.")
        if self.weight_concentration_prior_type not in {
            "dirichlet_process",
            "dirichlet_distribution",
        }:
            raise ValueError("Unsupported weight_concentration_prior_type.")
        if self.tol < 0.0:
            raise ValueError("tol must be non-negative.")
        if self.reg_covar < 0.0:
            raise ValueError("reg_covar must be non-negative.")
        if self.max_iter < 0:
            raise ValueError("max_iter must be non-negative.")
        if self.n_init < 1:
            raise ValueError("n_init must be >= 1.")

    def _prepare_input(
        self,
        X: ArrayLike,
        sample_weight: Optional[ArrayLike],
        mask: Optional[ArrayLike],
    ):
        x = self._to_tensor(X)
        input_was_batched = True

        if x.ndim == 1:
            x = x.unsqueeze(0)
            input_was_batched = False
        elif x.ndim == 2:
            if x.shape[1] == 1:
                x = x.squeeze(1).unsqueeze(0)
                input_was_batched = False
            else:
                input_was_batched = True
        elif x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
            input_was_batched = True
        else:
            raise ValueError(
                "Expected X with shape (n_samples,), (n_samples, 1), "
                "(batch_size, n_samples), or (batch_size, n_samples, 1)."
            )

        if mask is None:
            valid_mask = torch.isfinite(x)
        else:
            valid_mask = self._to_tensor(mask, dtype=torch.bool)
            if valid_mask.shape != x.shape:
                if valid_mask.ndim == 1 and x.ndim == 2 and valid_mask.shape[0] == x.shape[1]:
                    valid_mask = valid_mask.unsqueeze(0).expand_as(x)
                else:
                    raise ValueError("mask must have the same batch/sample shape as X.")
            valid_mask = valid_mask & torch.isfinite(x)

        x = torch.where(valid_mask, x, torch.zeros_like(x))

        if sample_weight is None:
            weights = valid_mask.to(self.dtype)
        else:
            weights = self._to_tensor(sample_weight)
            if weights.shape != x.shape:
                if weights.ndim == 1 and x.ndim == 2 and weights.shape[0] == x.shape[1]:
                    weights = weights.unsqueeze(0).expand_as(x)
                elif (
                    weights.ndim == 2
                    and x.ndim == 2
                    and weights.shape[0] == x.shape[0]
                    and weights.shape[1] == 1
                ):
                    weights = weights.expand_as(x)
                else:
                    raise ValueError("sample_weight must align with X.")
            weights = torch.where(valid_mask, weights, torch.zeros_like(weights))

        return x, valid_mask, weights, input_was_batched

    def _check_parameters(self, x: torch.Tensor, valid_mask: torch.Tensor):
        counts = valid_mask.sum(dim=1)
        if torch.any(counts < 2):
            raise ValueError("Each batch item must contain at least 2 valid samples.")

        self._n_components_fit = self.n_components
        self._active_components = torch.minimum(
            counts,
            torch.full_like(counts, self.n_components),
        )
        component_ids = torch.arange(self.n_components, device=self.device).unsqueeze(0)
        self._component_mask = component_ids < self._active_components.unsqueeze(1)
        self.n_components_fit_ = self._maybe_squeeze(
            self._active_components.detach().cpu().numpy()
        )

        if self.verbose and torch.any(self._active_components < self.n_components):
            min_count = int(self._active_components.min().item())
            print(
                "[BayesianGaussianMixtureGPU] per-row active components enabled "
                f"(max={self.n_components}, min_active={min_count})",
                flush=True,
            )

        if self.weight_concentration_prior is None:
            self.weight_concentration_prior_ = None
            self._weight_concentration_prior_tensor = (
                1.0 / self._active_components.to(self.dtype).clamp_min(1.0)
            ).unsqueeze(1)
        else:
            self.weight_concentration_prior_ = float(self.weight_concentration_prior)
            if self.weight_concentration_prior_ <= 0.0:
                raise ValueError("weight_concentration_prior must be > 0.")
            self._weight_concentration_prior_tensor = torch.as_tensor(
                self.weight_concentration_prior_, dtype=self.dtype, device=self.device
            )

        self.mean_precision_prior_ = (
            1.0 if self.mean_precision_prior is None else float(self.mean_precision_prior)
        )
        if self.mean_precision_prior_ <= 0.0:
            raise ValueError("mean_precision_prior must be > 0.")
        self._mean_precision_prior_tensor = torch.as_tensor(
            self.mean_precision_prior_, dtype=self.dtype, device=self.device
        )

        if self.mean_prior is None:
            denom = valid_mask.sum(dim=1).clamp_min(1).to(self.dtype)
            self._mean_prior = (x * valid_mask.to(self.dtype)).sum(dim=1) / denom
        else:
            mean_prior = self._to_tensor(self.mean_prior).reshape(-1)
            if mean_prior.numel() == 1:
                mean_prior = mean_prior.expand(x.shape[0])
            if mean_prior.shape[0] != x.shape[0]:
                raise ValueError("mean_prior must be scalar or have one value per batch item.")
            self._mean_prior = mean_prior

        if self.degrees_of_freedom_prior is None:
            self.degrees_of_freedom_prior_ = 1.0
        else:
            self.degrees_of_freedom_prior_ = float(self.degrees_of_freedom_prior)
            if self.degrees_of_freedom_prior_ <= 0.0:
                raise ValueError("degrees_of_freedom_prior must be > 0.")
        self._degrees_of_freedom_prior_tensor = torch.as_tensor(
            self.degrees_of_freedom_prior_, dtype=self.dtype, device=self.device
        )

        self._covariance_prior = self._resolve_covariance_prior(x, valid_mask)

    def _resolve_covariance_prior(self, x: torch.Tensor, valid_mask: torch.Tensor):
        if self.covariance_prior is None:
            counts = valid_mask.sum(dim=1).to(self.dtype)
            centered = x - self._mean_prior.unsqueeze(1)
            denom = (counts - 1.0).clamp_min(1.0)
            variance = (
                (centered.square() * valid_mask.to(self.dtype)).sum(dim=1) / denom
            ).clamp_min(self.reg_covar)
            return variance

        covariance_prior = self._to_tensor(self.covariance_prior).reshape(-1)
        if covariance_prior.numel() == 1:
            covariance_prior = covariance_prior.expand(x.shape[0])
        if covariance_prior.shape[0] != x.shape[0]:
            raise ValueError(
                "covariance_prior must be scalar or have one value per batch item."
            )
        if torch.any(covariance_prior <= 0.0):
            raise ValueError("covariance_prior must be strictly positive.")
        return covariance_prior

    def _initialize_parameters(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_weight: torch.Tensor,
        init_index: int,
    ):
        generator = self._make_generator(init_index)
        if self.init_params in {"kmeans", "k-means++"}:
            resp = self._init_resp_kmeans(x, valid_mask, generator)
        elif self.init_params == "random":
            n_components_fit = self._n_components_fit
            resp = torch.rand(
                x.shape[0],
                x.shape[1],
                n_components_fit,
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            )
            resp = resp / resp.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(self.dtype).eps)
            resp = resp * valid_mask.unsqueeze(-1)
        else:
            resp = self._init_resp_random_from_data(x, valid_mask, generator)

        self._initialize(x, resp, valid_mask, sample_weight)

    def _initialize(
        self,
        x: torch.Tensor,
        resp: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_weight: torch.Tensor,
    ):
        resp = resp * self._component_mask.unsqueeze(1).to(self.dtype)
        nk, xk, sk = self._estimate_gaussian_parameters(x, resp, valid_mask, sample_weight)
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _estimate_gaussian_parameters(
        self,
        x: torch.Tensor,
        resp: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del valid_mask
        effective_resp = resp * sample_weight.unsqueeze(-1)
        eps = 10.0 * torch.finfo(self.dtype).eps
        component_mask = self._component_mask.to(self.dtype)
        nk = effective_resp.sum(dim=1) + eps * component_mask
        safe_nk = nk.clamp_min(eps)
        xk = (effective_resp * x.unsqueeze(-1)).sum(dim=1) / safe_nk
        avg_x2 = (effective_resp * x.unsqueeze(-1).square()).sum(dim=1) / safe_nk
        sk = (avg_x2 - xk.square()).clamp_min(0.0) + self.reg_covar
        xk = torch.where(self._component_mask, xk, torch.zeros_like(xk))
        sk = torch.where(self._component_mask, sk, torch.full_like(sk, self.reg_covar))
        sk = torch.where(torch.isfinite(sk), sk, torch.full_like(sk, self.reg_covar))
        return nk, xk, sk

    def _estimate_weights(self, nk: torch.Tensor):
        if self.weight_concentration_prior_type == "dirichlet_process":
            suffix = torch.flip(torch.cumsum(torch.flip(nk, dims=[1]), dim=1), dims=[1])
            tail = torch.cat([suffix[:, 1:], torch.zeros_like(suffix[:, :1])], dim=1)
            self._weight_concentration = (
                1.0 + nk,
                self._weight_concentration_prior_tensor + tail,
            )
        else:
            self._weight_concentration = self._weight_concentration_prior_tensor + nk

    def _estimate_means(self, nk: torch.Tensor, xk: torch.Tensor):
        self._mean_precision = self._mean_precision_prior_tensor + nk
        self._means = (
            self._mean_precision_prior_tensor * self._mean_prior.unsqueeze(1) + nk * xk
        ) / self._mean_precision

    def _estimate_precisions(self, nk: torch.Tensor, xk: torch.Tensor, sk: torch.Tensor):
        self._degrees_of_freedom = self._degrees_of_freedom_prior_tensor + nk
        diff = xk - self._mean_prior.unsqueeze(1)
        covariance_prior = self._covariance_prior.unsqueeze(1)
        covariances = covariance_prior + nk * (
            sk + (self._mean_precision_prior_tensor / self._mean_precision) * diff.square()
        )
        self._covariances = (covariances / self._degrees_of_freedom).clamp_min(self.reg_covar)
        self._precisions_cholesky = torch.rsqrt(self._covariances)
        self._precisions = self._precisions_cholesky.square()

    def _estimate_log_weights(self):
        if self.weight_concentration_prior_type == "dirichlet_process":
            alpha, beta = self._weight_concentration
            digamma_sum = torch.digamma(alpha + beta)
            digamma_a = torch.digamma(alpha)
            digamma_b = torch.digamma(beta)
            prefix = torch.cumsum(digamma_b - digamma_sum, dim=1)
            prefix = torch.cat([torch.zeros_like(prefix[:, :1]), prefix[:, :-1]], dim=1)
            return digamma_a - digamma_sum + prefix

        concentration = self._weight_concentration
        return torch.digamma(concentration) - torch.digamma(concentration.sum(dim=1, keepdim=True))

    def _estimate_log_prob(self, x: torch.Tensor):
        means = self._means.unsqueeze(1)
        precisions_chol = self._precisions_cholesky.unsqueeze(1)
        diff = x.unsqueeze(-1) - means
        log_gauss = (
            -0.5 * (math.log(2.0 * math.pi) + (diff * precisions_chol).square())
            + torch.log(precisions_chol)
        )
        log_gauss = log_gauss - 0.5 * torch.log(self._degrees_of_freedom).unsqueeze(1)
        log_lambda = math.log(2.0) + torch.digamma(0.5 * self._degrees_of_freedom)
        return log_gauss + 0.5 * (
            log_lambda.unsqueeze(1) - (1.0 / self._mean_precision).unsqueeze(1)
        )

    def _estimate_weighted_log_prob(self, x: torch.Tensor):
        weighted = self._estimate_log_prob(x) + self._estimate_log_weights().unsqueeze(1)
        weighted = weighted.masked_fill(~self._component_mask.unsqueeze(1), -torch.inf)
        return weighted

    def _estimate_log_prob_resp(self, x: torch.Tensor, valid_mask: torch.Tensor):
        weighted_log_prob = self._estimate_weighted_log_prob(x)
        weighted_log_prob = weighted_log_prob.masked_fill(
            ~valid_mask.unsqueeze(-1), -torch.inf
        )
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=-1)
        log_resp = weighted_log_prob - log_prob_norm.unsqueeze(-1)
        log_resp = torch.where(valid_mask.unsqueeze(-1), log_resp, torch.zeros_like(log_resp))
        return log_prob_norm, log_resp

    def _e_step(self, x: torch.Tensor, valid_mask: torch.Tensor):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(x, valid_mask)
        denom = valid_mask.sum(dim=1).clamp_min(1).to(self.dtype)
        mean_log_prob_norm = (
            log_prob_norm.masked_fill(~valid_mask, 0.0).sum(dim=1) / denom
        )
        return mean_log_prob_norm, log_resp

    def _m_step(
        self,
        x: torch.Tensor,
        log_resp: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_weight: torch.Tensor,
    ):
        resp = torch.exp(log_resp)
        resp = resp * self._component_mask.unsqueeze(1).to(self.dtype)
        nk, xk, sk = self._estimate_gaussian_parameters(x, resp, valid_mask, sample_weight)
        self._estimate_weights(nk)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _compute_lower_bound(
        self,
        log_resp: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_weight: torch.Tensor,
    ):
        del valid_mask
        log_det_precisions_chol = (
            torch.log(self._precisions_cholesky) - 0.5 * torch.log(self._degrees_of_freedom)
        )
        log_wishart = -(
            self._degrees_of_freedom * log_det_precisions_chol
            + self._degrees_of_freedom * 0.5 * math.log(2.0)
            + torch.lgamma(0.5 * self._degrees_of_freedom)
        ).sum(dim=1)

        if self.weight_concentration_prior_type == "dirichlet_process":
            alpha, beta = self._weight_concentration
            log_norm_weight = -self._betaln(alpha, beta).sum(dim=1)
        else:
            concentration = self._weight_concentration
            log_norm_weight = torch.lgamma(concentration.sum(dim=1)) - torch.lgamma(
                concentration
            ).sum(dim=1)

        resp = torch.exp(log_resp) * sample_weight.unsqueeze(-1)
        log_resp_safe = torch.where(resp > 0.0, log_resp, torch.zeros_like(log_resp))
        entropy = -(resp * log_resp_safe).sum(dim=(1, 2))

        return (
            entropy
            - log_wishart
            - log_norm_weight
            - 0.5 * torch.log(self._mean_precision).sum(dim=1)
        )

    def _get_parameters(self):
        return (
            self._weight_concentration,
            self._mean_precision,
            self._means,
            self._degrees_of_freedom,
            self._covariances,
            self._precisions_cholesky,
        )

    def _set_parameters(self, params):
        (
            self._weight_concentration,
            self._mean_precision,
            self._means,
            self._degrees_of_freedom,
            self._covariances,
            self._precisions_cholesky,
        ) = params

        if self.weight_concentration_prior_type == "dirichlet_process":
            alpha, beta = self._weight_concentration
            weight_dirichlet_sum = alpha + beta
            tmp = beta / weight_dirichlet_sum
            prefix = torch.cumprod(tmp[:, :-1], dim=1)
            prefix = torch.cat([torch.ones_like(tmp[:, :1]), prefix], dim=1)
            self._weights = alpha / weight_dirichlet_sum * prefix
        else:
            concentration = self._weight_concentration
            self._weights = concentration

        # Keep inactive components fully disabled and renormalize active ones.
        self._weights = self._weights * self._component_mask.to(self.dtype)
        self._weights = self._weights / self._weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(self.dtype).eps
        )

        self._precisions = self._precisions_cholesky.square()

    def _sync_public_attributes(self):
        self.mean_prior_ = self._maybe_squeeze(self._mean_prior.detach().cpu().numpy())
        self.covariance_prior_ = self._maybe_squeeze(self._covariance_prior.detach().cpu().numpy())
        self.weight_concentration_prior_ = self.weight_concentration_prior_
        self.mean_precision_prior_ = self.mean_precision_prior_
        self.degrees_of_freedom_prior_ = self.degrees_of_freedom_prior_

        self.weights_ = self._maybe_squeeze(self._weights.detach().cpu().numpy())
        self.mean_precision_ = self._maybe_squeeze(self._mean_precision.detach().cpu().numpy())
        self.means_ = self._maybe_squeeze(self._means.unsqueeze(-1).detach().cpu().numpy())
        self.degrees_of_freedom_ = self._maybe_squeeze(
            self._degrees_of_freedom.detach().cpu().numpy()
        )
        self.covariances_ = self._maybe_squeeze(
            self._covariances.unsqueeze(-1).detach().cpu().numpy()
        )
        self.precisions_cholesky_ = self._maybe_squeeze(
            self._precisions_cholesky.unsqueeze(-1).detach().cpu().numpy()
        )
        self.precisions_ = self._maybe_squeeze(self._precisions.unsqueeze(-1).detach().cpu().numpy())
        if self.weight_concentration_prior_type == "dirichlet_process":
            self.weight_concentration_ = (
                self._maybe_squeeze(self._weight_concentration[0].detach().cpu().numpy()),
                self._maybe_squeeze(self._weight_concentration[1].detach().cpu().numpy()),
            )
        else:
            self.weight_concentration_ = self._maybe_squeeze(
                self._weight_concentration.detach().cpu().numpy()
            )

    def _init_resp_kmeans(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        generator: torch.Generator,
        num_iters: int = 10,
    ):
        n_components_fit = self._n_components_fit
        centers = self._initial_centers_1d(x, valid_mask, generator)
        prev_centers = centers

        for _ in range(num_iters):
            distances = torch.abs(x.unsqueeze(-1) - centers.unsqueeze(1))
            distances = distances.masked_fill(~valid_mask.unsqueeze(-1), torch.finfo(self.dtype).max)
            labels = torch.argmin(distances, dim=-1)
            resp = torch.nn.functional.one_hot(labels, num_classes=n_components_fit).to(self.dtype)
            resp = resp * valid_mask.unsqueeze(-1)
            nk = resp.sum(dim=1)
            updated_centers = (resp * x.unsqueeze(-1)).sum(dim=1) / nk.clamp_min(1.0)
            centers = torch.where(nk > 0, updated_centers, centers)
            if torch.max(torch.abs(centers - prev_centers)) < 1e-6:
                break
            prev_centers = centers

        return resp

    def _initial_centers_1d(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        generator: torch.Generator,
    ):
        n_components_fit = self._n_components_fit
        batch_size, _ = x.shape
        counts = valid_mask.sum(dim=1)
        batch_indices = torch.arange(batch_size, device=self.device)
        eps = torch.finfo(self.dtype).eps

        if self.init_params == "k-means++":
            random_scores = torch.rand(
                x.shape,
                device=self.device,
                dtype=self.dtype,
                generator=generator,
            ).masked_fill(~valid_mask, 2.0)
            first_idx = torch.argmin(random_scores, dim=1)
            centers = torch.empty(
                batch_size, n_components_fit, device=self.device, dtype=self.dtype
            )
            centers[:, 0] = x[batch_indices, first_idx]
            min_sq_dist = (x - centers[:, :1]).square().masked_fill(~valid_mask, 0.0)

            for component in range(1, n_components_fit):
                denom = min_sq_dist.sum(dim=1, keepdim=True)
                probs = min_sq_dist / denom.clamp_min(eps)
                fallback = first_idx
                next_idx = fallback.clone()
                valid_rows = denom.squeeze(1) > eps
                if torch.any(valid_rows):
                    next_idx[valid_rows] = torch.multinomial(
                        probs[valid_rows], 1, generator=generator
                    ).squeeze(1)
                centers[:, component] = x[batch_indices, next_idx]
                min_sq_dist = torch.minimum(
                    min_sq_dist,
                    (x - centers[:, component : component + 1]).square(),
                ).masked_fill(~valid_mask, 0.0)
            return centers

        masked_x = torch.where(
            valid_mask,
            x,
            torch.full_like(x, torch.finfo(self.dtype).max),
        )
        sorted_values = torch.sort(masked_x, dim=1).values
        positions = torch.linspace(
            0.0,
            1.0,
            n_components_fit,
            device=self.device,
            dtype=self.dtype,
        ).unsqueeze(0)
        indices = torch.round(
            positions * (counts.unsqueeze(1).to(self.dtype) - 1.0).clamp_min(0.0)
        ).to(torch.long)
        return sorted_values.gather(1, indices)

    def _init_resp_random_from_data(
        self, x: torch.Tensor, valid_mask: torch.Tensor, generator: torch.Generator
    ):
        n_components_fit = self._n_components_fit
        batch_size = x.shape[0]
        random_scores = torch.rand(
            x.shape,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        ).masked_fill(~valid_mask, 2.0)
        indices = torch.argsort(random_scores, dim=1)[:, : n_components_fit]
        resp = torch.zeros(
            batch_size, x.shape[1], n_components_fit, device=self.device, dtype=self.dtype
        )
        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1).expand_as(indices)
        component_idx = torch.arange(n_components_fit, device=self.device).unsqueeze(0).expand_as(indices)
        resp[batch_idx, indices, component_idx] = 1.0
        return resp

    def _make_generator(self, init_index: int):
        generator_device = self.device.type if self.device.type in {"cpu", "cuda"} else "cpu"
        generator = torch.Generator(device=generator_device)
        seed = 0 if self.random_state is None else int(self.random_state)
        generator.manual_seed(seed + init_index)
        return generator

    def _clone_parameters(self, params):
        weight_concentration = params[0]
        if isinstance(weight_concentration, tuple):
            weight_concentration = (
                weight_concentration[0].clone(),
                weight_concentration[1].clone(),
            )
        else:
            weight_concentration = weight_concentration.clone()
        return (
            weight_concentration,
            params[1].clone(),
            params[2].clone(),
            params[3].clone(),
            params[4].clone(),
            params[5].clone(),
        )

    def _select_parameters(self, best_params, current_params, improved: torch.Tensor):
        mask = improved.unsqueeze(1)
        best_wc = best_params[0]
        current_wc = current_params[0]
        if isinstance(best_wc, tuple):
            selected_wc = (
                torch.where(mask, current_wc[0], best_wc[0]),
                torch.where(mask, current_wc[1], best_wc[1]),
            )
        else:
            selected_wc = torch.where(mask, current_wc, best_wc)
        return (
            selected_wc,
            torch.where(mask, current_params[1], best_params[1]),
            torch.where(mask, current_params[2], best_params[2]),
            torch.where(mask, current_params[3], best_params[3]),
            torch.where(mask, current_params[4], best_params[4]),
            torch.where(mask, current_params[5], best_params[5]),
        )

    def _history_to_numpy(self, history: torch.Tensor, n_iter: torch.Tensor):
        if history is None:
            return [] if self._input_was_batched else np.array([], dtype=np.float64)

        history_np = history.detach().cpu().numpy()
        n_iter_np = n_iter.detach().cpu().numpy()
        if self._input_was_batched:
            return [history_np[: int(n_iter_np[i]), i].copy() for i in range(history_np.shape[1])]
        return history_np[: int(n_iter_np[0]), 0].copy()

    def _maybe_squeeze(self, value):
        if self._input_was_batched:
            return value
        return np.squeeze(value, axis=0)

    def _to_tensor(self, value, dtype: Optional[torch.dtype] = None):
        target_dtype = self.dtype if dtype is None else dtype
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=target_dtype)
        return torch.as_tensor(value, device=self.device, dtype=target_dtype)

    @staticmethod
    def _betaln(alpha: torch.Tensor, beta: torch.Tensor):
        return torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
