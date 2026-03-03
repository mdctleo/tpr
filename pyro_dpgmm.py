import math, torch, pyro, pyro.distributions as dist
from torch.distributions import constraints
from pyro.optim import ExponentialLR, Adam
from pyro.optim.clipped_adam import ClippedAdam
from sklearn.metrics import pairwise_distances

from pyro.infer import SVI, TraceEnum_ELBO
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def expected_log_sticks(alpha, beta):
    """
    Compute E_q[log v_k] and the running sum E_q[log(1-v_j)] analytically.
    Returns log π_k (shape [K]).
    """
    dig_sum   = torch.digamma(alpha + beta)
    e_log_v   = torch.digamma(alpha) - dig_sum
    e_log_1mv = torch.digamma(beta)  - dig_sum
    prefix    = torch.cumsum(e_log_1mv, dim=0)           # Σ_{j≤k} …
    prefix    = torch.cat([prefix.new_zeros(1), prefix[:-1]])
    return e_log_v + prefix                               # log π_k

class PyroDPGMM():
    """
    Truncated DP-GMM with stick-breaking prior.
    Data are min-max scaled to [0,1].
    """
    def __init__(self, K=100, gamma=5.0, global_variance=0, total_data_length=0):
        self.K = K  # truncation level
        self.gamma = gamma
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Fixed prior hyperparameters
        self.mu0 = torch.tensor(0.5, device=self.device, dtype=torch.float32)  # mean of prior
        self.lambda0 = torch.tensor(1.0, device=self.device, dtype=torch.float32)  # precision of prior
        
        alpha0 = 3.0  
        expected_variance = 1e-4  # Encourage tight, localized clusters
        beta0 = expected_variance * (alpha0 - 1)
        
        self.alpha0 = torch.tensor(alpha0, device=self.device, dtype=torch.float32)
        self.beta0  = torch.tensor(beta0,  device=self.device, dtype=torch.float32) 

    def kmeans_init(self, data):
        """
        Perform K-Means initialization using the Nearest Neighbor Center Heuristic.
        This dynamically scales variances without relying on global dataset metrics,
        and safely preserves single-point outlier clusters.
        """
        X_np = data.cpu().numpy().reshape(-1, 1)
        global_variance = np.var(X_np) # Kept purely as an emergency fallback

        while True:           
            # Perform K-Means clustering
            kmeans = KMeans(n_clusters=self.K, random_state=42).fit(X_np)
            labels = kmeans.predict(X_np)
            means = kmeans.cluster_centers_.reshape(-1)

            if self.K > 1:
                centers_2d = means.reshape(-1, 1)
                dists = pairwise_distances(centers_2d)
                
                # Ignore distance to itself (diagonal)
                np.fill_diagonal(dists, np.inf)
                
                # Distance to the absolute closest neighboring cluster
                min_center_dists = np.min(dists, axis=1)
            else:
                # If K somehow drops to 1, we can't do pairwise distance. Fallback safely.
                min_center_dists = np.array([np.sqrt(global_variance) * 3.0])

            variances = []
            
            for k in range(self.K):
                cluster_points = X_np[labels == k]
                
                # Calculate topological variance: (Distance / 3)^2
                # This assumes a 3-sigma tail just touches the nearest cluster center.
                topological_variance = (min_center_dists[k] / 3.0) ** 2
                
                if len(cluster_points) > 1:
                    local_variance = np.var(cluster_points)
                    # Blend the local variance with a small fraction of the topological variance.
                    # This prevents 0.0 underflow for perfectly overlapping points.
                    adjusted_variance = local_variance + (topological_variance * 0.1)
                else:
                    # N=1 Trap Fixed: Rely entirely on the topological variance!
                    adjusted_variance = topological_variance
                
                # Ultimate safety floor to prevent any exact 0.0s sneaking into Pyro
                adjusted_variance = max(adjusted_variance, 1e-6)
                variances.append(adjusted_variance)

            print("Initial means:", means, flush=True)
            print("Initial variances:", variances, flush=True)

            # Sklearn's KMeans rarely drops clusters unless there are heavy duplicates,
            # but we keep your safety check just in case.
            if len(means) == self.K:
                break  
            else:
                self.K = max(len(means), 1)
                print(f"Retrying K-Means with updated K = {self.K}", flush=True)

        # Convert valid means and variances to tensors
        self.init_means = torch.tensor(means, device=self.device, dtype=torch.float32)
        self.init_variances = torch.tensor(variances, device=self.device, dtype=torch.float32)

        # Debugging: Print initial and final variances
        print("Final variances after filtering and adjustments:", self.init_variances, flush=True)
        print("Final number of clusters (K):", self.K, flush=True)


    def model(self, data, weights=None):
        if weights is None:
            weights = torch.ones(data.shape[0], device=self.device)
        N = data.shape[0]
        K = self.K

        with pyro.plate("components", K):
            v = pyro.sample("v", dist.Beta(torch.ones(K, device=self.device, dtype=torch.float32) * self.gamma,
                                           torch.full((K,), self.gamma, device=self.device, dtype=torch.float32)))
            mu = pyro.sample("mu", dist.Normal(self.mu0, 1.0 / torch.sqrt(self.lambda0)))
            sigma2 = pyro.sample("sigma2", dist.InverseGamma(self.alpha0, self.beta0)).to(self.device)
        
        
        pi = self.stick_breaking(v)

        with pyro.plate("data", N, subsample_size=self.batch_size) as ind:
            w_batch = weights[ind]
            if torch.any(w_batch > 1):
                print("Weights contain values greater than 1:", w_batch[w_batch > 1], flush=True)            
            x_batch = data[ind]
            z = pyro.sample("z", dist.Categorical(pi).expand([x_batch.shape[0]]), infer={"enumerate": "parallel", "scale": w_batch})
            safe_sigma = torch.clamp(sigma2[z], min=1e-8)
            pyro.sample("obs", dist.Normal(mu[z], safe_sigma.sqrt()), obs=x_batch)

    def guide(self, data, weights=None):
        K = self.K
        N = data.shape[0]

        m_loc = pyro.param("m_loc", self.init_means.clone(),
                           constraint=constraints.unit_interval)
        
        alpha_q = pyro.param("alpha_q", torch.ones(K, device=self.device, dtype=torch.float32) * self.gamma,
                     constraint=constraints.greater_than(1e-3))
        
        beta_q = pyro.param("beta_q", torch.ones(K, device=self.device, dtype=torch.float32) * self.gamma,
                            constraint=constraints.greater_than(1e-3))

        m_scl = pyro.param("m_scl", self.init_variances.clone().sqrt().clamp(min=1e-5),
                        constraint=constraints.greater_than(1e-6))

        # Prevent infinite variance in the InverseGamma distribution
        a_q = pyro.param("a_q", torch.full((K,), max(float(self.alpha0), 1.05), device=self.device),
                        constraint=constraints.greater_than(1.01))

        b_q = pyro.param("b_q", self.init_variances.clone().clamp(min=1e-6), 
                        constraint=constraints.greater_than(1e-8))
        
        
        with pyro.plate("components", K):
            pyro.sample("v", dist.Beta(alpha_q, beta_q))
            pyro.sample("mu", dist.Normal(m_loc, m_scl))
            pyro.sample("sigma2", dist.InverseGamma(a_q, b_q)).to(self.device)

        with pyro.plate("data", N, subsample_size=self.batch_size):
            pass  # required for consistent guide structure

    def stick_breaking(self, v):
        # Construct mixture weights from stick-breaking process
        eps = 1e-6
        remaining_stick = torch.cumprod(1 - v + eps, dim=-1).to(self.device)  # Ensure device consistency
        remaining_stick = torch.roll(remaining_stick, shifts=1, dims=0).to(self.device)
        remaining_stick[0] = torch.tensor(1.0, device=self.device)  # Explicitly set device for scalar
        pi = (v * remaining_stick).to(self.device)  # Ensure device consistency
        pi = pi / pi.sum(-1, keepdim=True)  # Normalize
        return pi
    
    def fit(self, data, weights=None, num_epochs=10000, batch_size=10000):

        if weights is not None:
            print("Weights provided: ", flush=True)
            # Repeat data according to weight
            data = np.repeat(data, weights.astype(int), axis=0)
            weights = torch.ones(len(data), device=self.device)
        else:
            weights = torch.ones(len(data), device=self.device)

        self.N = len(data)  # total number of data points
        self.batch_size = min(batch_size, len(data))
        self.K = min(self.K, len(data))  # Ensure K does not exceed data size
        
        data = torch.tensor(data, device=self.device, dtype=torch.float32)
        data = data.squeeze(-1)  # Convert [batch_size, 1] to [batch_size]
        data = data.to(self.device)
        
        print("Fitting PyroDPGMM with K =", self.K, flush=True)
        self.kmeans_init(data)
        self.K = self.init_means.shape[0]  # Update K based on KMeans initialization
        print("K after KMeans initialization:", self.K, flush=True)

        pyro.clear_param_store()  # Clear previous parameters
        optimizer = Adam({"lr": 0.001})
        svi = SVI(self.model, self.guide, optimizer, loss=TraceEnum_ELBO())

        print("Number of data points: ", len(data), flush=True)
        print("Batch size: ", self.batch_size, flush=True)
        steps_per_epoch = math.ceil(len(data) / self.batch_size)    # ~1 full sweep on average
        print(f"Steps per epoch: {steps_per_epoch}", flush=True)
        best_elbo = float('-inf')
        no_improvement_epochs = 0
        
        ema_elbo = None
        alpha_ema = 0.1  # Smoothing factor (lower = smoother, ignores spikes)
        best_params = None

        print(f"Steps per epoch: {steps_per_epoch}", flush=True)
        for epoch in tqdm(range(num_epochs), total=num_epochs, desc="Training..."):
            epoch_loss = 0.0
            num_batches = 0
            for _ in range(steps_per_epoch):
                loss = svi.step(data, weights)
                epoch_loss += loss
                num_batches += 1
            
            # Calculate raw ELBO for this epoch
            raw_elbo = -epoch_loss / steps_per_epoch
            
            # --- NEW: Smooth the ELBO ---
            if ema_elbo is None:
                ema_elbo = raw_elbo
            else:
                ema_elbo = (1 - alpha_ema) * ema_elbo + alpha_ema * raw_elbo

            if epoch % 10 == 0: # Print every 10 epochs to reduce spam
                print(f"Epoch {epoch}: Raw = {raw_elbo:.2f} | EMA = {ema_elbo:.2f}", flush=True)            

            # --- NEW: Early stopping logic based on smoothed ELBO ---
            # Using a relative tolerance is usually safer than an absolute 1e-3
            # but we can stick to absolute if you prefer!
            if ema_elbo > best_elbo + 1e-3:  
                best_elbo = ema_elbo
                no_improvement_epochs = 0
                
                # Snapshot the best parameters!
                best_params = pyro.get_param_store().get_state()
            else:
                no_improvement_epochs += 1

            if no_improvement_epochs >= 100:
                print(f"\nEarly stopping at epoch {epoch}. Best smoothed ELBO = {best_elbo:.2f}")
                # Restore the best parameters so the model isn't degraded!
                pyro.get_param_store().set_state(best_params)
                break   

        # If the loop finishes all epochs without early stopping, 
        # still ensure we fall back to the absolute best state we found.
        if best_params is not None:
            pyro.get_param_store().set_state(best_params)

        truncated_means, truncated_stds, truncated_weights = self.truncate_clusters()
        return truncated_means, truncated_stds, truncated_weights
    
    def get_params(self):
        with torch.no_grad():
            # Extract weights
            log_pi = expected_log_sticks(pyro.param("alpha_q"), pyro.param("beta_q"))
            pi = torch.exp(log_pi)
            weights = (pi / pi.sum()).detach().cpu().numpy()

            # Extract means
            means = pyro.param("m_loc").detach().cpu().numpy()

            # Extract variances
            variances = pyro.param("b_q") / (pyro.param("a_q") - 1)  # Variance from InverseGamma
            variances = variances.detach().cpu().numpy()

            return means, variances, weights
    
    def truncate_clusters(self, threshold=0.99):
        """
        Truncate clusters based on cumulative weights using Pyro's learned parameters.
        Args:
            threshold (float): Cumulative weight threshold (e.g. 0.999 for 99.9%).
        Returns:
            truncated_means, truncated_stds, truncated_weights
        """
        with torch.no_grad():
            # 1) get and normalize
            means, variances, weights = self.get_params()
            weights = weights / weights.sum()

            print("Means: ", means, flush=True)
            print("Variances: ", variances, flush=True)
            print("Weights: ", weights, flush=True)

            # 2) sort descending
            idx = np.argsort(weights)[::-1]
            w_sorted = weights[idx]
            m_sorted = means[idx]
            std_sorted = np.sqrt(variances[idx])

            # 3) find how many components to keep
            cumulative = np.cumsum(w_sorted)
            # np.searchsorted tells us the first index where cumulative >= threshold
            cut = np.searchsorted(cumulative, threshold) + 1  

            # 4) truncate and renormalize
            kept_means   = m_sorted[:cut]
            kept_stds    = std_sorted[:cut]
            kept_weights = w_sorted[:cut]
            kept_weights = kept_weights / kept_weights.sum()

        return kept_means, kept_stds, kept_weights


    def create_gmm(self, means, covs, weights):        
        if means.shape[0] == 0:
            return None
        
        num_components = means.shape[0]
        n_features = means.shape[1] if means.ndim > 1 else 1

        means_reshaped = means
        covs_reshaped = covs
        if n_features == 1 and means.ndim == 1: # Ensure 2D for (K,1) shape
            means_reshaped = means.reshape(num_components, 1)
            covs_reshaped = covs.reshape(num_components, 1)

        new_gmm = GaussianMixture(
            n_components=num_components,
            covariance_type='diag', # Ensure this matches your covs structure
        )
        
        # Manually set the fitted parameters
        new_gmm.weights_ = weights
        new_gmm.means_ = means_reshaped
        new_gmm.covariances_ = covs_reshaped # For 'diag', this is (K, D) array of variances
        
        # GaussianMixture computes precisions_cholesky_ internally when needed.
        # Forcing it can be done like this:
        safe_covs = np.maximum(covs_reshaped, np.finfo(covs_reshaped.dtype).eps)
        new_gmm.precisions_cholesky_ = 1.0 / np.sqrt(safe_covs)
        
        # Set n_features_in_ for consistency with scikit-learn checks
        new_gmm.n_features_in_ = n_features
        # n_components_ is automatically set by virtue of setting means_, etc.
        # but explicitly:
        new_gmm.n_components_ = num_components

        return new_gmm
