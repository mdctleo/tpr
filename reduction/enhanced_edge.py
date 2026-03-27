import numpy as np
from tqdm import tqdm 
from dbstream import OnlineDBStreamClustering, tune_dbstream_params
from kmeans import KMeansElbow
from pyro_dpgmm import PyroDPGMM, truncate_clusters
from sklearn.neighbors import KernelDensity
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from scipy.stats import norm
from scipy.integrate import quad
from matplotlib import pyplot as plt
from datetime import datetime
import pytz
from scipy.stats import wasserstein_distance




class EnhancedEdge():
    def __init__(self, u, v, syscall, args, global_variance=0, total_data_length=0):
        self.u = u.split("/")[-1]
        self.v = v.split("/")[-1]
        self.syscall = syscall
        
        self.num_clusters = 0
        self.count = 0
        self.means = None
        self.covariances = None
        self.weights = None
        self.kde = None
        self.timestamps = None
        self.reduction_threshold = 2000
        self._metric_cache = {}

        self.global_variance = global_variance
        self.total_data_length = total_data_length

    def fit(self, timestamps, args, data_weights=None):
        self._metric_cache.clear()
        self.timestamps = np.asarray(timestamps).reshape(-1, 1)
        self.count += len(timestamps)

        if args.use_dbstream:
            bandwidth = self._sheather_jones_bandwidth_1d(timestamps)
            print("in kde edge fit with dbstream", flush=True)
            print("max timestamp: ", np.max(timestamps), flush=True)
            print("min timestamp: ", np.min(timestamps), flush=True)
            clustering_threshold, fading_factor, cleanup_interval, intersection_factor, minimum_weight, fallback_variances = tune_dbstream_params(timestamps, bandwidth=bandwidth)
            print("max timestamp after tuning: ", np.max(timestamps), flush=True)
            print("min timestamp after tuning: ", np.min(timestamps), flush=True)

            dbstream = OnlineDBStreamClustering(
                clustering_threshold=clustering_threshold,
                fading_factor=fading_factor,
                cleanup_interval=cleanup_interval,
                intersection_factor=intersection_factor,
                minimum_weight=minimum_weight,
                fallback_variances= fallback_variances,
            )

            print("max timestamp before learning: ", np.max(timestamps), flush=True)
            print("min timestamp before learning: ", np.min(timestamps), flush=True)

            for timestamp in tqdm(timestamps, desc="Fitting DBStream Learn..."):
                dbstream.learn_data(timestamp)

            print("max timestamp after learning: ", np.max(timestamps), flush=True)
            print("min timestamp after learning: ", np.min(timestamps), flush=True)

            for timestamp in tqdm(timestamps, desc="Fitting DBStream predict..."):
                dbstream.predict_data(timestamp)

            print("max timestamp after predicting: ", np.max(timestamps), flush=True)
            print("min timestamp after predicting: ", np.min(timestamps), flush=True)

            means, covs, weights, gmm = dbstream.get_gmm_params()

            if gmm is None:
                print("edge:", self.u, self.v, self.syscall, "==GMM is None after fitting DPGMM", flush=True)

            # assert gmm is not None, "GMM is None after fitting DPGMM"

        
            print("Means shape: ", means.shape, flush=True)
            print("Covs shape: ", covs.shape, flush=True)
            print("Weights shape: ", weights.shape, flush=True)
            print("Number of components: ", len(means), flush=True)

            # self.kde = gmm
            # self.num_clusters = len(means)
            self.means = means
            self.covariances = covs
            self.weights = weights
            self.num_clusters = len(means)
            self.kde = gmm

            print("Min timestamp after DBStream: ", np.min(timestamps), flush=True)
            print("Max timestamp after DBStream: ", np.max(timestamps), flush=True)
        elif args.use_kmeans:
            kmeans_elbow = KMeansElbow(100)
            kmeans_elbow.fit(timestamps)
            optimal_k = kmeans_elbow.get_optimal_k()
            kmeans = kmeans_elbow.fit_optimal_k(timestamps, optimal_k)

            means, covs, weights, gmm = kmeans_elbow.get_gmm_params(kmeans, timestamps)

            if gmm is None:
                print("edge:", self.u, self.v, self.syscall, "==GMM is None after fitting KMeans", flush=True)

            print("Means shape: ", means.shape, flush=True)
            print("Covs shape: ", covs.shape, flush=True)
            print("Weights shape: ", weights.shape, flush=True)
            print("Means : ", means)
            print("Stds : ", np.sqrt(covs))
            print("Weights : ", weights)
            print("Number of components: ", len(means), flush=True)

            self.means = means
            self.covariances = covs
            self.weights = weights
            self.num_clusters = len(means)
            self.kde = gmm
        elif args.use_dpgmm:

            kde = self.fit_kde(timestamps, data_weights)
            model = self.fitted_kde_to_bgmm_simple(kde, x_min=np.min(timestamps), x_max=np.max(timestamps), n_components=args.K, grid_size=500, total_count=5000, grid_padding=0.10)
            # model = self.fitted_kde_to_gmm_bic_simple(kde, x_min=np.min(timestamps), x_max=np.max(timestamps), k_max=args.K, grid_size=500, total_count=5000, grid_padding=0.10)

            means, covs, weights = model.means_.flatten(), model.covariances_.flatten(), model.weights_
            means, covs, weights = truncate_clusters(means=means, variances=covs, weights=weights)

            self.means = means
            self.covariances = covs
            self.weights = weights
            self.num_clusters = len(self.means)
            self.kde = model  # Store the model for later use
        else:
            if len(timestamps) >= 2:
               self.kde = self.fit_kde(timestamps, data_weights)


    def fit_kde(self, data, weights):
        normalized_data = data
        print("Normalized data shape: ", normalized_data.shape, flush=True)
        
        bandwidth = self._sheather_jones_bandwidth_1d(normalized_data)
        return KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(normalized_data, sample_weight=weights)

    
    def _sheather_jones_bandwidth_1d(self, data):
        """
        Compute a Sheather-Jones-style bandwidth using adaptive quadrature.
        """
        data = np.asarray(data, dtype=np.float64).ravel()

        if len(data) < 2:
            return None
        if not np.all(np.isfinite(data)):
            return None

        n = len(data)
        std_dev = np.std(data)
        iqr = np.subtract(*np.percentile(data, [75, 25]))
        sigma = min(std_dev, iqr / 1.34)

        if sigma == 0:
            return None

        def pilot_density_second_derivative(x):
            return np.mean(
                norm.pdf((x - data) / sigma) * ((x - data) ** 2 - sigma**2) / sigma**5
            )

        def integrand(x):
            return pilot_density_second_derivative(x) ** 2

        integral, _ = quad(integrand, np.min(data) - 3 * sigma, np.max(data) + 3 * sigma)
        integral += 0.01 * np.var(data)

        if integral <= 0 or not np.isfinite(integral):
            fallback = 1.06 * std_dev * n ** (-1 / 5)
            return fallback if np.isfinite(fallback) and fallback > 0 else None

        bandwidth = (1.06 * sigma * n ** (-1 / 5)) / (integral ** (1 / 5))
        return bandwidth if np.isfinite(bandwidth) and bandwidth > 0 else None
    
    def gh3_pseudo_points(self):
        """Return *deterministic* pseudo‑observations that exactly match the first
        two moments of a univariate Gaussian sample.

        Parameters
        ----------
        mean : float
            Cluster mean (µ).
        var : float
            Cluster variance (σ²).
        count : int
            Effective number of original data points represented by this cluster
            (N). This determines the *sample_weight* values.

        Returns
        -------
        coords : ndarray, shape (3,)
            The three support points: [µ‑√3σ,  µ,  µ+√3σ].
        weights : ndarray, shape (3,)
            Corresponding *sample_weight* values: [N/6, 4N/6, N/6].

        Notes
        -----
        *  The rule is the 3‑point Gauss–Hermite quadrature. It preserves the
        0‑th, 1‑st and 2‑nd moments of the full batch exactly.
        *  Suitable when you want to feed a very small, deterministic set of
        pseudo‑observations (plus sample weights) to scikit‑learn’s
        ``BayesianGaussianMixture`` instead of passing a full history or
        random Monte‑Carlo draws.
        """

        points = []
        points_weights = []
        means = self.means
        covs = self.covariances
        weights = self.weights

        for mean, var, component_weight in zip(means, covs, weights):
            # Calculate the effective count for this component
            effective_count = component_weight * self.count

            # Calculate the standard deviation
            sigma = np.sqrt(var)
            sqrt3 = np.sqrt(3.0)

            # Generate the 3 Gauss-Hermite points
            coords = np.array([mean - sqrt3 * sigma, mean, mean + sqrt3 * sigma], dtype=float)

            # Calculate the weights for the 3 points
            gh3_weights = np.array([effective_count / 6, 4 * effective_count / 6, effective_count / 6], dtype=float)

            # Append the points and weights
            points.append(coords)
            points_weights.append(gh3_weights)

        # Concatenate all points and weights
        all_points = np.concatenate(points)
        all_weights = np.concatenate(points_weights)

        # Normalize the weights relative to the total dataset size
        # normalized_weights = all_weights / self.total_data_length

        return all_points, all_weights
    
    def compute_average_log_likelihood(self, timestamps=None, sample_size=10000):
        if self.kde is None:
            return None
        data_2d, _ = self._resolve_timestamps(timestamps)
        if data_2d is None or data_2d.shape[0] == 0:
            return None

        # Uniformly subsample for speed on large edges.
        if data_2d.shape[0] > sample_size:
            idx = np.random.choice(data_2d.shape[0], size=sample_size, replace=False)
            sampled_data = data_2d[idx]
        else:
            sampled_data = data_2d

        return -np.mean(self.kde.score_samples(sampled_data))

    def _resolve_timestamps(self, timestamps):
        if timestamps is None:
            timestamps = self.timestamps
        if timestamps is None:
            return None, None

        arr = np.asarray(timestamps, dtype=np.float64)
        flat = arr.ravel()
        if flat.size == 0:
            return None, None

        return flat.reshape(-1, 1), flat

    def _build_density_context(self, timestamps=None, grid_size=10000):
        if self.kde is None:
            return None

        data_2d, flat = self._resolve_timestamps(timestamps)
        if data_2d is None:
            return None

        grid_size = int(max(128, grid_size))
        data_min, data_max = np.min(flat), np.max(flat)
        if data_min == data_max:
            return {
                "grid": np.array([data_min], dtype=np.float64),
                "true_density": np.array([1.0], dtype=np.float64),
                "kde_density": np.array([1.0], dtype=np.float64),
                "dx": 0.0,
            }

        cache_key = (
            id(self.kde),
            grid_size,
            flat.size,
            float(data_min),
            float(data_max),
            float(np.mean(flat)),
            float(np.var(flat)),
        )
        cached = self._metric_cache.get(cache_key)
        if cached is not None:
            return cached

        grid = np.linspace(data_min, data_max, grid_size, dtype=np.float64)
        bandwidth = self._sheather_jones_bandwidth_1d(flat)
        if bandwidth is None or bandwidth <= 0:
            std_dev = np.std(flat)
            bandwidth = 1.06 * std_dev * flat.size ** (-1 / 5) if std_dev > 0 else 1e-6

        true_kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(data_2d)
        true_density = np.exp(true_kde.score_samples(grid[:, np.newaxis]))
        kde_density = np.exp(self.kde.score_samples(grid[:, np.newaxis]))

        true_sum = true_density.sum()
        kde_sum = kde_density.sum()
        if true_sum <= 0 or kde_sum <= 0:
            return None

        true_density /= true_sum
        kde_density /= kde_sum
        dx = (data_max - data_min) / (grid_size - 1)

        context = {
            "grid": grid,
            "true_density": true_density,
            "kde_density": kde_density,
            "dx": dx,
        }
        self._metric_cache[cache_key] = context
        return context
    
    def compute_kl_divergence(self, timestamps=None, grid_size=10000):
        """
        Compute the KL divergence between the true density and the KDE density.

        Parameters:
        grid_size (int): Number of points to evaluate the densities on.

        Returns:
        float: KL divergence value.
        """
        if self.kde is None:
            return None

        context = self._build_density_context(timestamps=timestamps, grid_size=grid_size)
        if context is None:
            return None

        true_density = context["true_density"]
        kde_density = context["kde_density"]
        epsilon = 1e-10  # Small value to avoid division by zero or log of zero
        p = true_density + epsilon
        q = kde_density + epsilon

        # Compute KL divergence
        kl_divergence = np.sum(p * np.log(p / q))

        return kl_divergence
    
    def compute_wasserstein_distance(self, timestamps=None, grid_size=10000):
        """
        Compute the Wasserstein distance between the true density and the KDE density.

        Parameters:
        timestamps (np.ndarray): The input data points.
        grid_size (int): Number of points to evaluate the densities on.

        Returns:
        float: Wasserstein distance value.
        """
        if self.kde is None:
            return None

        context = self._build_density_context(timestamps=timestamps, grid_size=grid_size)
        if context is None:
            return None

        grid = context["grid"]
        true_density = context["true_density"]
        kde_density = context["kde_density"]

        # Compute Wasserstein distance
        wasserstein_dist = wasserstein_distance(grid, grid, u_weights=true_density, v_weights=kde_density)

        return wasserstein_dist
    
    def compute_integrated_square_error(self, timestamps=None, grid_size=10000):
        """
        Compute the Integrated Square Error (ISE) between the true density and the KDE density.

        Parameters:
        timestamps (np.ndarray): The input data points.
        grid_size (int): Number of points to evaluate the densities on.

        Returns:
        float: Integrated Square Error value.
        """
        if self.kde is None:
            return None

        context = self._build_density_context(timestamps=timestamps, grid_size=grid_size)
        if context is None:
            return None

        grid = context["grid"]
        true_density = context["true_density"]
        kde_density = context["kde_density"]

        # Compute the squared difference between the densities
        squared_difference = (true_density - kde_density) ** 2

        # Compute the Integrated Square Error (ISE) by integrating the squared difference
        ise = np.trapezoid(squared_difference, x=grid)

        return ise


    def visualize_distribution(self, timestamps, log_scale=False, name=None, scaler=None):
        if self.kde is None:
            raise ValueError("KDE is not fitted")

        # Inverse transform timestamps if a scaler is provided
        if scaler is not None:
            original_timestamps = scaler.inverse_transform(timestamps.reshape(-1, 1)).flatten()
        else:
            original_timestamps = timestamps

        # Debug: Print the raw timestamps
        print("Original timestamps (raw):", original_timestamps[:10], flush=True)

        # Create a grid in the scaled space (0 to 1)
        scaled_grid = np.linspace(0, 1, 10000)

        # Evaluate KDE density on the scaled grid
        kde_density = np.exp(self.kde.score_samples(scaled_grid[:, np.newaxis]))

        # Create a histogram for the true density
        iqr = np.subtract(*np.percentile(timestamps, [75, 25]))
        bin_width = 2 * iqr / np.cbrt(len(timestamps))
        bins = int(np.ceil((np.max(timestamps) - np.min(timestamps)) / bin_width))
        max_bins = 1000  # Adjust this value as needed
        bins = min(bins, max_bins)  # Ensure the number of bins does not exceed max_bins
        hist, bin_edges = np.histogram(timestamps, bins=bins, density=True)

        # Plot the true density
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), color='gray', alpha=0.7, label='Histogram Density (Binned)')

        # Plot the KDE density
        plt.plot(scaled_grid, kde_density, color='blue', label='DPGMM Density')

        # Format the x-axis to show human-readable time
        if scaler is not None:
            def format_func(x, pos):
                # Convert scaled tick positions back to original timestamps
                timestamp = scaler.inverse_transform([[x]]).flatten()[0]
                return datetime.fromtimestamp(timestamp, tz=pytz.timezone('Canada/Atlantic')).strftime('%H:%M')

            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_func))

        # Add labels and title
        plt.xlabel('Time (HH:MM)')
        plt.ylabel('Density')
        plt.title(name)
        if log_scale:
            plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(name)
        plt.close()
