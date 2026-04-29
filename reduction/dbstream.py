from custom_dbstream import DBSTREAM
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
import copy

class OnlineDBStreamClustering:
    def __init__(self, clustering_threshold=1.5, fading_factor=0.05,
                 cleanup_interval=4, intersection_factor=0.5, minimum_weight=10, fallback_variances=1e-10):
        
        print("Clustering threshold: ", clustering_threshold, flush=True)
        print("Fading factor: ", fading_factor, flush=True)
        print("Cleanup interval: ", cleanup_interval, flush=True)
        print("Intersection factor: ", intersection_factor, flush=True)
        print("Minimum weight: ", minimum_weight, flush=True)
        print("Fallback variances: ", fallback_variances, flush=True)
        self.dbstream = DBSTREAM(clustering_threshold = clustering_threshold,
                             fading_factor = fading_factor,
                             cleanup_interval = cleanup_interval,
                             intersection_factor = intersection_factor,
                             minimum_weight = minimum_weight,
                             fallback_variances= fallback_variances)


    def learn_data(self, timestamp):
        """Process streaming data points one at a time."""
        timestamp_dict = {"timestamp": timestamp}
        self.dbstream.learn_one(copy.deepcopy(timestamp_dict))
        # self.dbstream.predict_one(timestamp_dict)

    def predict_data(self, timestamp):
        """Predict the cluster for a new data point."""
        timestamp_dict = {"timestamp": timestamp}
        self.dbstream.predict_one(copy.deepcopy(timestamp_dict))


    def get_gmm_params(self):
        # Access macro-clusters
        macro_clusters = self.dbstream.clusters
        if not macro_clusters:
            return np.array([]), np.array([]), np.array([])
        

        # print("Macro clusters: ", macro_clusters, flush=True)
        print("Length of macro clusters: ", len(macro_clusters), flush=True)

        # Extract means, stds, and weights from macro-clusters
        # for mc in macro_clusters.values():
        #     print("Macro cluster information: ", flush=True)
        #     print(mc, flush=True)
        #     print(mc.center, flush=True)
        #     print(mc.variance, flush=True)
        #     print(mc.weight, flush=True)


        means = np.array([mc.center["timestamp"] for mc in macro_clusters.values()])
        # stds = np.array([np.std(mc.points) for mc in macro_clusters.values()])
        # stds = np.array([np.sqrt(macro_cluster_variances[macro_cluster_id]) 
        #              for macro_cluster_id in macro_clusters.keys()])
        variances = np.array([mc.variance for mc in macro_clusters.values()])
        weights = np.array([mc.weight for mc in macro_clusters.values()])
        weights = weights.astype(float)  # Convert weights to float

        print("Means: ", means, flush=True)
        print("Variances: ", variances, flush=True)
        # # Validate means
        if np.isnan(means).any():
            raise ValueError("Means contain NaN values.")
        if len(means) == 0:
            raise ValueError("No macro-clusters found. Means array is empty.")

        # Validate variances
        if np.isnan(variances).any():
            raise ValueError("Variances contain NaN values.")
        if (variances <= 0).any():
            raise ValueError("Variances must be positive. Found zero or negative variances.")

        # Validate weights
        if np.isnan(weights).any():
            raise ValueError("Weights contain NaN values.")
        if weights.sum() == 0:
            raise ValueError("Weights sum to zero. Check macro-cluster weights.")

        weights /= weights.sum()  # Normalize weights


        gmm = GaussianMixture(n_components=len(means), covariance_type='diag')
        gmm.means_ = means.reshape(-1, 1)  # Reshape to match GMM's expected format
        gmm.covariances_ = variances  # Variances (std^2)
        gmm.weights_ = weights
        gmm.precisions_cholesky_ = 1 / np.sqrt(variances)  # Precomputed precision

        return means, variances, weights, gmm
    

def compute_pairwise_distance_stats(data, chunk_size=1000, sample_size=100000):
    n = len(data)
    data = np.array(data).reshape(-1, 1)  # Ensure data is 2D

    # Subsample the data
    if n > sample_size:
        sampled_indices = np.random.choice(n, sample_size, replace=False)
        data = data[sampled_indices]

    avg_distance = 0
    std_distance = 0
    total_pairs = 0

    for i in range(0, len(data), chunk_size):
        chunk_i = data[i:i+chunk_size]
        for j in range(0, len(data), chunk_size):
            chunk_j = data[j:j+chunk_size]
            distances = pairwise_distances(chunk_i, chunk_j)
            avg_distance += np.sum(distances)
            std_distance += np.sum((distances - np.mean(distances)) ** 2)
            total_pairs += distances.size

    avg_distance /= total_pairs
    std_distance = np.sqrt(std_distance / total_pairs)

    return avg_distance, std_distance
    
def tune_dbstream_params(data, bandwidth):
    # print(data)
    data = np.array(data).reshape(-1, 1)  # Ensure data is 2D for pairwise_distances
    # print("Tune database data shape: ", data.shape, flush=True)

    # Compute pairwise distances
    avg_distance, std_distance = compute_pairwise_distance_stats(data, chunk_size=10000)

    # # Clustering threshold
    # clustering_threshold = avg_distance * 0.15
    clustering_threshold = bandwidth

    # Fading factor    # fading_factor = 1 / time_span if time_span > 0 else 1  # Avoid division by zero
    fading_factor = 0

    # Cleanup interval
    cleanup_interval = len(data) + 1
    
    # Intersection factor
    # data_density = len(data) / (np.max(data) - np.min(data)) if np.max(data) != np.min(data) else len(data)
    # intersection_factor = 0.5 if data_density > 10 else 0.3  # Adjust thresholds as needed
    intersection_factor = std_distance / (std_distance + avg_distance)

    # Minimum weight
    minimum_weight = len(data) * 0.1

    fallback_variances = np.var(data, axis=0) * 0.1

    return clustering_threshold, fading_factor, cleanup_interval, intersection_factor, minimum_weight, fallback_variances
    