import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


class KMeansElbow:
    def __init__(self, max_k=100):
        """
        Initialize the KMeansElbow class.

        Parameters:
            max_k (int): The maximum number of clusters to test. 
        """
        self.max_k = max_k
        self.wcss = []  # To store Within-Cluster Sum of Squares
        self.optimal_k = None

    def fit(self, data):
        """
        Run K-Means for a range of K values and compute WCSS.

        Parameters:
            data (ndarray): The dataset to cluster (shape: [n_samples, n_features]).
        """
        data = np.unique(data, axis=0)  # Remove duplicated points
        self.max_k = min(self.max_k, data.shape[0])  # Ensure max_k is not greater than n_samples
        # data_variances = np.var(data, axis=0)
        # tol = np.mean(data_variances) * 0.1  # Use 10% of the mean variance
        self.wcss = []
        # print("tolerance: ", tol, flush=True)
        for k in range(1, self.max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', max_iter=10000)
            kmeans.fit(data)
            self.wcss.append(kmeans.inertia_)  # Inertia is the WCSS

        # Automatically determine the optimal K
        self.optimal_k = self._find_elbow_point()

    def _find_elbow_point(self):
        """
        Find the elbow point using the first derivative method.

        Returns:
            int: The optimal number of clusters (K).
        """
        y = np.array(self.wcss)
        x = np.arange(1, self.max_k + 1)

        # Compute the first derivative (rate of change)
        first_derivative = np.diff(y)

        # Find the point where the rate of change decreases significantly
        elbow_point = np.argmax(-first_derivative) + 1  # +1 to adjust for diff offset
        return elbow_point

    def plot_elbow(self):
        """
        Plot the WCSS vs. K and mark the optimal K.
        """
        if not self.wcss:
            raise ValueError("You must call `fit` before plotting.")

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, self.max_k + 1), self.wcss, marker='o', linestyle='--', label='WCSS')
        plt.axvline(self.optimal_k, color='r', linestyle='--', label=f'Optimal K = {self.optimal_k}')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('WCSS')
        plt.xticks(range(1, self.max_k + 1))
        plt.legend()
        plt.grid()
        plt.show()

    def get_optimal_k(self):
        """
        Get the optimal number of clusters (K).

        Returns:
            int: The optimal number of clusters.
        """
        if self.optimal_k is None:
            raise ValueError("You must call `fit` before getting the optimal K.")

        print("Optimal K: ", self.optimal_k, flush=True)

        return self.optimal_k
    

    def fit_optimal_k(self, data, optimal_k):
        """
        Fit the model and return the optimal number of clusters.

        Parameters:
            data (ndarray): The dataset to cluster (shape: [n_samples, n_features]).

        Returns:
            int: The optimal number of clusters.
        """
        # data_variances = np.var(data, axis=0)
        # tol = np.mean(data_variances) * 0.1  # Use 10% of the mean variance
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto', max_iter=10000)
        kmeans.fit(data)

        return kmeans
    
    def get_gmm_params(self, kmeans, data):
        """
        Extract parameters for a Gaussian Mixture Model (GMM) from a fitted KMeans model
        and create a GMM.

        Parameters:
            kmeans (KMeans): A fitted KMeans model.

        Returns:
            tuple: Means, variances, weights, and the GaussianMixture model.
        """
        # Extract means (cluster centers)
        means = kmeans.cluster_centers_

        # Compute variances for each cluster
        labels = kmeans.labels_
        n_clusters = kmeans.n_clusters
        variances = np.zeros((n_clusters, means.shape[1]))
        for i in range(n_clusters):
            # Get points assigned to cluster i
            cluster_points = data[labels == i]
            # Compute variance along each feature dimension
            variances[i] = np.var(cluster_points, axis=0)

        # Compute weights (proportion of points in each cluster)
        weights = np.bincount(labels) / len(labels)

        # Create a GaussianMixture model
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='diag')
        gmm.means_ = means
        gmm.covariances_ = variances
        gmm.weights_ = weights
        gmm.precisions_cholesky_ = 1 / np.sqrt(variances)  # Precomputed precision

        return means, variances, weights, gmm