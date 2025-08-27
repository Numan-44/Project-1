import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any


class CentroidInitializer:
    """Initialize centroids for K-Means clustering using different methods."""

    @staticmethod
    def random_init(X: np.ndarray, k: int) -> np.ndarray:
        """
        Initialize centroids by randomly selecting k data points.

        Args:
            X: Input data array of shape (n_samples, n_features)
            k: Number of centroids to initialize

        Returns:
            Array of shape (k, n_features) containing initial centroids
        """
        n_samples = len(X)
        random_indices = np.random.choice(n_samples, k, replace=False)
        return X[random_indices].copy()

    @staticmethod
    def kmeans_plus_plus_init(X: np.ndarray, k: int) -> np.ndarray:
        """
        Initialize centroids using the K-Means++ algorithm.

        Args:
            X: Input data array of shape (n_samples, n_features)
            k: Number of centroids to initialize

        Returns:
            Array of shape (k, n_features) containing initial centroids
        """
        n_samples = X.shape[0]
        centroids = np.empty((k, X.shape[1]), dtype=X.dtype)

        # First centroid chosen randomly
        first_idx = np.random.randint(n_samples)
        centroids[0] = X[first_idx]

        # Distances to nearest centroid
        closest_dist_sq = np.linalg.norm(X - centroids[0], axis=1) ** 2

        for i in range(1, k):
            # Probabilities proportional to squared distances
            total_dist_sq = closest_dist_sq.sum()
            if total_dist_sq == 0:
                # fallback: random choice among remaining
                remaining_indices = np.setdiff1d(np.arange(n_samples), 
                                                np.where((X[:, None] == centroids[:i]).all(-1))[0])
                if len(remaining_indices) == 0:
                    next_idx = np.random.randint(n_samples)
                else:
                    next_idx = np.random.choice(remaining_indices)
            else:
                probabilities = closest_dist_sq / total_dist_sq
                next_idx = np.random.choice(n_samples, p=probabilities)

            centroids[i] = X[next_idx]

            # Update closest distances
            new_dist_sq = np.linalg.norm(X - centroids[i], axis=1) ** 2
            closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

        return centroids

class ClusterCalculator:
    """Helper class for cluster assignment and centroid recalculation."""

    def __init__(self, k: int, precision: int = 6):
        """
        Initialize ClusterCalculator.

        Args:
            k: Number of clusters
            precision: Floating point precision for calculations (default: 6)
        """
        self.k = k
        self.precision = precision

    def assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each data point to the nearest centroid.

        Args:
            X: Input data array of shape (n_samples, n_features)
            centroids: Current centroids array of shape (k, n_features)

        Returns:
            Array of cluster labels for each data point
        """
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def recalculate_centroids(
        self, 
        X: np.ndarray, 
        labels: np.ndarray, 
        current_centroids: np.ndarray
    ) -> np.ndarray:
        """
        Recalculate centroids based on current cluster assignments.

        Args:
            X: Input data array of shape (n_samples, n_features)
            labels: Cluster labels for each data point
            current_centroids: Current centroids array

        Returns:
            New centroids array of shape (k, n_features)
        """
        new_centroids = []
        for k in range(self.k):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroid = np.mean(cluster_points, axis=0)
                # Round to specified precision
                new_centroid = np.round(new_centroid, decimals=self.precision)
            else:
                new_centroid = current_centroids[k]
            new_centroids.append(new_centroid)
        return np.array(new_centroids)

    def calculate_sse(
        self, 
        X: np.ndarray, 
        labels: np.ndarray, 
        centroids: np.ndarray
    ) -> float:
        """
        Calculate Sum of Squared Errors (SSE) for the clustering.

        Args:
            X: Input data array of shape (n_samples, n_features)
            labels: Cluster labels for each data point
            centroids: Current centroids array

        Returns:
            Sum of squared errors value rounded to specified precision
        """
        # Assign each point's centroid
        diffs = X - centroids[labels]
        sse = np.sum(diffs ** 2)
        return np.round(sse, decimals=self.precision)



class KMeans:
    """K-Means clustering algorithm implementation."""

    def __init__(
        self, 
        k: int = 3, 
        max_iters: int = 100, 
        init: str = 'random', 
        max_history: int = 20,
        precision: int = 6,
        convergence_tol: float = 1e-4
    ):
        """
        Initialize K-Means clustering algorithm.

        Args:
            k: Number of clusters (default: 3)
            max_iters: Maximum number of iterations (default: 100)
            init: Initialization method ('random' or 'kmeans++') (default: 'random')
            max_history: Maximum number of iterations to keep in history (default: 20)
            precision: Floating point precision for calculations (default: 6 decimal places)
            convergence_tol: Convergence tolerance for centroid changes (default: 1e-4)
        """
        self.k = k
        self.max_iters = max_iters
        self.init = init
        self.max_history = max_history
        self.precision = precision
        self.convergence_tol = convergence_tol
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.iteration_history_ = []
        self.calculator = ClusterCalculator(k, precision)

    def fit(self, X: np.ndarray, dtype: type = np.float64) -> 'KMeans':
        """
        Fit K-Means clustering to the input data.

        Args:
            X: Input data array of shape (n_samples, n_features)
            dtype: Data type for floating point calculations (default: np.float64)

        Returns:
            self: Fitted KMeans instance
        """
        X = np.array(X, dtype=dtype)
        
        if self.init == 'kmeans++':
            self.centroids_ = CentroidInitializer.kmeans_plus_plus_init(X, self.k)
        else:
            self.centroids_ = CentroidInitializer.random_init(X, self.k)
        
        # Round initial centroids to specified precision
        self.centroids_ = np.round(self.centroids_, decimals=self.precision)
        
        self.iteration_history_.append({
            "centroids": self.centroids_.copy(),
            "labels": self.labels_.copy() if self.labels_ is not None else []
        })

        for _ in range(self.max_iters):
            labels = self.calculator.assign_clusters(X, self.centroids_)
            new_centroids = self.calculator.recalculate_centroids(
                X, labels, self.centroids_
            )
            
            self.iteration_history_.append({
                'centroids': new_centroids.copy(),
                'labels': labels.copy()
            })
            
            # Implement max_history limit here
            if len(self.iteration_history_) > self.max_history:
                self.iteration_history_.pop(0)
            
            # Check convergence with specified tolerance
            if np.allclose(
                self.centroids_, 
                new_centroids, 
                rtol=self.convergence_tol,
                atol=10**(-self.precision)
            ):
                break
            
            self.centroids_ = new_centroids
        
        self.labels_ = labels
        self.inertia_ = self.calculator.calculate_sse(X, labels, self.centroids_)
        return self


class ElbowAnalyzer:
    """Analyzer for determining optimal number of clusters using elbow method."""

    @staticmethod
    def calculate_elbow_data(
        X: np.ndarray, 
        max_k: int = 10, 
        init: str = 'random',
        precision: int = 6,
        dtype: type = np.float64
    ) -> Tuple[List[int], List[float]]:
        """
        Calculate SSE values for different numbers of clusters.

        Args:
            X: Input data array of shape (n_samples, n_features)
            max_k: Maximum number of clusters to test (default: 10)
            init: Initialization method for K-Means (default: 'random')
            precision: Floating point precision for SSE values (default: 6)
            dtype: Data type for floating point calculations (default: np.float64)

        Returns:
            Tuple containing list of k values and corresponding SSE values
        """
        k_values = []
        sse_values = []
        
        for k in range(1, max_k + 1):
            if k > len(X):
                break
            
            kmeans = KMeans(k=k, init=init, precision=precision)
            kmeans.fit(X, dtype=dtype)
            k_values.append(k)
            sse_values.append(kmeans.inertia_)
        
        return k_values, sse_values