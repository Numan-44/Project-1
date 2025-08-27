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
        n_samples = len(X)
        centroids = []
        first_idx = np.random.choice(n_samples)
        centroids.append(X[first_idx].copy())
        
        for _ in range(1, k):
            distances = []
            for point in X:
                min_dist = float('inf')
                for centroid in centroids:
                    dist = np.linalg.norm(point - centroid) ** 2
                    if dist < min_dist:
                        min_dist = dist
                distances.append(min_dist)
            
            distances = np.array(distances)
            if distances.sum() == 0:
                remaining_indices = list(range(n_samples))
                for centroid in centroids:
                    for i, point in enumerate(X):
                        if np.allclose(point, centroid):
                            if i in remaining_indices:
                                remaining_indices.remove(i)
                if remaining_indices:
                    next_idx = np.random.choice(remaining_indices)
                else:
                    next_idx = np.random.choice(n_samples)
            else:
                probabilities = distances / distances.sum()
                next_idx = np.random.choice(n_samples, p=probabilities)
            
            centroids.append(X[next_idx].copy())
        
        return np.array(centroids)


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
        sse = 0.0
        for k in range(self.k):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroid = centroids[k]
                cluster_sse = np.sum((cluster_points - centroid) ** 2)
                sse += np.round(cluster_sse, decimals=self.precision)
        return sse


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