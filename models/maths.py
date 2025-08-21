import numpy as np

class KMeans:
    """Simple K-Means clustering implementation (loop-based, with animation history)."""

    def __init__(self, k=3, max_iters=100, init='random'):
        self.k = k
        self.max_iters = max_iters
        self.init = init  # 'random' or 'kmeans++'
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.iteration_history_ = []  # stores snapshots of clustering progress

    def fit(self, X):
        """Run K-Means clustering and record iteration history for animation."""
        X = np.array(X, dtype=float)
        n_samples = len(X)

        # STEP 1: Initialize centroids based on method
        if self.init == 'kmeans++':
            self.centroids_ = self._init_kmeans_plus_plus(X)
        else:  # random initialization
            random_indices = np.random.choice(n_samples, self.k, replace=False)
            self.centroids_ = X[random_indices].copy()

        # Save initial state for animation
        self.iteration_history_.append({
         "centroids": self.centroids_.copy(),
         "labels": self.labels_.copy() if self.labels_ is not None else []
        })

        # STEP 2: Repeat until centroids don't change or max_iters reached
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            labels = self._assign_clusters(X)

            # Calculate new centroids
            new_centroids = self._recalculate_centroids(X, labels)

            # Save this step for animation
            self.iteration_history_.append({
                'centroids': new_centroids.copy(),
                'labels': labels.copy()
            })

            # Stop if centroids have not changed
            if np.allclose(self.centroids_, new_centroids, rtol=1e-4):
                break

            self.centroids_ = new_centroids

        # Save final results
        self.labels_ = labels
        self.inertia_ = self._calculate_sse(X, labels)

        return self

    def _init_kmeans_plus_plus(self, X):
        """Initialize centroids using K-means++ algorithm."""
        n_samples = len(X)
        centroids = []
        
        # Choose first centroid randomly
        first_idx = np.random.choice(n_samples)
        centroids.append(X[first_idx].copy())
        
        # Choose remaining centroids
        for _ in range(1, self.k):
            # Calculate distances from each point to nearest centroid
            distances = []
            for point in X:
                min_dist = float('inf')
                for centroid in centroids:
                    dist = np.linalg.norm(point - centroid) ** 2
                    if dist < min_dist:
                        min_dist = dist
                distances.append(min_dist)
            
            # Convert to probabilities
            distances = np.array(distances)
            if distances.sum() == 0:
                # All points are at existing centroids, choose randomly
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

    def _assign_clusters(self, X):
        """Assign each point to the nearest centroid."""
        distances = []
        for point in X:
            point_distances = []
            for centroid in self.centroids_:
                distance = np.linalg.norm(point - centroid)
                point_distances.append(distance)
            distances.append(point_distances)
        return np.argmin(distances, axis=1)

    def _recalculate_centroids(self, X, labels):
        """Recalculate centroids as the mean of points in each cluster."""
        new_centroids = []
        for k in range(self.k):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroid = np.mean(cluster_points, axis=0)
            else:
                new_centroid = self.centroids_[k]  # keep old centroid if cluster empty
            new_centroids.append(new_centroid)
        return np.array(new_centroids)

    def _calculate_sse(self, X, labels):
        """Calculate Sum of Squared Errors."""
        sse = 0
        for k in range(self.k):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroid = self.centroids_[k]
                sse += np.sum((cluster_points - centroid) ** 2)
        return sse


def calculate_elbow_data(X, max_k=10, init='random'):
    """Calculate SSE for different k values (Elbow Method)."""
    k_values = []
    sse_values = []
    for k in range(1, max_k + 1):
        if k > len(X):
            break
        kmeans = KMeans(k=k, init=init)
        kmeans.fit(X)
        k_values.append(k)
        sse_values.append(kmeans.inertia_)
    return k_values, sse_values