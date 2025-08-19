import numpy as np


class KMeans:
    """Simple K-Means clustering implementation (loop-based)."""

    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.iteration_history_ = []

    def fit(self, X):
        """Run K-Means clustering."""
        X = np.array(X, dtype=float)
        n_samples = len(X)
        #hello
        # STEP 1: Initialize random centroids
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids_ = X[random_indices].copy()

        # STEP 2: Repeat until centroids don't change or max_iters reached
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            labels = self._assign_clusters(X)

            # Calculate new centroids
            new_centroids = self._recalculate_centroids(X, labels)

            # Save history for debugging/visualization
            self.iteration_history_.append({
                'centroids': new_centroids.copy(),
                'labels': labels.copy()
            })

            # Stop if centroids have not changed
            if np.allclose(self.centroids_, new_centroids, rtol=1e-4):
                break

            self.centroids_ = new_centroids

        # Save results
        self.labels_ = labels
        self.inertia_ = self._calculate_sse(X, labels)

        return self

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
                new_centroid = self.centroids_[k]
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


def calculate_elbow_data(X, max_k=10):
    """Calculate SSE for different k values (Elbow Method)."""
    k_values = []
    sse_values = []
    for k in range(1, max_k + 1):
        if k > len(X):
            break
        kmeans = KMeans(k=k)
        kmeans.fit(X)
        k_values.append(k)
        sse_values.append(kmeans.inertia_)
    return k_values, sse_values


# if __name__ == "__main__":
#     data = np.array([
#         [1, 2], [2, 1],
#         [8, 9], [9, 8]
#     ])

#     kmeans = KMeans(k=2)
#     kmeans.fit(data)
#     print("Centroids:", kmeans.centroids_)
#     print("Labels:", kmeans.labels_)
#     print("SSE:", kmeans.inertia_)
