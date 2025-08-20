import numpy as np
import plotly.graph_objs as go

class KMeans:
    """Simple K-Means clustering implementation (loop-based, with animation history)."""

    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.iteration_history_ = []  # stores snapshots of clustering progress

    def fit(self, X):
        """Run K-Means clustering and record iteration history for animation."""
        X = np.array(X, dtype=float)
        n_samples = len(X)

        # STEP 1: Initialize random centroids
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


def create_animation(kmeans, X):
    frames = []
    for i, step in enumerate(kmeans.iteration_history_):
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=X[:,0], y=X[:,1],
                    mode="markers",
                    marker=dict(color=step["labels"], colorscale="Viridis"),
                    name="Points"
                ),
                go.Scatter(
                    x=step["centroids"][:,0],
                    y=step["centroids"][:,1],
                    mode="markers",
                    marker=dict(color="red", size=15, symbol="x"),
                    name="Centroids"
                )
            ],
            name=str(i)
        )
        frames.append(frame)

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, {"frame": {"duration": 1000, "redraw": True}}]
                )]
            )]
        ),
        frames=frames
    )
    return fig
