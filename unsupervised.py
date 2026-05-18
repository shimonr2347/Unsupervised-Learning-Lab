import numpy as np

class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.X_fit_ = None
        self.labels_ = None
        self.centroids = None
        self.labels_history = []
        self.centroids_history = []
        self.costs = []

    def fit(self, X):
        self.X_fit_ = X
        np.random.seed(42)
        random_idx = np.random.permutation(X.shape[0])
        self.centroids = X[random_idx[:self.n_clusters]]
        
        for i in range(self.max_iter):
            self.labels_ = self._get_labels(X)
            old_centroids = self.centroids.copy()
            self.centroids = self._get_centroids(X, self.labels_)
            
            self.costs.append(self._calculate_cost(X))
            self.labels_history.append(self.labels_)
            self.centroids_history.append(self.centroids)
            
            if np.all(self.centroids == old_centroids):
                break

    def predict(self, X):
        return self._get_labels(X)

    def _get_distances(self, X):
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

    def _get_labels(self, X):
        distances = self._get_distances(X)
        return np.argmin(distances, axis=1)

    def _get_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.any(labels == k):
                centroids[k] = X[labels == k].mean(axis=0)
        return centroids

    def _calculate_cost(self, X):
        distances = self._get_distances(X)
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances ** 2)

class PCA:
    def __init__(self):
        self.V = None
        self.mean = None

    def fit(self, X):
        self.mean = X.mean(axis=0)
        X_centered = X - self.mean
        # Using eigh for symmetric covariance matrix is more stable
        covariance_matrix = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        self.V = eigenvectors[:, idx]

    def transform(self, X, n_dimensions):
        X_centered = X - self.mean
        return np.dot(X_centered, self.V[:, :n_dimensions])
    
    def inverse_transform(self, X_transformed, n_dimensions):
        return np.dot(X_transformed, self.V[:, :n_dimensions].T) + self.mean