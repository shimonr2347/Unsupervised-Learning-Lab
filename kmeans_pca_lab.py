import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from unsupervised import KMeans, PCA

# ==========================================
# PART 1: K-MEANS CLUSTERING
# ==========================================
print("Starting Part 1: K-Means...")

# 1. Generate 2D data
X_kmeans = np.concatenate([
    np.random.normal([0, 0], size=(500, 2)),
    np.random.normal([5, 5], size=(500, 2)),
    np.random.normal([5, 0], size=(500, 2)),
    np.random.normal([0, 5], size=(500, 2)),
])
np.random.shuffle(X_kmeans)

# 2. Train K-Means
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X_kmeans)

# 3. Plot Cost Function
plt.figure(figsize=(8, 4))
plt.plot(kmeans.costs, marker='o')
plt.title("K-Means Cost over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Sum of Squared Distances")
plt.grid(True)
plt.show()

# 4. Final Clustering Visualization
def plot_clusters(X, labels, centroids, title):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']
    plt.figure(figsize=(8, 6))
    for i in range(len(centroids)):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], 
                    color=colors[i % len(colors)], label=f'Cluster {i}')
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=100, label='Centroids')
    plt.title(title)
    plt.legend()
    plt.show()

plot_clusters(X_kmeans, kmeans.labels_, kmeans.centroids, f'Final Clustering (K={n_clusters})')


# ==========================================
# PART 2: PCA (MNIST)
# ==========================================
print("\nStarting Part 2: PCA...")

# 1. Load Data
print("Loading MNIST (this may take a minute)...")
X_mnist, y_mnist = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
y_mnist = y_mnist.astype(int)

# 2. Train PCA
pca = PCA()
pca.fit(X_mnist)

# 3. Visualize 2D Projection
X_reduced = pca.transform(X_mnist, 2)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_mnist, cmap='viridis')
plt.colorbar(scatter, label='Digit Label')
plt.title("MNIST Projection onto Top 2 Principal Components")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.show()

# 4. VTV vs VVT Comparison (Orthogonality check)
n_dims = 40
V = pca.V[:, :n_dims]
VTV = np.dot(V.T, V)
VVT = np.dot(V, V.T)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
im1 = ax1.imshow(VTV, cmap='viridis')
ax1.set_title('$V^T V$ (Subspace Basis)')
fig.colorbar(im1, ax=ax1)

im2 = ax2.imshow(VVT, cmap='viridis')
ax2.set_title('$V V^T$ (Projection Matrix)')
fig.colorbar(im2, ax=ax2)
plt.show()

# 5. Image Reconstruction
sample_idx = np.random.choice(X_mnist.shape[0])
sample = X_mnist[sample_idx].reshape(1, -1)

dims_to_test = [3, 10, 100]
fig, axes = plt.subplots(1, len(dims_to_test) + 1, figsize=(15, 5))

# Original
axes[0].imshow(sample.reshape(28, 28), cmap='gray')
axes[0].set_title("Original")
axes[0].axis('off')

# Reconstructed versions
for i, d in enumerate(dims_to_test):
    reconstructed = pca.inverse_transform(pca.transform(sample, d), d)
    axes[i+1].imshow(reconstructed.reshape(28, 28), cmap='gray')
    axes[i+1].set_title(f"Dim: {d}")
    axes[i+1].axis('off')

plt.suptitle(f"PCA Reconstruction Comparison (Digit: {y_mnist[sample_idx]})")
plt.show()

print("Lab Execution Complete.")