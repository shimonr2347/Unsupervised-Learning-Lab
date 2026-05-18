## Unsupervised Learning: K-Means Clustering & PCA
This project implements two core unsupervised learning algorithms from scratch using NumPy: K-Means for data clustering and Principal Component Analysis (PCA) for dimensionality reduction and image reconstruction.

## 🛠 Part 1: K-Means Clustering
The K-Means algorithm partitions a dataset into $K$ distinct, non-overlapping clusters. It works by minimizing the inertia (sum of squared distances between data points and their respective cluster centroid).

### Key Observations
**Cost Convergence**: As iterations increase, the cost function decreases monotonically untilcentroids stabilize.

**Cluster Tuning**: We tested $K \in \{2, 3, 4, 6, 8, 10, 20\}$. While higher $K$ reduces cost, the "Elbow Method" suggests $K=4$ is optimal for our generated dataset.

## 📈 Part 2: Principal Component Analysis (PCA)
PCA is used to project high-dimensional data (MNIST images, $28 \times 28 = 784$ pixels) into a lower-dimensional space while preserving as much variance as possible.

**Mathematical Analysis: $V^TV$ vs $VV^T$** 
In our implementation, $V$ is the matrix of the top $r$ eigenvectors.
* **$V^TV$ (Identity Matrix)**: Since eigenvectors are orthonormal, $V^TV$ results in an $r \times r$ Identity matrix. This represents the projection into the subspace.

* **$VV^T$ (Projection Matrix)**: This is a $d \times d$ matrix (e.g., $784 \times 784$). It is a projection operator that maps the original data onto the principal subspace. It is not an identity matrix, which explains why reconstruction is lossy.

**Image Reconstruction**
We demonstrated image reconstruction from 3, 10, and 100 dimensions:
* 3 Dimensions: Captures the general "blob" of the digit; unrecognizable.
* 10 Dimensions: The digit becomes legible but blurry.
* 100 Dimensions: High-fidelity reconstruction; nearly identical to the original.


## 📂 Project Structure
```
Unsupervised-Learning-Lab/
├── docs/                 # Generated plots
│   ├── cost_over_iterations.png
│   ├── final_clustering.png
│   ├── matrices_plotting.png
│   ├── MNIST_projection.png
│   └── pca_reconstruction.png
├── kmeans_pca_lab.py     # Main execution script
├── unsupervised.py       # Algorithm implementations (KMeans & PCA classes)
└── README.md             # Project documentation & analysis
```


## 💻 Setup & Usage
### Clone the repository:

```bash
git clone https://github.com/shimonr2347/
```
### Install dependencies: 

``` bash
pip install numpy matplotlib sklearn
```

### Run the lab:

``` bash
python kmeans_pca_lab.py
```

## 📈 Results
