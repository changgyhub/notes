# 17 Principal Component Analysis

## 1. Outline

1. Dimensionality Reduction
   1. High-dimensional data
   2. Learning \(low dimensional\) representations
2. Principal Component Analysis \(PCA\)
   1. Examples: 2D and 3D
   2. Data for PCA
   3. PCA Definition
   4. Objective functions for PCA
   5. PCA, Eigenvectors, and Eigenvalues
   6. Algorithms for finding Eigenvectors / Eigenvalues
3. PCA Examples
   1. Face Recognition
   2. Image Compression

## 2. Principal Component Analysis

### 2.1 Learning Representations

PCA, Kernel PCA, ICA: Powerful unsupervised learning techniques for extracting hidden \(potentially lower dimensional\) structure from high dimensional datasets.

Useful for:

1. Visualization
2. More efficient use of resources \(e.g., time, memory, communication\)
3. Statistical: fewer dimensions $$\rightarrow$$ a better generalization
4. Noise removal \(improving data quality\)
5. Further processing by machine learning algorithms

### 2.2 Definition of PCA

In case where data lies on or near a low d-dimensional linear subspace, axes of this subspace are an effective representation of the data.

Identifying the axes is known as Principal Components Analysis, and can be obtained by using classic matrix computation tools \(Eigen or Singular Value Decomposition\).

### 2.3 Data for PCA

![](../../.gitbook/assets/image%20%28373%29.png)

### 2.4 Sample Covariance Matrix

![](../../.gitbook/assets/image%20%2854%29.png)

### 2.5 Vector Projection

Length of project $$\bm{x}$$to $$\bm{v}$$ is $$ a = \frac{\bm{v}^T\bm{x}}{||\bm{v}||_2}$$. If $$||\bm{v}|| = 1$$, then $$a = \bm{v}^T\bm{x}$$

Then we have the projection as $$\bm{u} = \frac{\bm{v}^T\bm{x}}{||\bm{v}||_2} \bm{v}$$

### 2.6 Equivalence of Maximizing Variance and Minimizing Reconstruction Error

![](../../.gitbook/assets/image%20%28403%29.png)

Example: Option B is the correct reconstruction. To view this, we project points onto the line and check their variance.

![](../../.gitbook/assets/image%20%28838%29.png)

### 2.7 Lemma for PCA

The vector that maximizes the variance is the eigenvector of sample covariance matrix $$\Sigma = \frac{1}{N}X^TX$$ with largest eigenvalue.

![](../../.gitbook/assets/image%20%28741%29.png)

Recall: $$\bm{v}$$ is an eigenvector of $$\Sigma$$ if $$\Sigma\bm{v} = \lambda\bm{v}$$ for same eigenvalue $$\lambda \in \mathbb{R}$$. The eigenvectors of a symmetric matrix \(covariance matrix\) are orthogonal to each other.

### 2.8 PCA Algorithm

Process: Repeatedly chooses a next vector $$\bm{v}_j$$ that minimizes the reconstruction error s.t. $$\bm{v}_1, \bm{v}_2, \cdots, \bm{v}_{j-1}$$ are orthogonal to $$\bm{v}_j$$.

Algorithm:

1. With projection matrix $$V \in \mathbb{R}^{K \times M}$$ as the first $$K$$ eigenvectors of $$\Sigma$$
2. Project down: $$\bm{v}^{(i)} = V \bm{x}^{(i)}$$
3. Reconstruct up: $$\hat{\bm{x}}^{(i)} = V^T \bm{v}^{(i)}$$

Ways to compute $$V$$:

1. Power Iteration
2. SVD: first $$K$$ columns of $$V$$ in $$X = USV^T$$
3. Approximate \(Random\) Methods

### 2.9 How Many PCs?

![](../../.gitbook/assets/image%20%28171%29.png)

## 3. PCA Examples

### 3.1 Projecting MNIST digits

![](../../.gitbook/assets/image%20%28656%29.png)

### 3.2 Applying PCA: Eigenfaces

![](../../.gitbook/assets/image%20%28267%29.png)

PCA make training faster if no glasses worn and in same lighting conditions.

Shortcomings:

1. Requires carefully controlled data
   1. All faces centered in frame
   2. Same size
   3. Some sensitivity to angle
2. Alternative
   1. “Learn” one set of PCA vectors for each angle
   2. Use the one with lowest error
3. Method is completely knowledge free \(sometimes this is good!\)
   1. Doesn’t know that faces are wrapped around 3D objects \(heads\)
   2. Makes no effort to preserve class distinctions

### 3.3 Image Compression

PCA can be applied for image compression.

## 4. PCA Learning Objectives

1. Define the sample mean, sample variance, and sample covariance of a vector-valued dataset
2. Identify examples of high dimensional data and common use cases for dimensionality reduction
3. Draw the principal components of a given toy dataset
4. Establish the equivalence of minimization of reconstruction error with maximization of variance
5. Given a set of principal components, project from high to low dimensional space and do the reverse to produce a reconstruction
6. Explain the connection between PCA, eigenvectors, eigenvalues, and covariance matrix
7. Use common methods in linear algebra to obtain the principal components



