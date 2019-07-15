# 16 Kernels & K-Means

## 1. Kernels

### 1.1 Real-world Data

Most real-world problems exhibit data that is not linearly separable.

Question: When your data is not linearly separable, how can you still use a linear classifier?

Answer: Preprocess the data to produce nonlinear features

### 1.2 Kernels: Motivation

Motivation \#1: Inefficient Features

* Non-linearly separable data requires **high dimensional representation**
* Might be prohibitively expensive to compute or store

Motivation \#2: Memory-based Methods

* k-Nearest Neighbors \(KNN\) for facial recognition allows a **distance metric** between images
* no need to worry about linearity restriction at all

### 1.3 Kernel Methods

![](../../.gitbook/assets/image%20%2816%29.png)

### 1.4 SVM: Kernel Trick

![](../../.gitbook/assets/image%20%28818%29.png)

![](../../.gitbook/assets/image%20%28602%29.png)

### 1.5 Example: Polynomial Kernel

![](../../.gitbook/assets/image%20%28516%29.png)

![](../../.gitbook/assets/image%20%28460%29.png)

### **1.6** Kernel Examples

![](../../.gitbook/assets/image%20%28469%29.png)

### 1.7 SVM + Kernels: Takeaways

1. Maximizing the margin of a linear separator is a good training criteria
2. Support Vector Machines \(SVMs\) learn a max-margin linear classifier
3. The SVM optimization problem can be solved with black-box Quadratic Programming \(QP\) solvers
4. Learned decision boundary is defined by its support vectors
5. Kernel methods allow us to work in a transformed feature space without explicitly representing that space
6. The kernel-trick can be applied to SVMs, as well as many other algorithms

## 2. Kernels Learning Objectives

1. Employ the kernel trick in common learning algorithms
2. Explain why the use of a kernel produces only an implicit representation of the transformed feature space
3. Use the "kernel trick" to obtain a computational complexity advantage over explicit feature transformation
4. Sketch the decision boundaries of a linear classifier with an RBF kernel

## 3. K-Means

### 3.1 Outline

1. Clustering: Motivation / Applications
2. Optimization Background
   1. Coordinate Descent
   2. Block Coordinate Descent
3. Clustering
   1. Inputs and Outputs
   2. Objective-based Clustering
4. K-Means
   1. K-Means Objective
   2. Computational Complexity
   3. K-Means Algorithm / Lloyd’s Method
5. K-Means Initialization
   1. Random
   2. Farthest Point
   3. K-Means++

### 3.2 Clustering - Informal Goals

Goal: Automatically partition unlabeled data into groups of similar datapoints.

Question: When and why would we want to do this?

Useful for

1. Automatically organizing data
2. Understanding hidden structure in data
3. Preprocessing for further analysis: Representing high-dimensional data in a low-dimensional space \(e.g. for visualization purposes\)

### 3.3 Clustering Applications

1. Cluster news articles or web pages or search results by topic
2. Cluster protein sequences by function or genes according to expression profile
3. Cluster users of social networks by interest \(community detection\)
4. Cluster customers according to purchase history
5. Cluster galaxies or nearby stars \(e.g. Sloan Digital Sky Survey\)

### 3.4 Coordinate Descent

Coordinate Descent:

1. Goal: Minimize a $$\mathcal{J}(\boldsymbol{\Theta})$$, $$\hat{\boldsymbol{\Theta}} = \arg \min_{\boldsymbol{\Theta}} \mathcal{J}(\boldsymbol{\Theta})$$
2. Algorithm: Pick one dimension $$\Theta_i$$ of $$\boldsymbol{\Theta}$$, and minimize along that dimension

Block Coordinate Descent:

1. Goal: Minimize a $$\mathcal{J}(\boldsymbol{\alpha}, \boldsymbol{\beta})$$, $$\hat{\boldsymbol{\alpha}}, \hat{\boldsymbol{\beta}} = \arg \min_{\boldsymbol{\alpha}, \boldsymbol{\beta} } \mathcal{J}(\boldsymbol{\alpha}, \boldsymbol{\beta})$$
2. Idea: Minimize over entire group of parameters
3. Algorithm:
   1. Initialize $$\boldsymbol{\alpha}$$ and $$\boldsymbol{\beta}$$
   2. Repeat until stopping criterion satisfied
      1. $$\boldsymbol{\alpha} = \arg \min_{\boldsymbol{\alpha}} \mathcal{J}(\boldsymbol{\alpha}, \boldsymbol{\beta})$$
      2. $$\boldsymbol{\beta} = \arg \min_{\boldsymbol{\beta}} \mathcal{J}(\boldsymbol{\alpha}, \boldsymbol{\beta})$$

Note the recipe for objective:

1. write objective for "good" parameters
2. optimize the objective

### 3.5 K-Means

Input: Unlabeled data $$\mathcal{D} = \{\boldsymbol{x}^{(1)}, \boldsymbol{x}^{(2)}, \cdots, \boldsymbol{x}^{(N)}\}$$

Goal: Find an assignment of points to $$K$$ clusters $$\{\boldsymbol{z}^{(1)}, \boldsymbol{z}^{(2)}, \cdots, \boldsymbol{z}^{(N)}\}$$ where $$\boldsymbol{z}^{(i)} \in \{1, 2, \cdots, K\}$$

Decision Rule: Assign $$\boldsymbol{x}^{(i)}$$ to the nearest cluster center $$\boldsymbol{c}_j$$

Objective: $$\hat{\boldsymbol{c}}, \hat{\boldsymbol{z}} = \arg\min_{\boldsymbol{c}, \boldsymbol{z}} \sum_{i=1}^N || \boldsymbol{x}^{(i)} - \boldsymbol{c}_{\boldsymbol{z}^{(i)}}  ||_2^2$$

Algorithm \(Lloyd’s method\):

1. Initialize cluster centers $$\boldsymbol{c}_1, \boldsymbol{c}_2, \cdots, \boldsymbol{c}_K$$ and cluster assignments $$\boldsymbol{z}^{(1)}, \boldsymbol{z}^{(2)}, \cdots, \boldsymbol{z}^{(N)}$$
2. Repeat until objective stops changing
   1.  $$\boldsymbol{c} = \arg\min_{\boldsymbol{c}} \sum_{i=1}^N || \boldsymbol{x}^{(i)} - \boldsymbol{c}_{\boldsymbol{z}^{(i)}}  ||_2^2$$
      * $$\boldsymbol{c}_j = \arg\min_{\boldsymbol{c}_j} \sum_{i: \boldsymbol{z}^{(i)}=j} || \boldsymbol{x}^{(i)} - \boldsymbol{c}_j  ||_2^2 = \text{mean}(\sum_{i: \boldsymbol{z}^{(i)}=j}\boldsymbol{x}^{(i)})$$
   2.  $$\boldsymbol{z} = \arg\min_{\boldsymbol{z}} \sum_{i=1}^N || \boldsymbol{x}^{(i)} - \boldsymbol{c}_{\boldsymbol{z}^{(i)}}  ||_2^2$$
      * $$\boldsymbol{z}^{(j)} = \arg\min_{i} || \boldsymbol{x}^{(j)} - \boldsymbol{c}_i  ||_2^2 =$$ closest cluster center to $$\boldsymbol{x}^{(j)}$$

Local optimum: every point is assigned to its nearest center and every center is the mean value of its points.

It always converges, but it may converge at a local optimum that is different from the global optimum, and in fact could be arbitrarily worse in terms of its score.

### 3.6 K-Means Initialization

**Random**

There are possibly bad cases:

**Furthest Traversal**

1. Choose $$\boldsymbol{c}_1$$ arbitrarily \(or at random\)
2. For $$j = 2, 3, \cdots, K$$, pick $$\boldsymbol{c}_j$$ among data that is farthest from previously chosen $$\boldsymbol{c}$$s.

It fixes the Gaussian problem, but it can be thrown off by outliers.

**K-Means++**

![](../../.gitbook/assets/image%20%28834%29.png)

## 4. K-Means Learning Objectives

1. Distinguish between coordinate descent and block coordinate descent
2. Define an objective function that gives rise to a "good" clustering
3. Apply block coordinate descent to an objective function preferring each point to be close to its nearest objective function to obtain the K-Means algorithm
4. Implement the K-Means algorithm
5. Connect the nonconvexity of the K-Means objective function with the \(possibly\) poor performance of random initialization

