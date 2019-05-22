# 15 SVM

## 1. Constrained Optimization

### 1.1  Definition

$$
\min_\bm{\Theta} \mathcal{J}(\bm{\Theta})\\
\text{s.t. } \bm{g}(\bm{\Theta}) \le \bm{b}
$$

### 1.2 Linear Programming

$$
\min_\bm{x} \bm{c}^T\bm{x}\\
\text{s.t. } \bm{A}\bm{x} \le \bm{b}\\[3pt]
\text{where } \bm{c}\in\mathbb{R}^m, \bm{A}\in\mathbb{R}^{n \times m}, \bm{b}\in\mathbb{R}^n
$$

We can use interior point method or simplex algorithm to solve.

### 1.3 Quadratic Programming

$$
\min_\bm{x} \frac{1}{2} \bm{x}^T \bm{Q}\bm{x} + \bm{c}^T\bm{x}\\
\text{s.t. } \bm{A}\bm{x} \le \bm{b}\\[3pt]
\text{where } \bm{c}\in\mathbb{R}^m, \bm{A}\in\mathbb{R}^{n \times m}, \bm{b}\in\mathbb{R}^n,  \bm{Q} \in \mathbb{R}^{m \times m}
$$

We can use interior point method, ellipsoid method \(assume $$\bm{Q}$$ convex, in polynomial time\), conjugate gradient method.

![](../../.gitbook/assets/image%20%28653%29.png)

## 2. Support Vector Machine \(SVM\)

### 2.1 Linearly Separable Case

#### 2.1.1 Margin

![](../../.gitbook/assets/image%20%28100%29.png)

#### 2.1.2 Lagrangian

![](../../.gitbook/assets/image%20%28589%29.png)

![](../../.gitbook/assets/image%20%28210%29.png)

#### 2.1.3 Primal vs Dual

![](../../.gitbook/assets/image%20%28434%29.png)



### 

