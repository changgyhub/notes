# 04 Linear Regression & Optimization for ML

## 1. Regression Problem

### 1.1 Definitions

$$D = \{(\bm{x}^{(1)}, y^{(1)}), \cdots, (\bm{x}^{(N)}, y^{(N)})\}$$ where $$\bm{x}^{(i)} \in \mathbb{R}^M$$ \(input, features\) and $$y^{(i)} \in \mathbb{R}$$ \(output, values\) 

For linear regression, $$ y = \bm{w}^T \bm{x} + b$$

Residual $$e_i = y^{(i)} - (\bm{w}^T \bm{x}^{(i)} + b)$$ is the distance \(vertically\) from observed value to predicted value

### 1.2 Notation Trick

Define $$x_0 = 1$$, $$ y = \bm{\theta}^T \bm{x}$$ where $$\bm{\theta} = [b, w_1, \cdots, w_M]$$and $$\bm{x} = [1, x_1, \cdots, x_M]$$

### 1.3 Linear Regression as Function Approximation

\(1\) Assume D is generated as $$\bm{x}^{(i)} \sim p^* (\bm{x}), ~y^{(i)} \sim c^* (\bm{x}^{(i)})$$, where $$p^*$$ and $$c^*$$ are unknown

\(2\) Choose hypothesis space $$\mathcal{H} = \{  h_{\bm{\theta}} |  h_{\bm{\theta}}(\bm{x}) =  \bm{\theta}^T \bm{x}, ~ \bm{\theta} \in \mathbb{R}^{M+1}, x_0 = 1\}$$

\(3\) Choose a objective function with a goal of minimizing the mean squared error $$\mathcal{J}_D(\bm{\theta}) = \frac{1}{N} \sum_{i=1}^N e_i^2 = \frac{1}{N} \sum_{i=1}^N ( y^{(i)} - (\bm{w}^T \bm{x}^{(i)} + b))^2$$ 

\(4\) Solve optimization problem $$\hat{\bm{\theta}} = \text{argmin}_{\bm{\theta}} \mathcal{J}_D(\bm{\theta})$$

\(5\) Predict, given next $$\bm{x}$$, $$y = h_{\hat{\bm{\theta}}}(\bm{x}) = \hat{\bm{\theta}}^T \bm{x}$$

## 2. Optimization for ML

### 2.1 Not Quite the Same Setting As Other Fields

1. Function we are optimizing might not be the true goal \(e.g. likelihood vs generalization error\)
2. Precision might not matter \(e.g. data is noisy, so optimal up to 1e-16 might not help\)
3. Stopping early can help generalization error \(i.e. “early stopping” is a technique for regularization – discussed more next time\)

### 2.2 Definitions

Derivatives: Many ways of understanding:

1. Derive as tangent $$\frac{d \mathcal{J}(\bm{\theta})}{d\bm{\theta}}$$
2. Derive as s a limit  $$\lim_{e \rightarrow 0} \frac{\mathcal{J(\bm{\theta} + e) - J(\bm{\theta})}}{e}$$
3. Derive as tangent plane

The gradient of $$\mathcal{J}(\bm{\theta})$$ is $$\nabla \mathcal{J}(\bm{\theta}) = [\frac{d \mathcal{J}(\bm{\theta})}{d\bm{\theta}_1}, \cdots, \frac{d \mathcal{J}(\bm{\theta})}{d\bm{\theta}_M}]^T$$

Zero derivative: maxima, minima, or saddle point

Convexity:

![](../../.gitbook/assets/image%20%2864%29.png)

### 2.3 Closed Form Optimization

For a M-dim function $$\mathcal{J}(\bm{\theta})$$

1. Solve $$\nabla \mathcal{J}(\bm{\theta}) = \bm{0}$$ for $$\hat{\bm{\theta}}$$
2. Test for min, max, or saddle point using Hessian matrix

### 2.4 Closed-form Solution to Linear Regression

Define the "design matrix" $$\bm{X}$$, where $$i$$-th row is $$[1, x_1^{(i)}, \cdots, x_M^{(i)}]$$

Then $$\mathcal{J}_D(\bm{\theta})   = \sum_{i=1}^N ( y^{(i)} - (\bm{w}^T \bm{x}^{(i)} + b))^2 =  (\bm{X} \bm{\theta} - y)^T(\bm{X} \bm{\theta} - y)$$,

$$\nabla \mathcal{J}_D(\bm{\theta}) = \bm{X}^T\bm{X} \bm{\theta} - \bm{X}^T y = \bm{0} \Longrightarrow \bm{\theta} = (\bm{X}^T\bm{X})^{-1} \bm{X}^T y$$

Hence , $$\hat{\bm{\theta}} = \text{argmin}_{\bm{\theta}} \mathcal{J}_D(\bm{\theta}) = (\bm{X}^T\bm{X})^{-1} \bm{X}^T y $$, which is called the Normal Equation

### 2.5 Regression Examples

1. Stock price prediction
2. Forecasting epidemics
3. Speech synthesis
4. Generation of images \(e.g. Deep Dream\)
5. Predicting the number of tourists on Machu Picchu on a given day

## 3. Root Finding and Gradient Descent

### 3.1 Optimal Method No.0: Random Guessing

1. Pick a random $$\bm{\theta}$$
2. Evaluate $$\mathcal{J}(\bm{\theta})$$
3. Repeat
4. Return best $$\hat{\bm{\theta}}$$

### 3.2 Motivation for Gradient Descent

![](../../.gitbook/assets/image%20%28600%29.png)

Cases to consider gradient descent

1. What if we can not find a closed-form solution?
2. What if we can, but it’s inefficient to compute?
3. What if we can, but it’s numerically unstable to compute?

### 3.3 Pros and Cons of Gradient Descent

* Simple and often quite effective on ML tasks
* Often very scalable
* Only applies to smooth functions \(differentiable\)
* Might find a local minimum, rather than a global one

### 3.4 Gradient Descent

![](../../.gitbook/assets/image%20%28174%29.png)

### 3.5 Details of Gradient Descent

Starting Point: Either

1. $$\bm{\theta} = 0$$
2. $$\bm{\theta}$$ randomly

Convergence: There are many possible ways to detect convergence. For example

1. We could check whether the L2 norm of the gradient is below some small tolerance $$||\nabla_{\bm{\theta}} \mathcal{J}(\bm{\theta})||_2 < \epsilon$$
2. Alternatively we could check that the reduction in the objective function from one iteration to the next is small

### 3.6 Gradient Descent for Linear Regression

$$ \mathcal{J}(\bm{\theta}) = \frac{1}{N} \sum_{i=1}^n \frac{1}{2} (y^{(i)} - \bm{\theta}^T \bm{x}^{(i)})^2$$

$$ \frac{\partial \mathcal{J}_i(\bm{\theta})}{\partial \bm{\theta}_j} = - (y^{(i)} - \bm{\theta}^T \bm{x}^{(i)}) \bm{x}_m^{(i)}$$

$$\nabla \mathcal{J}_i(\bm{\theta}) = - (y^{(i)} - \bm{\theta}^T \bm{x}^{(i)}) \bm{x}^{(i)}$$

$$\nabla \mathcal{J}(\bm{\theta}) = - \sum_{i=1}^n (y^{(i)} - \bm{\theta}^T \bm{x}^{(i)}) \bm{x}^{(i)}$$

### 3.7 Stochastic Gradient Descent \(SGD\)

![](../../.gitbook/assets/image%20%28430%29.png)

![](../../.gitbook/assets/image%20%28209%29.png)

### 3.8 Expectations of Gradients

![](../../.gitbook/assets/image%20%28223%29.png)

### 3.9 Convergence of Optimizers

![](../../.gitbook/assets/image%20%28673%29.png)

## 4. Linear Regression Objectives

* Design k-NN Regression and Decision Tree Regression
* Implement learning for Linear Regression using three optimization techniques: \(1\) closed form, \(2\) gradient descent, \(3\) stochastic gradient descent
* Choose a Linear Regression optimization technique that is appropriate for a particular dataset by analyzing the tradeoff of computational complexity vs. convergence speed
* Distinguish the three sources of error identified by the bias-variance decomposition: bias, variance, and irreducible error.

## 5. Optimization Objectives

* Apply gradient descent to optimize a function
* Apply stochastic gradient descent \(SGD\) to optimize a function
* Apply knowledge of zero derivatives to identify a closed-form solution \(if one exists\) to an optimization problem
* Distinguish between convex, concave, and nonconvex functions
* Obtain the gradient \(and Hessian\) of a \(twice\) differentiable function

