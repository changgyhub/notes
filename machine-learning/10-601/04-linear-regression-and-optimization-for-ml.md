# 04 Linear Regression & Optimization for ML

## 1. Regression Problem

### 1.1 Definitions

$$D = \{(\boldsymbol{x}^{(1)}, y^{(1)}), \cdots, (\boldsymbol{x}^{(N)}, y^{(N)})\}$$ where $$\boldsymbol{x}^{(i)} \in \mathbb{R}^M$$ \(input, features\) and $$y^{(i)} \in \mathbb{R}$$ \(output, values\) 

For linear regression, $$ y = \boldsymbol{w}^T \boldsymbol{x} + b$$

Residual $$e_i = y^{(i)} - (\boldsymbol{w}^T \boldsymbol{x}^{(i)} + b)$$ is the distance \(vertically\) from observed value to predicted value

### 1.2 Notation Trick

Define $$x_0 = 1$$, $$ y = \boldsymbol{\theta}^T \boldsymbol{x}$$ where $$\boldsymbol{\theta} = [b, w_1, \cdots, w_M]$$and $$\boldsymbol{x} = [1, x_1, \cdots, x_M]$$

### 1.3 Linear Regression as Function Approximation

\(1\) Assume D is generated as $$\boldsymbol{x}^{(i)} \sim p^* (\boldsymbol{x}), ~y^{(i)} \sim c^* (\boldsymbol{x}^{(i)})$$, where $$p^*$$ and $$c^*$$ are unknown

\(2\) Choose hypothesis space $$\mathcal{H} = \{  h_{\boldsymbol{\theta}} |  h_{\boldsymbol{\theta}}(\boldsymbol{x}) =  \boldsymbol{\theta}^T \boldsymbol{x}, ~ \boldsymbol{\theta} \in \mathbb{R}^{M+1}, x_0 = 1\}$$

\(3\) Choose a objective function with a goal of minimizing the mean squared error $$\mathcal{J}_D(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^N e_i^2 = \frac{1}{N} \sum_{i=1}^N ( y^{(i)} - (\boldsymbol{w}^T \boldsymbol{x}^{(i)} + b))^2$$ 

\(4\) Solve optimization problem $$\hat{\boldsymbol{\theta}} = \text{argmin}_{\boldsymbol{\theta}} \mathcal{J}_D(\boldsymbol{\theta})$$

\(5\) Predict, given next $$\boldsymbol{x}$$, $$y = h_{\hat{\boldsymbol{\theta}}}(\boldsymbol{x}) = \hat{\boldsymbol{\theta}}^T \boldsymbol{x}$$

## 2. Optimization for ML

### 2.1 Not Quite the Same Setting As Other Fields

1. Function we are optimizing might not be the true goal \(e.g. likelihood vs generalization error\)
2. Precision might not matter \(e.g. data is noisy, so optimal up to 1e-16 might not help\)
3. Stopping early can help generalization error \(i.e. “early stopping” is a technique for regularization – discussed more next time\)

### 2.2 Definitions

Derivatives: Many ways of understanding:

1. Derive as tangent $$\frac{d \mathcal{J}(\boldsymbol{\theta})}{d\boldsymbol{\theta}}$$
2. Derive as s a limit  $$\lim_{e \rightarrow 0} \frac{\mathcal{J(\boldsymbol{\theta} + e) - J(\boldsymbol{\theta})}}{e}$$
3. Derive as tangent plane

The gradient of $$\mathcal{J}(\boldsymbol{\theta})$$ is $$\nabla \mathcal{J}(\boldsymbol{\theta}) = [\frac{d \mathcal{J}(\boldsymbol{\theta})}{d\boldsymbol{\theta}_1}, \cdots, \frac{d \mathcal{J}(\boldsymbol{\theta})}{d\boldsymbol{\theta}_M}]^T$$

Zero derivative: maxima, minima, or saddle point

Convexity:

![](../../.gitbook/assets/image%20%2864%29.png)

### 2.3 Closed Form Optimization

For a M-dim function $$\mathcal{J}(\boldsymbol{\theta})$$

1. Solve $$\nabla \mathcal{J}(\boldsymbol{\theta}) = \boldsymbol{0}$$ for $$\hat{\boldsymbol{\theta}}$$
2. Test for min, max, or saddle point using Hessian matrix

### 2.4 Closed-form Solution to Linear Regression

Define the "design matrix" $$\boldsymbol{X}$$, where $$i$$-th row is $$[1, x_1^{(i)}, \cdots, x_M^{(i)}]$$

Then $$\mathcal{J}_D(\boldsymbol{\theta})   = \sum_{i=1}^N ( y^{(i)} - (\boldsymbol{w}^T \boldsymbol{x}^{(i)} + b))^2 =  (\boldsymbol{X} \boldsymbol{\theta} - y)^T(\boldsymbol{X} \boldsymbol{\theta} - y)$$,

$$\nabla \mathcal{J}_D(\boldsymbol{\theta}) = \boldsymbol{X}^T\boldsymbol{X} \boldsymbol{\theta} - \boldsymbol{X}^T y = \boldsymbol{0} \Longrightarrow \boldsymbol{\theta} = (\boldsymbol{X}^T\boldsymbol{X})^{-1} \boldsymbol{X}^T y$$

Hence , $$\hat{\boldsymbol{\theta}} = \text{argmin}_{\boldsymbol{\theta}} \mathcal{J}_D(\boldsymbol{\theta}) = (\boldsymbol{X}^T\boldsymbol{X})^{-1} \boldsymbol{X}^T y $$, which is called the Normal Equation

### 2.5 Regression Examples

1. Stock price prediction
2. Forecasting epidemics
3. Speech synthesis
4. Generation of images \(e.g. Deep Dream\)
5. Predicting the number of tourists on Machu Picchu on a given day

## 3. Root Finding and Gradient Descent

### 3.1 Optimal Method No.0: Random Guessing

1. Pick a random $$\boldsymbol{\theta}$$
2. Evaluate $$\mathcal{J}(\boldsymbol{\theta})$$
3. Repeat
4. Return best $$\hat{\boldsymbol{\theta}}$$

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

1. $$\boldsymbol{\theta} = 0$$
2. $$\boldsymbol{\theta}$$ randomly

Convergence: There are many possible ways to detect convergence. For example

1. We could check whether the L2 norm of the gradient is below some small tolerance $$||\nabla_{\boldsymbol{\theta}} \mathcal{J}(\boldsymbol{\theta})||_2 < \epsilon$$
2. Alternatively we could check that the reduction in the objective function from one iteration to the next is small

### 3.6 Gradient Descent for Linear Regression

$$ \mathcal{J}(\boldsymbol{\theta}) = \frac{1}{N} \sum_{i=1}^n \frac{1}{2} (y^{(i)} - \boldsymbol{\theta}^T \boldsymbol{x}^{(i)})^2$$

$$ \frac{\partial \mathcal{J}_i(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}_j} = - (y^{(i)} - \boldsymbol{\theta}^T \boldsymbol{x}^{(i)}) \boldsymbol{x}_m^{(i)}$$

$$\nabla \mathcal{J}_i(\boldsymbol{\theta}) = - (y^{(i)} - \boldsymbol{\theta}^T \boldsymbol{x}^{(i)}) \boldsymbol{x}^{(i)}$$

$$\nabla \mathcal{J}(\boldsymbol{\theta}) = - \sum_{i=1}^n (y^{(i)} - \boldsymbol{\theta}^T \boldsymbol{x}^{(i)}) \boldsymbol{x}^{(i)}$$

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

