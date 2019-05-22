# 11 Probabilistic Learning

## 1. Classification and Regression

### 1.1 ML Big Picture

![](../../.gitbook/assets/image%20%2823%29.png)

### 1.2 ML Recipe

1. Decision Rules/Functions/Models \(probabilistic generative, probabilistic discriminative, perceptron, SVM, regression\)
2. Objective Functions \(likelihood, conditional likelihood, hinge loss, mean squared error\)
3. Regularization \(L1, L2, priors for MAP
4. Update Rules \(SGD, perceptron\)
5. Nonlinear Features \(preprocessing, kernel trick\)

### 1.3 Decision Functions

Perceptron: $$h_{ \bm{\theta}} (\bm{x}) = \text{sign}(\bm{\theta}^T \bm{x})$$

Linear Regression: $$h_{ \bm{\theta}} (\bm{x}) = \bm{\theta}^T \bm{x}$$

Neural Network \(classical\):  $$h_{ \bm{\theta}} (\bm{x}) = \sigma (\bm{w}_2^T \sigma (\bm{w}_1^T \bm{x} + b_1) + b_2)$$

Discriminative Models: $$h_{ \bm{\theta}} (\bm{x}) = \text{argmax}_y~p_{ \bm{\theta}}(y|\bm{x})$$

* Binary Logistic Regression: $$p_{ \bm{\theta}}(y = 1|\bm{x}) = \text{sigmoid}(\bm{\theta}^T \bm{x})$$
* Multinomial Logistic Regression: $$p_{ \bm{\theta}}(y = 1|\bm{x}) \propto \exp(\bm{\theta}^T \bm{x})$$

Generative Models: $$h_{\bm{\theta}} (\bm{x}) = \text{argmax}_y~p_{ \bm{\theta}}(y, \bm{x})$$

* Naïve Bayes: $$p(y, \bm{x}) = p(y) \prod_{x=i}^M p(x_i | y)$$

### 1.4 Objective Functions

MSE: $$\mathcal{J}(\bm\theta) = \frac{1}{2} \sum_{i=1}^N (y^{(i)} - h_{ \bm{\theta}} (\bm{x}^{(i)}))^2$$

MCLE: $$\mathcal{J}(\bm\theta) = - \log \prod_{i=1}^N p_{ \bm{\theta}}(y^{(i)}|\bm{x}^{(i)})$$

MLE: $$\mathcal{J}(\bm\theta) = - \log \prod_{i=1}^N p_{ \bm{\theta}}(y^{(i)}, \bm{x}^{(i)})$$

L2 Regularization: $$\mathcal{J}'(\bm\theta) = \mathcal{J}(\bm\theta) + \lambda ||\bm\theta||_2^2$$

L1 Regularization: $$\mathcal{J}'(\bm\theta) = \mathcal{J}(\bm\theta) + \lambda ||\bm\theta||_1$$

### 1.5 Optimization Method

Gradient Descent: while not converge, $$\bm\theta = \bm\theta - \eta \nabla \mathcal{J}(\bm\theta)$$

Stochastic Gradient Descent

Closed Form

## 2. Probabilistic Learning

### 2.1 Function \(Deterministic\) Learning vs Probabilistic Learning

![](../../.gitbook/assets/image%20%2811%29.png)

### 2.2 Example: Robotic Farming

![](../../.gitbook/assets/image%20%2888%29.png)

### 2.3 Categorical Distribution

$$x \sim \text{Categorical}(\Theta)$$where $$\sum_{k=1}^K \theta_k = 1, \theta_K \in [0, 1]$$.

For categorical distribution

* pmf: $$p(a) = P(x = a) = \theta_a$$
* $$K = |\Theta|$$
* $$x \in \mathcal{X} = {1,2, \cdots, K}$$

Example: Uniform case: $$\theta_k = \frac{1}{K}$$

### 2.4 Sample from Categorical Distribution

Suppose we have a $$\texttt{rand()}$$function, we can obtain a random value from \[0, 1\], get its corresponding position at cmf, and finally return the corresponding $$k$$. For instance, when in uniform case, if we get a random number of 0.54, it should return $$\theta_6$$.

### 2.5 Takeaways

1. One view of what ML is trying to accomplish is function approximation
2. The principle of maximum likelihood estimation provides an alternate view of learning
3. Synthetic data can help debug ML algorithms
4. Probability distributions can be used to model real data that occurs in the world

## 3. Probabilistic View of Linear Regression

![](../../.gitbook/assets/image%20%28325%29.png)

![](../../.gitbook/assets/image%20%28125%29.png)

![](../../.gitbook/assets/image%20%28448%29.png)

substitute $$\mu$$ with $$\theta^T x$$, and set MLE target to $$\theta$$, we know that

$$
\theta = \text{argmax}_\theta \ell(\theta) = \text{argmin}_\theta \frac{1}{2}\sum_i(y_i - \theta^Tx_i)^2
$$

## 4. Generative Classifiers and Discriminative Classifiers

### 4.1 Definitions

![](../../.gitbook/assets/image%20%28559%29.png)

### 4.2 Finite Sample Analysis \(Ng & Jordan, 2002\)

Assume that we are learning from a finite training dataset

1. If model assumptions are correct: Naive Bayes is a more efficient learner \(requires fewer samples\) than Logistic Regression
2. If model assumptions are incorrect: Logistic Regression has lower asymptotic error, and does better than Naïve Bayes

Notes: Naïve Bayes makes stronger assumptions about the data but needs fewer examples to estimate the parameters.

![](../../.gitbook/assets/image%20%28253%29.png)

### 4.3 Naïve Bayes vs. Logistic Regression

Learning \(Parameter Estimation\):

1. Naïve Bayes: Parameters are decoupled -&gt; a Closed form solution for MLE
2. Logistic Regression: Parameters are coupled -&gt; a No closed form solution, must use iterative optimization techniques instead

Learning \(MAP Estimation of Parameters\):

1. Bernoulli Naïve Bayes: Parameters are probabilities -&gt; a Beta prior \(usually\) pushes probabilities away from zero / one extremes
2. Logistic Regression: Parameters are not probabilities -&gt; a Gaussian prior encourages parameters to be close to zero \(effectively pushes the probabilities away from zero / one extremes\), can proved to be identical to L1/L2 regularization

Features:

1. Naïve Bayes: Features x are assumed to be conditionally independent given y. \(i.e. Naïve Bayes Assumption\)
2. Logistic Regression: No assumptions are made about the form of the features x. They can be dependent and correlated in any fashion.

## 5. Oracles, Sampling, Generative vs. Discriminative Learning Objectives

1. Sample from common probability distributions
2. Write a generative story for a generative or discriminative classification or regression model
3. Pretend to be a data generating oracle
4. Provide a probabilistic interpretation of linear regression
5. Use the chain rule of probability to contrast generative vs. discriminative modeling
6. Define maximum likelihood estimation \(MLE\) and maximum conditional likelihood estimation \(MCLE\)

