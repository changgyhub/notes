# 10 Naïve Bayes

## 1. Applications

### 1.1 Fake News Detector

To define a generative model of emails of two different classes \(e.g. real vs. fake news\)

### 1.2 Bernoulli Naïve Bayes - Weighted Coin

![](../../.gitbook/assets/image%20%28764%29.png)

## 2. Bernoulli Naïve Bayes

### 2.1 Problem Definition for Weighted Coin

![](../../.gitbook/assets/image%20%28736%29.png)

![](../../.gitbook/assets/image%20%28484%29.png)

Note: Classification for Bernoulli Naïve Bayes is same as Generic Naïve Bayes

Note: $$p_{\phi, \boldsymbol{\theta}}(\boldsymbol{x}, y) = p(\boldsymbol{x}, y | \phi, \boldsymbol{\theta})$$

### 2.2 Naïve Bayes Assumption

Naïve Bayes Assumption: $$p(\boldsymbol{x}|y) = \prod_{x=i}^M p(x_i | y)$$, each pair of $$x_p, x_q$$ are conditionally independent given $$y$$

### 2.3 What’s wrong with the Naïve Bayes Assumption?

The features might not be independent!

Example: If "Donald" is in an article, it is extremely likely to be before a "Trump".

### 2.4 Why would we use Naïve Bayes? Isn’t it too Naïve?

Naïve Bayes has one key advantage over methods like Perceptron, Logistic Regression, Neural Nets: **Training is lightning fast**! While other methods require slow iterative training procedures that might require hundreds of epochs, Naïve Bayes computes its parameters in closed form by counting.

## 3. Other Types of Naïve Bayes

### 3.1 Overview

1. Bernoulli Naïve Bayes: – for binary features
2. Gaussian Naïve Bayes: – for continuous features
3. Multinomial Naïve Bayes: – for integer features
4. Multi-class Naïve Bayes: – for classification problems with &gt; 2 classes; event model could be any of Bernoulli, Gaussian, Multinomial, depending on features

### 3.2 Gaussian Naïve Bayes

![](../../.gitbook/assets/image%20%28792%29.png)

### 3.3 Multinomial Naïve Bayes

![](../../.gitbook/assets/image%20%28443%29.png)

### 3.4 Multiclass Naïve Bayes

![](../../.gitbook/assets/image%20%2872%29.png)

### 3.5 Generic Naïve Bayes

![](../../.gitbook/assets/image%20%2852%29.png)

![](../../.gitbook/assets/image%20%28459%29.png)

## 4. MLE, MAP & Smoothing

### 4.1 MLE for Naïve Bayes

For Bernoulli Naïve Bayes: take log of $$p_{\phi, \boldsymbol{\theta}}(\boldsymbol{x}, y)$$, which is the log likelihood; then take derivative with respect to $$\phi, \boldsymbol{\theta}$$  and set them to zero, we have

![](../../.gitbook/assets/image%20%28616%29.png)

### 4.2 What does MLE accomplish?

* There is only a finite amount of probability mass \(i.e. sum-to-one constraint\)
* MLE tries to allocate as much probability mass as possible to the things we have observed at the expense of the things we have not observed

### 4.3 Problem of MLE

For Fake News Detector:

Problem of MLE: Suppose we never observe, the word "climate" in fake articles is $$\forall y^{(i)} = \text{Fake}, x_{\text{climate}}^{(i)} = 0$$. Then MLE of $$\theta_{\text{climate, Fake}} = 0$$.

![](../../.gitbook/assets/image%20%28101%29.png)

### 4.4 Add-1 Smoothing

![](../../.gitbook/assets/image%20%2821%29.png)

![](../../.gitbook/assets/image%20%28658%29.png)

### 4.5 Add-λ Smoothing

![](../../.gitbook/assets/image%20%28531%29.png)

### 4.6 MAP Estimation \(Beta Prior\)

![](../../.gitbook/assets/image%20%28106%29.png)

## 5. Visualizing Naïve Bayes

### 5.1 Fisher Iris Dataset

Fisher \(1936\) used 150 measurements of flowers from 3 different species: Iris setosa \(0\), Iris virginica \(1\), Iris versicolor \(2\) collected by Anderson \(1936\)

### 5.2 Decision Boundary

Naïve Bayes has a linear decision boundary if variance \(sigma\) is constant across classes

![](../../.gitbook/assets/image%20%28527%29.png)

![](../../.gitbook/assets/image%20%28505%29.png)

![](../../.gitbook/assets/image%20%28622%29.png)

## 6. Naïve Bayes Summary

1. Naïve Bayes provides a framework for generative modeling
2. Choose p\(xm \| y\) appropriate to the data \(e.g. Bernoulli for binary features, Gaussian for continuous features\)
3. Train by MLE or MAP
4. Classify by maximizing the posterior

