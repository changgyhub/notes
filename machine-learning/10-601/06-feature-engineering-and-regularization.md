# 06 Feature Engineering & Regularization

## 1. Feature Engineering

### 1.1 Feature Learning and Engineering for NLP: Overview

![](../../.gitbook/assets/image%20%28776%29.png)

### 1.2 Feature Engineering for NLP

![](../../.gitbook/assets/image%20%28411%29.png)

![](../../.gitbook/assets/image%20%28637%29.png)

### 1.3 Feature Engineering for CV

Edge detection \(Canny\)

Corner Detection \(Harris\)

Scale Invariant Feature Transform \(SIFT\)

## 2. Nonlinear Features

### 2.1 Definitions

![](../../.gitbook/assets/image%20%28558%29.png)

### 2.2 Example: Linear Regression

![](../../.gitbook/assets/image%20%28753%29.png)

![](../../.gitbook/assets/image%20%28597%29.png)

### 2.3 Over-fitting

![](../../.gitbook/assets/image%20%28771%29.png)

## 3. Regularization

### 3.1 Definition of Overfitting

Definition: The problem of overfitting is when the model captures the noise in the training data instead of the underlying structure

Overfitting can occur in all the models we’ve seen so far

1. KNN \(e.g. when k is small\)
2. Naïve Bayes \(e.g. without a prior\)
3. Linear Regression \(e.g. with basis function\)
4. Logistic Regression \(e.g. with many rare features\)

### 3.2 Motivation: Regularization

![](../../.gitbook/assets/image%20%28258%29.png)

### 3.3 Occam’s Razor

Occam’s Razor: prefer the simplest hypothesis

What does it mean for a hypothesis \(or model\) to be simple?

1. small number of features \(model selection\)
2. small number of “important” features \(shrinkage\)

### 3.4 L0, L1, L2 norms

Regularization target becomes $$\hat{\theta} = \text{argmin}_{\theta} \mathcal{J}(\theta)+ \lambda r(\theta)$$ 

L2 norm: $$\sum_{m} (\theta_m)^2$$, L1 norm: $$\sum_{m} |\theta_m|$$, L0 norm: $$\sum_{m} \mathbb I (\theta_m \neq 0)$$

L2 prefer small value, L0 prefer zero value, L1 prefer small and zero values

### 3.5 Notes for Regularization

![](../../.gitbook/assets/image%20%28369%29.png)

### 3.6 Regularization as MAP

L1 regularization equals MAP estimation of parameters with Laplace prior $f(x) = \frac{1}{2b}\exp(-\frac{|x - a|}{b})$ with location parameter $\alpha = 0$.

L2 regularization equals MAP estimation of parameters with Gaussian prior with zero mean.

### 3.7 Takeaways

1. Nonlinear basis functions allow linear models \(e.g. Linear Regression, Logistic Regression\) to capture nonlinear aspects of the original input
2. Nonlinear features are require no changes to the model \(i.e. just preprocessing\)
3. Regularization helps to avoid overfitting
4. Regularization and MAP estimation are equivalent for appropriately chosen priors

## 4. Feature Engineering / Regularization Objectives

1. Engineer appropriate features for a new task
2. Use feature selection techniques to identify and remove irrelevant features
3. Identify when a model is overfitting
4. Add a regularizer to an existing objective in order to combat overfitting
5. Explain why we should not regularize the bias term
6. Convert linearly inseparable dataset to a linearly separable dataset in higher dimensions
7. Describe feature engineering in common application areas

