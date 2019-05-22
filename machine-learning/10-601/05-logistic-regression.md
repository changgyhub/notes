# 05 Logistic Regression

## 1. Probabilistic Learning

### 1.1 Principle of Maximum Likelihood Estimation \(MLE\)

Choose the parameters that make the data "most likely"

Assumptions: $$\mathcal{D} = \{\bm{v}^{(i)}\}_{i=1}^N$$, data generated iid from distribution $$p^*(\bm{v}|\bm{\theta}^*)$$, i.e. $$\bm{v} \sim p^*(\cdot|\bm{\theta}^*)$$, and $$p^*$$ comes from a family of distributions parameterized by $$\bm{\theta}^* \in \bm{\Theta}$$.

We can also write $$p(\mathcal{D}|\bm{\theta}) = \prod_{i=1}^N p(\bm{v}^{(i)}|\bm{\theta})$$, which is the likelihood.

MLE is defined as $$\hat{\bm{\theta}} = \text{argmax}_{\bm{\theta} \in \bm{\Theta}} p(\mathcal{D}|\bm{\theta}) = \text{argmax}_{\bm{\theta} \in \bm{\Theta}} \log p(\mathcal{D}|\bm{\theta})$$

### 1.2 Bernoulli Classifier

For data $$\{(\bm{x}^{(i)}, y^{(i)})\}$$ describing \(features, label\), we

1. Ignore features $$\bm{x}$$
2. Model $$y \sim \text{Bernoulli}(\phi)$$such that $$p(y|\bm{x}) = \phi$$ if $$y=1$$ otherwise $$1-\phi$$
3. Use conditional log likelihood: $$\mathcal{L}(\phi) = \sum_{i=1}^N \log p(y^{(i)} | \bm{x}^{(i)}) = \sum_{i=1}^N y^{(i)}\log\phi + (1-y^{(i)})\log(1-\phi))$$
4. $$\hat{\phi} = \text{argmax}_{\phi \in [0,1]} \mathcal{L}(\phi) = \text{sum}(Y)/\text{len}(Y)$$

If we use Bayes Classifier based on this, it turns out to be a majority vote classifier: $$h(\bm{x}) = \text{argmax}_{y \in [0,1]} p(y | \bm{x})$$

## 2. Logistic Regression

### 2.1 Example: Image Classification

Softmax here is just logistic regression.

![](../../.gitbook/assets/image%20%28310%29.png)

### 2.2 Logistic Regression

![](../../.gitbook/assets/image%20%28820%29.png)

![](../../.gitbook/assets/image%20%28149%29.png)

### 2.3 Binary Logistic Regression

Base on section 1.2

1. We use $$\bm{x}$$
2. Model $$\phi = \sigma(\bm{\theta}^T \bm{x})$$ where $$\sigma(u) = \frac{1}{1 + e^{-u}}$$, $$y \sim \text{Bernoulli}(\phi)$$ such that $$p(y|\bm{x}) = \phi$$ if $$y=1$$ otherwise $$1-\phi$$
3. $$\mathcal{J}(\bm{\theta})  = - \frac{1}{N}\mathcal{L}(\bm{\theta}) = - \frac{1}{N}\sum_{i=1}^N \log p(y^{(i)} | \bm{x}^{(i)}) $$
4. We derive that $$\nabla \mathcal{J}_i(\bm{\theta}) = - (y^{(i)} - \sigma(\bm{\theta}^T \bm{x}^{(i)})) \bm{x}^{(i)}$$to solve $$\hat{\bm{\theta}} = \text{argmax}_{\bm{\theta}} \mathcal{L}(\bm{\theta}) = \text{argmin}_{\bm{\theta}} \mathcal{J}(\bm{\theta})$$
5. Use gradient descent or SGD to optimize

## 3. Details of Logistic Regression

### 3.1 Maximum Conditional Likelihood Estimation

![](../../.gitbook/assets/image%20%28340%29.png)

![](../../.gitbook/assets/image%20%28193%29.png)

### 3.2 SGD for Logistic Regression

![](../../.gitbook/assets/image%20%28614%29.png)

### 3.3 Summary

1. Discriminative classifiers directly model the conditional, p\(y\|x\)
2. Logistic regression is a simple linear classifier, that retains a probabilistic semantics
3. Parameters in LR are learned by iterative optimization \(e.g. SGD\)

## 4. Logistic Regression Objectives

1. Apply the principle of maximum likelihood estimation \(MLE\) to learn the parameters of a probabilistic model
2. Given a discriminative probabilistic model, derive the conditional log-likelihood, its gradient, and the corresponding Bayes Classifier
3. Explain the practical reasons why we work with the log of the likelihood
4. Implement logistic regression for binary or multiclass classification
5. Prove that the decision boundary of binary logistic regression is linear
6. For linear regression, show that the parameters which minimize squared error are equivalent to those that maximize conditional likelihood

## 5. Multinomial Logistic Regression

Note

1. in multinomial case, $$\bm{\theta}$$ is of shape $$k \times M$$, where $$k$$ is the number of classes and $$M$$ is the dimension of $$\bm{x}$$
2. Conditional log likelihood is defined as $$\mathcal{L}(\bm{\theta}) = \sum_{i=1}^M \log p(y^{(i)} | \bm{x}^{(i)})$$
3. Loss function is $$\mathcal{J}(\bm{\theta}) =- \sum_{i=1}^M \log p(y^{(i)} | \bm{x}^{(i)})$$

![](../../.gitbook/assets/image%20%28445%29.png)

