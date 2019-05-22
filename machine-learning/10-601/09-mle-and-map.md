# 09 MLE & MAP

## 1. Probability

### 1.1 Random Variables: Definitions

![](../../.gitbook/assets/image%20%2891%29.png)

![](../../.gitbook/assets/image%20%28185%29.png)

### 1.2 Common Probability Distributions

For Discrete Random Variables:

1. Bernoulli
2. Binomial
3. Multinomial
4. Categorical
5. Poisson

For Continuous Random Variables:

1. Exponential
2. Gamma
3. Beta
4. Dirichlet
5. Laplace
6. Gaussian \(1D\)
7. Multivariate Gaussian

Beta Distribution:

![](../../.gitbook/assets/image%20%28590%29.png)

Dirichlet Distribution

![](../../.gitbook/assets/image%20%2865%29.png)

### 1.3 Expectation and Variance

![](../../.gitbook/assets/image%20%28363%29.png)

![](../../.gitbook/assets/image%20%28115%29.png)

### 1.4 Multiple Random Variables

Joint Probability

![](../../.gitbook/assets/image%20%28568%29.png)

Marginal Probabilities

![](../../.gitbook/assets/image%20%28824%29.png)

Conditional Probability

![](../../.gitbook/assets/image%20%287%29.png)

Independence and Conditional Independence

![](../../.gitbook/assets/image%20%28642%29.png)

## 2. MLE & MAP

### 2.1 MLE

![](../../.gitbook/assets/image%20%28735%29.png)

What does maximizing likelihood accomplish?

* There is only a finite amount of probability mass \(i.e. sum-to-one constraint\)
* MLE tries to allocate as much probability mass as possible to the things we have observed at the expense of the things we have not observed

### 2.2 Example: MLE of Exponential Distribution

![](../../.gitbook/assets/image%20%28152%29.png)

![](../../.gitbook/assets/image%20%28136%29.png)

![](../../.gitbook/assets/image%20%28790%29.png)

### 2.3 Example: MLE of Bernoulli Distribution

Similarly, we can derive the MLE of parameter $$\phi$$ for $$N_0+N_1$$ samples drawn from $$\text{Bernoulli}(\phi)$$, where there are $$N_1$$ samples of $$x_i = 1$$ and $$N_0$$ samples of $$x_i = 0$$:

$$
\ell(\phi) = \log p(D|\phi) = N_1 \log \phi + N_0\log ( 1 - \phi )\\[3pt]
\frac{d \ell(\phi)}{d\phi} = \frac{N_1}{\phi} - \frac{N_0}{1-\phi} = 0\\[3pt]
N_1 (1-\phi) - N_0\phi = 0\\[3pt]
\phi_{\text{MLE}} = \frac{N_1}{N_0 + N_1}
$$

### 2.4 Example: MLE of Poisson Distribution

Given $$P(X=x) = \frac{\lambda^x e^{-\lambda}}{x!}$$, we have

$$
L(D;\lambda) = p(D|\lambda) = \prod_{i=1}^N \lambda^{X_i}e^{-\lambda}(X_i!)^{-1}\\[3pt]
\ell(D;\lambda) = \sum_{i=1}^N (X_i \log\lambda - \lambda) ~~~~~(X! \text{ term dropped})\\[3pt]
\frac{d\ell(D;\lambda)}{d\lambda} = \sum_{i=1}^N \frac{X_i}{\lambda} - N = 0\\[3pt]
\lambda_{\text{MLE}} = \frac{\sum_{i=0}^N X_i}{N}
$$

Note: Since $$E[\frac{\sum_{i=1}^N X_i}{N}] = \frac{1}{N} \sum_{i=1}^N E[X_i]$$, $$E[X_i] = \lambda_{\text{MLE}}$$.

### 2.5 MAP

![](../../.gitbook/assets/image%20%28165%29.png)

### 2.6 Example: MAP of Bernoulli—Beta

Let $$\phi \sim \text{Beta}(\alpha, \beta)$$, then

$$
\ell_{\text{MAP}}(\phi) = \log p(D|\phi)p(\phi)\\[3pt]
=\log((\phi^{N_1}(1-\phi)^{N_0})(\frac{1}{B(\alpha, \beta)}\phi^{\alpha-1}(1-\phi)^{\beta-1}))\\[3pt]
= - \log B(\alpha, \beta) + (N_1 + \alpha - 1)\log\phi + (N_0 + \beta - 1) \log (1-\phi)\\[9pt]
\frac{d \ell_{\text{MAP}}(\phi)}{d\phi} = \frac{N_1 + \alpha - 1}{\phi} -  \frac{N_0 + \beta - 1}{1-\phi} = 0\\[6pt]
\phi_{\text{MAP}} = \frac{N_1 + \alpha - 1}{N_1 + N_0 + \alpha + \beta - 2}
$$

Example: Flip a coin, 8 heads and 2 tails. Suppose a prior of $$\phi \sim \text{Beta}(101, 101)$$, that is, we have already observed 101 heads and 101 tails before. Then $$\phi_{\text{MAP}} = \frac{108}{108+102} \approx 0.514$$, while $$\phi_{\text{MLE}} = \frac{8}{8+2} = 0.8$$.

### 2.7 Example: MAP of Gamma-Poisson

If we use Gamma distribution $$\text{Gamma}(\alpha=1, \beta) = \beta e^{-\beta \lambda}$$ as the prior distribution for $$\lambda$$, then

$$
\ell_{\text{MAP}}(\lambda) = \log p(D|\lambda)p(\lambda)\\[3pt]
=\sum_{i=1}^N (X_i \log\lambda - \lambda) + \log p(\lambda)\\[3pt]
= \sum_{i=1}^N X_i \log\lambda - N \lambda +\log \beta - \beta \lambda
\\[3pt]
\frac{d\ell_{\text{MAP}}(\lambda)}{d\lambda} = \frac{\sum_{i=1}^N X_i}{N} - N - \beta = 0\\[3pt]
\lambda_{\text{MAP}} = \frac{\sum_{i=1}^N X_i}{N + \beta}
$$

### 2.8 Takeaways

1. One view of what ML is trying to accomplish is function approximation
2. The principle of maximum likelihood estimation provides an alternate view of learning
3. Synthetic data can help debug ML algorithms
4. Probability distributions can be used to model real data that occurs in the world

## 3. Learning Objectives

1. Recall probability basics, including but not limited to: discrete and continuous random variables, probability mass functions, probability density functions, events vs. random variables, expectation and variance, joint probability distributions, marginal probabilities, conditional probabilities, independence, conditional independence
2. Describe common probability distributions such as the Beta, Dirichlet, Multinomial, Categorical, Gaussian, Exponential, etc.
3. State the principle of maximum likelihood estimation and explain what it tries to accomplish
4. State the principle of maximum a posteriori estimation and explain why we use it
5. Derive the MLE or MAP parameters of a simple model in closed form

