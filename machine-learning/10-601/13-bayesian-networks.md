# 13 Bayesian Networks

## 1. Bayes Nets Outline

1. Motivation
   1. Structured Prediction
2. Background
   1. Conditional Independence
   2. Chain Rule of Probability
3. Directed Graphical Models
   1. Writing Joint Distributions
   2. Definition: Bayesian Network
   3. Qualitative Specification
   4. Quantitative Specification
   5. Familiar Models as Bayes Nets
4. Conditional Independence in Bayes Nets
   1. Three case studies
   2. D-separation
   3. Markov blanket
5. Learning
   1. Fully Observed Bayes Net
   2. \(Partially Observed Bayes Net\)
6. Inference
   1. Background: Marginal Probability
   2. Sampling directly from the joint distribution
   3. Gibbs Sampling

## 2. Direct Graphical Models \(Bayes Nets\)

### 2.1 Definition

![](../../.gitbook/assets/image%20%28102%29.png)

![](../../.gitbook/assets/image%20%28348%29.png)

### 2.2 Qualitative Specification

Where does the qualitative specification come from? Examples:

1. Prior knowledge of causal relationships
2. Prior knowledge of modular relationships
3. Assessment from experts
4. Learning from data \(i.e. structure learning\) – We simply link a certain architecture \(e.g. a layered graph\)

### 2.3 Example: Conditional probability tables \(CPTs\) for discrete random variables

![](../../.gitbook/assets/image%20%28252%29.png)

### 2.4 Example: Conditional probability density functions \(CPDs\) for continuous random variables

![](../../.gitbook/assets/image%20%28135%29.png)

### 2.5 Example: Combination of CPTs and CPDs for a mix of discrete and continuous variables

![](../../.gitbook/assets/image%20%28438%29.png)

### 2.6 Familiar Models as Bayes Nets

Bernoulli Naïve Bayes \(one $$y$$ to M $$x$$\): $$p(\boldsymbol{x}, y) = p(y) \prod_{i=1}^M p(x_i|y)$$

Logistic Regression \(M $$x$$ to one $$y$$\): $$p(\boldsymbol{x}, y) = p(y|\boldsymbol{x})\prod_{i=1}^M p(x)$$, where $$p(y|\boldsymbol{x})$$ is LR result

1D Gaussian \($$\mu, \sigma^2$$ to $$x$$\): $$p(x, \mu, \sigma^2) = p(\mu)p(\sigma^2)p(x|\mu, \sigma^2)$$

## 3. Graphical Models: Determining Conditional Independence

### 3.1 What Independencies does a Bayes Net Model?

![](../../.gitbook/assets/image%20%2866%29.png)

![](../../.gitbook/assets/image%20%28108%29.png)

Cascade:

$$P(X, Y, Z) = P(Z) P(Y|Z)P(X|Y) \Longrightarrow P(X, Z|Y) = \frac{P(Z) P(Y|Z)}{P(Y)} P(X|Y) \Longrightarrow P(Z|Y) P(X|Y)$$

Common Parent:

$$P(X, Y, Z) = P(Y)P(X|Y)p(Z|Y) \Longrightarrow P(X, Z|Y) = P(X|Y)P(Z|Y)$$

### 3.2 Markov Blanket

![](../../.gitbook/assets/image%20%28188%29.png)

### 3.3 D-Separation

![](../../.gitbook/assets/image%20%28852%29.png)

![](../../.gitbook/assets/image%20%28198%29.png)

## 4. Supervised Learning for Bayesian Network

### 4.1 Learning Fully Observed BNs

![](../../.gitbook/assets/image%20%28342%29.png)

![](../../.gitbook/assets/image%20%28596%29.png)

![](../../.gitbook/assets/image%20%28215%29.png)



### 4.2 Example: Learning Fully Observed BNs

![](../../.gitbook/assets/image%20%28595%29.png)

![](../../.gitbook/assets/image%20%288%29.png)

## 5. Inference for Bayesian Network \(Won't Test\)

### 5.1 A Few Problems for Bayes Nets

![](../../.gitbook/assets/image%20%28681%29.png)

### 5.2 Gibbs Sampling

![](../../.gitbook/assets/image%20%2842%29.png)

![](../../.gitbook/assets/image%20%28256%29.png)

## 6. Learning Objectives

1. Identify the conditional independence assumptions given by a generative story or a specification of a joint distribution
2. Draw a Bayesian network given a set of conditional independence assumptions
3. Define the joint distribution specified by a Bayesian network
4. User domain knowledge to construct a \(simple\) Bayesian network for a real- world modeling problem
5. Depict familiar models as Bayesian networks
6. Use d-separation to prove the existence of conditional indenpendencies in a Bayesian network
7. Employ a Markov blanket to identify conditional independence assumptions of a graphical model
8. Develop a supervised learning algorithm for a Bayesian network
9. Use samples from a joint distribution to compute marginal probabilities
10. Sample from the joint distribution specified by a generative story
11. Implement a Gibbs sampler for a Bayesian network

