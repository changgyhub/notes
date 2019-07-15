# 12 Hidden Markov Models

## 1. Structured Prediction

### 1.1 Definition

![](../../.gitbook/assets/image%20%28456%29.png)

### 1.2 Examples

Examples of structured prediction

1. Part-of-speech \(POS\) tagging
2. Handwriting recognition
3. Speech recognition
4. Word alignment
5. Congressional voting

Examples of latent structure

1. Object recognition

### 1.3 Case Study: Object Recognition

![](../../.gitbook/assets/image%20%28451%29.png)

### 1.4 Another Example: Finding the most probable assignment to the output

![](../../.gitbook/assets/image%20%28566%29.png)

### 1.5 Review of Machine Learning

![](../../.gitbook/assets/image%20%28510%29.png)

![](../../.gitbook/assets/image%20%28652%29.png)

## 2. Background

### 2.1 Chain Rule of Probability

![](../../.gitbook/assets/image%20%28151%29.png)

### 2.2 Conditional Independence

![](../../.gitbook/assets/image%20%28496%29.png)

## 3. Hidden Markov Model \(HMM\)

### 3.1 Outline

Motivation

1. Time Series Data

Hidden Markov Model \(HMM\)

1. Example: Squirrel Hill Tunnel Closures \[courtesy of Roni Rosenfeld\]
2. Background: Markov Models
3. From Mixture Model to HMM
4. History of HMMs
5. Higher-order HMMs

Training HMMs

1. \(Supervised\) Likelihood for HMM
2. Maximum Likelihood Estimation \(MLE\) for HMM
3. EM for HMM \(aka. Baum-Welch algorithm\)

Forward-Backward Algorithm

1. Three Inference Problems for HMM
2. Great Ideas in ML: Message Passing
3. Example: Forward-Backward on 3-word Sentence
4. Derivation of Forward Algorithm
5. Forward-Backward Algorithm
6. Viterbi algorithm

### 3.2 Example: Squirrel Hill Tunnel Closures

![](../../.gitbook/assets/image%20%28718%29.png)

![](../../.gitbook/assets/image%20%28276%29.png)

### 3.3 From Mixture Model to HMM

![](../../.gitbook/assets/image%20%28580%29.png)

![](../../.gitbook/assets/image%20%28763%29.png)

### 3.4 Hidden Markov Model - Form 1

![](../../.gitbook/assets/image%20%2869%29.png)

### 3.5 Supervised Learning for HMMs - Form 1

![](../../.gitbook/assets/image%20%28805%29.png)

![](../../.gitbook/assets/image%20%28733%29.png)

Note: Solving the MLE needs Lagrange multiplier since we have the constraint $$\sum = 1$$in rows of the above matrices.

### 3.6 Hidden Markov Model - Form 2

![](../../.gitbook/assets/image%20%28457%29.png)

![](../../.gitbook/assets/image%20%28433%29.png)

### 3.7 Supervised Learning for HMMs - Form 2

![](../../.gitbook/assets/image%20%28748%29.png)

### 3.8 Higher-order HMMs

![](../../.gitbook/assets/image%20%2838%29.png)

### 3.9 Shortcomings of Hidden Markov Models

1. HMM models capture dependences between each state and only its corresponding observation 
   * NLP example: In a sentence segmentation task, each segmental state may depend not just on a single word \(and the adjacent segmental stages\), but also on the \(nonlocal\) features of the whole line such as line length, indentation, amount of white space, etc.
2. Mismatch between learning objective function and prediction objective function
   * HMM learns a joint distribution of states and observations $$P(Y, X)$$, but in a prediction task, we need the conditional probability $$P(Y|X)$$

## 4. Message Passing

### 4.1 Count the soldiers

![](../../.gitbook/assets/image%20%28393%29.png)

### 4.2 Each soldier receives reports from all branches of tree

![](../../.gitbook/assets/image%20%2834%29.png)

## 5. Inference Problems

### 5.1 Four Inference Problems for an HMM

#### 5.1.1 Evaluation

Compute the probability of a given sequence of observations: $$p(\boldsymbol{x}) = \sum_{\boldsymbol{y}} p(\boldsymbol{x}, \boldsymbol{y})$$

A brute force solution:

```python
def eval(x):
    return sum([joint(x, y) for y in all_y(x)])
```

#### 5.1.2. Viterbi Decoding

Find the most-likely sequence of hidden states, given a sequence of observations: $$\boldsymbol{y} = \arg \max_{\boldsymbol{y}} p(\boldsymbol{y}|\boldsymbol{x})$$

#### 5.1.3. Marginals

Compute the marginal distribution for a hidden state, given a sequence of observations:

$$p(y_t = k | \boldsymbol{x}) = \sum_{\boldsymbol{y} \text{ s.t.}y_t = k} p(\boldsymbol{y}|\boldsymbol{x})$$

#### 5.1.4 Minimum Bayes Risk \(MBR\) Decoding

Find the lowest loss sequence of hidden states, given a sequence of observations \(Viterbi decoding is a special case\)

### 5.2 Derivation of Forward Algorithm

![](../../.gitbook/assets/image%20%28130%29.png)

Note:

1. Before implementing forward backward algorithm, first implement the brute force algorithm to check the result
2. For $$|\boldsymbol{y}| = T$$ and $$y_t \in \{1, \cdots, K\}$$, there are $$K^T$$possible values of $$\boldsymbol{y}$$
3. Some probability formulas:

$$
P(x_1, x_2, x_3) = \sum_{y_1} \sum_{y_2} \sum_{y_3} P(x_1, x_2, x_3, y_1, y_2, y_3)\\[3pt]
P(y_2 | x_1, x_2, x_3) = \sum_{y_1} \sum_{y_3} P(y_1, y_2, y_3|x_1, x_2, x_3)
$$

### 5.3 Forward-Backward Algorithm

![](../../.gitbook/assets/image%20%28623%29.png)

### 5.4 Viterbi Algorithm

![](../../.gitbook/assets/image%20%28849%29.png)

### 5.5 Inference Computational Complexity

The naïve \(brute force\) computations for Evaluation, Decoding, and Marginals take exponential time, $$O(K^T)$$

The forward-backward algorithm and Viterbi algorithm run in polynomial time, $$O(TK^2)$$ – Thanks to dynamic programming

### 5.6 Minimum Bayes Risk Decoding

![](../../.gitbook/assets/image%20%28605%29.png)

![](../../.gitbook/assets/image%20%28155%29.png)

![](../../.gitbook/assets/image%20%28694%29.png)

## 6. HMM Learning Objectives

1. Show that structured prediction problems yield high-computation inference problems
2. Define the first order Markov assumption
3. Draw a Finite State Machine depicting a first order Markov assumption
4. Derive the MLE parameters of an HMM
5. Define the three key problems for an HMM: evaluation, decoding, and marginal computation
6. Derive a dynamic programming algorithm for computing the marginal probabilities of an HMM
7. Interpret the forward-backward algorithm as a message passing algorithm
8. Implement supervised learning for an HMM
9. Implement the forward-backward algorithm for an HMM
10. Implement the Viterbi algorithm for an HMM
11. Implement a minimum Bayes risk decoder with Hamming loss for an HMM

