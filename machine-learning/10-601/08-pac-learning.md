# 08 PAC Learning

## 1. Learning Theory

### 1.1 PAC/SLT models for Supervised Learning

![](../../.gitbook/assets/image%20%28305%29.png)

### 1.2 Error Types

![](../../.gitbook/assets/image%20%28328%29.png)

### 1.3 PAC / SLT Model

![](../../.gitbook/assets/image%20%28843%29.png)

### 1.4 Three Hypotheses of Interest

![](../../.gitbook/assets/image%20%28722%29.png)

## 2. PAC Learning

### 2.1 Sample Complexity

Definition of sample complexity: the minimum value N of training examples such that the PAC criterion holds for a given value of $$\epsilon$$ and $$\delta$$. An alternative definition is:

![](../../.gitbook/assets/image%20%28816%29.png)

Note:

1. Realizable mean s $$c^* \in \mathcal{H}$$, Agnostic means $$c^*$$ could be anywhere.
2. For realizable cases, their contrapositives are more intuitive \(if $$\hat{R}(h) = 0$$ then $$R(h) < \epsilon$$\)
3. For decision tree with $$n$$ features, $$|\mathcal{H}| = 2^{2^n}$$

### 2.2 Proof for finite and realizable:

1. Assume $$k$$ bad hypothesis $$h_1, \cdots, h_k$$ with $$R(h_i) \ge \epsilon$$
2. The probability that at least one $$h_i$$is consistent with first $$N$$ training example exist is $$k (1-\epsilon)^N \le |\mathcal{H}| (1-\epsilon)^N$$
3. Calculate value of $$N$$ such that$$|\mathcal{H}| (1-\epsilon)^N \le \delta$$
4. Since $$1-x \le e^{-x}$$, we have $$|\mathcal{H}| e^{-\epsilon N} \le \delta$$, and the result follows.

### 2.3 Example: Conjunctions

Suppose $$\mathcal{H}$$ is class of conjunctions over x in $$\{0,1\}^M$$ If $$M = 10, \epsilon = 0.1, \delta = 0.01$$, how many examples suffice?

Solution: Use the formula. Note that$$|\mathcal{H}| = 3^M$$. This is because in $$h$$, it may set $$x_i = 0$$ or $$1$$ or not contain $$x_i$$. Example: $$h_1 = x_1 (1-x_2) x_4 \cdots$$

### 2.4 Proof for finite and agnostic:

1. For finite and realizable, bound is inversely linear in epsilon \(e.g. halving the error requires double the examples\), and bound is only logarithmic in $$\mathcal{H}$$ \(e.g. quadrupling the hypothesis space only requires double the examples\)
2. For finite and agnostic, bound is inversely quadratic in epsilon \(e.g. halving the error requires 4x the examples\), and bound is only logarithmic in $$\mathcal{H}$$ \(i.e. same as Realizable case\)

### 2.5 Shattering and VC-dimension

Shattering:

![](../../.gitbook/assets/image%20%28170%29.png)

Vapnik-Chervonenkis dimension

![](../../.gitbook/assets/image%20%28654%29.png)

### 2.6 Examples of VC-dimension

![](../../.gitbook/assets/image%20%28475%29.png)

![](../../.gitbook/assets/image%20%28571%29.png)

![](../../.gitbook/assets/image%20%28632%29.png)

![](../../.gitbook/assets/image%20%28297%29.png)

![](../../.gitbook/assets/image%20%28680%29.png)

### 2.7 Probably Approximately Correct \(PAC\) Learning

![](../../.gitbook/assets/image%20%2815%29.png)

## 3. Questions for Learning Theory

1. Given a classifier with zero training error, what can we say about generalization error? \(Sample Complexity, Realizable Case\)
2. Given a classifier with low training error, what can we say about generalization error? \(Sample Complexity, Agnostic Case\)
3. Is there a theoretical justification for regularization to avoid overfitting? \(Structural Risk Minimization\)

## 4. Learning Theory Objectives

1. Identify the properties of a learning setting and assumptions required to ensure low generalization error
2. Distinguish true error, train error, test error
3. Define PAC and explain what it means to be approximately correct and what occurs with high probability
4. Apply sample complexity bounds to real-world learning examples
5. Distinguish between a large sample and a finite sample analysis
6. Theoretically motivate regularization

