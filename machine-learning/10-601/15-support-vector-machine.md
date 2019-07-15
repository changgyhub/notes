# 15 Support Vector Machine

## 1. Constrained Optimization

### 1.1  Definition

$$
\min_\boldsymbol{\Theta} \mathcal{J}(\boldsymbol{\Theta})\\
\text{s.t. } \boldsymbol{g}(\boldsymbol{\Theta}) \le \boldsymbol{b}
$$

### 1.2 Linear Programming

$$
\min_\boldsymbol{x} \boldsymbol{c}^T\boldsymbol{x}\\
\text{s.t. } \boldsymbol{A}\boldsymbol{x} \le \boldsymbol{b}\\[3pt]
\text{where } \boldsymbol{c}\in\mathbb{R}^m, \boldsymbol{A}\in\mathbb{R}^{n \times m}, \boldsymbol{b}\in\mathbb{R}^n
$$

We can use interior point method or simplex algorithm to solve.

### 1.3 Quadratic Programming

$$
\min_\boldsymbol{x} \frac{1}{2} \boldsymbol{x}^T \boldsymbol{Q}\boldsymbol{x} + \boldsymbol{c}^T\boldsymbol{x}\\
\text{s.t. } \boldsymbol{A}\boldsymbol{x} \le \boldsymbol{b}\\[3pt]
\text{where } \boldsymbol{c}\in\mathbb{R}^m, \boldsymbol{A}\in\mathbb{R}^{n \times m}, \boldsymbol{b}\in\mathbb{R}^n,  \boldsymbol{Q} \in \mathbb{R}^{m \times m}
$$

We can use interior point method, ellipsoid method \(assume $$\boldsymbol{Q}$$ convex, in polynomial time\), conjugate gradient method.

![](../../.gitbook/assets/image%20%28676%29.png)

## 2. Support Vector Machine \(SVM\)

### 2.1 Linearly Separable Case

#### 2.1.1 Margin

![](../../.gitbook/assets/image%20%28104%29.png)

#### 2.1.2 Lagrangian

![](../../.gitbook/assets/image%20%28609%29.png)

![](../../.gitbook/assets/image%20%28218%29.png)

#### 2.1.3 Primal vs Dual

![](../../.gitbook/assets/image%20%28450%29.png)

### 2.2 Soft-Margin SVM \(Not Covered\)

![](../../.gitbook/assets/image%20%28848%29.png)

![](../../.gitbook/assets/image%20%28226%29.png)

### 2.3 Multiclass SVMs \(Not Covered\)

The SVM is inherently a binary classification method, but can be extended to handle K-class classification in many ways.

1. one-vs-rest
   1. build K binary classifiers
   2. train the kth classifier to predict whether an instance has label k or something else
   3. predict the class with largest score
2. one-vs-one
   1. build \(K choose 2\) binary classifiers
   2. train one classifier for distinguishing between each pair of labels
   3. predict the class with the most “votes” from any given classifier

## 4. SVM Learning Objectives

1. Motivate the learning of a decision boundary with large margin
2. Compare the decision boundary learned by SVM with that of Perceptron
3. Distinguish unconstrained and constrained optimization
4. Compare linear and quadratic mathematical programs
5. Derive the hard-margin SVM primal formulation
6. Derive the Lagrangian dual for a hard-margin SVM
7. Describe the mathematical properties of support vectors and provide an intuitive explanation of their role
8. Draw a picture of the weight vector, bias, decision boundary, training examples, support vectors, and margin of an SVM
9. Employ slack variables to obtain the soft-margin SVM
10. Implement an SVM learner using a black-box quadratic programming \(QP\) solver

