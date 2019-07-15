# 02 KNN and Model Selection

## 1. Classification

$$D = \{(x^{(i)}, y^{(i)}\}_{i=1}^N, ~\forall i~~x^{(i)} \in \mathbb{R}^M, y^{(i)} \in\{1, 2, \cdots, L\}$$. $$x$$ is called features \(input\) and $$y$$ is the label \(output\)

For binary classification, $$y^{(i)} \in\{0, 1\} / \{-1, 1\} / \{+, -\} / \{T, F\} / \{blue, red\}$$

Decision boundary \(Decision rule\) $$h$$ for binary classification is $$h: \mathbb{R}^M \rightarrow \{+, -\}$$

At test time, given some vector $$x$$, we predict $$\hat{y} = h(x)$$

## 2. KNN

### 2.1 Nearest Neighbor Classifier

Train: Store $$D$$

Predict: Assign the label of the nearest point in $$D$$

### 2.2. K-Nearest Neighbor Classifier

Train: Store $$D$$

Predict: Assign the most common label of the nearest $$k$$ points in $$D$$

### 2.3 Distance Functions

KNN requires a distance function $$g: \mathbb{R}^M \times \mathbb{R}^M \rightarrow \mathbb{R}$$

Euclidean distance: $$g(\boldsymbol{u}, \boldsymbol{v}) = \sqrt{\sum_{m=1}^M (u_m - v_m)^2}$$

Manhattan distance: $$g(\boldsymbol{u}, \boldsymbol{v}) = \sqrt{\sum_{m=1}^M |u_m - v_m|}$$

### 2.4 Questions on KNN

1. How to handle even values of k?
   1. keep adding points
   2. distance-weighted
   3. remove furthest point...
2. What is the inductive bias of KNN?
   1. Similar points should have similar labels
   2. All dimensions are created equally
3. Computational Efficiency of KNN?
   * Suppose we have $$N$$ training examples, and each one has $$M$$ features. 
   * When training, Naive \(k = 1\) is $$O(1)$$, k-d Tree is $$O(MN \log N)$$
   * When testing, Naive \(k = 1\) is $$O(MN)$$, k-d Tree is $$O(2^M \log N)$$
4. Theoretical Guarantees of KNN?
   * $$error_{true}(h) < 2 \times \text{Bayes error rate}$$ \(Cover & Hart, 1967\)
   * Bayes error rate is like "the best you could possibly do"
5. How could k-Nearest Neighbors \(KNN\) be applied to regression?
6. Can we do better than majority vote? \(e.g. distance-weighted KNN\)
7. Where does the Cover & Hart \(1967\) Bayes error rate bound come from?

## 3. KNN Learning Objectives

1. Describe a dataset as points in a high dimensional space \[CIML\]
2. Implement k-Nearest Neighbors with O\(N\) prediction
3. Describe the inductive bias of a k-NN classifier and relate it to feature scale \[a la. CIML\]
4. Sketch the decision boundary for a learning algorithm \(compare k-NN and DT\)
5. State Cover & Hart \(1967\)'s large sample analysis of a nearest neighbor classifier
6. Invent "new" k-NN learning algorithms capable of dealing with even k
7. Explain computational and geometric examples of the curse of dimensionality

## 4. Model Selection

### 4.1 Statistics vs Machine Learning

#### 4.1.1 Statistics:

1. A **model** defines the data generation process \(i.e. a set or family of parametric probability distributions\)
2. **Model parameters** are the values that give rise to a particular probability distribution in the model family
3. **Learning** \(aka. estimation\) is the process of finding the parameters that best fit the data
4. **Hyperparameters** are the parameters of a prior distribution over parameters

#### 4.1.2 Machine Learning:

1. \(loosely\) A **model** defines the hypothesis space over which learning performs its search
2. **Model parameters** are the numeric values or structure selected by the learning algorithm that give rise to a hypothesis
3. The **learning algorithm** defines the data-driven search over the hypothesis space \(i.e. search for good parameters\)
4. **Hyperparameters** are the tunable aspects of the model, that the learning algorithm does not select

#### 4.1.3 Machine Learning Examples

Decision Tree

1. model = set of all possible trees, possibly restricted by some hyperparameters \(e.g. max depth\)
2. parameters = structure of a specific decision tree
3. learning algorithm = ID3, CART, etc.
4. hyperparameters = max- depth, threshold for splitting criterion, etc.

K-Nearest Neighbors

1. model = set of all possible nearest neighbors classifiers
2. parameters = none \(KNN is an instance-based or non-parametric method\)
3. learning algorithm = for naïve setting, just storing the data
4. hyperparameters = k, the number of neighbors to consider

Perceptron

1. model = set of all linear separators
2. parameters = vector of weights \(one for each feature\)
3. learning algorithm = mistake based updates to the parameters
4. hyperparameters = none \(unless using some variant such as averaged perceptron\)

#### 4.1.4 Similarities

Two very similar definitions:

1. **Model selection** is the process by which we choose the “best” model from among a set of candidates
2. **Hyperparameter** **optimization** is the process by which we choose the “best” hyperparameters from among a set of candidates \(could be called a special case of model selection\)

Both assume access to a function capable of measuring the quality of a model. Both are typically done “outside” the main training algorithm --- typically training is treated as a black box.

### 4.2 Train, Test, Validation and Cross Validation

For KNN, when k = 1 =&gt; nearest neighbor; when k = N =&gt; majority vote.

Suppose in dataset $$D$$, $$40\%$$ of $$y^{(i)} = 0$$, and $$60\%$$ of $$y^{(i)} = 1$$.

For train error, $$error_{k=1} = 0, error_{k=N} = 40\%$$. For k between 1 and N, the training error oscillates up. For test error, it is generally above train error.

We can choose k at lowest test error as our choice for k. However, we do not know the label for test data. Hence we separate a part of training data as validation data. Then we choose k at **lowest validation error**.

We can also perform cross-validation: leave out different folds as validation sets, and get different validation error curves. Then we can average the curves and get a smoother validation error curve, and finally choose k with lowest validation error.

Note that for cross-validation with N different validation folds, we call it N-fold cross validation.

![](../../.gitbook/assets/image%20%28270%29.png)

## 5. Model Selection Learning Objectives

1. Plan an experiment that uses training, validation, and test datasets to predict the performance of a classifier on unseen data \(without cheating\)
2. Explain the difference between \(1\) training error, \(2\) validation error, \(3\) cross-validation error, \(4\) test error, and \(5\) true error
3. For a given learning technique, identify the model, learning algorithm, parameters, and hyperparamters
4. Define "instance-based learning" or "nonparametric methods"
5. Select an appropriate algorithm for optimizing \(aka. learning\) hyperparameters

