# 01 Decision Trees

## 1. Example: Medical Diagnosis

Doctor decide $$y \in \{+, -\}$$whether to give a vaccine or not based on attributes of patient $$X_1, X_2, \dots, X_N$$. For instance:

| Travel Internationally | Age | Bite Mark | VacCine? |
| :--- | :--- | :--- | :--- |
| N | senior | N | - |
| Y | child | N | + |
| Y | senior | N | - |
| N | adult | Y | + |
| Y | child | Y | + |

## 2. Decision Tree

### 2.1 Decision Tree Prediction

$$f : [x_1, x_2, x_3] \rightarrow y$$ where $$y \in \{+, -\}$$.

Internal node: test an attribute.

Branch from nodes: select value for attribute

Leaf node: predict y

![](../../.gitbook/assets/image%20%28655%29.png)

### 2.2 Function Approximation - Problem Setting

* Set of possible inputs $$X$$, $$\boldsymbol{x} \in X$$ is a discrete vector of attributes
* Set of possible outputs $$Y$$, $$\boldsymbol{y} \in Y$$ is discrete
* Unknown target function $$C^*: X \rightarrow Y$$, the oracle or a perfect doctor
* Set of candidate hypothesis $$H = \{h | h: X \rightarrow Y\}$$, a decision tree

### 2.3 Example: Tree to Predict C-Section Risk

![](../../.gitbook/assets/image%20%28500%29.png)

### 2.4 Majority Vote Classifier

def train\(D\): store D

def test\($$\boldsymbol{x}$$: attributes of a patient\): return y = majority\_class\(D\), where majority\_class means the output class that appears most often in D

### 2.5 Error Rate \(Misclassification Rate\)

Let D be a dataset $$D = {(\boldsymbol{x}^{(1)}, \boldsymbol{y}^{(1)}), (\boldsymbol{x}^{(2)}, \boldsymbol{y}^{(2)}), \dots, (\boldsymbol{x}^{(N)}, \boldsymbol{y}^{(N)})}$$, $$h(\boldsymbol{x})$$be a classifier

$$error(h, D) = [\sum_{i=1}^N \mathbb{I}(h(\boldsymbol{x^{(i)}}) \neq \boldsymbol{y}^{(i)})]/N$$, which is the number of mistakes/examples in D

### 2.6 Decision Tree Learning \(ID3, CART, etc\)

```text
def train(D):
    root = new Node (D)
    train_tree(root)

procedure train_tree(node):
    m = best attribute on which to split node's data
    Let m be decision attribute for node
    for value in m:
        create a descendant node
    partition node's data into descendants
    stop if is_perfectly_classified(D), otherwise recurse on descendants
```

To choose the best attribute, we may use

1. Error Rate
2. Mutual Information
3. Gini Impurity

### 2.7 Information Theory

Let X, Y be random variables, where $$x \in X$$and $$y \in Y$$, then

\(marginal\) entropy is

$$
H(Y) = - \sum_{y \in Y} P(Y= y) \log_2 P(Y= y)
$$

\(specific\) conditional entropy is 

$$
H(Y|X = x) = - \sum_{y \in Y} P(Y= y|X = x) \log_2 P(Y= y|X = x)
$$

conditional entropy is 

$$
H(Y|X) = \sum_{x 
\in X} P(X = x) H(Y|X = x)
$$

mutual information/information gain is

$$
I(Y; X) = H(Y) - H(Y|X)
$$

### 2.8 Example: Using Mutual Information

![](../../.gitbook/assets/image%20%28435%29.png)

$$H(Y) = -(\frac{2}{8} \log_2 (\frac{2}{8}) + \frac{6}{8} \log_2 (\frac{6}{8})) \approx 0.8113$$

$$H(Y|B = 0) = -(\frac{2}{4} \log_2 (\frac{2}{4}) + \frac{2}{4} \log_2 (\frac{2}{4})) = 1$$

$$H(Y|B = 1) = -(\frac{0}{4} \log_2 (\frac{0}{4}) + \frac{4}{4} \log_2 (\frac{4}{4})) = 0$$

$$H(Y|B) = \frac{4}{8} \times 1 + \frac{4}{8} \times 0 = 0.5$$

$$I(Y; B) = H(Y) - H(Y|B) \approx 0.3113$$

$$I(Y; A) = H(Y) - H(Y|A)  = 0$$

## 3. Greedy Search and ID3

### 3.1 Greedy Search

Decision trees are like greedy search: may miss shortest paths

![](../../.gitbook/assets/image%20%28641%29.png)

### 3.2 ID3

Search space: all possible decision tress, each node is a tree.

ID3 is an instance of greedy search that is trying to maximize mutual information at each split.

ID3 searches for the smallest tree that is consistent \(i.e. zero error rate\) with the training data.

### 3.3 Inductive Bias

The inductive bias \(also known as learning bias\) of a learning algorithm is the set of assumptions that the learner uses to predict outputs given inputs that it has not encountered.

That is to say, the principle of the algorithm to generalize to unseen examples.

### 3.3 Occam's Razor - An example of Inductive Bias of ID3

We should prefer the simplest hypothesis classifier that explains the data.

## 4. Overfitting

### 4.1  Underfitting vs Overfitting

![](../../.gitbook/assets/image%20%28485%29.png)

### 4.2 Definition

Consider a hypothesis $$h$$, its error rate over training data $$error_{train}(h)$$ and the true error rate over all data $$error_{true}(h)$$, we say $$h$$ overfits the training data if $$error_{true}(h) > error_{train}(h)$$, where the amount of overfitting is $$error_{true}(h) - error_{train}(h)$$.

![](../../.gitbook/assets/image%20%28266%29.png)

### 4.3 How to Avoid Overfitting

For Decision Trees:

1. Do not grow tree beyond some maximum depth
2. Do not split if splitting criterion \(e.g. Info. Gain\) is below some threshold
3. Stop growing when the split is not statistically significant
4. Grow the entire tree, then prune

### 4.4 Reduced-Error Pruning

Split data into training and validation set.

Create tree that classifies training set correctly.

Do until further pruning is harmful:

1. Evaluate impact on validation set of pruning each possible node \(plus those below it\)
2. Greedily remove the one that most improves validation set accuracy.

It produces smallest version of most accurate subtree.

![](../../.gitbook/assets/image%20%28269%29.png)

## 5. Decision Trees \(DTs\) in the Wild

![](../../.gitbook/assets/image%20%28626%29.png)

## 6. DT Learning Objectives

1. Implement Decision Tree training and prediction
2. Use effective splitting criteria for Decision Trees and be able to define entropy, conditional entropy, and mutual information / information gain
3. Explain the difference between memorization and generalization \[CIML\]
4. Describe the inductive bias of a decision tree
5. Formalize a learning problem by identifying the input space, output space, hypothesis space, and target function
6. Explain the difference between true error and training error
7. Judge whether a decision tree is underfitting or overfitting
8. Implement a pruning or early stopping method to combat overfitting in Decision Tree learning



