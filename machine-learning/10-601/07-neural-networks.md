# 07 Neural Networks

## 1. Outline

1. Logistic Regression \(Recap\)
   1. Data, Model, Learning, Prediction
2. Neural Networks
   1. A Recipe for Machine Learning
   2. Visual Notation for Neural Networks
   3. Example: Logistic Regression Output Surface
   4. 2-Layer Neural Network
   5. 3-Layer Neural Network
3. Neural Net Architectures
   1. Objective Functions
   2. Activation Functions
4. Backpropagation
   1. Basic Chain Rule \(of calculus\)
   2. Chain Rule for Arbitrary Computation Graph
   3. Backpropagation Algorithm
   4. Module-based Automatic Differentiation \(Autodiff\)

## 2. Logistic Regression \(Recap\)

### 2.1 A Recipe for Machine Learning

![](../../.gitbook/assets/image%20%28447%29.png)

For gradient: Backpropagation can compute this gradient! And it’s a special case of a more general algorithm called **reverse mode automatic differentiation** that can compute the gradient of any differentiable function efficiently!

### 2.2 Decision Functions - Linear Regression

![](../../.gitbook/assets/image%20%28744%29.png)

### 2.3 Decision Functions - Logistic Regression

![](../../.gitbook/assets/image%20%28370%29.png)

### 2.4 Decision Functions - Perceptron

![](../../.gitbook/assets/image%20%28534%29.png)

## 3. Neural Network

### 3.1 Example: Neural Network with 1 Hidden Layer

![](../../.gitbook/assets/image%20%28142%29.png)

### 3.2 Decision Boundary Examples

See Lecture Slides

Main idea: Multi-layer NN with non-linear activation can fit tricky data distributions well

### 3.3 Architectures

Even for a basic Neural Network, there are many design decisions to make

1. \# of hidden layers \(depth\)
2. \# of units per hidden layer \(width\)
3. Type of activation function \(nonlinearity\)
4. Form of objective function

#### 3.3.1 How many layers should we use?

![](../../.gitbook/assets/image%20%2894%29.png)

### 3.4 Different Levels of Abstraction

![](../../.gitbook/assets/image%20%28754%29.png)

### 3.5 Activation Functions

1. tanh: Like logistic function but shifted to range \[-1, +1\]
2. rectified linear unit \(ReLU\):
   1. Hard version: Linear with a cutoff at zero
      * Implementation: clip the gradient when you pass zero
   2. Soft version: log\(exp\(x\)+1\)
      * Doesn’t saturate \(at one end\)
      * Sparsifies outputs Helps with vanishing gradient

### 3.6 Objective Functions for NNs

![](../../.gitbook/assets/image%20%28551%29.png)

Note: Softmax Loss

![](../../.gitbook/assets/image%20%28380%29.png)

### 3.7 Multi-Class Output

![](../../.gitbook/assets/image%20%28265%29.png)

### 3.8 Neural Networks Objectives

1. Explain the biological motivations for a neural network
2. Combine simpler models \(e.g. linear regression, binary logistic regression, multinomial logistic regression\) as components to build up feed-forward neural network architectures
3. Explain the reasons why a neural network can model nonlinear decision boundaries for classification
4. Compare and contrast feature engineering with learning features
5. Identify \(some of\) the options available when designing the architecture of a neural network
6. Implement a feed-forward neural network

## 4. Backpropagation

### 4.1 Approaches to Differentiation

![](../../.gitbook/assets/image%20%28840%29.png)

### 4.2 Finite Difference Method

![](../../.gitbook/assets/image%20%28245%29.png)

### 4.3 Symbolic Differentiation and Chain Rule

![](../../.gitbook/assets/image%20%2841%29.png)

### 4.4 Backpropagation

![](../../.gitbook/assets/image%20%2856%29.png)

![](../../.gitbook/assets/image%20%28397%29.png)

![](../../.gitbook/assets/image%20%283%29.png)

![](../../.gitbook/assets/image%20%28489%29.png)

![](../../.gitbook/assets/image%20%28488%29.png)

![](../../.gitbook/assets/image%20%2813%29.png)

![](../../.gitbook/assets/image%20%2898%29.png)

### 4.5 Backpropagation Objectives

1. Construct a computation graph for a function as specified by an algorithm
2. Carry out the backpropagation on an arbitrary computation graph
3. Construct a computation graph for a neural network, identifying all the given and intermediate quantities that are relevant
4. Instantiate the backpropagation algorithm for a neural network
5. Instantiate an optimization method \(e.g. SGD\) and a regularizer \(e.g. L2\) when the parameters of a model are comprised of several matrices corresponding to different layers of a neural network
6. Apply the empirical risk minimization framework to learn a neural network
7. Use the finite difference method to evaluate the gradient of a function
8. Identify when the gradient of a function can be computed at all and when it can be computed efficiently

## 5. Summary

Neural Networks

1. provide a way of learning features
2. are highly nonlinear prediction functions
3. \(can be\) a highly parallel network of logistic regression classifiers
4. discover useful hidden representations of the input

Backpropagation

1. provides an efficient way to compute gradients
2. is a special case of reverse-mode automatic differentiation

