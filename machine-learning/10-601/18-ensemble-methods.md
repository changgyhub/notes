# 18 Ensemble Methods

## 1. Weighted Majority Algorithm

### 1.1 Example: Recommender Systems

500,000 users, 20,000 movies, 100 million ratings.

Goal: To obtain lower root mean squared error \(RMSE\) than Netflix’s existing system on 3 million held out ratings

![](../../.gitbook/assets/image%20%28190%29.png)

### 1.2 Weighted Majority Algorithm

![](../../.gitbook/assets/image%20%28761%29.png)

![](../../.gitbook/assets/image%20%28712%29.png)

### 1.3 Mistake Bound

![](../../.gitbook/assets/image%20%28131%29.png)

## 2. AdaBoost

### 2.1 Comparison

![](../../.gitbook/assets/image%20%28368%29.png)

### 2.2 AdaBoost

![](../../.gitbook/assets/image%20%28415%29.png)

### 2.3 AdaBoost: Toy Example

![](../../.gitbook/assets/image%20%28192%29.png)

![](../../.gitbook/assets/image%20%28238%29.png)

![](../../.gitbook/assets/image%20%28730%29.png)

![](../../.gitbook/assets/image%20%28281%29.png)

![](../../.gitbook/assets/image%20%28670%29.png)

### 2.4 Error curves

Training error overfits, but test error seems continue to go down. It tends no to overfit.

![](../../.gitbook/assets/image%20%28665%29.png)

## 3. Ensemble Methods / Boosting Learning Objectives

1. Implement the Weighted Majority Algorithm
2. Implement AdaBoost
3. Distinguish what is learned in the Weighted Majority Algorithm vs. Adaboost
4. Contrast the theoretical result for the Weighted Majority Algorithm to that of Perceptron
5. Explain a surprisingly common empirical result regarding Adaboost train/test curves

