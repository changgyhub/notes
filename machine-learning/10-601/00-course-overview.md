# 00 Course Overview

## Course Info

* **Instructor**: [Matt Gormley](http://www.cs.cmu.edu/~mgormley/)
* **Meetings**:
  * **10-601B:** MWF, 3:00 PM - 4:20 PM \(GHC 4401\)
  * **10-601D:** Same times as Section B \(online, or in GHC 4401 as seats permit\)
  * For all sections, lectures are on Mondays and Wednesdays.
  * Occasional recitations are on Fridays and will be announced ahead of time.
* **Assistant Instructors Email:** [10601-assistant-instructors@cs.cmu.edu](mailto:10601-assistant-instructors@cs.cmu.edu)
* **Piazza:** [https://piazza.com/cmu/fall2018/10601bd](https://piazza.com/cmu/fall2018/10601bd)
* **Autolab:** [https://autolab.andrew.cmu.edu/courses/10601b-f18](https://autolab.andrew.cmu.edu/courses/10601b-f18)
* **Gradescope:** [https://www.gradescope.com/courses/20983](https://www.gradescope.com/courses/20983)
* **Video:** See [Schedule](http://www.cs.cmu.edu/~mgormley/courses/10601bd-f18/schedule.html) page.
* **Office Hours Queue:** [https://cmu.ohqueue.com](https://cmu.ohqueue.com/)

## 1. General Questions

### 1.1 What is AI?

The basic goal of AI is to develop intelligent machines. This consists of many sub-goals:

* Perception
* Reasoning
* Control / Motion / Manipulation
* Planning
* Communication
* Creativity
* Learning

Machine Learning is a part of AI.

### 1.2 What is ML?

![](../../.gitbook/assets/image%20%28150%29.png)

Examples:

* Learning to recognize spoken words
* Learning to drive an autonomous vehicle
* Learning to beat the masters at board games
* Learning to recognize images
* In what cases and how well can we learn?

### 1.3 Topics

![](../../.gitbook/assets/image%20%28324%29.png)

### 1.4 ML Big Picture

![](../../.gitbook/assets/image%20%28420%29.png)

## 2. Defining Learning Problems

### 2.1 Well-Posed Learning Problems

**Three components &lt;T, P, E&gt;:**

1. Task, T
2. Performance measure, P. \(Time, \# moves, distance, score, winning probability, etc\)
3. Experience, E

**Definition of learning**: 

A computer program **learns** if its performance at tasks in T, as measured by P, improves with experience E.

### 2.2 Capturing the Knowledge of Experts

Solution \#1: Expert Systems

* ~30 years ago, rule based systems
* Example
  * If: “X directions”
  * Then: directions\(here, nearest\(X\)\)

Solution \#2: Annotate Data and Learn

* Very good at answering questions about specific cases
* Not very good at telling HOW they do it
* Method
  1. Collect raw sentences {x1, …, xn}
  2. Experts annotate their meaning {y1, …, yn}
* Example
  * x1: "How do I get to X?"
  * y1: directions\(here, nearest\(X\)\)
* &lt;T, P, E&gt; example on Siri:
  * Task, T: predicting action from speech
  * Performance measure, P: percent of correct actions taken in user pilot study
  * Experience, E: examples of \(speech, action\) pairs

### 2.3 Problem Formulation

Often, the same task can be formulated in more than one way

Example: Loan applications

* creditworthiness/score \(regression\)
* probability of default \(density estimation\)
* loan decision \(classification\)

### 2.4 Machine Learning & Ethics

What ethical responsibilities do we have as machine learning experts?

## 3. Big Ideas

1. How to formalize a learning problem
2. How to learn an expert system \(i.e. Decision Tree\)
3. Importance of inductive bias for generalization
4. Overfitting

## 4. Function Approximation

### 4.1 Example: Interpolation

Implement a simple function which returns sin\(x\): call existing implementation of sin\(x\) a few times to get some points, then when query new points, do weighted average of nearby solved points.

### 4.2  ML as Function Approximation

#### 4.2.1 Problem Setting

* Set of possible inputs $$X$$
* Set of possible outputs $$Y$$
* Unknown target function $$C^*: X \rightarrow Y$$
* Set of candidate hypothesis $$H = \{h | h: X \rightarrow Y\}$$

#### 4.2.2 Input

* Training examples $$D = {(x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \dots, (x^{(N)}, y^{(N)})}$$of unknown target function $$C^*$$, where $$y^{(i)} = C^*(x^{(i)})$$

#### 4.2.3 Output

* Hypothesis $$h\in H$$that best approximates $$C^*$$
* Loss function $$l : Y \times Y \rightarrow \mathbb{R}$$ measures how "bad" predictions $$\hat{y} = h(x)$$are compared to $$C^*(x)$$,

#### 4.2.4 Example: Loss functions

* Regression: $$y \in \mathbb{R}, l(\hat{y}, y) = (\hat{y}, y)^2$$ \(squared loss\)
* Classification: $$y \in \{+, -\}, l(\hat{y}, y) = 0 \text{ if } \hat{y} = y  \text{ else } 1$$

#### 4.2.5. Example Algorithm: Memorizer

Train with dataset $$D$$, and store dataset $$D$$. Test on data $$x$$, if there exists $$x^{(i)} \in D$$ such that $$x = x^{(i)}$$, the return $$y^{(i)}$$; otherwise pick a random $$y \in Y$$and return.



