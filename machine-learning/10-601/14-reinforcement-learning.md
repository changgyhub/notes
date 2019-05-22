# 14 Reinforcement Learning

## 1. Learning Paradigms

1. Supervised
   1. Regression
   2. Classification
   3. Binary Classification
   4. Structured Prediction
2. Unsupervised
3. Semi-supervised
4. Online
5. Active Learning
6. Reinforcement Learning

## 2. Markov Decision Process

### 2.1 What is special about RL?

1. RL is learning how to map states to actions, so as to maximize a numerical reward over time
2. Unlike other forms of learning, it is a multistage decision-making process \(often Markovian\)
3. An RL agent must learn by trial-and-error \(Not entirely supervised, but interactive\)
4. Actions may affect not only the immediate reward but also subsequent rewards \(Delayed effect\)

### 2.2 Elements of RL

1. A policy
   * A map from state space to action space
   * May be stochastic
2. A reward function
   * It maps each state \(or, state-action pair\) to a real number, called reward
3. A value function
   * Value of a state \(or, state-action pair\) is the total expected reward, starting from that state \(or, state-action pair\)

### 2.3 The Precise Goal

1. To find a policy that maximizes the Value function
   * transitions and rewards usually not available
2. There are different approaches to achieve this goal in various situations
3. Value iteration and Policy iteration are two more classic approaches to this problem. But essentially both are dynamic programming
4.  Q-learning is a more recent approaches to this problem. Essentially it is a temporal-difference method

### 2.4 Markov Decision Process

#### 2.4.1 Components

1. $$\mathcal{S}$$ :set of states
2. $$\mathcal{A}$$ :set of actions
3. $$R(s, a)$$: reward
4. $$p(s'|s,a)$$

#### 2.4.2 Model

1. Start in state $$s_0$$
2. At each time $$t$$, agent observes $$s_t$$ then choose $$a_t$$ then receives $$r_t$$ and change to $$s_{t+1} \sim p(\cdot| s_t, a_t)$$
3. Total payoff is $$r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots$$, where $$\gamma$$ is the discount factor

#### 2.4.3 Goal

Learn a policy $$\pi: \mathcal{S} \rightarrow \mathcal{A}$$ for choosing actions to maximize "infinite-horizon discounted reward": $$\mathbb{E}[r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots] = \sum_{t=0}^{\infty} \gamma^t \mathbb{E}[r_t]$$

* The finite-horizon discounted reward is $$\sum_{t=0}^{h} \gamma^t \mathbb{E}[r_t]$$ or $$\sum_{t=0}^{h}\mathbb{E}[r_t]$$ which can prevent non-convergence
* Average reward is $$\lim_{h\rightarrow \infty} \frac{1}{h} \sum_{t=0}^h \mathbb{E}[r_t]$$

### 2.5 Exploration vs. Exploitation Tradeoff

Example: k-Armed Bandit Problem

1. $$|\mathcal{S}| = 1$$
2. $$|\mathcal{A}| = k$$
3. $$\mathcal{R}$$ is nondeterministic
4. Always transition to same state

### 2.6 Fixed Point Iteration for Optimization \(Should Not Use Since Slow\) 

![](../../.gitbook/assets/image%20%2843%29.png)

## 3. Value Iteration

### 3.1 State trajectory

Given $$s_0, \pi, p(s_{t+1}|s_t, a_t)$$, there exists a distribution over state trajectories of $$s_0 \rightarrow_{a_0} s_1 \rightarrow_{a_1} s_2 \rightarrow_{a_2} \cdots $$

### 3.2 The value function \(expected future disordered reward\)

$$
V^\pi (s) =\mathbb{E}_{\pi, p( \cdot | s, a)} [R(s_0, a_0) + \gamma R(s_1, a_1)  + \gamma^2 R(s_2, a_2) + \cdots | s_0]\\[6pt]
= R(s_0, a_0) + \gamma \mathbb{E}_{\pi, p(\cdot|s, a)} [ R(s_1, a_1) + \gamma R(s_2, a_2) + \cdots | s_0]\\[6pt]
= R(s_0, a_0) + \gamma \sum_{s_1 \in S} \left( p(s_1|s_0, a_0) R(s_1, a_1)+ \gamma \mathbb{E}_{\pi, p(\cdot|s, a)}[R(s_2, a_2) + \cdots | s_1] \right)
$$

### 3.3 Bellman Equations

$$
V^\pi (s) = r(s, \pi(s)) + \gamma \sum_{s' \in S} p(s'|s, \pi(s)) V^\pi (s')
$$

### 3.4 Optimal Policy and  Value Function

$$
\pi^* = \arg \max_{\pi} V^\pi (s), \forall s \in S\\
V^* \triangleq V^{\pi^*}
$$

### 3.5 Computing the Optimal Policy

$$
\pi^*(s) = \arg \max_{a \in \mathcal{A}} \left( R(s, a) + \gamma \sum_{s' \in S} p(s'|s, a)V^*(s')\right)
$$

where $$a^*$$is the best action, $$R(s, a)$$ is immediate reward, $$\gamma \sum_{s' \in S} p(s'|s, a)V^*(s')$$ is the expected discounted reward.

### 3.6 Value Iteration Algorithm

![](../../.gitbook/assets/image%20%28306%29.png)

### 3.7 Async vs Sync Update

![](../../.gitbook/assets/image%20%28205%29.png)

### 3.8 Convergence

![](../../.gitbook/assets/image%20%28793%29.png)

## 4. Policy Iteration

### 4.1 Policy Iteration Algorithm

![](../../.gitbook/assets/image%20%289%29.png)

### 4.2 Policy Iteration Convergence

How many policies are there for a finite sized state and action space? $$|A|^{|S|}$$

Suppose policy iteration is shown to improve the policy at every iteration. Can you bound the number of iterations it will take to converge? $$ n \le |A|^{|S|}$$

### 4.3 Value Iteration vs. Policy Iteration

Value iteration requires $$O(|A| |S|^2)$$ computation per iteration

Policy iteration requires $$O(|A| |S|^2 + |S|^3)$$ computation per iteration

* In practice, policy iteration converges in fewer iterations

## 5. Value and Policy Iteration Learning Objectives

1. Compare the reinforcement learning paradigm to other learning paradigms
2. Cast a real-world problem as a Markov Decision Process
3. Depict the exploration vs. exploitation tradeoff via MDP examples
4. Explain how to solve a system of equations using fixed point iteration
5. Define the Bellman Equations
6. Show how to compute the optimal policy in terms of the optimal value function
7. Explain the relationship between a value function mapping states to expected rewards and a value function mapping state-action pairs to expected rewards
8. Implement value iteration
9. Implement policy iteration
10. Contrast the computational complexity and empirical convergence of value iteration vs. policy iteration
11. Identify the conditions under which the value iteration algorithm will converge to the true value function
12. Describe properties of the policy iteration algorithm

## 6. Q-Learning

### 6.1 Motivation: What if we have zero knowledge of the environment?

If $$R(s, a)$$ and/or $$p(s'|s, a)$$ are not available, then value iteration and policy iteration do not work. We can use Q-learning instead.

### 6.2 Q-Function: Expected Discounted Reward

Let $$Q^*(s, a)$$ be the expected discount reward for taking $$a$$ at $$s$$, then we have

$$
V^*(s) = \max_a Q^*(s, a)\\[3pt]
Q^*(s,a) = R(s, a) + \gamma \sum_{s'} p(s'|s, a) V^*(s) \Rightarrow \text{}\\
Q^*(s,a) = R(s, a) + \gamma \sum_{s'} p(s'|s, a) \max_{a'} Q^*(s', a')\\
\pi^*(s) = \arg\max_a Q^*(s, a)
$$

If we can learn $$Q^*$$, we can define $$\pi^*$$ without $$R(s, a)$$ and $$p(s'|s, a)$$.

### 6.3 Example: Robot Localization

![](../../.gitbook/assets/image%20%28257%29.png)

### 6.4 Case 1: Deterministic Environment

Assume $$p(s'|s, a) = 1$$ if $$\delta(s, a) = s'$$ else 0. Then we have

$$
Q^*(s,a) = R(s, a) + \gamma \sum_{s'} \max_{a'} Q^*(\delta(s, a), a')
$$

The algorithm is

1. Initialize $$Q(s,a) = 0 ~~\forall s, a$$
2. Do forever
   1. Select action $$a$$ and execute
   2. receive reward $$r = R(s,a)$$
   3. observe new state $$s' = \delta (s,a)$$
   4. update table entry $$Q(s,a) = r + \gamma \max_{a'} Q(s', a')$$

$$\epsilon$$-greedy variant: at step 2.2

* with probability $$1-\epsilon$$ select an action $$a = \max_a Q(s, a')$$
* with probability $$\epsilon$$ select action $$a$$ at random

### 6.5 Case 2: Nondeterministic Environment

Let $$K_n(s,a)$$ be the number of visits of $$(s,a)$$ at time $$n$$.When $$p(s'|s, a)$$ is stochastic, the algorithm is

1. Initialize $$Q(s,a) = 0 ~~\forall s, a$$
2. Do forever
   1. Select action $$a$$ and execute
   2. receive reward $$r = R(s,a)$$
   3. observe new state $$s' \sim p(\cdot|s,a)$$
   4. update table entry $$Q(s, a) = (1 - \alpha_n)Q(s, a) + \alpha_n (r + \gamma \max_{a'} Q(s', a'))$$, where $$\alpha_n = 1/(1 + K_n(s,a))$$

It is essentially weighted average of Q-value: the more we update $$Q(s,a)$$, the less we will take new Q-values into account. It will make Q-values less sensitive to noise in the stochastic state transition probability.

Remarks:

1. $$Q$$converges to $$Q^*$$with probability 1 under certain assumptions \(e.g. visit each state infinitely often\)
2. Q-learning is exploration-insensitive. That is, any visitation strategy \(step 2.2\) will work as long as it has certain properties \(e.g. visit each state infinitely often\)
3. Usually needs many thousands of iterations or more to converge.
4. For visitation $$\langle s, a, r, s' \rangle$$, we can store them and use in any order \(e.g. experience replay\)

## 7. Q-Learning Learning Objectives

1. Apply Q-Learning to a real-world environment
2. Implement Q-learning
3. Identify the conditions under which the Q-learning algorithm will converge to the true value function
4. Adapt Q-learning to Deep Q-learning by employing a neural network approximation to the Q function
5. Describe the connection between Deep Q- Learning and regression

