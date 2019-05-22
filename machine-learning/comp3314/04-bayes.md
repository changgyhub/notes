# 04 Bayes

Note：期望值的计算公式为$$E(x) = \int x f(x)~dx$$。

## 1. Bayes Theory

对于分类c，样本$$\bm{x}$$，通过贝叶斯定理可得，后验概率 \(posterior\)$$P(c|\bm{x})$$为

$$
P(c|\bm{x}) = \frac{P(c)P(\bm{x}|c)}{P(\bm{x})}
$$

其中$$P(c)$$是先验 \(prior\)，$$P(\bm{x}|c)$$是似然 \(likelihood\)，$$P(\bm{x})$$是证据因子 \(evidence\)。

## 2. Maximum Likelihood Estimation \(MLE\)

假设似然$$P(\bm{x}|c)$$被参数$$\bm{\theta}_c$$唯一确定，我们记$$P(\bm{x}|c)$$为$$P(\bm{x}|\bm{\theta}_c)$$。令$$D_c$$表示训练集$$D$$中第$$c$$类样本组成的集合，假设这些样本是独立同分布的，则$$\bm{\theta}_c$$对于$$D_c$$的似然是

$$
P(D_c|\bm{\theta}_c) = \prod_{\bm{x} \in D_c} P(\bm{x}|\bm{\theta}_c)
$$

我们希望得到使得似然值$$P(D_c|\bm{\theta}_c)$$最大的参数$$\hat\bm{\theta}_c$$，由于连乘容易造成下溢，使用对数似然得到

$$
LL(\bm{\theta}_c) = \log P(D_c|\bm{\theta}_c) = \log \sum_{\bm{x} \in D_c} P(\bm{x}|\bm{\theta}_c)\\
\hat\bm{\theta}_c = {\arg \max}_{\bm{\theta}_c} LL(\bm{\theta}_c)
$$

若$$P(\bm{x}|\bm{\theta}_c)$$可导，则一般情况下取$$\frac{\partial LL(\bm{\theta}_c)}{\partial \bm{\theta}_c} = 0$$即可得到$$\hat\bm{\theta_c}$$。

使用案例：根据样本来源得到的集合$$D$$，估计样本来源的均值和方差。

## 3. Maximum a Posteriori Estimation \(MAP\)

已知MLE的原始形式为

$$
\hat\bm{\theta}_{MLE} = {\arg \max}_{\bm{\theta}_c} P(D_c|\bm{\theta}_c)
$$

则优化后验的目标为

$$
\hat\bm{\theta}_{MAP} = {\arg \max}_{\bm{\theta}_c} P(\bm{\theta}_c|D_c) = {\arg \max}_{\bm{\theta}_c} \frac{P(D_c|\bm{\theta}_c) P(\bm{\theta}_c)}{P(D_c)} \\
\hat\bm{\theta}_{MAP}= {\arg \max}_{\bm{\theta}_c} P(D_c|\bm{\theta}_c) P(\bm{\theta}_c)
$$

## 4. Naïve Bayes

朴素贝叶斯被用来估计单个样本$$\bm{x}$$，假设$$\bm{x}$$有$$d$$个对分类结果独立影响的变量$${x_i}$$，对后验概率的贝叶斯公式变形则可得到

$$
P(c|\bm{x}) = \frac{P(c)P(\bm{x}|c)}{P(\bm{x})} =\frac{P(c)}{P(\bm{x})} \prod_{i = 1}^d P(x_i | c)
$$

由于对所有类别的证据因子$$P(\bm{x})$$是恒定值，最终的估计为

$$
\hat\bm{c} = {\arg \max}_c P(c) \prod_{i = 1}^d P(x_i | c)
$$

如果为离散属性，则可改写为

$$
\hat\bm{c}= {\arg \max}_c \frac{|D_c|}{|D|} \prod_{i = 1}^d \frac{|D_{c, x_i}|}{|D_c|}
$$

使用案例：根据样本集合$$D$$，估计测试集样本$$\bm{x}$$的分类。

## 4. EM Algorithm

令$$\bm{X}, \bm{Z}, \Theta$$分别表示已观测变量集合、隐变量集、模型参数。若对$$\Theta$$做极大似然，需要最大化$$LL(\Theta|\bm{X}, \bm{Z}) = \log P(\Theta|\bm{X}, \bm{Z} )$$。由于$$\bm{Z}$$是隐变量无法求解极大似然，可以对$$\bm{Z}$$计算期望，最大化已观测数据的对数“边际似然“ \(marginal likelihood\)：

$$
LL(\Theta|\bm{X}) = \log P(\bm{X}|\Theta) = \log \sum_Z P(\bm{X}, \bm{Z} | \Theta)
$$

于是我们初始化模型参数为$$\Theta_0$$，交替进行E步和M步：

Expectation：以当前参数$$\Theta^t$$推断隐变量分布$$P(\bm{Z} | \bm{X}, \Theta^t)$$，并计算对数似然$$LL(\Theta|\bm{X}, \bm{Z})$$关于$$\bm{Z}$$的期望

$$
Q(\Theta|\Theta^t) = \mathbb{E}_{\bm{Z} | \bm{X}, \Theta^t} LL(\Theta|\bm{X}, \bm{Z}) = P(\bm{Z} | \bm{X}, \Theta^t) LL(\Theta|\bm{X}, \bm{Z})\\
= \sum_Z P(\bm{Z} | \bm{X}, \Theta^t) \log P(\bm{X}, \bm{Z} | \Theta)
$$

Maximization：寻找最大化期望似然，即

$$
\Theta^{t+1} = {\arg \max}_{\Theta}Q(\Theta|\Theta^t)
$$

使用案例：高斯混合模型的优化

