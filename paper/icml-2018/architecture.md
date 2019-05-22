# Architecture

## 1. Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples

Best Paper

### 简介

模糊梯度（obfuscated gradients）作为一种梯度掩码（grading masking），可以用来对抗迭代优化式（iterative optimization-based）的反例攻击，这是由于它们会造成不成功的loss优化，导致无法获得最优解。目前的模糊梯度方法可以分为三种：shattered gradients（由于有意的不可导操作、或者无意的数值上的不稳定性，造成的不存在或者不正确的梯度）、stochastic gradients（测试时的网络参数的随机，或输入的随机造成的）和vanishing/exploding gradients（由于过深的计算导致的不可用的梯度，通常是通过多迭代输入实现的，即网络的输出再次作为输入）。本文提出了克服这三种情况的方法：通过正常计算正向pass、同时模拟一个可微分的反向pass来近似微分；通过Expectation Over Transformation来克服随机防御；通过reparameterization和在梯度不爆炸或不消失的空间上优化，来克服梯度爆炸和消失。本文成功攻破了7个ICLR论文的反例防御。

### 算法

攻击样本（adversarial examples）的定义为，对一个训练集的数据x，它的反例x'需满足，在某距离标准D下，D\(x, x'\)足够小且 x != x'。

* 训练准备：本文采用L2和L∞ norm作为衡量距离D，使用MNIST，CIFAR-10，和一部分的ImageNet做测试，网络分别选用了流行的AlexNet，ResNet和InceptionV3。
* Threat Model：包含有攻击样本的威胁模型（threat model）通常可以被分类成白箱和黑箱。本文采用白箱模型，即攻击样本完全知道网络的参数和防御措施，知道测试集的统计分布，但不知道测试集的随机数生成法。
* Attack Methods：本文通过迭代优化生成攻击样本，在某个high level c上，我们希望取得一个很小的dealta，使得c\(x+delta\)差异于c\(x\)；或者等价地，最大化x+delta的loss。为了生成L∞距离上的攻击样本，本文采用Projected Gradient Descent \(PGD\)去限制一个L∞球；为了生成L2距离上的攻击样本，本文采用Lagrangian relaxation of Carlini & Wagner。

梯度掩码防御一般有如下的几个特点：

1. 一步完成的攻击比迭代式攻击要好：因为迭代式攻击的效果应该严格大于一步完成的攻击，所以当一步完成的攻击比迭代式攻击好的时候，很有可能迭代陷入了极小值。
2. 黑箱比白箱好：因为白箱的效果应该严格大于黑箱，所以当黑箱比白箱效果好的时候，很有可能遇到了模糊梯度。
3. 无约束的攻击未达到100%成功率：在没有无约束扭曲（unbounded distortion）下，分类器理应对于攻击无任何抵抗力。如果未达到100%成功率，很有可能没有攻破防御。
4. 随机采样的过程中发现了攻击样本：如果在样本附近的球状区域做一个暴力扰动搜索时能够发现攻击样本，则说明很可能有梯度掩码防御。

针对梯度掩码防御的三种情况，可以分别作出如下对抗：

#### 1. 对抗shattered gradients - 反向微分模拟（Backward Pass Differentiable Approximation，BPDA）

目前的防御策略大致为，对于一个训练好的分类器f，制造一个预处理器g，来形成一个安全分类器f~\(x\) = f\(g\(x\)\)，且g\(x\)约等于x。如果g是不平滑不可导的，那么就不能通过反向求导产生攻击样本。在这种情况下，本文用x的导数替代g\(x\)的导数，即将g\(x\)换成identity function。更直接的，f\(g\(x\)\)的导数可以用f\(x\)在g\(x\)点的导数替代。虽然这样的替代不是很准确，但是由于多次迭代优化的平均效应，最终能获得很好的攻击样本。BPDA对此进一步延伸，通过学习一个网络层来近似这个预处理器，正向的时候过安全分类器的网络层，反向的时候过近似层，从而模拟反向微分。

#### 2. 对抗randomized classifiers - Expectation over Transformation（EOT）

如果网络对于输入做了随机扰动t，EOT的目标是最优化期望值E\(f\(t\(x\)\)\)。这个优化目标可以通过梯度下降来完成，因为E\(f\(t\(x\)\)\)的梯度等于f\(t\(x\)\)的梯度的期望值。

#### 3. 对抗vanishing/exploding - Reparameterization

对于一个训练好的分类器f\(g\(x\)\)，若g是一个带有loop的优化器，经过迭代后梯度很容易弥散或消失。Reparameterization的解决办法是，训练一个可导的函数h，使得g\(h\(x\)\)近似h\(x\)，这样就可以避免梯度问题了。



