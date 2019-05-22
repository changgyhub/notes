# Object Tracking

## 背景简介

Object tracking 的任务是给定第一帧的物体位置，需要系统给出各帧的物体位置， 需要注意的是，target理论上是class aganostic的，也就是可能属于之前从来没见过的物体类别，有着和detection本质的区别。

Object Tracking的主要难度在于：

1. 训练正样本少，理论上只有第一帧是supervision的信息
2. 物体appearance在不断变化，需要一定的adaptation
3. tracking过程中无监督， 当遇到被污染的target，比如occlusion，模型的adaptation可能使得模型被污染，发生drift现象

## 1. VITAL: Visual Tracking via Adversarial Learning

Spotlight

### 简介

本文主要分析了现有的检测式跟踪的框架在模型在线学习过程中的两个弊病，即：

1. 每一帧中正样本高度重叠，他们无法捕获物体丰富的变化表征；
2. 正负样本之间存在严重的不均衡分布的问题；

针对上述问题，本文提出 VITAL 这个算法来解决，主要思路如下：

1. 为了丰富正样本，作者采用生成式网络来随机生成mask，且这些mask作用在输入特征上来捕获目标物体的一系列变化。在对抗学习的作用下，作者的网络能够识别出在整个时序中哪一种mask保留了目标物体的鲁棒性特征；
2. 在解决正负样本不均衡的问题中，本文提出了一个高阶敏感损失来减小简单负样本对于分类器训练的影响。

### 算法

分为网络结构、不平衡优化和检测流程三部分。

#### 1. 网络结构

![](../../.gitbook/assets2/image%20%2890%29.png)

#### 2. 不平衡优化

本文提出两个方法来减少跟踪中的drift和正负训练样本不平衡的问题。

**Adversial Feature Generator**：输入特征图，用网络G生成一个spatial的mask, 来filter掉单帧的discriminative的信息，保留temporal robust的信息： $$C_{ijk}^o = C_{ijk} M_{ij}$$ ，其中C为feature map， M为生成的mask。Mask之后的特征将被传入一个二分类器 D，来分类该patch是前景\(target\)或背景。 在训练中，G与D互为adversial的关系进行训练，目的是让D尽量依赖temporal robust的信息进行分类，防止在单帧信息上过拟合。整合的Loss为

![](../../.gitbook/assets2/image%20%2863%29.png)

【分析： 个人认为，这个contribution不太靠谱，因为训练中并没有任何信息来约束mask的选取】

**Cost Sensitive Loss**：简单来说，为了减小简单样本的loss占的权重，防止简单的负样本overrun整个优化器，本文在正常的cross-entropy中增加了一个权重，由原来的

$$
\mathcal{L}(p, y) = - y \log(p) - (1-y) \log(1-p)
$$

变成

$$
\mathcal{L}(p, y) = - y (1-p) \log(p) - (1-y) p \log(1-p)
$$

#### 3. 检测流程

本文的baseline pipeline用的是[MDNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Nam_Learning_Multi-Domain_Convolutional_CVPR_2016_paper.pdf)，对各个target proposal做分类然后选取得分最高的。在第一帧和每t帧之后对模型做fine-tune来adapt。

### 实验结果

Ablation结果如图4所示

![&#x56FE;4](../../.gitbook/assets2/image%20%2847%29.png)

Benchmark结果如图7所示

![&#x56FE;7](../../.gitbook/assets2/image%20%2819%29.png)

### 存在问题

个人认为，虽然本文propose的adversial training的想法很好，但并没有说清为什么训练过程可以达到想要的效果。 另外，虽然结果很好，但很有可能是犯了和MDNet一样的问题，使用过于接近测试集的数据进行训练，导致overfit的情况存在。

