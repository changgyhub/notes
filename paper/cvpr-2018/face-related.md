# Face Related

## 1. Finding Tiny Faces in the Wild with Generative Adversarial Network

Oral

### 简介

本文主要解决无限制场景下的微小人脸检测问题，tiny face指尺寸小，分辨率低，模糊并缺乏必要信息的人脸patch，tiny face检测问题如图1所示。通过利用GAN模型对tiny face进行高分辨和去模糊的人脸生成，在通过判别器来进行识别。生成网络包含super-resolving和refining两个过程，这两个过程分别解决分辨率低和人脸模糊两个问题，分别通过借鉴SR-GAN和cycle-GAN。

![&#x56FE;1](../../.gitbook/assets2/image%20%2861%29.png)

### 算法

整体网络结构主要包括MB-FCN和GAN两部分，如图2所示。本文采用MB-FCN检测器作为基准检测模型，训练时为GAN网络提供样本，测试时为GAN提供ROI。GAN网络使用生成器和判别器构成。生成器网络由两个sub-network，一个是up-sample sub-network，另一个是refinement sub-network，其目的是生成高分辨率和去模糊的人脸patch。判别器网络有两个判别任务，一个是人脸是否是高分辨，另一个是patch是否是人脸。

![&#x56FE;2](../../.gitbook/assets2/image%20%2832%29.png)

#### 1. Generator network

由于tiny face缺乏细节信息和重建误差MSE loss的影响\(保留低频信息，丢失高频信息，这是由误差函数导致的\)，生成的超分辨人脸往往比较模糊。需要设计refinement sub-network来去除模糊现象。up-sample sub-network中有两个de-conv layer用来做上采样，使分辨率提升4倍。refinement sub-network中除了最后一层，其余每个conv之后都有BN和Relu。

#### 2. Discriminator network

判别网络使用VGG作为backbone，同时去除conv5之后的max-pooling，替换全连接层fc6，fc7，fc8为两个并行的fc\_GAN和fc\_clc。

#### 3. Loss function

网络结构的多任务学习通过混合多个loss来实现。整体loss主要分为三个部分图像重构误差pixel-wise loss，GAN网络误差adversarial loss以及分类误差classification loss。

1. Pixel-wise loss：生成网络的输入是模糊的tiny face加上随机noise，后面加上一个与ground-truth高分辨人脸的MSE loss。
2. Adversarial loss：生成网络的损失，该loss引导图像生成尽可能多含有高频信息。
3. Classification loss 该loss是判别器的损失函数，为了区分人脸与非人脸，人脸样本主要包括高分辨人脸和有生成器生成的低分辨人脸。该loss可以使生成网络重建image更sharper。

整体Loss可以分为G和D两部分，如下：

![](../../.gitbook/assets2/image%20%2822%29.png)

### 实验结果

如图3所示，优于目前其他模型

![](../../.gitbook/assets2/image%20%2825%29.png)

## 2. Learning Face Age Progression: A Pyramid Architecture of GANs

Oral

### 简介

现有方案大多将年龄信息优先，而identity信息次之；换句话说，就是生成不同年龄的同时identity信息不能很好保留。本文在做人脸回归时，对面部信息和年龄变化分别建模，并且用一个金字塔形式的对抗器模拟细节性的年龄变化。

### 算法

整体框架如图2所示。注：在训练开始之前，人脸都根据眼睛的位置对齐好了，要么数据集有位置标注，要么可以使用Face++的API。

![&#x56FE;2](../../.gitbook/assets2/image%20%2896%29.png)

#### 1. Generator

是一个encoder-decoder架构，输入是年轻的脸。注意年龄是框架的超参数，以间隔年龄为10取样训练集，不输入网络。

#### 2. Discriminator

提取不同深度的feature得到特征金字塔，然后再把这些特征缩小并concat，预测出最终的输出。由于GAN原始的log loss是JS散度，会使得D收敛太快而回传到G的梯度几乎为零，作者用L2 loss，即Pearson Chi-square divergence，来取代原始的GAN loss。具体公式如下：

![](../../.gitbook/assets2/image%20%2859%29.png)

其中正样本标为1，负样本标为0；负样本既包含G合成的年老的脸G\(x\)，也包括真实年轻的脸x；而正样本只有真实年老的脸x。

#### 3. Identity Preservation Loss

使用一个deep face descriptor网络 $$\phi_{id}$$ ，分别输入真实年轻的脸x和合成的年老的脸G\(x\)，最小化他们的欧式距离：![](file:////Users/irsisyphus/Library/Group%20Containers/UBF8T346G9.Office/TemporaryItems/msohtmlclip/7BA4DF94-548C-D545-9A89-722E1BD142E7.png)

$$
\mathcal{L}_{identity} = \mathbb{E}_{x\in P_{young(x)}}[d(\phi_{id}(x), \phi_{id}(G(x)))]
$$

#### 4. Pixel Level Loss

对真实年轻的脸x和合成的年老的脸G\(x\)做一个pixel-level的L2 norm。

【思考：Identity Preservation Loss和perceptual loss很像，Pixel Level Loss是常见trick，没有觉得这个网络在loss上有什么特别大的新意。而且年龄竟然不是GAN的conditonal的输入，这样生成的结果没有办法控制年龄。不知道这样一篇新意不大且可提升空间很多的文章是怎么评上Oral的。】

### 实验结果

训练过程需要pretrain一个deep face descriptor网络，且不能自由地定义年龄。与FG-Net的比较也只是问卷调查，可信度值得商榷。

