# Architecture

## 1. A Variational U-Net for Conditional Appearance and Shape Generation

用于条件式生成外貌和形状的变分 U-Net，Spotlight

### 简介

深度生成模型在图像合成领域展现了优异的性能。然而，由于它们是直接生成目标的图像，而没有对其本质形状和外观之间的复杂相互影响进行建模，所以在空间转换时就会存在性能退化。我们针对形状指导图像生成提出了条件 U-Net，将变分自编码器输出的外观条件化。这个方法在图像数据集上进行端到端的训练，不需要同一个物体在不同的姿态或者外观下的采样。实验证明，这个模型能够完成条件图像生成和转换。所以，查询图像的外观或者形状能够被保留，同时能够自由地改变未被保留的另一个。此外，在保留形状的时候，由于外观的随机潜在表征，它可以被采样。在 COCO、 DeepFashion,、shoes、 Market-1501 以及 handbags 数据集上进行的定性和定量实验表明，我们的方法比目前最先进的方法都有所提升。

### 算法

生成目标的图像需要对它们的外观和空间布局的详细理解。因此，我们必须分辨基本的目标特征。一方面，与观察者视角相关的有目标的形状和几何轮廓（例如，一个人坐着、站着、躺着或者拎着包）。另一方面，还有由颜色和纹理为特征的本质外观属性（例如棕色长卷发、黑色短平头，或者毛茸茸的样式）。很明显，目标可以自然地改变其形状，同时保留本质外观（例如，将鞋子弄弯曲也不会改变它的样式）。然而，由于变换或者自遮挡等原因，目标的图像特征会在这个过程中发生显著变化。相反，衣服的颜色或者面料的变化对其形状是没有影响的，但是，它还是很清晰地改变了衣服的图像特征。

图 1展示了我们的变分 U-Net 模型学习从左边的查询中进行推理，然后生成相同外观的目标在第一行所示的不同姿态下的图像。

![&#x56FE;1](../../.gitbook/assets2/image%20%2810%29.png)

模型架构如图2。x是数据库里一张带有目标物的图片，y是它的形状，z是它的外表。

![&#x56FE;2](../../.gitbook/assets2/image%20%2880%29.png)

### 实验结果

下图是一个和pix2pix的比较。

![](../../.gitbook/assets2/image%20%285%29.png)

## 2. Practical Block-wise Neural Network Architecture Generation

BlockQNN自动网络设计方法，Oral

### 简介

近期的网络结构自动设计/搜索算法通常需要耗费巨大的计算资源，而且生成的模型可迁移性不强，难以做到真正的实用化。本文提出的BlockQNN算法能够解决现有网络结构自动设计/搜索方法效率和泛化性的问题。本文借鉴了现代主流深度神经网络的设计思想，比如ResNet、Inception等网络。这些网络是由同样结构的子网络重复组合在一起形成，本文把这种能重复组合的子结构称为block。通过设计block结构，可以让网络结构的搜索空间大大减小，并且block结构本身具有强大的泛化性，针对不同的数据集或者任务，只需要叠加不同个数的block即可完成。一个和其他模型的比较如图1所示。

![&#x56FE;1](../../.gitbook/assets2/image%20%2812%29.png)

### 算法

#### 1. 网络结构编码

为了表示网络block结构，本文设计了一套网络结构编码，把神经网络看做一个有向无环图，每个节点表示网络中的每一层，而边就表示数据流动的方向。整个编码包括神经网络的层数序号，类型，核的大小，以及两个前序节点的序号。使用这种编码方式就可以表示任意的神经网络结构，例如ResNet和Inception的block结构就能使用如图2所示的编码进行表示。

![&#x56FE;2](../../.gitbook/assets2/image%20%2846%29.png)

#### 2. 基于强化学习的网络结构自动设计

接下来的核心问题即是如何获得最优的网络结构。尽管网络结构的搜索空间已经通过设计block大大减小，但是直接暴力搜索所有可能结构，依然十分耗费计算资源。本文因此提出一种基于强化学习的网络设计方法，自动学习得到网络结构。

在网络设计强化学习中，本文把当前神经网络层定义为增强学习中的目前状态（current state），而下一层结构的决策定义为增强学习中的动作（action）。这里使用之前定义的神经网络结构编码来表示每一层网络。这样，通过一系列的动作决策，就能获得一条表示block结构的编码（如图4所示），而提出的强化学习算法通过优化寻获最优的动作决策序列。本文使用Q-learning算法来进行学习，具体的公式不再展开。

![&#x56FE;4](../../.gitbook/assets2/image%20%2840%29.png)

值得注意的一点是，与一般的强化学习问题不同，该任务只在结束整个序列的决策后（即生成完整网络结构后）才会得到一个reward，而之前的每个决策是对应reward。由于获得最终reward的成本非常高（需要在数据上重新训练新获得的网络结构），为了加快它的收敛，作者使用了reward shaping的技巧（r' = r/T，T为迭代数即层数），因而训练初始阶段终止层的Q值不会过高，让算法不会在训练初始阶段倾向于生成层数过浅的网络结构。

#### 3. 提前停止策略

虽然能够使用多种技巧来使自动化网络结构设计变的更加高效。但是自动网络设计中耗费时间的关键还是在于每次获得reward的时间成本非常高，需要将生成的网络结构在对应的数据集上训练至收敛，然后获得相应的准确度来表示结构的好坏并且用作reward。本文作者发现，通过调整学习率，只需要正常训练30分之一的过程（例如，CIFAR-100数据集上训练12个epoch），就可以得到网络的大致最终精度，这样可以大大降低时间成本。但是，这样的网络结构精度及其关联的reward会有误差，导致无法精细区分网络结构的优劣，本文提出一个凭经验的解决公式：

$$
reward = ACC_{\text{early stop}}−\mu \log⁡(\text{FLOPs})−\rho \log⁡(\text{Density})
$$

即真实的reward和提前停止的准确度成正比，但是和网络结构的计算复杂度和结构连接复杂度（block中边数除以点数）成反比。通过这样的公式矫正，得到的reward对网络结构的好坏更加具备可鉴别性。

### 实验结果

本文使用了32个GPU，经过3天的搜索，可以在CIFAR数据集上找到性能达到目前先进水平的网络结构。另一方面，学习获得的网络结构也可以更容易的迁移到ImageNet任务上，取得了不错的精度（浅层介于resent和xception之间，深层介于resnet和resnext之间）。对整个搜索过程和结果网络结构进行分析，作者发现学习得到的优异结构拥有一些共性。比如multi-branch结构、short-cut连接方式等这些现在常用的设计思想。同时，作者也发现了一些不太常见的结构共性，比如卷积层之间的addition操作出现的十分频繁，这些学习得到的网络结构还有待进一步的分析和研究。

## 3. Deep Layer Aggregation

Berkeley，Oral

### 简介

视觉识别任务中，往往同时需要底层和高层、小范围和大范围、以及低分辨率和高分辨率的信息，所以人们通常复合和汇聚网络中的多个部分来提升预测效果。虽然skip connections经常被用来融合层，但是这些链接自己本身就是shallow的（注：梯度反向传播经过一个聚合点便能传回到第一个subnetwork），且只能融合一步的简单结果。本文提出了更深的deep layer aggregation \(DLA\)来更好的融合不同层之间的信息，它iteratively and hierachically 汇聚特征结构来获得更高预测精度且更少参数的网络。

### 算法

当前创造网络的几个方向有更深、更宽、联系更紧密等等，且使用不同block来解决这些问题，如residual，bottleneck，gated等。作者认为，深度和宽度上的聚合（aggregation）是网络结构的一个核心。如图2所示，本文为DLA提出了两种架构，iterative deep aggregation \(IDA\) 和 hierarchical deep aggregation \(HDA\)：IDA侧重融合resolution和scale，而HDA侧重融合不同module和channel的信息；IDA从base hierarchy出发然后一步一步地优化resolution、聚合scale，而HDA在树状的连接里汇聚和交叉不同层的信息。例如Feature Pyramid属于IDA，DenseNet属于HDA。

![&#x56FE;2](../../.gitbook/assets2/image%20%2824%29.png)

其中IDA的公式如下

![](../../.gitbook/assets2/image%20%2879%29.png)

HDA的公式如下

![](../../.gitbook/assets2/image%20%2884%29.png)

在DLA中，IDA和HDA可以一同使用，如图3：

![&#x56FE;3](../../.gitbook/assets2/image%20%2844%29.png)

IDA也可以和插值一起用，做high resolution相关的模型，如图4：

![&#x56FE;4](../../.gitbook/assets2/image%20%2886%29.png)

### 实验结果

在分类、细颗粒度识别、语义分割、边框检测等任务，可以大大减少参数，并达到和resnext差不多的效果。

## 4. Detail-Preserving Pooling in Deep Networks

Oral

### 简介

max pooling只选取最大而忽略与周围像素的关联性，avg pooling重视关联性却又直接抹平，并且在实际梯度计算中也有一些drawback。本文提出了一个新方法，在池化过程中学一个动态的weight。

### 算法

本文提出了保留细节的下采样方法和池化方法：

#### 1. Detail-Preserving Image Downscaling

对于图像I，在每个像素p上，先做box filter，然后下采样，再用高斯滤波（可不加），得到线性下采样图像I~。最后对I和I～做inverse bilateral filter得到输出O。这么做的启发是，小细节相比大的同色区域，运输了更多的信息。流程如图2所示。

![&#x56FE;2](../../.gitbook/assets2/image%20%2895%29.png)

其中inverse bilateral filter为

$$
O[p] = \frac{1}{k_p}\sum_{q \in \Omega_p} I[q]||I[q] - \tilde{I}[p]||^\lambda
$$

目的是激励而并非惩罚两个输入的差距。简而言之，DPID计算的是输入的weighted average，周围的像素中，差越大的值能够给最终的输出更高的contribution。

#### 2. Detail-Preserving Pooling

基于DPID，本文进一步提出了DPP。简而言之，就是将上部分中的L2 NORM替换成一个可学习的 generic scalar reward function。

$$
\mathcal{D}_{\alpha, \lambda}(I)[p] = \frac{1}{\sum_{q' \in \Omega_p} w_{\alpha, \lambda}[p, q']}\sum_{q \in \Omega_p} w_{\alpha, \lambda}[p,q]I[q],
$$

$$
w_{\alpha, \lambda}[p,q] = \alpha + \rho_\lambda(I[q] - \tilde{I}[p])
$$

其中$$\rho$$是激励函数，有两种选择：

$$
\rho_{Sym}(x) = (\sqrt{x^2 + \epsilon^2})^{\lambda}, \rho_{Asym}(x) = (\sqrt{\max(0, x)^2 + \epsilon^2})^{\lambda}
$$

后者更倾向与给比中心像素高的像素更高权重，前者则是给差距大的更高权重。注意公式里的I～不同于DPID，是由一个2D filter F处理的，具体公式为$$\tilde{I}_F[p]=\sum_{q \in \tilde{\Omega_p}} F[q]I[q]$$，有三种F的选择：

1. Full-DPP：F是一个学得的、未正则化过的filter，kernal为3x3，即使用3x3的区域；
2. Lite-DPP：F是一个简单的box filterF，即$$F(p) = 1/|\Omega_p|$$，大小变为2x2；
3. Stochastic spatial sampling DPP \(S3DPP\)：先采用stride = 1进行Full-DPP或者Lite-DPP，然后均匀取样（uniform sampling）得到下采样的输出。

### 实验结果：

在CIFAR10上，Lite-S3DPP-Sym的效果最好，能降低0.8%左右的错误率。在ImageNet上，Lite-S3DPP-Sym也可略微降低错误率。

## 5. Squeeze-and-Excitation Networks

Oral

### 简介

卷积神经网络建立在卷积运算的基础上，通过融合局部感受野内的空间信息和通道信息来提取信息特征。为了提高网络的表示能力，许多现有的工作已经显示出增强空间编码的好处。在这项工作中，我们专注于通道，并提出了一种新颖的架构单元，我们称之为“Squeeze-and-Excitation”（SE）块，通过显式地建模通道之间的相互依赖关系，自适应地重新校准通道式的特征响应。通过将这些块堆叠在一起，我们证明了我们可以构建SENet架构，在具有挑战性的数据集中可以进行泛化地非常好。关键的是，我们发现SE块以微小的计算成本为现有的最先进的深层架构产生了显著的性能改进。SENets是我们ILSVRC 2017分类提交的基础，它赢得了第一名，并将top-5错误率显著减少到2.251%，相对于2016年的获胜成绩取得了∼25%的相对改进。

### 算法

由于一般卷积层的输出是通过所有通道的和来产生的，所以通道依赖性被隐式地嵌入到滤波器中，但是这些依赖性与滤波器捕获的空间相关性纠缠在一起。我们的目标是确保能够提高网络对信息特征的敏感度，以便后续转换可以利用这些功能，并抑制不太有用的功能。我们建议通过显式建模通道依赖性来实现这一点，以便在进入下一个转换之前通过两步重新校准滤波器响应，两步为：squeeze和excitation。

SE构建块的基本结构如图1所示。对于任何给定的变换$$F_{tr}$$，例如卷积或一组卷积，我们可以构造一个相应的SE块来执行特征重新校准。特征U首先通过squeeze操作，该操作跨越空间维度W×H聚合特征映射来产生通道描述符。这个描述符嵌入了通道特征响应的全局分布，使来自网络全局感受野的信息能够被其较低层利用。这之后是一个excitation操作，其中通过基于通道依赖性的自门（self-gating）机制为每个通道学习特定采样的激活，控制每个通道的激励。然后特征映射U被重新加权以生成SE块的输出，然后可以将其直接输入到随后的层中。

![&#x56FE;1](../../.gitbook/assets2/image%20%2823%29.png)

#### 1. Squeeze - 全局信息嵌入

为了解决利用通道依赖性的问题，我们首先考虑输出特征中每个通道的信号。每个学习到的滤波器都对局部感受野进行操作，因此变换输出U的每个单元都无法利用该区域之外的上下文信息。在网络较低的层次上其感受野尺寸很小，这个问题变得更严重。为了解决这一问题，我们简单地采用全局平均池化得到每一个通道的属性，也可以用其他的办法。

#### 2. Excitation - 自适应重新校正

为了利用压缩操作中汇聚的信息，我们接下来通过第二个操作来全面捕获通道依赖性。为了实现这个目标，这个功能必须符合两个标准：第一，它必须是灵活的（特别是它必须能够学习通道之间的非线性交互）；第二，它必须学习一个非互斥的关系，因为我们想保证多个通道可以同时被激活。为了满足这些标准，我们选择采用一个简单的门机制，并使用sigmoid激活：

$$
s = F_{ex}(z, W) = \sigma(g(z, W)) = \sigma(W_2 \delta(W_1 z))
$$

其中$$\delta$$是指ReLU函数，W1和W2为两个FC层，W1大小为$$C \times r / C$$，W2大小为$$C \times C / r$$，其中r为降维比例（实验结果得出，r=16比较好）。块的最终输出通过重新调节带有激活的变换输出U得到：

$$
\tilde{x_c} = F_{scale}(u_c, s_c) = s_c u_c
$$

#### 3. 模型示例 - SE-Inception和SE-ResNet

如图2、3所示

![&#x56FE;2](../../.gitbook/assets2/image%20%2813%29.png)

![&#x56FE;3](../../.gitbook/assets2/image%20%2834%29.png)

### 实验结果

以ResNet为例，在ResNet-50上加入SE只会增加很少的运算量，却能得到媲美ResNet-101的ImageNet准确率。

在ILSVRC 2017时，我们提出了SENets。我们的获胜输入由一小群SENets组成，它们采用标准的多尺度和多裁剪图像融合策略，在测试集上获得了2.251%的top-5错误率。这个结果表示在2016年获胜输入（2.99%的top-5错误率）的基础上相对改进了∼25%。我们的高性能网络之一是将SE块与修改后的ResNeXt集成在一起构建的（附录A提供了这些修改的细节）。我们的模型在每一张图像使用224×224中间裁剪评估（短边首先归一化到256）取得了18.68%的top-1错误率和4.47%的top-5错误率。为了与以前的模型进行公平的比较，我们也提供了320×320的中心裁剪图像评估，在top-1\(17.28%\)和top-5\(3.79%\)的错误率度量中获得了最低的错误率。

## 6. Convolutional Neural Networks with Alternately Updated Clique

Oral

### 简介

相比DenseNet等参数大的模型，在 CliqueNet 中，每个 Clique Block 只有固定通道数的特征图会馈送到下一个 Clique Block，这样就大大增加了参数效率。CliqueNet 最大的特点是其不仅有前传的部分，同时还能根据后面层级的输出对前面层级的特征图做优化。这种网络架构受到了循环结构与注意力机制的启发，即卷积输出的特征图可重复使用，经过精炼的特征图将注意更重要的信息。在同一个 Clique 模块内，任意两层间都有前向和反向连接，这也就提升了深度网络中的信息流。

### 算法

#### 1. Clique Block

CliqueNet 的每一个模块可分为多个阶段，但更高的阶段需要更大的计算成本，因此该论文只讨论两个阶段。第一个阶段如同 DenseNet 那样传播，这可以视为初始化过程。而第二个阶段如图1所示，每一个卷积运算的输入不仅包括前面所有层的输出特征图，同样还包括后面层级的输出特征图。第二阶段中的循环反馈结构会利用更高级视觉信息精炼前面层级的卷积核，因而能实现空间注意力的效果。

![&#x56FE;1](../../.gitbook/assets2/image%20%2842%29.png)

由此可以看出，每一步更新时，都利用最后得到的几个特征图去精炼相对最早得到的特征图。因为最后得到的特征图相对包含更高阶的视觉信息，所以该方法用交替的方式，实现了对各个层级特征图的精炼。

具体来说，对于第二阶段中的第 i 层和第 k 个循环，交替更新的表达式为：

$$
X_i^{(k)} = g(\sum_{l<i}W_{li}*X_l^{(k)}+\sum_{i<m}W_{mi}*X_m^{(k-1)})
$$

#### 2. CliqueNet

基于Clique Block的Stage-I和Stage-II，可以得到用注意力机制精修并且吸收了高级视觉信息的的Stage-II feature。论文中作者们采用的多尺度策略是：首先把各个Block的输入和Stage-II feature连结，再经过池化之后构成Block feature，最后所有的Block feature连结起来构成最终的final represent，如图2所示。

![&#x56FE;2](../../.gitbook/assets2/image%20%2854%29.png)

其中Translation的设定延续自DenseNet。用这种方法得到的final represent即为多尺度特征图，并且各个block的维度并没有激增。由于更高阶段显著增加计算量和模型复杂度，所以该论文的作者们仅考虑了前两个阶段。

#### 3. 其他技术

CliqueNet还运用了其他技术，如

1. Squeeze and Excitation，对特征层加attention
2. Bottleneck and compression，进一步缩小参数量

### 实验结果

在Cifar-10和Cifar-100上明显低于DenseNet和其他压缩循环式的网络。

## 7. StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation

Oral

### 简介

CycleGAN等模型不能解决多领域迁移的问题，对于多个数据集，或者多种属性的互相转化，就需要O\(n^2\)的生成器，而且无法利用他们之间的共同信息。本文提出了StarGAN来处理多个domain之间互相生成图像的问题，只需要一个生成器。

### 算法

首先要解释一下domain的定义，这里的domain是根据数据集中的attribute来划分的。比如就性别这个attribute而言，男是一个domain，女是一个；相对于发色而言，金发是一个domain，黑发是一个domain。本文提出的是一个可以解决multiple domain translation的translator，如图3所示。

![&#x56FE;3](../../.gitbook/assets2/image%20%2833%29.png)

#### 损失函数

1. Adversarial Loss：这部分设计无太多新意，采用WGAN的loss。
2. Domain Classification Loss：对real image，激励D分类到正确domain label；对fake image，激励G向target domain label靠近。
3. Reconstruction Loss：等同于cycleGAN采用的，x 和 G1\(G2\(x\)\) 之间的L1 loss，保证cross-domain过程中只更改我们想要更改的部分。

#### Mask vector \(domain label\)

采用one-hot的方法记录属性。如下图所示，左边是CelebA dataset产生的label，右边是RaFD dataset产生的label。除非是通用属性（如图中mask第三行的男女），属于本dataset的label才有值，否则取0。这种采用n维mask vector的方法可以同时使用n个dataset/信息源进行训练和标注。注意只有在训练real image的时候，D才输出domain classification：

![](../../.gitbook/assets2/image%20%2826%29.png)

### 实验结果

远超DIAT、CycleGAN、IcGAN等模型并极大减少了训练量和参数。 

## 8. PairedCycleGAN: Asymmetric Style Transfer for Applying and Removing Makeup

Oral

### 简介

本文提出了一个自动上妆/去妆的GAN框架。框架采用两个不对称的风格转换函数，前向函数根据样例风格上妆，反向函数去妆，并用两个相关联的网络实现它们。

### 算法

网络结构如图2所示

![&#x56FE;2](../../.gitbook/assets2/image%20%2866%29.png)

其中source X包含了不同面部特征、表情、肤色、性别的广泛的无妆人脸数据；reference Y包含了不同化妆风格的人脸数据，对于每种妆容B，yB代表其中的一张人脸。框架的第一步是把\(x, yB\)转化成\(xB, y\)，第二部是把\(xB, y\)转化回\(x, yB\)。

生成器G为上妆器，接受\(x, yB\)，产生xB；或者接受\(xB, y\)，产生yB。生成器F为卸妆器，接受yB产生y，或者接受xB产生x。对抗器DY鉴别真假妆，把yB作为真，把xB作为假。对抗器DX鉴别真假无妆，把x作为真，y作为假。

除了GAN loss之外，还引入了indentity loss，最小化x和F\(G\(x, yB\)\)的距离，旨在最小化重构无妆人脸的差距；也引入了style loss，最小化yB和G\(F\(yB\), G\(x, yB\)\)的距离，旨在最小化重构有妆人脸的差距。

网络结构方面，为了避免出现把高像素的全脸放到GPU而引起的内存瓶颈，文章首先分割了人脸的不同部位，然后分别进入不同的生成器进行操作，如图4、5所示。

![&#x56FE;5](../../.gitbook/assets2/image%20%2869%29.png)

## 9. Wasserstein Introspective Neural Networks

Oral

### 简介

Wasserstein Introspective Neural Networks（WINN）把生成器和对抗器融入到了一个模型，具有三个新颖特点：一，将INN和WGAN在数学上联系到一起；二，将Wasserstein距离引入INN节省了20倍的运算量；三，在监督领域，WINN能够鲁棒地鉴别反例和减少错误率。实验证明WINN在无监督的texture、人脸、物体建模等领域，以及监督的（且包含对抗样本攻击的）分类问题上有很好的效果。

### 算法

WGAN大家都熟悉，是把Wasserstein距离引入GAN，极大提高了稳定性。

INN的目的是定义discriminative classifier自己产生的pseudo-negative样本。具体来说，对于样本集合X的每一个样本x，定义它的标签为y；若x是训练集的样本，y = 1；若x是自我生成的，y = -1。为了学习$$p(x|y = 1)$$，网络利用pseudo-negative样本学习$$p_t(x|y=-1; W_t)$$，简称$$p_{W_t}(x)$$，其公式如下：

$$
\bar{p_{W_t}}(x) = \frac{1}{Z_t}\exp(w_t^{(1)} \phi(x; w_t^{(0)})) \bar{p_0}(x), t = 1, 2, \cdots, T
$$

其中 $$Z_t = \int\exp(w_t^{(1)} \phi(x; w_t^{(0)})) \bar{p_0}(x) ~dx$$ ，$$p_0(x)$$选用高斯分布。$$W_t = (w_{t_0}, w_{t_1})$$代表网络参数，其中$$w_{t_1}$$代表最上层的网络，用来整合下面各层的参数$$\phi(x; w_{t_0})$$，即softmax整合；$$w_{t_0}$$表示网络的内部特征参数。通过对x进行如下的BP即可将$$p_{W_t}(x)$$逐渐靠近$$p(x|y = 1)$$：

$$
\Delta x = \frac{\epsilon}{2}\nabla(w_t^{(1)} \phi(x;w_t^{(0)}))+\eta
$$

而在WINN里面，用Wasserstein距离进行BP，公式和框架结构如下

![](../../.gitbook/assets2/image%20%2891%29.png)

![](../../.gitbook/assets2/image%20%2831%29.png)

注意在合成部分，WINN的$$p_{W_t}(x)$$变更为

$$
\bar{p_{W_t}}(x) = \frac{1}{Z_t}\exp(f_{W_t}(x)) \bar{p_0}(x), t = 1, 2, \cdots, T
$$

其中$$\exp(f_{W_t}(x)) = \frac{p(y=+1|x; W_t)}{p(y=-1|x; W_t)}$$，同样是以生成图与原图的相似度作为衡量标准。在合成步里以$$p_{W_t}(x)$$做对象进行SGA，最大化y=1的概率。

### 实验结果

在无监督的texture、人脸、物体建模等领域更逼真了；在监督的MNIST数据集上比单纯使用同样的网络的效果要好。

## 10. SeGAN: Segmenting and Generating the Invisible

Spotlight

### 简介

图片中的物体的被遮挡部分对于后续的识别和分析都可能是是重要的信息。作者通过一个segmentor + generator + discriminator来补全图像中物体被遮挡的部分。流程主要分为两部分：将物体不可见的部分分割出来，得到一个物体整体的mask；利用GAN生成不可见的部分的图像。实验结果说明，SeGAN能够比较好的补上被遮挡的物体，且超过了目前最好的对被遮挡物体的语义分割模型；SeGAN能够在自然图片上有较好的效果，且可以做图片深度分割。

### 算法

网络结构如图2所示

![&#x56FE;2](../../.gitbook/assets2/image%20%2876%29.png)

其中分割器的输入是原始图片和visible mask，通过roi pooling，输出对应物体的visible + invisible mask。生成器采用U-Net结构，输入是一张融合图：对于visible mask，取原始图片的像素；对invisible mask，取红色；对mask外的，取蓝色。生成器和对抗器的架构类似于Pix2Pix网络。

### 实验结果

对带遮挡物体的语义分割远高于现有的模型，且在visible和invisible区域都比它们高。对带遮挡物体的外观重构上，也远好于pix2pix等现有模型。同时，模型也能在自然图片的语义分割上取得不错的效果，也可以在深度预测上取得不错的结果。

