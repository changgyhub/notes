---
description: 计算机视觉
---

# Computer Vision

## 1. 图像分割

### 1.1 图像的不连续性

#### 1.1.1 边缘检测

1. 基本步骤：平滑滤波 -&gt; [锐化滤波 ](https://www.cnblogs.com/wangguchangqing/p/6947727.html)-&gt; 边缘判定 -&gt; 边缘连接
2. 边缘检测算子
   1. 基于微分（需要先去除噪声）：Roberts、Sobel、LoG
   2. [Canny Edge Detector](https://en.wikipedia.org/wiki/Canny_edge_detector)：高斯平滑滤波 -&gt; 利用一阶偏导的有限差分来计算梯度的幅值和方向 -&gt; 非极大值抑制 -&gt; 用双阈值算法检测和连接边缘 -&gt; 滞后筛选，英文表述为
      1. Apply Gaussian filter to smooth the image in order to remove the noise
      2. Find the intensity gradients of the image
      3. Apply non-maximum suppression to get rid of spurious response to edge detection
      4. Apply double threshold to determine potential edges
      5. Track edge by [hysteresis](https://en.wikipedia.org/wiki/Hysteresis): Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.
3. 便捷跟踪
4. Hough变换
   1. 直线检测：直接叠加\(a, b\)
   2. 极坐标参数空间：同样是直接叠加，注意取值范围的改变
   3. 曲线检测：类似，选择非\(x, y\)的其它项作为待叠加点
   4. 广义检测

#### 1.1.2 斑点检测

1. Difference of Gaussians \(DOG\)
   * 宽高斯-窄高斯
   * 有人claim比LOG快，其实结果差不太多
   * In 2D: $$DOG(x, y; \sigma, K) =  I (\frac{1}{2 \pi \sigma^2} e^{-(x^2 + y^2)/(2 \sigma^2)} - \frac{1}{2 \pi K^2\sigma^2} e^{-(x^2 + y^2)/(2 K^2 \sigma^2)})$$
2. Laplacian of Gaussian \(LoG\)
   * Laplacian: $$L(x, y) = \nabla^2 F(x, y) = \frac{\delta^2 F(x,y)}{\delta x^2} + \frac{\delta^2 F(x,y)}{\delta y^2}$$

![](.gitbook/assets/image%20%28750%29.png)

#### 1.1.3 角点检测

1. [Harris Corner Detector](https://zhuanlan.zhihu.com/p/42490675)
   1. Color to grayscale
   2. Spatial derivative calculation
      * For image $$I$$, compute $$I_x(x, y)$$ and $$I_y(x, y)$$
   3. Structure tensor setup
      * construct structure tensor $$M = \sum_{(x, y) \in N}[I_x^2, I_x I_y; I_x I_y, I_y^2] $$ for each $$(x, y)$$in a neighborhood
   4. Harris response calculation
      * $$(x, y)$$ is a corner if $$\min(\lambda_1, \lambda_2) \approx \frac{\lambda_1 \lambda_2}{\lambda_1+ \lambda_2} = \det(M)/\text{tr}(M)$$ of $$M$$ is larger than threshold
   5. Non-maximum suppression:
      * keep local maxima within 3 by 3 filter as corners
2. [Features from accelerated segment test \(FAST\)](https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test)
   * 原理：若某像素点的灰度值比周围足够多的像素点的灰度值大或小，则该点可能为角点
   * High Speed Test \(when N = 12\)：
     * 对于图像中一个像素点p，其灰度值为Ip
     * 以该像素点为中心考虑一个半径为3的离散化的Bresenham圆，圆边界上有16个像素
     * 设定一个合适的阈值t，如果圆上有n个连续像素点的灰度值小于Ip−t或者大于Ip+t，那么这个点即可判断为角点\(n的值可取12或9\)
   * 对n &lt; 12的值可以用决策树，在多个图片上训练一个FAST分类器，进行新图片的快速预测
   * 问题：
     * FAST不产生多尺度特征，不具备旋转不变性，而且检测到的角点不是最优

#### 1.1.4 特征描述、匹配

1. [Scale-invariant feature transform \(SIFT\)](https://blog.csdn.net/dcrmg/article/details/52577555)
   1. Scale-space extrema detection \(尺度空间的极值侦测\)
      * Key points are taken as maxima/minima of the DoG pyramids
   2. Key point localization \(关键点定位\)
      * Interpolation of nearby data for accurate position \(邻近资料插补\)
        * 以关键点作为原点，设变数![\textbf{x} = \left\( x, y, \sigma \right\)](https://wikimedia.org/api/rest_v1/media/math/render/svg/0bd62f77681647d41d51ccb4eceb598577c19c90)为点到此关键点的偏移量。DoG影像的二次泰勒级数$$D(\textbf{x}) = D + \frac{\partial D^T}{\partial \textbf{x}}\textbf{x} + \frac{1}{2}\textbf{x}^T \frac{\partial^2 D}{\partial \textbf{x}^2} \textbf{x}$$对$$\textbf{x}$$微分后设为零，便可求出极值![\hat{\textbf{x}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/a51ff2ef64578ad2d691347cba94d67b57c2e9ab)的位置。若![\hat{\textbf{x}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/a51ff2ef64578ad2d691347cba94d67b57c2e9ab)的任一项参数大于0.5，代表此时有另一关键点更接近极值，将舍弃此关键点并对新的点作插补。反之若所有参数皆小于0.5，此时将偏移量加回关键点中找出极值的位置
      * Discarding low-contrast key points \(舍弃不明显关键点\)
        * 此步骤将计算上述二次泰勒级数![D\(\textbf{x}\)](https://wikimedia.org/api/rest_v1/media/math/render/svg/999fbffd563d66720a18ff91c66e7702d54e98b7)在![\hat{\textbf{x}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/a51ff2ef64578ad2d691347cba94d67b57c2e9ab)的值。若此值小于0.03，则舍弃此关键点。反之保留此关键点，并记录其位置为![\textbf{y} + \hat{\textbf{x}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/c8478657a4454df2961bcc2c2ae5211656be2eb0)，其中![{\textbf  {y}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/9217602eb948f6bfca224665a8b6aac54e15725b)是一开始关键点的位置
      * Eliminating edge responses \(消除边缘响应\)
        * DoG函数对于侦测边缘上的点相当敏感，需要消除有高度边缘响应但其位置不佳不符合需求的关键点。 因为穿过边缘方向的主曲率会远大于沿着边缘方向上的主曲率，因此其主曲率比值远大于位于角落的比值。 为了算出关键点的主曲率，可解其二次海森矩阵$$H = [D_{xx}, D_{xy}; D_{yx}, D_{yy}]$$的特征值，若特征值比例$$\lambda_1/ \lambda_2$$大于一定范围则舍弃。
   3. Orientation assignment \(为关键点确定方向\)
      * 计算每个关键点与其相邻像素之梯度的量值与方向后，为其建立一个以10度为单位36条的直方图。每个相邻像素依据其量值大小与方向加入关键点的直方图中，最后直方图中最大值的方向即为此关键点的方向。若最大值与局部极大值的差在20%以内，则此判断此关键点含有多个方向，因此将会再额外建立一个位置、尺寸相同方向不同的关键点
   4. Key point descriptor \(生成关键点描述子\)
      * 首先在关键点周围16x16的区域内，对每个4x4的子区域内建立一个8方向的直方图，一共16个子区域，产生一个16x8=128维的资料。为了使描述子在不同光线下保有不变性，需将描述子正规化为一128维的单位向量。此外为了减少非线性亮度的影响，把大于0.2的向量值设为0.2，最后将正规化后的向量乘上256以8位元无号数储存，可有效减少储存空间 
   5. Matching \(关键点匹配\)
      * 采用欧式距离匹配，距离越小相似度越高，小于阈值即可判定匹配成功
2. Speeded Up Robust Features \(SURF\)
   1. Detector
      * SURF使用了方型滤波器取代SIFT中的高斯滤波器，借此达到高斯糢糊的近似。其滤波器可表示为:![{\displaystyle S\(x,y\)=\sum \_{i=0}^{x}\sum \_{j=0}^{y}I\(i,j\)}](https://wikimedia.org/api/rest_v1/media/math/render/svg/ca6962b58054faac2706d9092e70bb2eebdd201a)。此外使用方型滤波器可利用积分图大幅提高运算速度，仅需计算位于滤波器方型的四个角落値即可。SURF使用了斑点侦测的海森矩阵来侦测特征点，其行列式值代表像素点周围的变化量，因此特征点需取行列式值为极大、极小值。除此之外，为了达到尺度上的不变，SURF还使用了尺度σ的行列式值作特征点的侦测，给定图形中的一点p=\(x, y\)，在尺度σ的海森矩阵为H\(p, σ\):![{\displaystyle H\(p,\sigma \)={\begin{pmatrix}L\_{xx}\(p,\sigma \)&amp;L\_{xy}\(p,\sigma \)\\L\_{xy}\(p,\sigma \)&amp;L\_{yy}\(p,\sigma \)\end{pmatrix}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/9c513011dc2c80c15bb2fbaa57442380ff7144ef)，其中矩阵内的![{\displaystyle L\_{xx}\(p,\sigma \)}](https://wikimedia.org/api/rest_v1/media/math/render/svg/79cdde8daa0bdd4cf5aa3a65c3a424b7d35cdae7)等函数为二阶微分后的灰阶图像。9\*9的方型滤波器被作为SURF最底的尺度，近似于σ=1.2的高斯滤波器。
   2. Discriptor
      * 方位定向：SURF的描述子计算特征点周围半径维6σ个像素点的x,y方向的哈尔小波转换，其中σ是此特征点位于的尺度。所得到的小波响应以特征点为中心的高斯函数作加权，并将其值标于一xy作标平面上作图。最后在xy作标平面上以π/3为一个区间，将区间内小波响应的x、y分量加总得到一向量，在所有的向量当中最长的\(即x、y分量最大的\)即为此特征点的方向。
      * 描述子：选定了特征点的方向后，其周围像素点需要以此方向为基准来建立描述子。此时以5\*5个像素点为一个子区域，取特征点周围20\*20个像素点的范围共16个子区域，计算子区域内的x、y方向\(此时以平行特征点方向为x、垂直特征点方向为y\)的[哈尔小波转换](https://zh.wikipedia.org/wiki/%E5%93%88%E7%88%BE%E5%B0%8F%E6%B3%A2%E8%BD%89%E6%8F%9B)总合![{\displaystyle \sum dx}](https://wikimedia.org/api/rest_v1/media/math/render/svg/c0380aae1d113b85dfc1b29032a5b84f46fefd96)、![{\displaystyle \sum dy}](https://wikimedia.org/api/rest_v1/media/math/render/svg/4f89343dd1f91f7da2bf64cecebe3a9243d62f08)与其向量长度总合![{\displaystyle \sum \|dx\|}](https://wikimedia.org/api/rest_v1/media/math/render/svg/963e5679faa904df4a8b4f70d66b42c8a44b3969)、![{\displaystyle \sum \|dy\|}](https://wikimedia.org/api/rest_v1/media/math/render/svg/e7a0ae989ea1ffd7850db04accdb78cd862e96e6)共四个量值，共可产生一个64维资料的描述子。
   3. Matching

#### 1.1.5 特征跟踪

1. [Kanade–Lucas–Tomasi \(KLT\) feature tracker](https://zh.wikipedia.org/wiki/%E7%9B%A7%E5%8D%A1%E6%96%AF-%E5%8D%A1%E7%B4%8D%E5%BE%B7-%E6%89%98%E9%A6%AC%E5%B8%8C%E7%89%B9%E5%BE%B5%E8%BF%BD%E8%B9%A4) 估计两个图片/特征点云的transformation

### 1.2 图像的相似性

1. 区域生长：如八邻域生长算法
2. 区域分割与合并：如四叉树分解
3. 阈值分割
   1. 实验法、直方图谷底法等简单方法
   2. 迭代选择阈值法（分两块，取每块平均灰度，新的阈值为其和的一半）
   3. 最小均方误差法
   4. 最大类间方差法

## 2. 特征提取

1. 基本特征：周长、面积、致密性（周长平方除以面积）、质心、灰度均值中值、包含区域的最小矩形、最小或最大灰度级、大于或小于均值的像素数、欧拉数等
2. 直方图特征：均值、标准方差、平滑度、三阶矩、一致性、熵等
3. 灰度共现矩阵
4. PCA： [http://blog.codinglabs.org/articles/pca-tutorial.html?from=timeline&isappinstalled=0](http://blog.codinglabs.org/articles/pca-tutorial.html?from=timeline&isappinstalled=0)
5. 白化和ZCA：见UFLDL

