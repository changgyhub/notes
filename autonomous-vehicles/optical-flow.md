# Optical Flow

### Unsupervised Deep Learning for Optical Flow Estimation

提出了架构DSTFlow，用类似FlowNet的结构预测光流。不同于FlowNet所使用的endpoint error \(EPE\)，网络用传统unsupervised的、基于Charbonnier penalty的smooth loss + data loss \(也就是photometric loss\)，来进行优化。网络采用可BP的bilinear sampling kernel来把flow warp到图片上，而不是直接upsample。

另一篇文章Back to Basics: Unsupervised Learning of Optical Flow via Brightness Constancy and Motion Smoothness的内容也大致相同。

![](../.gitbook/assets3/image%20%2811%29.png)

