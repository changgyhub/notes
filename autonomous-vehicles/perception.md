# Perception

### 3D Detection

1. Voxel-based, 3D convolution
   * VoxelNet
2. Treat height as channel, 2D convolution, usually use bird eye view slides
   * Fast and Furious, MV3D
3. Bridge 3D and 2D
   * SurfConv
4. Pixel-based, 3D convolution
   * PointNet

### Deep Continuous Fusion for Multi-Sensor 3D Object Detection <a id="deep-continuous-fusion-for-multi-sensor-3d-object-detection"></a>

Raquel ECCV 2018的paper，把camera image的信息整合到BEV的信息里，效果好于MV3D和PIXOR。

![&#x7F51;&#x7EDC;&#x7ED3;&#x6784;](../.gitbook/assets3/image%20%289%29.png)

![Continuous Fusion &#x6A21;&#x5757;](../.gitbook/assets3/image%20%282%29.png)

### PIXOR: Real-time 3D Object Detection from Point Clouds <a id="pixor-real-time-3d-object-detection-from-point-clouds"></a>

Raquel CVPR 2018的paper，直接用BEV，切成0-1 voxel，把高度作为channel，进行single-stage detection。效果比MV3D好，速度可达到100FPS。‌

### End-to-end Learning of Multi-sensor 3D Tracking by Detection <a id="end-to-end-learning-of-multi-sensor-3d-tracking-by-detection"></a>

Raquel ICRA 2018的paper，用MV3D做Detection，用一个Matching Net来预测前帧物体a和后帧物体b是否是同一个物体，用Scoring Net算分类结果。

![](../.gitbook/assets3/image%20%2810%29.png)

### Fast and Furious: Real Time End-to-End 3D Detection, Tracking and Motion Forecasting with a Single Convolutional Net <a id="fast-and-furious-real-time-end-to-end-3d-detection-tracking-and-motion-forecasting-with-a-single-convolutional-net"></a>

Raquel CVPR 2018的paper，网络的输是入前n帧的BEV 3D点云。对点云先做0-1 Voxel，把高度作为通道进行2D卷积，输出是当前帧数的bounding box及classification，还有之后n-1帧的bounding box预测。这样某一帧的预测结果就是前n帧时输出的结果的加权平均。

![](../.gitbook/assets3/image%20%2815%29.png)

### Multi-View 3D Object Detection Network for Autonomous Driving <a id="multi-view-3d-object-detection-network-for-autonomous-driving"></a>

百度大名鼎鼎的MV3D模型，其中BV和FV都是2D的图片，预测的是3D的BBox。

![](../.gitbook/assets3/image%20%2816%29.png)

![](../.gitbook/assets3/image%20%285%29.png)

### VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection <a id="voxelnet-end-to-end-learning-for-point-cloud-based-3d-object-detection"></a>

![](../.gitbook/assets3/image%20%286%29.png)

### PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation <a id="pointnet-deep-learning-on-point-sets-for-3d-classification-and-segmentation"></a>

![](../.gitbook/assets3/image%20%283%29.png)

### SurfConv: Bridging 3D and 2D Convolution for RGBD Images‌ <a id="surfconv-bridging-3d-and-2d-convolution-for-rgbd-images"></a>

![](../.gitbook/assets3/image%20%281%29.png)

### Deep Multi-Sensor Lane Detection <a id="deep-multi-sensor-lane-detection"></a>

先通过LiDAR图像预测地表高度，然后把Camera投影到BEV上做马路线预测。注意regress马路线的时候crop到一个\[0, r\]的值，且是一个linear的过渡，使得网络不用预测很sharp的边界值。‌

![](../.gitbook/assets3/image%20%2817%29.png)

### End-to-End Deep Structured Models for Drawing Crosswalks <a id="end-to-end-deep-structured-models-for-drawing-crosswalks"></a>

一个用CNN画斑马线区域的方法，需要预测一些角度和斑马线左右起止。

![](../.gitbook/assets3/image%20%287%29.png)

### MultiNet: Real-time Joint Semantic Reasoning for Autonomous Driving <a id="multinet-real-time-joint-semantic-reasoning-for-autonomous-driving"></a>

没什么新意，就是给Detection加了个0-1 Segmentation的任务。

![](../.gitbook/assets3/image%20%284%29.png)

### Efficient Convolutions for Real-Time Semantic Segmentation of 3D Point Clouds <a id="efficient-convolutions-for-real-time-semantic-segmentation-of-3d-point-clouds"></a>

介绍了一种利用3D点云的方法：先voxelize，对每个高度取C个channel，第一个channel代表0-1，后几个可选channel可以放如RGB等信息；然后对L x W x \(HC\)做2D卷积，出来L x W x H x K，其中K表示预测的class。‌

### HDNET: Exploiting HD Maps for 3D Object Detection <a id="hdnet-exploiting-hd-maps-for-3d-object-detection"></a>

把HD Map的信息加到了BEW LiDAR上进行Detection，如果没有高清地图也可以用3D点云扫描结果来估计geometric & semantic prior，而无需生成整个新地图。

![](../.gitbook/assets3/image%20%2812%29.png)

![](../.gitbook/assets3/image%20%2813%29.png)

