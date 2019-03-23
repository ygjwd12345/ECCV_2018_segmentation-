# ECCV_2018_segmentation-
自己找了ECCV 2018 关于 semantic segmentation、 instance segmentation、video相关的文章，做了点笔记，写了点感想。

————————————————————————————————————————

PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model

Work：box free bottom-up approach for the task of pose estimation and instance segmentation of
People in multi-person images using an efficient single-shot model。感觉是基于SSD的

result：COCO test-dev keypoint average precision of 0.665 using single-scale inference and 0.687 using multi-scale inference    We get a mask AP of 0.417, which outperforms the strong top-down FCIS method , which gets 0.386. he first bottom-up method to report competitive results on the person class for the COCO instance segmentation task.
 
method：Hough voting of multiple predictions(通过一定的规则把 False positive剔除 如在 Depth-Encoded Hough Voting for Joint ObjectDetection and Shape Recovery 就是利用物体的景深排出 错误预测点提高准确率) 未详细讲解
OKS score  目前最为常用的就是OKS（Object Keypoint Similarity）指标，这个指标启发于目标检测中的IoU指标，目的就是为了计算真值和预测人体关键点的相似度。

![image](https://github.com/ygjwd12345/ECCV_2018_segmentation-/raw/master/img/1.png)

Expected-OKS score 算区域 新的 sore 方法 基于OKS 着实没看懂
但大义应该是计算了short-range offset的 Hough score应用于关键点检测 pose estimation，再基于 long-range offests的 OKS sore 实现了 instance segmentation，但OKS是计算关键点的怎么用于segmentation 还有那个积分公式没看懂

keypoint：instance segmentation

3DMV: Joint 3D-Multi-View Prediction for 3D Semantic Scene Segmentation

work：combination of RGB and geometric features with our joint 2D-3D architecture achieves indoor environments（2D+3D 室内场景分割）

result：our final result on the ScanNet 3D segmentation benchmark increases from 52.8% to 75% accuracy compared to existing volumetric architectures.we improve 22.2% over existing volumetric and 14.8% over the state-of-the-art PointNet++ architecture（点）.

Method：利用 voxel max pool 解决多个二维image 输入融合问题再通过 Backprojection  layer e learned 2D features from each of the input RGB views with the 3D network 3DMV network 

Language： pytorch Backprojection  layer pytorch实现的

keypoint：3D semantic scene segmentation

Joint Task-Recursive Learning for Semantic Segmentation and Depth Estimation

Work: depth estimation and semantic segmentation on monocular. Considering the difficulty of high-resolution pixel-resolution pixel-level prediction, we derive the recursive task learning on a sequence of the estimation results.
文章认为当前得joint task learning多是浅层的权值共享，另一方面由于各个任务的独立性也难以实现真正的 muti-task learning。To address such problem, in this paper, we propose a novel joint Task-Recursive Learning (TRL) framework to closely-loop semantic segmentation and depth estimation on indoor scenes. 

Result: 在 NYU Depth V2 and SUN RGBD datasets  state of the art. NYU Depth V2 在depth estimate rms=0.501达到最优,在segmentation  pixel-acc=76.2 but IoU mean acc 并不是最好
IoU 不是最优 只是 PA是最优可以认为是最优吗?

Pixel accuracy(PA 像素精度) 正确像素占总像素的比
Mean Pixel Accuracy(MPA)每个类PA 求平均

Method: TRL(task recursive learning) framework Task-Attentional Module(TAM)--layer.假设 中间层更关注深度信息,起始的层更关注 segmentation, 用TAM 将两种信息融合起来,做blance(怎么能做到一部分层focus on depth other focus on segmentation)

language:pytorch

Keypoint: Depth Estimation ,semantic Segmentation,RGB--D

Leveraging Motion Priors in Videos for Improving Human Segmentation

work:we propose to leverage “motion prior” in videos for improving human segmentation in a weakly-supervised active learning setting.mitigate the performance drop caused by distribution mismatch


Given a longer sequence of frames, sparse long-term trajectories轨迹 of pixels
can be extracted. In the rest of the paper, we refer to these motion information
in a video as “motion prior”.

Result:对比偏弱,在自己设定的 PAL Source only Random中 PAL的架构最好, source only is direct application of pre-trained model on target domain data,感觉应该是指没有PAL 直接训练 segmentation,这样证明的仅仅是PAL加上了好一些,但还是比不上人,而且人工筛查这里列上也没啥意思,现在又没人关心比人强不强,而且这个也不是本文的强调的

对于Domain Adaptation, PAL+MMD/CMD会更好但缺乏量化,只是自身比,可能是因为做的人少.

method:propose to learn a memory-network-based policy model to select good candidate segments through reinforcement learning.unlabeled videos and a set of labeled images. Policy CNN用来鉴别 good/bad pathces.需要训练两个网络 policy model, 用于鉴别motion priors,作为 human segmenter的 labels.Domain adaptation leverages information from one or more source domains to improve the performance on target domain.

A Dataset for Lane Instance Segmentation in Urban Environments

Work：propose a semi-automated method that allows for efficient labelling of image sequences by utilising an estimated road plane in 3D based on where the car has driven and projecting labels from this plane into all images of the sequence. 

Resoult：(1) The release of a new dataset for lane instance and road segmentation, (2) A semi-automated annotation method for lane instances in 3D, requiring only inexpensive dash-cam equipment

结果是用自己的模型训练得到的结果在交叉集中表现最好 三个实验 EGO lane segmentation lane与 instance segmentation

表格中列出的AP比较没看含义什么，没有横向可比性，这里用到的ap 是coco的计算方法AP[.50:.05:.95]

AP

在目標檢測中，我們通常會認為預測的目標(通常是一個四四方方的bounding box)和Ground truth進行IoU計算，如果IoU大於一個閾值(threshold，通常是0.5)，則認為這個目標被是一個TP，如果小於這個閾值就是一個FP。AP就是計算這條precision-recall curve下的面積(area under curve, AUC)

VOC 2010之前的方法在算AP部分時會做一個小轉換。
就是選取Recall>=0, 0.1, 0.2,…, 1，這11的地方的Precision的最大值。VOC 2010之後的方法在recall的決斷值又做了修正，就是選取Recall>=0, 0.14, 0.29, 0.43, 0.57, 0.71, 1，這7的地方的Precision的最大值。


Method：1.OpenSfM reconstructions (OpenSfM is a Structure from Motion library written in Python. The library serves as a processing pipeline for reconstructing camera poses and 3D scenes from multiple images. It consists of basic modules for Structure from Motion (feature detection/matching, minimal solvers) with a focus on building a robust and scalable reconstruction pipeline. It also integrates external sensor (e.g. GPS, accelerometer) measurements for geographical alignment and robustness. A JavaScript viewer is provided to preview the models and debug the pipeline.)，failure cases are filtered during the manual annotation process 2.通过模型 计算lane boundary， 分ego lane 与 non-ego lane

Keypoint：instance segmentation ，semantic segmentation， weakly supervised？

Key-Word-Aware Network for Referring Expression Image Segmentation

Work   通过输入语言进行图像分割，和我们的关系不大


Resoult：outperforms state-of-the-art methods on two common referring expression image segmentation databases. In

Method： Key- word- aware Network  利用CNN提取图像信息 RNN提取语言信息将视觉特征，语言特征以及两者的融合特征一起做判断。

Keypoint: refer expression image segmentation


Penalizing Top Performers: Conservative Loss for Semantic Segmentation Adaptation

Work :propose Conservative loss 应用于  synthetic data 合成
提到了 Computer Graphics 也可以解决label time-consuming问题，但在 实际环境应用中效果很差。这里它定义 source domain（synthetic image） target domain（real images）也就是训练集和测试集。
测试集是在GTA V中提取的



Result  在FCN8s-VGG19取得了art of state GTAV → Cityscapes，Synthia → Cityscapes

Method 采用 conservative loss 解决Domain Adaptation for Semantic Segmentation，采用CoGAN架构应用 conservative loss 最终解决问题。最终在利用GAN的编码单元与segmentation结合，输出分割结果

Keypoint Domain Adaptation Semantic Segmentation
提出了一种不是弱监督的解决标记问题的思路。

Learnable PINs: Cross-Modal Embeddings for Person Identity

Work ：to learn a joint embedding of faces and voices, and to do so using a virtually free and limitless source of unlabelled training data and trained jointly to predict whether a face corresponds to a voice or not

Result：在VoxCeleb下，通过自己比较 FV的融合要优于F only。

Method  VGG-M用于检测人脸， VGG-Vox用于检测声音，然后再匹配（Pair selection using curriculum Mining）

Keypoint ：Face and voice recognition

Weakly- and Semi-Supervised Panoptic Segmentation

Work ：our model jointly produces semantic- and instance-segmentations of the image, whereas the aforementioned works only output instance-agnostic semantic segmentations. Secondly,we consider the segmentation of both “thing” and “stuff”classes

PS： “Thing” classes are weakly-supervised with bounding boxes, and “stuff” with image-level tags.

Result：obtain state-of-the-art results on Pascal VOC, for both full and weak supervision (which achieves about 95% of fully- supervised performance). 

Method：1.use GrabCut (a classic foreground segmentation technique given a bounding- box prior) and MCG (a segment-proposal algorithm) to obtain a foreground mask，两个提出的区域的& 2.the ground truth can subsequently be iteratively refined by using the outputs of the network on the training set as the new approximate ground  
把output当label不太能够理解

                         Detector （GrabCut or MCG）    
input----->PSPnet(semantic)---->fast RCNN--->0utput

根据的是 Bansal的理论sampling only 4% of the pixels in the image for computing the loss during fully-supervised training yielded about the same results as sampling all pixels, as traditionally done. 

Keypoint weak supervision, instance segmentation, semantic segmentation,

TS 2 C: Tight Box Mining with Surrounding Segmentation Context for Weakly Supervised Object Detection

Work  simple approach to discover tight object bounding boxes with only image-level supervision 主要做的是object detection Weakly Supervised Object Detection 


Result With TS 2 C, we obtain 48.0% and 44.4% mAP scores on VOC2007 and 2012 benchmarks, which are the new state-of-the-arts.


Method 三个branch,一是 image classification  branch  produce object Localization Map,作为semantic segmentation的mask,用来训练第二个分支 semantic segmentation net , 生成分割图像,作为object detection的参考,使box尽量的小.

Keypoint weakly-supervised object detection

现在做分割的弱监督好像很喜欢找个一个object detection的好模型作为 labels的生成器,但这不有点像RCNN?反过来 object detection也利用了segmentation 把盒子做小

Bayesian Semantic Instance Segmentation in Open Set World

Work:a novel open-set semantic instance segmentation approach capable of segmenting all known and unknown object classes in images, based on the output of an object detector trained on known object classes. This capability is vitally useful for many vision-based robotic systems. 

Result 在保持 已知物体的ap的情况下，未知物体也得到了分割，用F1 sore 进行评价。它是选取了一些类进行识别，另一些作为未label的物体，在NYU数据集上。

Method:object Detector 检测 box， Boundary Detertor 生成轮廓，在用Bayesian Formution 进行预测，这里就没看明白了概率 阵亡。

Keypoint:Instance segmentation, Open-set conditions 

ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation  open code

Work: for semantic segmentation of high resolution images under resource constraints.可以快速分割高像素图片。开源

Result：在牺牲8%的category-wise accuracy下 process high resolution images at a rate of 112 and 9 frames per second on a standard GPU and edge device

Method：factorization principle， low-bit networks，network compression， sparse CNN全用。采用了降维N/k dimensional space，Hierarchical feature fusion for de-gridding
总的来说就是先乘以1*1的核得到size=N/k， 再做k个SPP 再sum
Keyword：SPP， semantic segmentation， acceleration

Sequential Clique Optimization for Video Object Segmentation

Work：Video object segmentation（VOS） to segment out primary objects from the background in a video sequence，extract multiple primary objects


Method ：sequential clique optimization
Instance generation---complete K-partite graph---finding salient object tracks---segmentation results
Instance generation：employ instance-aware semantic segmentation method FCIS，
complete K-partite graph：太多数学公式，没看懂
Result 在DAVIS dataset  and the FBMS dataset 实现了最优，semi-/un-supervisedVOS
评价指标为 Region similarity Contour accuracy。
Keypoint：VOS， video， semi-/un-supervised

Video Object Segmentation with Joint Re-identification and Attention-Aware Mask Propagation
http://mmlab.ie.cuhk.edu.hk/projects/DyeNet/.

Work :Solving video object segmentation with multiple instances requires template match-
ing for coping with occlusion and temporal propagation for ensuring temporal continuity. 

Method Our network hinges on two main modules, namely a re-identification (Re-ID) module and a recurrent mask propagation (Re-MP) module.
joints template matching and temporal propagation into a unified deep neural network for addressing video object segmentation with multiple instances. 

PK Li et al [21] adapt person re-identification approach [36]

主要是两个部分 Re-ID module，Bi-directional mask propagation，Re-MP Module，Re-ID module 主要是做 mask，Re-MP Module是一个RCNN模型

Result： Our approach achieves a new state-of-the-art G-mean of 68.2 on the challenging DAVIS  2017 benchmark (test-dev set), outperforming the winning solution. 


Keypoint ：video， semi supervised

DAVIS challenge开始于2017，测评标准在2016年就已经阐述A Benchmark Dataset and Evaluation Methodology forVideo Object Segmentation，主要是用于做unsupervised或semisupervised，videoHD，太先进了，标准制定者。
给出了三个评价标准
Region Similarity Jaccard index defined as the intersection-over-union of the estimation segmentation and the ground truth mask。Given an output segmentation M and the corresponding ground-truth mask G，J就是他们的IoU
Contour Accuracy ontour-based precision and recall Pc and Rc between the contour points of c(M)and c(G), via a bipartite graph matching in order to be robust to small inaccuracies
Pc Rc的调和平均
Temporal stability 一种评价边界稳定性的方法

Depth-aware CNN for RGB-D Segmentation

Work ：an end-to-end network, Depth-aware CNN (D-CNN), for RGB-D segmentation
https://github.com/laughtervv/DepthAwareCNN
Pytorch 

Result ：Comparison with the state-of-the-art methods and extensive ablation studies on RGB-D semantic segmentation demonstrate the effectiveness and efficiency of depth-aware CNN。

Method ：Depth-aware CNN 。depth-aware convolution and depth-aware average pooling 就是引入两个像素点之间 Depth 差的修正。
Baseline选择的是 Deeplab，VGG-16架构
实验时有RGB+D和RGB+HHA（horizontal disparity, height above ground, and the angle the pixel’s local surface normal makes with the inferred gravity direction. ）


Keypoint:RGB-D,Semantic Segmentation

写作思路很好，就是把deeplab应用到了 RGB-D分割中，提出了自己的改进，考虑了像素差，再加了个比较

SRDA: Generating Instance Segmentation Annotation Via Scanning, Reasoning And Domain Adaptation

work novel three-stage SRDA pipeline and we build up a database which contains 3D models of common objects and corresponding scenes (SOM dataset) and scene images with instance level annotations (instance-60K).
We

method SRDA pipeline scanning, physics reasoning, domain adaptation
scanning 就是指的真实场景的数据库
physics plausible and commonsense plausible is checked by commonsense likelihood (CL) function.
domain adaptation Geometry-guided GAN (GeoGAN) framework
LSGAN loss (GAN loss), Structure loss, Reconstruction loss (L1 loss), Geometry-guided loss (Geo loss)

result our pipeline (blue) can significantly reduce human labor cost by nearly 2000 folds and achieve reasonable accuracy in instance segmentation. 77.02 and 86.02 are average mAP@0.5 of 3 scenes.

keypoint： domain adaptation， instance segmentation

Efficient Uncertainty Estimation for Semantic Segmentation in Videos

work解决的是 deep learning中的 Uncertainty Estimation，虽然前人提出Monte Carlo Dropout (MC dropout)，但速度太慢，本文提出TA-MC/RTA-MC dropout，在精度约损失2%的前提下速度提高10x
method We propose two main methods called temporal aggregation (TA) and region-based temporal aggregation (RTA)
我对于 RTA的理解是 I（t-1）做预测  Ft-1再将 warping function 与 current
frame做对比计算 reconstruction error 再进行 aggregated prediction，没咋看懂这个架构，感觉需要一些贝叶斯网络的知识。

result Compared to using general MC dropout, RTA can achieve similar performance on CamVid dataset with only 1.2% drop on mean IoU metric and incredibly speed up the inference process 10.97 times.

keypoint Uncertainty, Segmentation, Video,

BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation

work Real-time Semantic Segmentation

result  We achieve impressive results on the benchmarks of Cityscapes, CamVid, and COCO-Stuff. More specifically, we obtain the results of 68.4% on the Cityscapes test dataset with the speed of 105 FPS.

method 
Bilateral Segmentation Network
Spatial Path (SP) and Context Path (CP)
SP只是说要保留 spatial feature 
CP采用了lightweight model 卷积的分解 and global average pooling
Feature Fusion Module (FFM) and Attention Refinement Module (ARM) respectively.
We use the pre-trained Xception model as the backbone of the Context Path
and three convolution layers with stride as the Spatial Path.

kepoint real-time semantic segmentation

这里可以进一步了解的点有：
1.所应用的数据集大多都是在分割数据集上的，而且没有突出数据集背景，105FPS我认为是指的图片处理，并与一定是视频，这里可以弄清楚 如果是在KITTI的视频数据集会怎样，这里最后选的是train 640*360 val from2048*1024 to1536*768 并未达到高清的要求，这里有待提高
归纳这一点就是，并没有真正意义上的实现高像素1080p（1920*1080 FHD）的处理，且没有对video进行处理
现在流行的video segmentation net是什么？
mask RCNN可以做 video是否可以借鉴其网络？
2.在framework上我认为其实是把U shape 和FCN 相结合，在主要两个 module上处理非常粗糙而且同质化严重，感觉就是把fuse多了点卷积 和 strip resnet的技巧几乎一样
对于图像卷积的因式分解到底是怎么个逻辑？
weight crop filter crop 因式分解都尝试加速？
3. 速度pk这个地方 也缺乏对比性。到底别net的速度什么样不知道，这里也可以进行细化

End-to-End Joint Semantic Segmentation of Actors and Actions in Video

work We train and benchmark our model on the Actor-Action Dataset (A2D) for joint actor-action semantic segmentation, and demonstrate state-of-the-art performance for both segmentation and detection. 这里的detection是action detection应该说是也就是 action recognition。In this work, we propose a new direct end-to-end architecture that combines video action recognition and actor segmentation in a single unified model

result our approach significantly outperforms prior state-of-the-art methods on both actor-action segmentation and detection in videos.

method Our approach takes both RGB and flow video clips as input streams, leveraging information from both appearance and motion in the video.

![image](https://github.com/ygjwd12345/ECCV_2018_segmentation-/raw/master/img/2.png)

Appearance Backbone Feature Pyramid Network (FPN)
Region Proposal Network. like mask rcnn
Backbones. For action recognition, different from actor branch, two backbones are used. encoder是相同的。as shown in [22], motion patterns are also valuable in action recognition. 
总体来说 基本上 appearance 部分大部分借鉴了 mask rcnn的 idea，加了一个 temporal convs 和 action recognition layer。
Zero-shot learning is being able to solve a task despite not having received any training examples of that task. 处理 uncertain 信息的能力
keypoint joint multitask learning ，video，segmentation， action recognition
1.这篇文章有意思的问题是它提到了video的一个核心的问题那就是 CNN只考虑了空间因素，在 video这种与时间相关的场景， temporal context如何被利用是个问题。
2.在本文最后它有非常有野性的的提到了 zero-shot learning问题，在segmentation的pk对象是 maskrcnn，而在Joint Spatial Detection of Actors and Actions pk的是[11]TSMT,也就是是改进型
3.作为人的识别 DAVIS数据集上会怎么样？

Characterizing Adversarial Examples Based on Spatial Consistency Information for Semantic Segmentation
网络攻防领域的文章，完全不懂 跳过
work characterize adversarial examples based on spatial context information in semantic segmentation. 
We propose the spatial consistency analysis for benign/adversarial images and conduct large scale experiments on two state-of-the-art attack strategies against both DRN and DLA segmentation models with diverse adversarial targets on different dataset, including Cityscapes and real-world autonomous driving video dataset.

result

method

keypoint Semantic segmentation, adversarial example

Semi-convolutional Operators for Instance Segmentation

work pk RCNN mask RCNN 中的 instance segmentation中的pv过程，提出了IC来解决，semi-convolution

result we apply the latter to a standard instance segmentation benchmark PASCAL VOC (sec. 4.3). We show in all such cases that the use of semi-convolutional features can improve the performance of state-of-the-art instance segmentation methods such as Mask RCNN.

method instance coloring can efficiently represent any number of objects of arbitrary shape by predicting a single label map. Intuitively, this can be done by painting different regions with different “colors” (aka pixel labels) making objects easy to recover in post-processing. We call this process instance coloring (IC).

We do so by defining semi- convolutional operators which mix information extracted from a standard convolutional network with information about the global location of a pixel
结果上来看是 讲卷积结果与 pixl信息一起运算得到新的函数，采用了高斯核，这个变换太学术。


又是一篇对于MaskRCNN的改进
keypoint instance segmentation

VideoMatch: Matching based Video Object Segmentation
 
work in this paper, we propose a novel end-to-end trainable approach for fast semi-supervised video object segmentation that does not require any fine-tuning. 

result On the recently released DAVIS-16 dataset [40], our algorithm achieves 81.03% in IoU while reducing the running time by one order of magnitude compared to the state-of- the-art, requiring on average only 0.32 seconds per frame

method We use the provided ground truth mask of the first frame to obtain the set of foreground and background features (mF and mB). After extracting the feature tensor xt from the current frame, we use the proposed soft matching layer to produce FG and BG similarity. We then concatenate the two similarity scores and generate the final prediction via softmax.
soft matching layer 采用的是 cosine similarity，再做 top k average
keypoint video object segmentation

ECCV2018 segmentation 读后感

segmentation 领域论文数量一共23篇，数量不小，仍属于热点话题。现在关注的重点 已经放到了两个方向，一个是在传统的 segmentation net上做 semi-supervised另一个是将segmentation 放到 video中，还有一种是 muti-task domain adaptation。对于在segmentation net上做semi-supervised，主要的思路是采用一些预训练好的一些特征提取模型如Weakly- and Semi-Supervised Panoptic Segmentation 还有就是采用一些辅助手段 比如 在RGB图中加入 Depth信息如Depth-aware CNN for RGB-D Segmentation，还有把 detection和segmentation结合相互促进如TS 2 C: Tight Box Mining with Surrounding Segmentation Context for Weakly Supervised Object Detection，这就有点多任务学习的味道了。更多的研究是在video或者为video做准备，video主要关心的问题是1.实时性处理速度问题，如何使segmentation net做到real time，总结来有三种方法一是 卷积分解，将正常的卷积过程分成更简便 weight light model 如ESPNet: Efficient Spatial Pyramid of Dilated Convolutions for Semantic Segmentation  open code，另一种方法是做 filter crop，如BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation。除此之外对于 video还出现了对于mask RCNN的改进变种，如End-to-End Joint Semantic Segmentation of Actors and Actions in Video，将改进了 mask rcnn实时性还有处理多任务的能力，再比如Semi-convolutional Operators for Instance Segmentation更改了高斯核。2.还有个问题是提高video分割的准确度，这就是一个基本问题，CNN是未考虑时间特征的网络，怎么能在 video分割的时候考虑 temporal feature就是一个核心问题。End-to-End Joint Semantic egmentation of Actors and Actions in Video，提出的方法是 提出了action recognition的task，在这个前提下利用 temporal feature 再声明这种方法也有利于提高分割精度。总体来说还是在deeplab v2 /FCN, Mask RCNN.这些net的改进工作。原创性的net并没有。
很明显的是segmentation 在向 human recognition or action recognition方向结合，并且提高到了很多的 domain adaptation 和 uncertainty detection，这是向 实际中的自动驾驶问题考虑的迹象。 DAVIS数据集现在是做 segmentation和human recognition的比较新也比较热的数据集。instance segmentation作为一个噱头出现，现在还不是研究的核心问题，更多的是关心在instance segmentation下的计算量 标记量等代价问题，很少关心，instance segmentation的分割精度问题
存在的问题是：
1.现在的加速学习基本上没有太多的横向可比性，更多的是在image 的FPS 并非我们意义上的video的FPS，因为数据集的问题，在video dataset的数据集 我看到的都是以mask rcnn为backbone，也就是说在传统的segmentation net 没有真正的想 video转化，只是提高了处理速度，这个地方要借鉴 mask rcnn的经验 将segmentation net 真正能处理 video是一个非常有前景的方向。
还有数据集上，DAVIS数据集的video分割是没有太多认作，而原来很火的KITTI数据集做自动驾驶的更已经无人问津，这些具有特别鲜明的应用场景的数据集，应当被测试发出好文章。

2.现在的instance segmentation更多的是 light model的考虑，对于 instance segmentation的精度问题很少讨论，而且 instance segmentation中就要有detection的部分，能实现 多少的IoU 和 能够在多目标的情况下识别多少目标都是测量标准，但是现在的instance segmentation问题中很少讨论，在instance中要借鉴一下detection的标准，这个工作也很有意义。

3.在domain adaptation ，和未知事物的分割出现了但是工作较少。

4.在video的一个简单的问题 1080P高像素视频处理问题，还没有得到真正的解决。

