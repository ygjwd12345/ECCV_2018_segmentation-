# ECCV_2018_segmentation-
自己找了ECCV 2018 关于 semantic segmentation、 instance segmentation、video相关的文章，做了点笔记，写了点感想。

————————————————————————————————————————

PersonLab: Person Pose Estimation and Instance Segmentation with a Bottom-Up, Part-Based, Geometric Embedding Model

Work：box free bottom-up approach for the task of pose estimation and instance segmentation of
People in multi-person images using an efficient single-shot model。感觉是基于SSD的

result：COCO test-dev keypoint average precision of 0.665 using single-scale inference and 0.687 using multi-scale inference    We get a mask AP of 0.417, which outperforms the strong top-down FCIS method , which gets 0.386. he first bottom-up method to report competitive results on the person class for the COCO instance segmentation task.
 
method：Hough voting of multiple predictions(通过一定的规则把 False positive剔除 如在 Depth-Encoded Hough Voting for Joint ObjectDetection and Shape Recovery 就是利用物体的景深排出 错误预测点提高准确率) 未详细讲解
OKS score  目前最为常用的就是OKS（Object Keypoint Similarity）指标，这个指标启发于目标检测中的IoU指标，目的就是为了计算真值和预测人体关键点的相似度。
