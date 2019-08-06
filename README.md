# Tricks-for-better-performance-of-YOLO-v3
My tricks for better performance of small targets detection, already tested on VEDAI dataset
## A:
Spatial self-attention and channel self-attention module for feature refinement: Attention plays an important role in the way humans perceive their surrounding environment. When looking at a picture, people always focus on the highlighted parts and selectively obtain information to make corresponding judgments. Inspired by this, we combine a dual selfattention network of position and channel to learn higher-quality convolutional features, where self means autonomous learning and adaptive weight allocation between feature maps and channels.
![demo](https://github.com/Realwhisky/Tricks-for-better-performance-of-YOLO-v3/blob/master/Attention.jpg)


## B:
Deconvolution structure for feature upsampling: The YOLO v3 network extracts depth features and encodes them. It generates feature maps of different resolutions through an interpolation method. However, this simple upsampling method cannot use the underlying semantic information felicitously. To solve this problem, the FCN and U-Net algorithms in the field of image segmentation proposed the deconvolution operation to recover deep semantic information and gained preferable effect. Inspired by this work, we adjust the upsampling mode by substituting deconvolution for interpolation.
![demo](https://github.com/Realwhisky/Tricks-for-better-performance-of-YOLO-v3/blob/master/deconvolution.jpg)

## C:
Online hard examples learning with Focal Loss for better classification. YOLO v3 uses the cross-entropy loss function for training. As a single-stage detector, the quantity gap between the positive and negative samples is huge. Moreover, since the number of simple samples is large and easy to be discriminated by the model, its parameter update cannot improve the judgment ability of the model, making the entire training inefficient. Focal Loss improves this situation by relatively suppressing simple samples, but the imbalance problem of samples is not well resolved. We introduce online hard example learning with Focal Loss in network training to improve this situation.
![demo](https://github.com/Realwhisky/Tricks-for-better-performance-of-YOLO-v3/blob/master/OH-Focal%20Loss.jpg)


## Test
Detection effect test of YOLO v3 and DEAN(improved) network. YOLO v3 is in the left side of first row, the picture on the right is the magnified details, DEAN is in second row correspondingly and the third row is the results of DEAN in more aerial imageries
![demo](https://github.com/Realwhisky/Tricks-for-better-performance-of-YOLO-v3/blob/master/test.jpg)


## references
[1] Redmon, J., & Farhadi, A., “Yolov3: An incremental improvement,” arXiv preprint arXiv:1804.02767 (2018).

[2] Fu, J., Liu, J., Tian, H., Li, Y., Bao, Y., Fang, Z., & Lu, H., “Dual attention network for scene segmentation,” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 3146-3154 (2019).

[3] Ronneberger, O., Fischer, P., & Brox, T., “U-net: Convolutional networks for biomedical image segmentation,” International Conference on Medical image computing and computer-assisted intervention, Springer, Cham, 234-241 (2015).

[4] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P., “Focal loss for dense object detection,” Proceedings of the IEEE international conference on computer vision, 2980-2988 (2017).

[5] Razakarivony, Sebastien, and Frederic Jurie., “Vehicle detection in aerial imagery: A small target detection benchmark.” Journal of Visual Communication and Image Representation, 34, 187-203 (2016).
