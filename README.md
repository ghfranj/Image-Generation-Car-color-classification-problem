# Image-Generation-Car-color-classification-problem
Steps:
1- Utilizing a pre-trained detector or segmenter to cut out cars from pictures. Thus, collecting a dataset of cut out cars.
2- Retraining two car color classifiers on the DVM dataset using models pretrained on ImageNet and Cityscapes.
3- Training our own classifier from scratch.
4- Assessing the quality using F1_macro, and F1_macro > 0.8 is required.
5- Comparing the quality obtained and drawing a conclusion.


1- The pre-trained segmenter:PSPNet (Pyramid Scene Parsing Network)
~~~
from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12
model = pspnet_101_voc12() # load the pretrained model trained on Pascal VOC 2012 dataset
~~~
