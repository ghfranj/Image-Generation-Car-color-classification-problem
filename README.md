# Image-Generation-Car-color-classification-problem

Colab link: 
~~~
https://colab.research.google.com/drive/1yqd9kLyzc_mcKRjgX7K8CedR1Fy7ZqLI?usp=sharing
~~~

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


2- Dataset Used: DVM dataset
~~~
!gdown 'https://figshare.com/ndownloader/files/38754867'
~~~

3- Pre-trained models for training on DVM: 
~~~
from torchvision import models
from torch import nn
import torch
def get_my_resnet_34(num_classes):
    resnet_features = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    resnet_features.requires_grad_(False)
    resnet_features.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.Dropout(0.4),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.Dropout(0.3),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )
    return resnet_features
~~~
