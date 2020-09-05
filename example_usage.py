# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.


import torch

# Run `bash weights/get_antialiased_models.sh` to get weights.

# filter_size = 2
# filter_size = 3
filter_size = 4
# filter_size = 5

pretrained = True

import models_lpf.resnet
model = models_lpf.resnet.resnet18(pretrained=pretrained, filter_size=filter_size)
model = models_lpf.resnet.resnet34(pretrained=pretrained, filter_size=filter_size)
model = models_lpf.resnet.resnet50(pretrained=pretrained, filter_size=filter_size)
model = models_lpf.resnet.resnet101(pretrained=pretrained, filter_size=filter_size)

import models_lpf.alexnet
model = models_lpf.alexnet.AlexNet(pretrained=pretrained, filter_size=filter_size)

import models_lpf.vgg
model = models_lpf.vgg.vgg16(pretrained=pretrained, filter_size=filter_size)
model = models_lpf.vgg.vgg16_bn(pretrained=pretrained, filter_size=filter_size)

import models_lpf.densenet
model = models_lpf.densenet.densenet121(pretrained=pretrained, filter_size=filter_size)

import models_lpf.mobilenet
model = models_lpf.mobilenet.mobilenet_v2(pretrained=pretrained, filter_size=filter_size)

