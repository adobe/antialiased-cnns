# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.


import torch

# Run `bash weights/get_antialiased_models.sh` to get weights.

# filter_size = 2;
filter_size = 3;
# filter_size = 5;

import models_lpf.resnet
model = models_lpf.resnet.resnet18(filter_size=filter_size)
model.load_state_dict(torch.load('weights/resnet18_lpf%i.pth.tar'%filter_size)['state_dict'])

model = models_lpf.resnet.resnet34(filter_size=filter_size)
model.load_state_dict(torch.load('weights/resnet34_lpf%i.pth.tar'%filter_size)['state_dict'])

model = models_lpf.resnet.resnet50(filter_size=filter_size)
model.load_state_dict(torch.load('weights/resnet50_lpf%i.pth.tar'%filter_size)['state_dict'])

model = models_lpf.resnet.resnet101(filter_size=filter_size)
model.load_state_dict(torch.load('weights/resnet101_lpf%i.pth.tar'%filter_size)['state_dict'])

import models_lpf.alexnet
model = models_lpf.alexnet.AlexNet(filter_size=filter_size)
model.load_state_dict(torch.load('weights/alexnet_lpf%i.pth.tar'%filter_size)['state_dict'])

import models_lpf.vgg
model = models_lpf.vgg.vgg16(filter_size=filter_size)
model.load_state_dict(torch.load('weights/vgg16_lpf%i.pth.tar'%filter_size)['state_dict'])

model = models_lpf.vgg.vgg16_bn(filter_size=filter_size)
model.load_state_dict(torch.load('weights/vgg16_bn_lpf%i.pth.tar'%filter_size)['state_dict'])

import models_lpf.densenet
model = models_lpf.densenet.densenet121(filter_size=filter_size)
model.load_state_dict(torch.load('weights/densenet121_lpf%i.pth.tar'%filter_size)['state_dict'])

import models_lpf.mobilenet
model = models_lpf.mobilenet.mobilenet_v2(filter_size=filter_size)
model.load_state_dict(torch.load('weights/mobilenet_v2_lpf%i.pth.tar'%filter_size)['state_dict'])

