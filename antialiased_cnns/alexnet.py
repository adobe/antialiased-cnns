# This code is built from the PyTorch examples repository: https://github.com/pytorch/vision/tree/master/torchvision/models.
# Copyright (c) 2017 Torch Contributors.
# The Pytorch examples are available under the BSD 3-Clause License.
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2019 Adobe. All rights reserved.
# Adobe’s modifications are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License (CC-NC-SA-4.0). To view a copy of the license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
#
# ==========================================================================================
#
# BSD-3 License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
from antialiased_cnns import *
from IPython import embed

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet_lpf2': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/alexnet_lpf2-da8aca74.pth',
    'alexnet_lpf3': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/alexnet_lpf3-f9bbc410.pth',
    'alexnet_lpf4': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/alexnet_lpf4-0114fe25.pth',
    'alexnet_lpf5': 'https://antialiased-cnns.s3.us-east-2.amazonaws.com/weights_v0.1/alexnet_lpf5-4fa3706a.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, filter_size=4, pool_only=False, relu_first=True):
        super(AlexNet, self).__init__()

        if(pool_only): # only apply LPF to pooling layers, so run conv1 at stride 4 as before
            first_ds = [nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),]
        else:
            if(relu_first): # this is the right order
                first_ds = [nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=2),
                    nn.ReLU(inplace=True),
                    BlurPool(64, filt_size=filter_size, stride=2),]
            else: # this is the wrong order, since it's equivalent to downsampling the image first
                first_ds = [nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=2),
                    BlurPool(64, filt_size=filter_size, stride=2),
                    nn.ReLU(inplace=True),]

        first_ds += [nn.MaxPool2d(kernel_size=3, stride=1), 
            BlurPool(64, filt_size=filter_size, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            BlurPool(192, filt_size=filter_size, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            BlurPool(256, filt_size=filter_size, stride=2)]
        self.features = nn.Sequential(*first_ds)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, filter_size=4, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        filter_size (int): [4] Antialiasing filter size
    """
    model = AlexNet(filter_size=filter_size, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet_lpf%i'%filter_size], map_location='cpu', check_hash=True)['state_dict'])
    return model


# replacing MaxPool with BlurPool layers
class AlexNetNMP(nn.Module):

    def __init__(self, num_classes=1000, filter_size=1):
        super(AlexNetNMP, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            BlurPool(64, filt_size=filter_size, stride=2, pad_off=-1, hidden=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            BlurPool(192, filt_size=filter_size, stride=2, pad_off=-1, hidden=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            BlurPool(256, filt_size=filter_size, stride=2, pad_off=-1, hidden=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        # embed()
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnetnmp(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNetNMP(**kwargs)
    # if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model




    # def __init__(self, num_classes=1000):
        # super(AlexNet, self).__init__()
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(64, 192, kernel_size=5, padding=2),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        #     nn.Conv2d(192, 384, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(384, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=3, stride=2),
        # )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(4096, num_classes),
        # )



