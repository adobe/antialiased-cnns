
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import numpy as np
from models_lpf import *
from IPython import embed

__all__ = ['AlexNet', 'alexnet']


# model_urls = {
    # 'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
# }


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, filter_size=1, pool_only=False, relu_first=True):
        super(AlexNet, self).__init__()

        if(pool_only): # only apply LPF to pooling layers, so run conv1 at stride 4 as before
            first_ds = [nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),]
        else:
            if(relu_first): # this is the right order
                first_ds = [nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=2),
                    nn.ReLU(inplace=True),
                    Downsample(filt_size=filter_size, stride=2, channels=64),]
            else: # this is the wrong order, since it's equivalent to downsampling the image first
                first_ds = [nn.Conv2d(3, 64, kernel_size=11, stride=2, padding=2),
                    Downsample(filt_size=filter_size, stride=2, channels=64),
                    nn.ReLU(inplace=True),]

        first_ds += [nn.MaxPool2d(kernel_size=3, stride=1), 
            Downsample(filt_size=filter_size, stride=2, channels=64),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            Downsample(filt_size=filter_size, stride=2, channels=192),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            Downsample(filt_size=filter_size, stride=2, channels=256)]
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


def alexnet(pretrained=False, **kwargs):
    """AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model


# replacing MaxPool with BlurPool layers
class AlexNetNMP(nn.Module):

    def __init__(self, num_classes=1000, filter_size=1):
        super(AlexNetNMP, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            Downsample(filt_size=filter_size, stride=2, channels=64, pad_off=-1, hidden=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            Downsample(filt_size=filter_size, stride=2, channels=192, pad_off=-1, hidden=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            Downsample(filt_size=filter_size, stride=2, channels=256, pad_off=-1, hidden=True),
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
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
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



