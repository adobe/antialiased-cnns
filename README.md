# <b>Antialiased CNNs</b> [[Project Page]](http://richzhang.github.io/antialiased-cnns/) [[Paper]](https://arxiv.org/abs/1904.11486)

<img src='https://richzhang.github.io/antialiased-cnns/resources/gifs2/video_00810.gif' align="right" width=300>

**Making Convolutional Networks Shift-Invariant Again** <br>
[Richard Zhang](https://richzhang.github.io/). <br>
To appear in [ICML, 2019](https://arxiv.org/abs/1904.11486).

This repository contains examples of anti-aliased convnets. We build off publicly available PyTorch [ImageNet](https://github.com/pytorch/examples/tree/master/imagenet) and [model](https://github.com/pytorch/vision/tree/master/torchvision/models) repositories, with antialiasing add-ons: <br>

- antialiased AlexNet, VGG, ResNet, DenseNet architectures, along with weights. Few lines to load a pretrained antialiased model

```python
import models_lpf
model = models_lpf.resnet.resnet50(filter_size=3)
model.load_state_dict(torch.load('./weights/resnet50_lpf3.pth.tar')['state_dict'])
```

- an [antialiasing layer](models_lpf/__init__.py) (called `BlurPool` in the paper), which can be easily plugged into your favorite architecture as a downsampling substitute

- [ImageNet training code and evaluation code](README_IMAGENET.md). This includes shift-invariant benchmarking code (`-es` flag). Achieving better consistency, while maintaining or improving accuracy, is an open problem. Help improve the results!

## Licenses

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

All material is made available under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license by Adobe Inc. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.

The repository builds off the PyTorch [examples repository](https://github.com/pytorch/examples) and torchvision [models repository](https://github.com/pytorch/vision/tree/master/torchvision/models). It is [BSD-style licensed](https://github.com/pytorch/examples/blob/master/LICENSE).

## (0) Getting started

### PyTorch
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`


### Download anti-aliased models


- Run `bash weights/get_antialiased_models.sh`

## (1) Quickstart: load an antialiased model

If you'd just like to load a pretrained antialiased model, perhaps as a backbone for your application, just do the following.

Run `bash weights/get_antialiased_models.sh` to get model weights. The following gives you an anti-aliased ResNet50 (filter size 3).

```python
import torch
import models_lpf.resnet

model = models_lpf.resnet.resnet50(filter_size=3)
model.load_state_dict(torch.load('weights/resnet50_lpf3.pth.tar')['state_dict'])
```

We also provide weights for antialiased `AlexNet`, `VGG16(bn)`, `Resnet18,34,50,101`, `Densenet121` (see [example_usage.py](example_usage.py)).

## (2) Antialias your own architecture

The methodology is simple -- first evaluate with stride 1, and then use our `Downsample` layer to do antialiased downsampling.

1. Copy `models_lpf` into your codebase. This [file](models_lpf/__init__.py) This contains the `Downsample` class which does blur+subsampling. Put the following into your header to get the `Downsample` class.

```python
from models_lpf import *
```

2. Make the following architectural changes to antialias your strided layers.

|   |Original|Anti-aliased replacement|
|:-:|---|---|
|**MaxPool --><br> MaxBlurPool** | `[nn.MaxPool2d(kernel_size=2, stride=2),]` | `[nn.MaxPool2d(kernel_size=2, stride=1),` <br> `Downsample(filt_size=M, stride=2, channels=C)]`|
|**StridedConv --><br> ConvBlurPool**| `[nn.Conv2d(Cin, C, kernel_size=3, stride=2, padding=1),` <br> `nn.ReLU(inplace=True)]` | `[nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),` <br> `nn.ReLU(inplace=True),` <br> `Downsample(filt_size=M, stride=2, channels=C)]` |
|**AvgPool --><br> BlurPool**| `nn.AvgPool2d(kernel_size=2, stride=2)` | `Downsample(filt_size=M, stride=2, channels=C)`|

We assume tensor has `C` channels. For blur kernel size `M`, 3 or 5 is typical.

Note that this requires computing a layer at stride 1 instead of stride 2, which adds memory and run-time. We typically skip this step at the highest-resolution (early in the network), to prevent large increases.

## (3) Results

<img src='https://richzhang.github.io/antialiased-cnns/resources/imagenet_ind2_noalex.jpg' align="right" width=400>

We show consistency (y-axis) vs accuracy (x-axis) for various networks. Up and to the right is good. Training and testing instructions are [here](README_IMAGENET.md).

We *italicize* a variant if it is not on the Pareto front -- that is, it is strictly dominated in both aspects by another variant. We **bold** a variant if it is on the Pareto front. We **bold** highest values per column.

Note that the current arxiv paper is slightly out of date; we will update soon.

**AlexNet [(plot)](https://richzhang.github.io/antialiased-cnns/resources/imagenet_ind_AlexNet.jpg)**

|          | Accuracy | Consistency |
| :------: | :------: | :---------: |
| *Baseline* | *56.55* | *78.18* | 
| **Rect-2** | **57.24** | 81.33 | 
| **Tri-3** | 56.90 | 82.15 | 
| **Bin-5** | 56.58 | **82.51** | 

**VGG16 [(plot)](https://richzhang.github.io/antialiased-cnns/resources/imagenet_ind_VGG16.jpg)**

|          | Accuracy | Consistency |
| :------: | :------: | :---------: |
| *Baseline* | *71.59* | *88.52* | 
| *Rect-2* | *72.15* | *89.24* | 
| *Tri-3* | *72.20* | *89.60* | 
| **Bin-5** | **72.33** | **90.19** | 

**VGG16bn [(plot)](https://richzhang.github.io/antialiased-cnns/resources/imagenet_ind_VGG16bn.jpg)**

|          | Accuracy | Consistency |
| :------: | :------: | :---------: |
| *Baseline* | *73.36* | *89.24* | 
| *Rect-2* | *74.01* | *90.72* | 
| *Tri-3* | *73.91* | *91.10* | 
| **Bin-5** | **74.05** | **91.35** | 

**ResNet18 [(plot)](https://richzhang.github.io/antialiased-cnns/resources/imagenet_ind_ResNet18.jpg)**

|          | Accuracy | Consistency |
| :------: | :------: | :---------: |
| *Baseline* | *69.74* | *85.11* | 
| *Rect-2* | 71.39 | 86.90 | 
| **Tri-3** | **71.69** | 87.51 | 
| **Bin-5** | 71.38 | **88.25** | 

**ResNet34 [(plot)](https://richzhang.github.io/antialiased-cnns/resources/imagenet_ind_ResNet34.jpg)**

|          | Accuracy | Consistency |
| :------: | :------: | :---------: |
| *Baseline* | *73.30* | *87.56* | 
| **Rect-2** | **74.46** | 89.14 | 
| **Tri-3** | 74.33 | 89.32 | 
| **Bin-5** | 74.20 | **89.49** | 

**ResNet50 [(plot)](https://richzhang.github.io/antialiased-cnns/resources/imagenet_ind_ResNet50.jpg)**

|          | Accuracy | Consistency |
| :------: | :------: | :---------: |
| *Baseline* | *76.16* | *89.20* | 
| *Rect-2* | *76.81* | *89.96* | 
| *Tri-3* | *76.83* | *90.91* | 
| **Bin-5** | **77.04** | **91.31** | 

**ResNet101 [(plot)](https://richzhang.github.io/antialiased-cnns/resources/imagenet_ind_ResNet101.jpg)**

|          | Accuracy | Consistency |
| :------: | :------: | :---------: |
| *Baseline* | *77.37* | *89.81* | 
| *Rect-2* | *77.82* | *91.04* | 
| **Tri-3** | **78.13** | 91.62 | 
| **Bin-5** | 77.92 | **91.74** | 


**DenseNet121 [(plot)](https://richzhang.github.io/antialiased-cnns/resources/imagenet_ind_DenseNet121.jpg)**

|          | Accuracy | Consistency |
| :------: | :------: | :---------: |
| *Baseline* | *74.43* | *88.81* | 
| *Rect-2* | *75.04* | *89.53* | 
| **Tri-3** | **75.14** | 89.78 | 
| **Bin-5** | 75.03 | **90.39** | 

## (A) Acknowledgments

This repository is built off the PyTorch [ImageNet training](https://github.com/pytorch/examples/tree/master/imagenet) and [torchvision models](https://github.com/pytorch/vision/tree/master/torchvision/models) repositories.

## (B) Citation, Contact

If you find this useful for your research, please consider citing this [bibtex](https://richzhang.github.io/index_files/bibtex_icml2019.txt). Please contact Richard Zhang \<rizhang at adobe dot com\> with any comments or feedback.


