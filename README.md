
**Diffs**
- Adding pre-trained flag


# <b>Antialiased CNNs</b> [[Project Page]](http://richzhang.github.io/antialiased-cnns/) [[Paper]](https://arxiv.org/abs/1904.11486) [[Talk]](https://www.youtube.com/watch?v=HjewNBZz00w)

<img src='https://richzhang.github.io/antialiased-cnns/resources/gifs2/video_00810.gif' align="right" width=300>

**Making Convolutional Networks Shift-Invariant Again** <br>
[Richard Zhang](https://richzhang.github.io/). <br>
In [ICML, 2019](https://arxiv.org/abs/1904.11486).

This repository contains examples of anti-aliased convnets. <br>

**Table of contents**<br>
1. [Pretrained antialiased models](#1-quickstart-load-an-antialiased-model)<br>
2. [Instructions for antialiasing your own model](#2-antialias-your-own-architecture), using the [`BlurPool`](models_lpf/__init__.py) layer<br>
3. [Results on Imagenet consistency + accuracy.](#3-results)<br>
4. [ImageNet training and evaluation code](README_IMAGENET.md). Achieving better consistency, while maintaining or improving accuracy, is an open problem. Help improve the results!

**Update (Sept 2020)** I have added kernel size 4 experiments. When downsampling an even sized feature map (e.g., a 128x128-->64x64), this is actually the correct size to use to keep the indices from drifting.

## Licenses

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

All material is made available under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license by Adobe Inc. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.

The repository builds off the PyTorch [examples repository](https://github.com/pytorch/examples) and torchvision [models repository](https://github.com/pytorch/vision/tree/master/torchvision/models). These are [BSD-style licensed](https://github.com/pytorch/examples/blob/master/LICENSE).

## (0) Getting started

### PyTorch
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`


### Download anti-aliased models


- Run `bash weights/download_antialiased_models.sh`

## (1) Quickstart: load an antialiased model

The following loads a pretrained antialiased model, perhaps as a backbone for your application.

```python
import torch
import models_lpf.resnet

model = models_lpf.resnet.resnet50(filter_size=3)
model.load_state_dict(torch.load('weights/resnet50_lpf3.pth.tar')['state_dict'])
```

We also provide weights for antialiased `AlexNet`, `VGG16(bn)`, `Resnet18,34,50,101`, `Densenet121`, and `MobileNetv2` (see [example_usage.py](example_usage.py)).

## (2) Antialias your own architecture

The methodology is simple -- first evaluate with stride 1, and then use our `Downsample` layer (also referred to as `BlurPool`) to do antialiased downsampling.

<img src='https://richzhang.github.io/antialiased-cnns/resources/antialias_mod.jpg' width=800><br>

1. Copy `models_lpf` into your codebase, which contains the `Downsample` [class](models_lpf/__init__.py), which does blur+subsampling. Put the following into your header.

```python
from models_lpf import *
```

2. Make the following architectural changes to antialias your strided layers. Typically, blur kernel `M` is 3 or 5.

|Baseline|Anti-aliased replacement|
|---|---|
| `[nn.MaxPool2d(kernel_size=2, stride=2),]` | `[nn.MaxPool2d(kernel_size=2, stride=1),` <br> `Downsample(channels=C, filt_size=M, stride=2)]`|
| `[nn.Conv2d(Cin,C,kernel_size=3,stride=2,padding=1),` <br> `nn.ReLU(inplace=True)]` | `[nn.Conv2d(Cin,C,kernel_size=3,stride=1,padding=1),` <br> `nn.ReLU(inplace=True),` <br> `Downsample(channels=C, filt_size=M, stride=2)]` |
| `nn.AvgPool2d(kernel_size=2, stride=2)` | `Downsample(channels=C, filt_size=M, stride=2)`|

We assume incoming tensor has `C` channels. Computing a layer at stride 1 instead of stride 2 adds memory and run-time. As such, we typically skip antialiasing at the highest-resolution (early in the network), to prevent large increases.

## (3) Results

<img src='https://richzhang.github.io/antialiased-cnns/resources/imagenet_ind2_noalex.jpg' align="right" width=380>

We show consistency (y-axis) vs accuracy (x-axis) for various networks. Up and to the right is good. Training and testing instructions are [here](README_IMAGENET.md).

We *italicize* a variant if it is not on the Pareto front -- that is, it is strictly dominated in both aspects by another variant. We **bold** a variant if it is on the Pareto front. We **bold** highest values per column.

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

**MobileNet-v2 [(plot)](https://richzhang.github.io/antialiased-cnns/resources/imagenet_ind_MobileNetv2.jpg)**

|          | Accuracy | Consistency |
| :------: | :------: | :---------: |
| *Baseline* | *71.88* | *86.50* | 
| **Rect-2** | **72.63** | 87.33 | 
| **Tri-3** | 72.59 | 87.46 | 
| **Bin-5** | 72.50 | **87.79** | 

**Extra Run-Time**

Antialiasing requires extra computation (but no extra parameters). Below, we measure run-time (x-axis, both plots) on a forward pass of batch of 48 images of 224x224 resolution on a RTX 2080 Ti. In this case, gains in accuracy (y-axis, left) and consistency (y-axis, right) end up justifying the increased computation.

<img src='https://richzhang.github.io/antialiased-cnns/resources/resnet_timing.jpg' width=800><br>

## (4) Training and Evaluation

To reduce clutter, this is linked [here](README_IMAGENET.md). Help improve the results!

## (A) Acknowledgments

This repository is built off the PyTorch [ImageNet training](https://github.com/pytorch/examples/tree/master/imagenet) and [torchvision models](https://github.com/pytorch/vision/tree/master/torchvision/models) repositories.

## (B) Citation, Contact

If you find this useful for your research, please consider citing this [bibtex](https://richzhang.github.io/index_files/bibtex_icml2019.txt). Please contact Richard Zhang \<rizhang at adobe dot com\> with any comments or feedback.


