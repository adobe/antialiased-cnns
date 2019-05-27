# <b>Antialiased CNNs</b> [[Project Page]](http://richzhang.github.io/antialiased-cnns/) [[Paper]](https://arxiv.org/abs/1904.11486)

<img src='https://richzhang.github.io/antialiased-cnns/resources/gifs2/video_00810.gif' align="right" width=300>

**Making Convolutional Networks Shift-Invariant Again** <br>
[Richard Zhang](https://richzhang.github.io/). <br>
To appear in [ICML, 2019](https://arxiv.org/abs/1904.11486).

This repository contains examples of anti-aliased convnets. We build off publicly available PyTorch [ImageNet](https://github.com/pytorch/examples/tree/master/imagenet) and [models](https://github.com/pytorch/vision/tree/master/torchvision/models) repositories, with add-ons for antialiasing: <br>
- a [low-pass filter layer](models_lpf/__init__.py) (called `BlurPool` in the paper), which can be easily plugged into any network
- antialiased AlexNet, VGG, ResNet, DenseNet architectures, along with pretrained weights
- benchmarking code and evaluation for shift-invariance (`-es` flag)

## Licenses

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

All material is made available under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license by Adobe Inc. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.

The repository builds off the PyTorch [examples repository](https://github.com/pytorch/examples) and torchvision [models repository](https://github.com/pytorch/vision/tree/master/torchvision/models). It is [BSD-style licensed](https://github.com/pytorch/examples/blob/master/LICENSE).

## (0) Getting started

### PyTorch + ImageNet
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## (1) Evaluating models

### Download anti-aliased models

- Run `bash weights/get_antialiased_models.py`

We provide models with filter sizes 2,3,5 for AlexNet, VGG16, VGG16bn, ResNet18,34,50,101 and DenseNet121.

### Evaluating accuracy

```bash
python main.py --data /PTH/TO/ILSVRC2012 -e -f 5 -a alexnet_lpf --weights ./weights/alexnet_lpf5.pth.tar --gpu 0
python main.py --data /PTH/TO/ILSVRC2012 -e -f 5 -a vgg16_lpf --weights ./weights/vgg16_lpf5.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -e -f 5 -a vgg16_bn_lpf --weights ./weights/vgg16_bn_lpf5.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -e -f 5 -a resnet18_lpf --weights ./weights/resnet18_lpf5.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -e -f 5 -a resnet34_lpf --weights ./weights/resnet34_lpf5.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -e -f 5 -a resnet50_lpf --weights ./weights/resnet50_lpf5.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -e -f 5 -a resnet101_lpf --weights ./weights/resnet101_lpf5.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -e -f 5 -a densenet121_lpf --weights ./weights/densenet121_lpf5.pth.tar
```

### Evaluating consistency

Same as above, but flag `-es` evaluates the shift-consistency -- how often two random `224x224` crops are classified the same.

```bash
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -f 5 -a alexnet_lpf --weights ./weights/alexnet_lpf5.pth.tar --gpu 0
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -f 5 -a vgg16_lpf --weights ./weights/vgg16_lpf5.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -f 5 -a vgg16_bn_lpf --weights ./weights/vgg16_bn_lpf5.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -f 5 -a resnet18_lpf --weights ./weights/resnet18_lpf5.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -f 5 -a resnet34_lpf --weights ./weights/resnet34_lpf5.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -f 5 -a resnet50_lpf --weights ./weights/resnet50_lpf5.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -f 5 -a resnet101_lpf --weights ./weights/resnet101_lpf5.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -f 5 -a densenet121_lpf --weights ./weights/densenet121_lpf5.pth.tar
```

Some notes:
- These line commands are very similar to the base PyTorch [repository](https://github.com/pytorch/examples/tree/master/imagenet). We simply add suffix `_lpf` to the architecture and specify `-f` for filter size.
- Substitute `-f 5` and appropriate filepath for different filter sizes.
- The example commands use our weights. You can them from your own training checkpoints by subsituting `--weights PTH/TO/WEIGHTS` for `--resume PTH/TO/CHECKPOINT`.

## (2) Training antialiased models

The following commands train antialiased AlexNet, VGG16, VGG16bn, ResNet18,34,50, and Densenet121 models with filter size 5. Output models will be in `[[OUT_DIR]]/model_best.pth.tar`

```bash
python main.py --data /PTH/TO/ILSVRC2012 -f 5 -a alexnet_lpf --out-dir alexnet_lpf5 --gpu 0 --lr .01
python main.py --data /PTH/TO/ILSVRC2012 -f 5 -a vgg16_lpf --out-dir vgg16_lpf5 --lr .01 -b 128 -ba 2
python main.py --data /PTH/TO/ILSVRC2012 -f 5 -a vgg16_bn_lpf --out-dir vgg16_bn_lpf5 --lr .05 -b 128 -ba 2
python main.py --data /PTH/TO/ILSVRC2012 -f 5 -a resnet18_lpf --out-dir resnet18_lpf5
python main.py --data /PTH/TO/ILSVRC2012 -f 5 -a resnet34_lpf --out-dir resnet34_lpf5
python main.py --data /PTH/TO/ILSVRC2012 -f 5 -a resnet50_lpf --out-dir resnet50_lpf5
python main.py --data /PTH/TO/ILSVRC2012 -f 5 -a resnet101_lpf --out-dir resnet101_lpf5
python main.py --data /PTH/TO/ILSVRC2012 -f 5 -a densenet121_lpf --out-dir densenet121_lpf5 -b 128 -ba 2
```

Some notes:
- As suggested by the official repository, AlexNet and VGG16 require lower learning rates of `0.01` (default is `0.1`). 
- VGG16_bn also required a slightly lower learning rate of `0.05`.
- I train AlexNet on a single GPU (the network is fast, so preprocessing becomes the limiting factor if multiple GPUs are used).
- Default batch size is `256`. Some extra memory is added for the antialiasing layers, so the default batchsize may no longer fit in memory. To get around this, we simply accumulate gradients over 2 smaller batches `-b 128` with flag `--ba 2`. You may find this useful, even for the default models, if you are training with smaller/fewer GPUs. It is not exactly identical to training with a large batch, as the batchnorm statistics will be computed with a smaller batch.

## (3) Make your own architecture more shift-invariant

The methodology is simple -- first evaluate with stride 1, and then use our `Downsample` layer to do antialiased downsampling.

1. Copy [models_lpf/\_\_init\_\_.py](models_lpf/__init__.py) into your codebase. This contains the `Downsample` layer which does blur+subsampling.

2. Put the following into your header to get the `Downsample` class.

```python
from models_lpf import *
```

3. Make the following architectural changes.

|   |Original|Anti-aliased replacement|
|:-:|---|---|
|**MaxPool --><br> MaxBlurPool** | `[nn.MaxPool2d(kernel_size=2, stride=2),]` | `[nn.MaxPool2d(kernel_size=2, stride=1),` <br> `Downsample(filt_size=M, stride=2, channels=C)]`|
|**StridedConv --><br> ConvBlurPool**| `[nn.Conv2d(Cin, C, kernel_size=3, stride=2, padding=1),` <br> `nn.ReLU(inplace=True)]` | `[nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),` <br> `nn.ReLU(inplace=True),` <br> `Downsample(filt_size=M, stride=2, channels=C)]` |
|**AvgPool --><br> BlurPool**| `nn.AvgPool2d(kernel_size=2, stride=2)` | `Downsample(filt_size=M, stride=2, channels=C)`|

We assume blur kernel size `M` (3 or 5 is typical) and that the tensor has `C` channels.

Note that this requires computing a layer at stride 1 instead of stride 2, which adds memory and run-time. We typically skip this step for at the highest-resolution (early in the network), to prevent large increases.

## (4) Results

<img src='https://richzhang.github.io/antialiased-cnns/resources/imagenet_ind2_noalex.jpg' align="right" width=400>

We show consistency (y-axis) vs accuracy(x-axis) for various networks. Up and to the right is good.

We *italicize* a variant if it is not on the Pareto front -- that is, it is strictly dominated in both aspects by another variant. We **bold** a variant if it is on the Pareto front. We **bold** highest values per column.

Achieving better consistency, while maintaining or improving accuracy, is an open problem. We invite you to participate!

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


