# <b>Antialiased CNNs</b> [[Project Page]](http://richzhang.github.io/antialiased-cnns/) [[Paper]](https://arxiv.org/abs/1904.11486) [[Talk]](https://www.youtube.com/watch?v=HjewNBZz00w)

<img src='https://richzhang.github.io/antialiased-cnns/resources/gifs2/video_00810.gif' align="right" width=300>

**Making Convolutional Networks Shift-Invariant Again** <br>
[Richard Zhang](https://richzhang.github.io/). In [ICML, 2019](https://arxiv.org/abs/1904.11486).

### Quick & easy start

Run `pip install antialiased-cnns`

```python
import antialiased_cnns
model = antialiased_cnns.resnet50(pretrained=True) 
```
<!-- model.load_state_dict(torch.load('resnet50_lpf4-994b528f.pth.tar')['state_dict']) # load weights; download it beforehand from https://www.dropbox.com/s/zqsudi0oz5ym8w8/resnet50_lpf4-994b528f.pth.tar?dl=0 -->

Now you are antialiased!

If you have a model trained and don't want to retrain the antialiased model from scratch, no problem! Simply load your old weights and fine-tune:

``` python
import torchvision.models as models
old_model = models.resnet50(pretrained=True) # old (aliased) model
antialiased_cnns.copy_params_buffers(old_model, model) # copy the weights over
```

If you want to antialias your own model, use the BlurPool layer.

```python
C = 10 # example feature channel size
blurpool = antialiased_cnns.BlurPool(C, stride=2) # BlurPool layer; use to downsample a feature map
ex_tens = torch.Tensor(1,C,128,128)
print(blurpool(ex_tens).shape) # 1xCx64x64 tensor
```

More information about our provided models and how to use BlurPool is below.

**Update (Sept 2020)** You can also now `pip install antialiased-cnns` and load models with the `pretrained=True` flag. I have added kernel size 4 experiments. When downsampling an even sized feature map (e.g., a 128x128-->64x64), this is actually the correct size to use to keep the indices from drifting.

### Table of contents

1. [More information about antialiased models](#1-more-information-loading-an-antialiased-model)<br>
2. [Instructions for antialiasing your own model](#2-more-information-how-to-antialias-your-own-architecture), using the [`BlurPool`](antialiased_cnns/__init__.py) layer<br>
3. [ImageNet training and evaluation code](README_IMAGENET.md). Achieving better consistency, while maintaining or improving accuracy, is an open problem. Help improve the results!

## (0) Preliminaries

Pip install this package

- `pip install antialiased-cnns`

Or clone this repository and install requirements (notably, PyTorch)

```bash

https://github.com/adobe/antialiased-cnns.git
cd antialiased-cnns
pip install -r requirements.txt
```


## (1) Loading an antialiased model

The following loads a pretrained antialiased model, perhaps as a backbone for your application.

```python
import antialiased_cnns
model = antialiased_cnns.resnet50(pretrained=True, filter_size=4)
```

We also provide weights for antialiased `AlexNet`, `VGG16(bn)`, `Resnet18,34,50,101`, `Densenet121`, and `MobileNetv2` (see [example_usage.py](example_usage.py)).

## (2) How to antialias your own architecture

The `antialiased_cnns` module contains the `BlurPool` [class](antialiased_cnns/downsample.py), which does blur+subsampling. Run `pip install antialiased-cnns` or copy the `antialiased_cnns` subdirectory.

The methodology is simple -- first evaluate with stride 1, and then use our `BlurPool` layer to do antialiased downsampling. Make the following architectural changes. Typically, blur kernel `M` is 4.

```python
import antialiased_cnns

# MaxPool --> MaxBlurPool
baseline = nn.MaxPool2d(kernel_size=2, stride=2)
antialiased = [nn.MaxPool2d(kernel_size=2, stride=1), 
    antialiased_cnns.BlurPool(C, filt_size=M, stride=2)]
    
# Conv --> ConvBlurPool
baseline = [nn.Conv2d(Cin, C, kernel_size=3, stride=2, padding=1), 
    nn.ReLU(inplace=True)]
antialiased = [nn.Conv2d(Cin, C, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    antialiased_cnns.BlurPool(C, filt_size=M, stride=2)]

# AvgPool --> BlurPool
baseline = nn.AvgPool2d(kernel_size=2, stride=2)
antialiased = antialiased_cnns.BlurPool(C, filt_size=M, stride=2)
```

We assume incoming tensor has `C` channels. Computing a layer at stride 1 instead of stride 2 adds memory and run-time. As such, we typically skip antialiasing at the highest-resolution (early in the network), to prevent large increases.

If you already trained a model, and then add antialiasing, you can fine-tune from that old model:

``` python
antialiased_cnns.copy_params_buffers(old_model, antialiased_model)
```

If this doesn't work, you can just copy the parameters (and not buffers). Adding antialiasing doesn't add any parameters, so the parameter lists are identical. (It does add buffers, so some heuristic is used to match the buffers, which may throw an error.)

``` python
antialiased_cnns.copy_params(old_model, antialiased_model)
```

<img src='https://richzhang.github.io/antialiased-cnns/resources/antialias_mod.jpg' width=800><br>

## (3) ImageNet Evaluation, Results, and Training code


**Accuracy**
|          | Baseline | Antialiased | Delta |
| :------: | :------: | :---------: | :---------: |
| AlexNet | 56.55 | 56.72 | +0.17 |
| VGG16 | 71.59 | 72.43 | +0.84 |
| VGG16bn | 73.36 | 74.12 | +0.76 |
| Resnet18 | 69.74 | 71.48 | +1.74 |
| Resnet34 | 73.30 | 74.38 | +1.08 |
| Resnet50 | 76.16 | 77.23 | +1.07 |
| Resnet101 | 77.37 | 78.22 | +0.85 |
| DenseNet121 | 74.43 | 75.29 | +0.86 |
| MobileNetv2 | 71.88 | 72.72 | +0.84 |


**Consistency**
|          | Baseline | Antialiased | Delta |
| :------: | :------: | :---------: | :---------: |
| AlexNet | 78.18 | 82.54 | +4.36 |
| VGG16 | 88.52 | 89.92  | +1.40 |
| VGG16bn | 89.24 | 91.22 | +1.98 |
| Resnet18 | 85.11 | 88.07 | +2.96 |
| Resnet34 | 87.56 | 89.53 | +1.97 |
| Resnet50 | 89.20 | 91.29 | +2.09 |
| Resnet101 | 89.81 | 91.85 | +2.04 |
| DenseNet121 | 88.81 | 90.29 | +1.48 |
| MobileNetv2 | 86.50 | 87.72 | +1.22 |


To reduce clutter, extended results are [here](README_IMAGENET.md). Help improve the results!

## Licenses

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

All material is made available under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license by Adobe Inc. You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicating any changes** that you've made.

The repository builds off the PyTorch [examples repository](https://github.com/pytorch/examples) and torchvision [models repository](https://github.com/pytorch/vision/tree/master/torchvision/models). These are [BSD-style licensed](https://github.com/pytorch/examples/blob/master/LICENSE).

## Citation, contact

If you find this useful for your research, please consider citing this [bibtex](https://richzhang.github.io/index_files/bibtex_icml2019.txt). Please contact Richard Zhang \<rizhang at adobe dot com\> with any comments or feedback.


