# <b>Making Convolutional Networks Shift-Invariant Again</b> [[Project Page]](http://richzhang.github.io/antialiased-cnns/) [[Paper]](https://arxiv.org/abs/1904.11486) <br>
[Richard Zhang](https://richzhang.github.io/). To appear in [ICML, 2019](https://arxiv.org/abs/1904.11486).


<img src='https://richzhang.github.io/antialiased-cnns/resources/gifs2/video_00810.gif' align="right" width=300>

This repository contains examples of anti-aliased convnets. We build off publicly available PyTorch ImageNet [repository](https://github.com/pytorch/examples/tree/master/imagenet). This repository contains add-ons related to anti-aliasing:

- a [low-pass filter layer](models_lpf/__init__.py#L8) (called `BlurPool` in the paper), which can be easily plugged into any network
- modified AlexNet, VGG, ResNet, DenseNet architectures, along with pretrained nets
- code for evaluating how shift-invariant a model is (`--evaluate-shift` flag)

## Getting started

### PyTorch + ImageNet
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

### Downloading anti-aliased models

- Run `bash weights/get_antialiased_models.py`


## Evaluating anti-aliased models

We provide models with filter sizes 2,3,5 for AlexNet, VGG16, ResNet50, and DenseNet121. Substitute `-f 5` and appropriate filepath. The example commands use our weights. You can them from your own training session.

These line commands are very similar to the base PyTorch [repository](https://github.com/pytorch/examples/tree/master/imagenet). We simply add suffix `_lpf` to the architecture and specify `-f` for filter size.

### Evaluating classification accuracy

```bash
python main.py --data /PTH/TO/ILSVRC2012 -a alexnet_lpf -f 5 --resume ./weights/alexnet_lpf5.pth.tar -e --gpu 0
python main.py --data /PTH/TO/ILSVRC2012 -a vgg16_lpf -f 5 --resume ./weights/vgg16_lpf5.pth.tar -e
python main.py --data /PTH/TO/ILSVRC2012 -a resnet50_lpf -f 5 --resume ./weights/resnet50_lpf5.pth.tar -e
python main.py --data /PTH/TO/ILSVRC2012 -a densenet121_lpf -f 5 --resume ./weights/densenet121_lpf5.pth.tar -e
```

### Evaluating classification consistency

Same as above, but flag `-es` evaluates the shift-consistency -- how often two random `224x224` crops are classified the same.

```bash
python main.py --data /PTH/TO/ILSVRC2012 -a alexnet_lpf -f 5 --resume ./weights/alexnet_lpf5.pth.tar -b 8 -es --gpu 0
python main.py --data /PTH/TO/ILSVRC2012 -a vgg16_lpf -f 5 --resume ./weights/vgg16_lpf5.pth.tar -b 8 -es
python main.py --data /PTH/TO/ILSVRC2012 -a resnet50_lpf -f 5 --resume ./weights/resnet50_lpf5.pth.tar -b 8 -es
python main.py --data /PTH/TO/ILSVRC2012 -a densenet121_lpf -f 5 --resume ./weights/densenet121_lpf5.pth.tar -b 8 -es
```

## Training

AlexNet and VGG16 require lower learning rates of `0.01` (default is `0.1`). I train AlexNet on a single GPU (the network is fast, so preprocessing becomes the limiting factor if multiple GPUs are used). Default batch size is `256`. Some extra memory is added for the low-pass filter layers, so a default batchsize may no longer fit in memory. To get around this, we simply accumulate gradients over 2 smaller batches with flag `--ba 2`.

Output models will be in `OUT_DIR/model_best.pth.tar`, which you can substitute in the test commands above.

```bash
python main.py --data /PTH/TO/ILSVRC2012 -a alexnet_lpf -f 5 --out-dir alexnet_lpf5 --gpu 0 --lr .01
python main.py --data /PTH/TO/ILSVRC2012 -a vgg16_lpf -f 5 --out-dir vgg16_lpf5 --lr .01 -b 128 -ba 2
python main.py --data /PTH/TO/ILSVRC2012 -a resnet50_lpf -f 5 --out-dir resnet50_lpf5
python main.py --data /PTH/TO/ILSVRC2012 -a densenet121_lpf -f 5 --out-dir densenet121_lpf5 -b 128 -ba 2
```

## Modifying your own architecture to be more shift-invariant

We show how to make your `MaxPool` and `Conv2d` more shift-invariant. The methodology is simple -- first evaluate with stride 1, and then use the `Downsample` layer to do the striding. To follow the paper, we will use blur kernel size `M`, pool/conv kernel size `K`, and stride `S`. Assume that the tensor as `C` channels.

```python
from models_lpf import *
```

|   |Original|Anti-aliased replacement|
|:-:|---|---|
|**MaxPool <br>--><br> MaxBlurPool** | `MaxPool2d(kernel_size=K, stride=S)`| `MaxPool2d(kernel_size=K, stride=1),` <br> `Downsample(filt_size=M, stride=S, channels=C)`|
|**StridedConv <br>--><br> ConvBlurPool**| `Conv2d(C_in, C_out, kernel_size=K, stride=S, padding=(K-1)/2),` <br> `ReLU(inplace=True)` | `Conv2d(C_in, C_out, kernel_size=K, stride=1, padding=(K-1)/2),` <br> `ReLU(inplace=True),` <br> `Downsample(filt_size=M, stride=S, channels=C_out)` |
|**AvgPool <br>--><br> BlurPool**| `AvgPool2d(kernel_size=K, stride=S),` | `Downsample(filt_size=M, stride=S, channels=C)`|


**`MaxPool` --> `MaxBlurPool`**

Replace:

- `nn.MaxPool2d(kernel_size=K, stride=S)`

with:

- `nn.MaxPool2d(kernel_size=K, stride=1)`
- `Downsample(filt_size=M, stride=S, channels=C)`

**`StridedConv` --> `ConvBlurPool`**

Replace:

- `nn.Conv2d(C_in, C_out, kernel_size=K, stride=S, padding=(K-1)/2), nn.ReLU(inplace=True)`

with:

- `nn.Conv2d(C_in, C_out, kernel_size=K, stride=1, padding=(K-1)/2)`
- `nn.ReLU(inplace=True), Downsample(filt_size=M, stride=S, channels=C_out)`

`AvgPool` is a special case of `BlurPool`. Replacing with `BlurPool` will make it more shift-invariant.

**`AvgPool` --> `BlurPool`**

Replace:

- `nn.AvgPool2d(kernel_size=K, stride=S)`

with:

- `Downsample(filt_size=M, stride=S, channels=C)`

### Some things to watch out for

The `Downsample` layer is simply a wrapper around a `Conv2d` layer, with hard-coded weights. As such, two things may happen:

**(1) Initialization code may accidentally overwrite the low-pass filter weights.** An example bypass is shown below.

```python
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
            # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        else:
            print('Not initializing')
```

**(2) Weights may accidentally start training** When initialized, the layer freezes the weights with `p.requires_grad = False` command. If you overwrite this, the fixed weights will start training and will not anti-alias properly for you.


### Acknowledgments

This repository is built off the PyTorch ImageNet training [repository](https://github.com/pytorch/examples/tree/master/imagenet).

### Citation

If you find this useful for your research, please consider citing this [bibtex](https://richzhang.github.io/index_files/bibtex_icml2019.txt).



