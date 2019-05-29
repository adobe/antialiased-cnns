
# Training & Evaluating on Antialiased models on Imagenet

We describe how to evaluate models for shift-invariance.

## Prepare ImageNet

- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## Evaluating models

We provide models with filter sizes 2,3,5 for AlexNet, VGG16, VGG16bn, ResNet18,34,50,101 and DenseNet121.

### Evaluating accuracy

```bash
python main.py --data /PTH/TO/ILSVRC2012 -e -f 3 -a alexnet_lpf --weights ./weights/alexnet_lpf3.pth.tar --gpu 0
python main.py --data /PTH/TO/ILSVRC2012 -e -f 3 -a vgg16_lpf --weights ./weights/vgg16_lpf3.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -e -f 3 -a vgg16_bn_lpf --weights ./weights/vgg16_bn_lpf3.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -e -f 3 -a resnet18_lpf --weights ./weights/resnet18_lpf3.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -e -f 3 -a resnet34_lpf --weights ./weights/resnet34_lpf3.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -e -f 3 -a resnet50_lpf --weights ./weights/resnet50_lpf3.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -e -f 3 -a resnet101_lpf --weights ./weights/resnet101_lpf3.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -e -f 3 -a densenet121_lpf --weights ./weights/densenet121_lpf3.pth.tar
```

### Evaluating consistency

Same as above, but flag `-es` evaluates the shift-consistency -- how often two random `224x224` crops are classified the same.

```bash
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -f 3 -a alexnet_lpf --weights ./weights/alexnet_lpf3.pth.tar --gpu 0
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -f 3 -a vgg16_lpf --weights ./weights/vgg16_lpf3.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -f 3 -a vgg16_bn_lpf --weights ./weights/vgg16_bn_lpf3.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -f 3 -a resnet18_lpf --weights ./weights/resnet18_lpf3.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -f 3 -a resnet34_lpf --weights ./weights/resnet34_lpf3.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -f 3 -a resnet50_lpf --weights ./weights/resnet50_lpf3.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -f 3 -a resnet101_lpf --weights ./weights/resnet101_lpf3.pth.tar
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -f 3 -a densenet121_lpf --weights ./weights/densenet121_lpf3.pth.tar
```

Some notes:
- These line commands are very similar to the base PyTorch [repository](https://github.com/pytorch/examples/tree/master/imagenet). We simply add suffix `_lpf` to the architecture and specify `-f` for filter size.
- Substitute `-f 3` and appropriate filepath for different filter sizes.
- The example commands use our weights. You can them from your own training checkpoints by subsituting `--weights PTH/TO/WEIGHTS` for `--resume PTH/TO/CHECKPOINT`.

## (3) Training antialiased models

The following commands train antialiased AlexNet, VGG16, VGG16bn, ResNet18,34,50, and Densenet121 models with filter size 5. Best checkpoint will be saved `[[OUT_DIR]]/model_best.pth.tar`.

```bash
python main.py --data /PTH/TO/ILSVRC2012 -f 3 -a alexnet_lpf --out-dir alexnet_lpf3 --gpu 0 --lr .01
python main.py --data /PTH/TO/ILSVRC2012 -f 3 -a vgg16_lpf --out-dir vgg16_lpf3 --lr .01 -b 128 -ba 2
python main.py --data /PTH/TO/ILSVRC2012 -f 3 -a vgg16_bn_lpf --out-dir vgg16_bn_lpf3 --lr .05 -b 128 -ba 2
python main.py --data /PTH/TO/ILSVRC2012 -f 3 -a resnet18_lpf --out-dir resnet18_lpf3
python main.py --data /PTH/TO/ILSVRC2012 -f 3 -a resnet34_lpf --out-dir resnet34_lpf3
python main.py --data /PTH/TO/ILSVRC2012 -f 3 -a resnet50_lpf --out-dir resnet50_lpf3
python main.py --data /PTH/TO/ILSVRC2012 -f 3 -a resnet101_lpf --out-dir resnet101_lpf3
python main.py --data /PTH/TO/ILSVRC2012 -f 3 -a densenet121_lpf --out-dir densenet121_lpf3 -b 128 -ba 2
```

Some notes:
- As suggested by the official repository, AlexNet and VGG16 require lower learning rates of `0.01` (default is `0.1`). 
- VGG16_bn also required a slightly lower learning rate of `0.05`.
- I train AlexNet on a single GPU (the network is fast, so preprocessing becomes the limiting factor if multiple GPUs are used).
- Default batch size is `256`. Some extra memory is added for the antialiasing layers, so the default batchsize may no longer fit in memory. To get around this, we simply accumulate gradients over 2 smaller batches `-b 128` with flag `--ba 2`. You may find this useful, even for the default models, if you are training with smaller/fewer GPUs. It is not exactly identical to training with a large batch, as the batchnorm statistics will be computed with a smaller batch.

Checkpoint vs weights:
- To resume training session, use flag `--resume [[OUT_DIR]]/checkpoint_[[NUM]].pth.tar`. This flag can be used instead of `--weights` in the evaluation scripts above.
- Saved checkpoints include model weights and optimizer parameters. Also, if you trained with parallelization, then the weights/optimizer dicts will include parallelization. To strip optimizer parameters away and 'deparallelize' the model weights, run the following command (with appropriate substitution) afterwards:

```bash
python main.py --data /PTH/TO/ILSVRC2012 -f 3 -a resnet18_lpf --resume resnet18_lpf3/model_best.pth.tar --save_weights resnet18_lpf3/weights.pth.tar
```

I used this postprocessing step to provide the pretrained weights. As seen [here](https://github.com/adobe/antialiased-cnns/blob/master/main.py#L265), weights should be loaded *before* parallelizing the model. Meanwhile, the [checkpoint](https://github.com/adobe/antialiased-cnns/blob/master/main.py#L308) is loaded *after* parallelizing the model.
 