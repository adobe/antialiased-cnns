
# Training & Evaluating on Antialiased models on Imagenet

We describe how to evaluate models for shift-invariance.

## (0) Prepare ImageNet

- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## (1) Evaluating models

We provide models with filter sizes 2,3,4,5 for AlexNet, VGG16, VGG16bn, ResNet18,34,50,101, DenseNet121, and MobileNetv2.

### Evaluating accuracy

```bash
python main.py --data /PTH/TO/ILSVRC2012 -e -a alexnet_lpf4 --pretrained --gpu 0
python main.py --data /PTH/TO/ILSVRC2012 -e -a vgg16_lpf4 --pretrained
python main.py --data /PTH/TO/ILSVRC2012 -e -a vgg16_bn_lpf4 --pretrained
python main.py --data /PTH/TO/ILSVRC2012 -e -a resnet18_lpf4 --pretrained
python main.py --data /PTH/TO/ILSVRC2012 -e -a resnet34_lpf4 --pretrained
python main.py --data /PTH/TO/ILSVRC2012 -e -a resnet50_lpf4 --pretrained
python main.py --data /PTH/TO/ILSVRC2012 -e -a resnet101_lpf4 --pretrained
python main.py --data /PTH/TO/ILSVRC2012 -e -a densenet121_lpf4 --pretrained
python main.py --data /PTH/TO/ILSVRC2012 -e -a mobilenet_v2_lpf4 --pretrained
```

<!-- **Ensembling**
- Add `-ens E` to ensemble over `E` random crops.
- By default, we average over logits; add `-ens_sm` to average after softmax (this works better empirically).
- By default, we use all random crops; add `-ens_cf` to always have center crop in the ensemble (this works worse). -->

<!-- ```bash
python main.py --data /PTH/TO/ILSVRC2012 -e  -ens 5 -ens_sm -a resnet34_lpf4 --weights ./weights/resnet34_lpf4.pth.tar
``` -->

### Evaluating consistency

Same as above, but flag `-es` evaluates the shift-consistency -- how often two random `224x224` crops are classified the same.

```bash
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -a alexnet_lpf4 --pretrained --gpu 0
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -a vgg16_lpf4 --pretrained
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -a vgg16_bn_lpf4 --pretrained
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -a resnet18_lpf4 --pretrained
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -a resnet34_lpf4 --pretrained
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -a resnet50_lpf4 --pretrained
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -a resnet101_lpf4 --pretrained
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -a densenet121_lpf4 --pretrained
python main.py --data /PTH/TO/ILSVRC2012 -es -b 8 -a mobilenet_v2_lpf4 --pretrained
```

Some notes:
- These line commands are very similar to the base PyTorch [repository](https://github.com/pytorch/examples/tree/master/imagenet). Change `_lpf#` with filter size (2,3,4,5).
- The example commands use our pretrained. You can them from your own training checkpoints by subsituting `--weights PTH/TO/WEIGHTS` for `--resume PTH/TO/CHECKPOINT`.


## (2) Training antialiased models

The following commands train antialiased AlexNet, VGG16, VGG16bn, ResNet18,34,50, and Densenet121 models with filter size 5. Best checkpoint will be saved `[[OUT_DIR]]/model_best.pth.tar`.

```bash
python main.py --data /PTH/TO/ILSVRC2012 -a alexnet_lpf4 --out-dir alexnet_lpf4 --gpu 0 --lr .01
python main.py --data /PTH/TO/ILSVRC2012 -a vgg16_lpf4 --out-dir vgg16_lpf4 --lr .01 -b 128 -ba 2
python main.py --data /PTH/TO/ILSVRC2012 -a vgg16_bn_lpf4 --out-dir vgg16_bn_lpf4 --lr .05 -b 128 -ba 2
python main.py --data /PTH/TO/ILSVRC2012 -a resnet18_lpf4 --out-dir resnet18_lpf4
python main.py --data /PTH/TO/ILSVRC2012 -a resnet34_lpf4 --out-dir resnet34_lpf4
python main.py --data /PTH/TO/ILSVRC2012 -a resnet50_lpf4 --out-dir resnet50_lpf4
python main.py --data /PTH/TO/ILSVRC2012 -a resnet101_lpf4 --out-dir resnet101_lpf4
python main.py --data /PTH/TO/ILSVRC2012 -a densenet121_lpf4 --out-dir densenet121_lpf4
python main.py --data /PTH/TO/ILSVRC2012 -a mobilenet_v2_lpf4 --out-dir mobilenet_v2_lpf4 --lr .05 --cos_lr --wd 4e-5 --ep 150
```

Some notes:
- As suggested by the official repository, AlexNet and VGG16 require lower learning rates of `0.01` (default is `0.1`). 
- VGG16_bn also required a slightly lower learning rate of `0.05`.
- I train AlexNet on a single GPU (the network is fast, so preprocessing becomes the limiting factor if multiple GPUs are used).
- MobileNet was trained with the training recipe from [here](https://github.com/tonylins/pytorch-mobilenet-v2#training-recipe).
- Default batch size is `256`. Some extra memory is added for the antialiasing layers, so the default batchsize may no longer fit in memory. To get around this, we simply accumulate gradients over 2 smaller batches `-b 128` with flag `--ba 2`. You may find this useful, even for the default models, if you are training with smaller/fewer GPUs. It is not exactly identical to training with a large batch, as the batchnorm statistics will be computed with a smaller batch.

Checkpoint vs weights:
- To resume training session, use flag `--resume [[OUT_DIR]]/checkpoint_[[NUM]].pth.tar`. This flag can be used instead of `--weights` in the evaluation scripts above.
- Saved checkpoints include model weights and optimizer parameters. Also, if you trained with parallelization, then the weights/optimizer dicts will include parallelization. To strip optimizer parameters away and 'deparallelize' the model weights, run the following command (with appropriate substitution) afterwards:

```bash
python main.py -a resnet18_lpf4 --resume resnet18_lpf4/model_best.pth.tar --save_weights resnet18_lpf4/weights.pth.tar
```

I used this postprocessing step to provide the pretrained weights. As seen [here](https://github.com/adobe/antialiased-cnns/blob/master/main.py#L265), weights should be loaded *before* parallelizing the model. Meanwhile, the [checkpoint](https://github.com/adobe/antialiased-cnns/blob/master/main.py#L308) is loaded *after* parallelizing the model.

## (3) Results

Results are [here](README.md#3-results).
