finetune_resnet18:
	NETWORK='resnet18'
	mkdir finetune_${NETWORK}_lr0p01_ep60
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 256 -j 16 -p 1000 --finetune --lr .01 -ep 60 --out-dir finetune_${NETWORK}_lr0p01_ep60 > finetune_${NETWORK}_lr0p01_ep60/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 256 -j 16 -p 1000 --resume finetune_${NETWORK}_lr0p01_ep60/model_best.pth.tar -e > finetune_${NETWORK}_lr0p01_ep60/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 -p 1000 --resume finetune_${NETWORK}_lr0p01_ep60/model_best.pth.tar -es > finetune_${NETWORK}_lr0p01_ep60/con

finetune_resnet34:
	NETWORK='resnet34'
	mkdir finetune_${NETWORK}_lr0p01_ep60
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 256 -j 16 -p 1000 --finetune --lr .01 -ep 60 --out-dir finetune_${NETWORK}_lr0p01_ep60 > finetune_${NETWORK}_lr0p01_ep60/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 256 -j 16 -p 1000 --resume finetune_${NETWORK}_lr0p01_ep60/model_best.pth.tar -e > finetune_${NETWORK}_lr0p01_ep60/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 -p 1000 --resume finetune_${NETWORK}_lr0p01_ep60/model_best.pth.tar -es > finetune_${NETWORK}_lr0p01_ep60/con

finetune_resnet50:
	NETWORK='resnet50'
	mkdir finetune_${NETWORK}_lr0p01_ep60
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 256 -j 16 -p 1000 --finetune --lr .01 -ep 60 --out-dir finetune_${NETWORK}_lr0p01_ep60 > finetune_${NETWORK}_lr0p01_ep60/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 256 -j 16 -p 1000 --resume finetune_${NETWORK}_lr0p01_ep60/model_best.pth.tar -e > finetune_${NETWORK}_lr0p01_ep60/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 -p 1000 --resume finetune_${NETWORK}_lr0p01_ep60/model_best.pth.tar -es > finetune_${NETWORK}_lr0p01_ep60/con

finetune_resnet101:
	NETWORK='resnet101'
	mkdir finetune_${NETWORK}_lr0p01_ep60
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 256 -j 16 -p 1000 --finetune --lr .01 -ep 60 --out-dir finetune_${NETWORK}_lr0p01_ep60 > finetune_${NETWORK}_lr0p01_ep60/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 256 -j 16 -p 1000 --resume finetune_${NETWORK}_lr0p01_ep60/model_best.pth.tar -e > finetune_${NETWORK}_lr0p01_ep60/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 -p 1000 --resume finetune_${NETWORK}_lr0p01_ep60/model_best.pth.tar -es > finetune_${NETWORK}_lr0p01_ep60/con

finetune_resnet152:
	NETWORK='resnet152'
	mkdir finetune_${NETWORK}_lr0p01_ep60
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 256 -j 16 -p 1000 --finetune --lr .01 -ep 60 --out-dir finetune_${NETWORK}_lr0p01_ep60 > finetune_${NETWORK}_lr0p01_ep60/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 256 -j 16 -p 1000 --resume finetune_${NETWORK}_lr0p01_ep60/model_best.pth.tar -e > finetune_${NETWORK}_lr0p01_ep60/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 -p 1000 --resume finetune_${NETWORK}_lr0p01_ep60/model_best.pth.tar -es > finetune_${NETWORK}_lr0p01_ep60/con

finetune_alexnet:
	NETWORK='alexnet'
	mkdir finetune_${NETWORK}
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --finetune --out-dir finetune_${NETWORK} --gpu 0 --lr .001 -ep 60 > finetune_${NETWORK}/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar --gpu 0 -e > finetune_${NETWORK}/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar --gpu 0 -es > finetune_${NETWORK}/con

finetune_vgg11:
	NETWORK='vgg11'
	mkdir finetune_${NETWORK}
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --finetune --out-dir finetune_${NETWORK} --lr .001 -b 128 -ba 2 -ep 60 > finetune_${NETWORK}/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -e > finetune_${NETWORK}/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -es > finetune_${NETWORK}/con

finetune_vgg11_bn:
	NETWORK='vgg11_bn'
	mkdir finetune_${NETWORK}
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --finetune --out-dir finetune_${NETWORK} --lr .005 -b 128 -ba 2 -ep 60 > finetune_${NETWORK}/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -e > finetune_${NETWORK}/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -es > finetune_${NETWORK}/con

finetune_vgg13:
	NETWORK='vgg13'
	mkdir finetune_${NETWORK}
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --finetune --out-dir finetune_${NETWORK} --lr .001 -b 128 -ba 2 -ep 60 > finetune_${NETWORK}/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -e > finetune_${NETWORK}/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -es > finetune_${NETWORK}/con

finetune_vgg13_bn:
	NETWORK='vgg13_bn'
	mkdir finetune_${NETWORK}
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --finetune --out-dir finetune_${NETWORK} --lr .005 -b 128 -ba 2 -ep 60 > finetune_${NETWORK}/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -e > finetune_${NETWORK}/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -es > finetune_${NETWORK}/con

finetune_vgg16:
	NETWORK='vgg16'
	mkdir finetune_${NETWORK}
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --finetune --out-dir finetune_${NETWORK} --lr .001 -b 128 -ba 2 -ep 60 > finetune_${NETWORK}/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -e > finetune_${NETWORK}/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -es > finetune_${NETWORK}/con

finetune_vgg16_bn:
	NETWORK='vgg16_bn'
	mkdir finetune_${NETWORK}
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --finetune --out-dir finetune_${NETWORK} --lr .005 -b 128 -ba 2 -ep 60 > finetune_${NETWORK}/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -e > finetune_${NETWORK}/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -es > finetune_${NETWORK}/con

finetune_vgg19:
	NETWORK='vgg19'
	mkdir finetune_${NETWORK}
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --finetune --out-dir finetune_${NETWORK} --lr .001 -b 128 -ba 2 -ep 60 > finetune_${NETWORK}/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -e > finetune_${NETWORK}/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -es > finetune_${NETWORK}/con

finetune_vgg19_bn:
	NETWORK='vgg19_bn'
	mkdir finetune_${NETWORK}
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --finetune --out-dir finetune_${NETWORK} --lr .005 -b 128 -ba 2 -ep 60 > finetune_${NETWORK}/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -e > finetune_${NETWORK}/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -es > finetune_${NETWORK}/con

finetune_densenet121:
	NETWORK='densenet121'
	mkdir finetune_${NETWORK}
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --finetune --out-dir finetune_${NETWORK} -ep 60 --lr .01 > finetune_${NETWORK}/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -e > finetune_${NETWORK}/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -es > finetune_${NETWORK}/con

finetune_densenet169:
	NETWORK='densenet169'
	mkdir finetune_${NETWORK}
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --finetune --out-dir finetune_${NETWORK} -ep 60 --lr .01 > finetune_${NETWORK}/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -e > finetune_${NETWORK}/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -es > finetune_${NETWORK}/con

finetune_densenet161:
	NETWORK='densenet161'
	mkdir finetune_${NETWORK}
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --finetune --out-dir finetune_${NETWORK} -ep 60 --lr .01 > finetune_${NETWORK}/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -e > finetune_${NETWORK}/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -es > finetune_${NETWORK}/con

finetune_mobilenet_v2:
	NETWORK='mobilenet_v2'
	mkdir finetune_${NETWORK}
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --finetune --out-dir finetune_${NETWORK} --lr .05 --cos_lr --wd 4e-5 --ep 150 --start-epoch 50 > finetune_${NETWORK}/log
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -e > finetune_${NETWORK}/acc
	~/anaconda3/bin/python main.py --data /mnt/ssd/tmp/rzhang/ILSVRC2012 -a ${NETWORK}_lpf4 -b 8 -j 16 --resume finetune_${NETWORK}/model_best.pth.tar -es > finetune_${NETWORK}/con




