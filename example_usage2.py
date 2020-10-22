# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

import antialiased_cnns

for force in [False, True]:
	model = antialiased_cnns.resnet18(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.resnet34(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.resnet50(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.resnet101(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.resnet152(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.wide_resnet50_2(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.wide_resnet101_2(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.resnext50_32x4d(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.resnext101_32x8d(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.alexnet(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.vgg11(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.vgg11_bn(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.vgg13(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.vgg13_bn(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.vgg16(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.vgg16_bn(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.vgg19(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.vgg19_bn(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.densenet121(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.densenet169(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.densenet201(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.densenet161(pretrained=True, _force_nonfinetuned=force)
	model = antialiased_cnns.mobilenet_v2(pretrained=True, _force_nonfinetuned=force)

