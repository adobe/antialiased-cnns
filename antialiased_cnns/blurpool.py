# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from functools import partial


class ZeroPad1d(torch.nn.modules.padding.ConstantPad1d):
    def __init__(self, padding):
        super(ZeroPad1d, self).__init__(padding, 0.)


class ZeroPad3d(torch.nn.modules.padding.ConstantPad3d):
    def __init__(self, padding):
        super(ZeroPad3d, self).__init__(padding, 0.)


def get_padding_layer(pad_type, dim):
    pad_layer_dict = {
        'reflect': {
            1: nn.ReflectionPad1d,
            2: nn.ReflectionPad2d,
        },
        'replicate': {
            1: nn.ReplicationPad1d,
            2: nn.ReplicationPad2d,
            3: nn.ReplicationPad3d
        },
        'zero': {
            1: ZeroPad1d,
            2: nn.ZeroPad2d,
            3: ZeroPad3d
        }
    }
    pad_layer_dict['refl'] = pad_layer_dict['reflect']
    pad_layer_dict['repl'] = pad_layer_dict['replicate']
    try:
        return pad_layer_dict[pad_type][dim]
    except KeyError:
        raise NotImplementedError


class BlurPoolND(nn.Module):

    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0, dims=2):

        super(BlurPoolND, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))] * dims
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        a = None
        if self.filt_size == 1:
            a = np.array([1., ])
        elif self.filt_size == 2:
            a = np.array([1., 1.])
        elif self.filt_size == 3:
            a = np.array([1., 2., 1.])
        elif self.filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif self.filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif self.filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = create_filter(a, dims)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', get_filter_for_registration(filt, self.channels, dims))
        self.pad = get_padding_layer(pad_type, dims)(self.pad_sizes)
        self.identity_inference = get_identity_inference_func(dims)
        self.conv_func = get_conv_func(dims)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return self.identity_inference(inp, self.stride)
            else:
                return self.identity_inference(self.pad(inp), self.stride)
        else:
            return self.conv_func(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def create_filter(a, dims):
    if dims == 1:
        return torch.Tensor(a)
    elif dims == 2:
        return torch.Tensor(a[:, None] * a[None, :])
    elif dims == 3:
        b = a[:, None] * a[None, :]
        return torch.Tensor(a[:, None, None] * b[None, :, :])
    return NotImplementedError


def get_conv_func(dims):
    if dims == 1:
        return F.conv1d
    elif dims == 2:
        return F.conv2d
    elif dims == 3:
        return F.conv3d
    return NotImplementedError


def get_filter_for_registration(filt, channels, dims):
    if dims == 1:
        return filt[None, None, :].repeat((channels, 1, 1))
    elif dims == 2:
        return filt[None, None, :, :].repeat((channels, 1, 1, 1))
    elif dims == 3:
        return filt[None, None, :, :, :].repeat((channels, 1, 1, 1, 1))
    return NotImplementedError


def get_identity_inference_func(dims):
    if dims == 1:
        return lambda x, s: x[:, :, ::s]
    elif dims == 2:
        return lambda x, s: x[:, :, ::s, ::s]
    elif dims == 3:
        return lambda x, s: x[:, :, ::s, ::s, ::s]
    return NotImplementedError


BlurPool1D = partial(BlurPoolND, dims=1)
BlurPool = BlurPoolND
BlurPool3D = partial(BlurPoolND, dims=3)
