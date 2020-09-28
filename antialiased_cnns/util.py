# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

import torch

def copy_params(src_model, dest_model):
    src_params = list(src_model.parameters())
    dest_params = list(dest_model.parameters())
    assert(len(src_params)==len(dest_params))
    with torch.no_grad():
        for params in zip(src_params, dest_params):
            params[1][...] = params[0][...]

    return dest_model