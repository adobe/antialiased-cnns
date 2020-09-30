# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.

import torch

def copy_params(src_model, dest_model):
    src_params = list(src_model.parameters())
    dest_params = list(dest_model.parameters())

    # antialiasing shouldn't change number of parameters, so these lists should be identical
    assert(len(src_params)==len(dest_params))
    with torch.no_grad():
        for params in zip(src_params, dest_params):
            params[1][...] = params[0][...]

def copy_buffers(src_model, dest_model):
	src_buffers = list(src_model.buffers())
	dest_buffers = list(dest_model.buffers())

	cc = 0
	for (bb,buffer) in enumerate(src_buffers):
		cond = False
		while(not cond): # antialiasing adds buffers, so these lists won't match
			cond = buffer.shape==dest_buffers[cc].shape
			cc+=1
			if(cc==len(dest_buffers) and not cond):
				raise ValueError('Could not find matching buffer in [dest_model]')
		with torch.no_grad():
			dest_buffers[cc-1][...] = buffer[...]

def copy_params_buffers(src_model, dest_model):
	copy_params(src_model, dest_model)
	copy_buffers(src_model, dest_model)
