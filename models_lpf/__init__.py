
import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
from IPython import embed

class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, gpu=None, pad_off=0, hidden=False):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [(pad_size+pad_off) for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
        self.hidden = hidden

        pad_amt = 0
        self.ds_ind = 1
        self.ds = [get_pad_layer(pad_type)(self.pad_sizes),nn.Conv2d(channels,channels,kernel_size=self.filt_size,padding=pad_amt,stride=stride,bias=False,groups=channels),]

        print('Filter size [%i]'%filt_size)
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.ds[self.ds_ind].weight.data[:,0,:,:] = filt[None,:,:]
        if(not hidden):
            self.ds = nn.Sequential(*self.ds)
            for p in self.ds[1].parameters():
                p.requires_grad = False
        else:
            self.ds = [nn.Sequential(*self.ds),]
            self.ds[0] = self.ds[0].cuda()

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                if(not self.hidden):
                    return self.ds[0](inp)[:,:,::self.stride,::self.stride]
                else:
                    return self.ds[0][0](inp)[:,:,::self.stride,::self.stride]
        else:
            if(not self.hidden):
                return self.ds(inp)
            else:
                return self.ds[0](inp)


def get_pad_layer(pad_type):
    if(pad_type in ['circ','circle']):
        PadLayer = CircPad
    elif(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer
