from torch.nn import init
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

from args import args as parser_args
import numpy as np

DenseConv = nn.Conv2d

def sparseFunction(x, s, activation=torch.relu, f=torch.sigmoid):
    return torch.sign(x)*activation(torch.abs(x)-f(s))

def initialize_sInit():

    if parser_args.sInit_type == "constant":
        return parser_args.sInit_value*torch.ones([1, 1])

class STRConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.activation = torch.relu

        if parser_args.sparse_function == 'sigmoid':
            self.f = torch.sigmoid
            self.sparseThreshold = nn.Parameter(initialize_sInit())
        else:
            self.sparseThreshold = nn.Parameter(initialize_sInit())
    
    def forward(self, x):
        # In case STR is not training for the hyperparameters given in the paper, change sparseWeight to self.sparseWeight if it is a problem of backprop.
        # However, that should not be the case according to graph computation.
        sparseWeight = sparseFunction(self.weight, self.sparseThreshold, self.activation, self.f)
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def getSparsity(self, f=torch.sigmoid):
        sparseWeight = sparseFunction(self.weight, self.sparseThreshold,  self.activation, self.f)
        temp = sparseWeight.detach().cpu()
        temp[temp!=0] = 1
        return (100 - temp.mean().item()*100), temp.numel(), f(self.sparseThreshold).item()

class ChooseEdges(autograd.Function):
    @staticmethod
    def forward(ctx, weight, prune_rate):
        output = weight.clone()
        _, idx = weight.flatten().abs().sort()
        p = int(prune_rate * weight.numel())
        # flat_oup and output access the same memory.
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class DNWConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        print(f"=> Setting prune rate to {prune_rate}")

    def forward(self, x):
        w = ChooseEdges.apply(self.weight, self.prune_rate)

        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x

def GMPChooseEdges(weight, prune_rate):
    output = weight.clone()
    _, idx = weight.flatten().abs().sort()
    p = int(prune_rate * weight.numel())
    # flat_oup and output access the same memory.
    flat_oup = output.flatten()
    flat_oup[idx[:p]] = 0
    return output

class GMPConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        self.curr_prune_rate = 0.0
        print(f"=> Setting prune rate to {prune_rate}")

    def set_curr_prune_rate(self, curr_prune_rate):
        self.curr_prune_rate = curr_prune_rate

    def forward(self, x):
        w = GMPChooseEdges(self.weight, self.curr_prune_rate)
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x
        
def sigmoid(x):
    return float(1./(1.+np.exp(-x)))

class LCSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, padding_mode="zeros", mask_initial_value=0.):
        super(LCSConv, self).__init__()
        self.mask_initial_value = mask_initial_value
        
        self.in_channels = in_channels
        self.out_channels = out_channels    
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.bias = bias
        self.padding_mode = padding_mode
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels//groups, kernel_size, kernel_size))
        nn.init.xavier_normal_(self.weight)
        self.init_weight = nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
        self.init_mask()

        self.ticket = False
        self.tp = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.init_tp = nn.Parameter(torch.tensor(1.), requires_grad=False)
        
    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels//self.groups, self.kernel_size, self.kernel_size))
        nn.init.constant_(self.mask_weight, self.mask_initial_value)

    def compute_mask(self, tp, ticket):
        if ticket: mask = (self.mask_weight >= 0).float()
        else: mask = F.sigmoid(torch.abs(1./tp) * self.mask_weight)
        return mask

    def prune(self):
        self.mask_weight.data = torch.clamp(self.mask_weight.data, max=self.mask_initial_value)

    def forward(self, x):
        self.mask = self.compute_mask(self.tp, self.ticket)
        masked_weight = self.weight * self.mask
        out = F.conv2d(x, masked_weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)        
        return out
        
    def checkpoint(self):
        self.init_weight.data = self.weight.clone()       
        
    def rewind_weights(self):
        self.weight.data = self.init_weight.clone()
        
    def getSparsity(self, f=torch.sigmoid):
        tkmask = self.compute_mask(self.tp, True)
        temp = tkmask.detach().cpu()
        temp[temp!=0] = 1
        return (100 - temp.mean().item()*100), temp.numel()

    def extra_repr(self):
        return '{}, {}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
