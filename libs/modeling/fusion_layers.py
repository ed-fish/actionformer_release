
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from .weight_init import trunc_normal_
from .blocks import ModalLocalMaskedMHCA

class SumFusion(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, v, a, masks):
        out = tuple()
        for v_i, a_i in zip(v, a):
            out += (v_i + a_i, )
        return out
    
    
class ConcatFusion(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, v, a, masks):
        out = tuple()
        for v_i, a_i in zip(v, a):
            f_out = torch.cat((v_i, a_i), dim=1) 
            out += (f_out, )
        return out
    
    
class CrossAttFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.modal_mhsa = ModalLocalMaskedMHCA(512, 512, 4, 9)
         
    def forward(self, v, a, masks):
        out = tuple()
        for v_i, a_i, mask in zip(v, a, masks):
            f_out, mask = self.modal_mhsa(v_i, a_i, mask)
            out += (f_out, )
        return out
    
    
class CrossAttFusionGated(nn.Module):
    def __init__(self):
        super().__init__()
        self.modal_mhsa = ModalLocalMaskedMHCA(512, 512, 4, 9)
         
    def forward(self, v, a, masks):
        out = tuple()
        for v_i, a_i, mask in zip(v, a, masks):
            f_out, mask = self.modal_mhsa(v_i, a_i, mask)
            out += (f_out, )
        return out
        