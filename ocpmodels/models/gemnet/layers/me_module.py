"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math

import numpy as np
import torch
from scipy.special import binom


class MEModule(torch.nn.Module):
    """

    Parameters
    ----------
    num_radial: int
        Controls maximum frequency.
    cutoff: float
        Cutoff distance in Angstrom.
    rbf: dict = {"name": "gaussian"}
        Basis function and its hyperparameters.
    envelope: dict = {"name": "polynomial", "exponent": 5}
        Envelope function and its hyperparameters.
    """

    def __init__(self, 
                num_modules = 3, emb_size_attention = 12):
        super().__init__()
        self.num_modules = num_modules
        self.mha_layers = torch.nn.ModuleList([torch.nn.MultiheadAttention(emb_size_attention*2, 1) for _ in range(self.num_modules)])
        self.linear = torch.nn.Linear(num_modules*emb_size_attention*2,128)
        
    def forward(self, rbf, h, idx_s, idx_t):
        me_blocks = []
        for i in range(self.num_modules):
            #h_emb_s = h[i][idx_s] 
            #h_emb_t = h[i][idx_t] 
            h_emb_s = h[idx_s] 
            h_emb_t = h[idx_t] 
            h_emb = torch.cat((h_emb_s, h_emb_t), dim=1)
            attention, _ = self.mha_layers[i](h_emb, h_emb, h_emb)
            output = rbf * attention
            me_blocks.append(output)
        #me_blocks_stack = torch.stack(me_blocks, dim=-1)
        me_blocks_stack = torch.hstack(me_blocks)
        output = torch.squeeze(self.linear(me_blocks_stack), dim = -1)     
        return output
