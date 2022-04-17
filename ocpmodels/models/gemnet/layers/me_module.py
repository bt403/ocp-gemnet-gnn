"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

class GatedLayer(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_size, in_size)
        self.linear2 = torch.nn.Linear(in_size, out_size)
    
    def forward(self, X):
        X = torch.nn.ReLU()(self.linear1(X))
        return torch.sigmoid(self.linear2(X))

class MEModule(torch.nn.Module):
    def __init__(self, 
                num_modules = 3, emb_size_attention = 12, num_radial = 32):
        super().__init__()
        self.num_modules = num_modules
        self.gated_layers = torch.nn.ModuleList([GatedLayer(emb_size_attention*2, num_radial*2) for _ in range(self.num_modules)])
        self.linear = torch.nn.Linear(num_modules*num_radial*2,num_radial)
        
    def forward(self, rbf, h, idx_s, idx_t):
        me_blocks = []
        for i in range(self.num_modules):
            h_emb_s = h[idx_s] 
            h_emb_t = h[idx_t] 
            h_emb = torch.cat((h_emb_s, h_emb_t), dim=1)
            gated_ml = self.gated_layers[i](h_emb)
            output = rbf * gated_ml
            me_blocks.append(output)
        me_blocks_stack = torch.hstack(me_blocks)
        output = torch.squeeze(self.linear(me_blocks_stack), dim = -1)     
        return output
