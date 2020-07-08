################################################################################
# MIT License
#
# Copyright (c) 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(VanillaRNN, self).__init__()
 
        self.seq_length = seq_length
        self.num_hidden = num_hidden
        self.input_dim = input_dim
        self.device = device
        
        Whx = torch.empty(input_dim, num_hidden, device=device)   #dim: D x H
        Whh = torch.empty(num_hidden, num_hidden, device=device)  #dim: H x H
        Wph = torch.empty(num_hidden, num_classes, device=device) #dim: H x C
        bh  = torch.zeros(num_hidden, device=device)  #dim: H
        bp  = torch.zeros(num_classes, device=device) #dim: C
        
        #init method?
        self.Whx = nn.Parameter( nn.init.xavier_uniform_(Whx) ) 
        self.Whh = nn.Parameter( nn.init.xavier_uniform_(Whh) ) 
        self.Wph = nn.Parameter( nn.init.xavier_uniform_(Wph) ) 
        self.bh  = nn.Parameter( bh ) 
        self.bp  = nn.Parameter( bp ) 

    def forward(self, x):
#     def forward(self, x, timestepGrad): #to get gradients
        
        # x dim: B x T x D
        h = torch.zeros(x.size(0), self.num_hidden, device=self.device) #dim: B x H
        
#         #to get gradients
#         if timestepGrad == 0:
#             hGrad = h.clone().detach().requires_grad_(True)
#             h = hGrad
            
        for t in range(self.seq_length):
            xt = x[:,t,:] if self.input_dim > 1 else x[:,t] #dim: B x D 
            #treat each batch independently
            h = torch.tanh(xt @ self.Whx + h @ self.Whh + self.bh) #dim: B x H
            
#             #to get gradients
#             if t+1 == timestepGrad:
#                 hGrad = h.clone().detach().requires_grad_(True) #keep track of grad
#                 h = hGrad
             
        pt = h @ self.Wph + self.bp #dim: B x C
        #softmax is done in cross entropy loss
        
        return pt
#         return pt, hGrad #to get gradients
