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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, device='cpu'):
        super(LSTM, self).__init__()
        
        self.seq_length = seq_length
        self.num_hidden = num_hidden
        self.input_dim = input_dim
        self.device = device
        
        Wgx = torch.empty(input_dim, num_hidden, device=device)   #dim: D x H
        Wgh = torch.empty(num_hidden, num_hidden, device=device)  #dim: H x H
        bg  = torch.zeros(num_hidden, device=device)              #dim: H
        
        Wix = torch.empty(input_dim, num_hidden, device=device)   #dim: D x H
        Wih = torch.empty(num_hidden, num_hidden, device=device)  #dim: H x H
        bi  = torch.zeros(num_hidden, device=device)              #dim: H
        
        Wfx = torch.empty(input_dim, num_hidden, device=device)   #dim: D x H
        Wfh = torch.empty(num_hidden, num_hidden, device=device)  #dim: H x H
#         bf  = torch.zeros(num_hidden, device=device) #not good
        bf  = torch.ones(num_hidden, device=device) #good          dim: H
        
        Wox = torch.empty(input_dim, num_hidden, device=device)   #dim: D x H
        Woh = torch.empty(num_hidden, num_hidden, device=device)  #dim: H x H
        bo  = torch.zeros(num_hidden, device=device)              #dim: H
        
        Wph = torch.empty(num_hidden, num_classes, device=device) #dim: H x C
        bp  = torch.zeros(num_classes, device=device)             #dim: C
        
        #init method?
        self.Wgx = nn.Parameter( nn.init.xavier_uniform_(Wgx) ) 
        self.Wgh = nn.Parameter( nn.init.xavier_uniform_(Wgh) ) 
        self.Wix = nn.Parameter( nn.init.xavier_uniform_(Wix) ) 
        self.Wih = nn.Parameter( nn.init.xavier_uniform_(Wih) ) 
        self.Wfx = nn.Parameter( nn.init.xavier_uniform_(Wfx) ) 
        self.Wfh = nn.Parameter( nn.init.xavier_uniform_(Wfh) ) 
        self.Wox = nn.Parameter( nn.init.xavier_uniform_(Wox) ) 
        self.Woh = nn.Parameter( nn.init.xavier_uniform_(Woh) ) 
        self.Wph = nn.Parameter( nn.init.xavier_uniform_(Wph) ) 
        self.bg  = nn.Parameter( bg ) 
        self.bi  = nn.Parameter( bi )
        self.bf  = nn.Parameter( bf )
        self.bo  = nn.Parameter( bo )
        self.bp  = nn.Parameter( bp ) 

    def forward(self, x):
#     def forward(self, x, timestepGrad):   #to get gradients
        # x dim: B x T x D
        
        c = torch.zeros(x.size(0), self.num_hidden, device=self.device) #dim: B x H 
        h = torch.zeros(x.size(0), self.num_hidden, device=self.device) #dim: B x H
        
#         #to get gradients
#         if timestepGrad == 0:
#             hGrad = h.clone().detach().requires_grad_(True)
#             h = hGrad
        
        for t in range(self.seq_length):
            xt = x[:,t,:].to(self.device) if self.input_dim > 1 else x[:,t].to(self.device) #dim: B x D 
            
            gt = torch.tanh(xt @ self.Wgx + h @ self.Wgh + self.bg)    #dim: B x H 
            it = torch.sigmoid(xt @ self.Wix + h @ self.Wih + self.bi) #dim: B x H 
            ft = torch.sigmoid(xt @ self.Wfx + h @ self.Wfh + self.bf) #dim: B x H 
            ot = torch.sigmoid(xt @ self.Wox + h @ self.Woh + self.bo) #dim: B x H 

            c = gt * it + c * ft    #dim: B x H
            h = torch.tanh(c) * ot  #dim: B x H
            
#             #to get gradients
#             if t+1 == timestepGrad:
#                 hGrad = h.clone().detach().requires_grad_(True) #keep track of grad
#                 h = hGrad
                
        pt = h @ self.Wph + self.bp #dim: B x C
        #softmax is done in cross entropy loss
        return pt
#         return pt, hGrad #to get gradients
    