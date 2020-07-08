"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
      neg_slope: negative slope parameter for LeakyReLU

    TODO:
    Implement initialization of the network.
    """
    super(MLP, self).__init__()
    nodes = [n_inputs]+n_hidden+[n_classes]
    self.linears = nn.ModuleList([nn.Linear(nodes[i], nodes[i+1]) for i in range(len(nodes)-1)])
    self.LReLU = nn.LeakyReLU(neg_slope)
#     self.SoftMax = nn.SoftMax(dim=1) #SoftMax included in CrossEntropyLoss
    for layer in self.linears:
      nn.init.normal_(layer.weight, mean=0.0, std=.0001)
      nn.init.zeros_(layer.bias)

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    for layer in self.linears[:-1]:
        x = layer(x)
        x = self.LReLU(x)
    out = self.linears[-1](x)
#     out = self.SoftMax(x) #handles by cross entropy loss

    return out
