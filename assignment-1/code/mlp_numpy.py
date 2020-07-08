"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
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
    
    nodes = [n_inputs]+n_hidden+[n_classes]
    self.linears = [LinearModule(nodes[i],nodes[i+1]) for i in range(len(nodes)-1)]
    self.LReLU = LeakyReLUModule(neg_slope)
    self.SoftMax = SoftMaxModule()
    
    
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
        x = layer.forward(x)
        x = self.LReLU.forward(x)
    x = self.linears[-1].forward(x)
    out = self.SoftMax.forward(x)

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    dout = self.SoftMax.backward(dout)
    dout = self.linears[-1].backward(dout)
    for layer in self.linears[-2::-1]: #reversed order
        dout = self.LReLU.backward(dout)
        dout = layer.backward(dout)
    
    return
