"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    weight = np.random.normal(0,.0001,(in_features, out_features))
    bias   = np.zeros(out_features)
    grads  = np.zeros((in_features, out_features))
    
    self.params = {'weight': weight, 'bias': bias}
    self.grads = {'weight': grads, 'bias': bias}

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    self.x = np.copy(x)
    out = x @ self.params["weight"] + self.params["bias"].squeeze()
    self.out = out
    
    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """
    
    self.grads["weight"] = self.x.T @ dout  
    self.grads["bias"]   = np.mean(dout, axis = 0).squeeze()
    
    dx =  dout @ self.params["weight"].T
   
    
    return dx

class LeakyReLUModule(object):
  """
  Leaky ReLU activation module.
  """
  def __init__(self, neg_slope):
    """
    Initializes the parameters of the module.

    Args:
      neg_slope: negative slope parameter.

    TODO:
    Initialize the module.
    """

    self.neg_slope = neg_slope 

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    self.x = np.copy(x)
    out = np.copy(x)
    out[out<0] = self.neg_slope * out[out<0]
    
#     print("lrelu out", np.shape(out))
    
    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """
    dlrelu = np.copy(self.x)
    dlrelu[dlrelu > 0] = 1
    dlrelu[dlrelu <= 0] = self.neg_slope
    dx = dout*dlrelu
    
    return dx


class SoftMaxModule(object):
  """
  Softmax activation module.
  """

  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
    """

    b = np.max(x,axis=1)
    b = b.reshape(len(b),1) @ np.ones((1,len(x[0,:])))
    y = np.exp(x - b)
    norm = np.sum(y,axis=1).reshape(len(b),1) @ np.ones((1,len(x[0,:])))
    out = y / norm
    self.out = out

    return out

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """
    
    #hard to do vectorized
    dx = np.zeros(np.shape(dout))
    for i in range(len(dout[:,0])):
        self_out = self.out[i,:].reshape((len(self.out[i,:]),1))
        dsoftm = np.diag(self_out.squeeze()) - self_out@self_out.T
        dx[i,:] = dout[i,:] @ dsoftm
    
    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """

  def forward(self, x, y):
    """
    Forward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    TODO:
    Implement forward pass of the module.
    """

    out = np.mean(np.sum(-np.multiply(y,np.log(x+1e-30)), axis=1))
    
    return out

  def backward(self, x, y):
    """
    Backward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """

    dx = -y/(x+1e-30) / np.shape(x[:,0])

    return dx