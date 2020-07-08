"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import os.path

import torch
import torch.nn as nn
import torch.optim as optim

N_CLASSES = 10
N_INPUTS = 3*32*32

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """
  with torch.no_grad():
      correct = np.sum(np.argmax(predictions.cpu().detach().numpy(),axis=1) == np.argmax(np.array(targets),axis=1))
      accuracy = correct / len(targets[:,0])

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope
  learning_rate = FLAGS.learning_rate
  batch_size = FLAGS.batch_size
  max_steps = FLAGS.max_steps

  results = open("results.dat","w+")
  results.write("#torch_mlp \n#neg_slope : {}\n#learning_rate : {}\n#batch_size : {}\n#hidden_units : {}\
\n#max_steps : {}\n".format(neg_slope, learning_rate, batch_size, dnn_hidden_units, max_steps))

  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  
  x_test, t_test = cifar10["test"].images, cifar10["test"].labels
  x_test = torch.tensor(x_test.reshape(np.size(x_test[:,0,0,0]), N_INPUTS)).to(device)
  t_test_indx = torch.tensor(np.where(t_test==1)[1]).to(device)
  
  mlp = MLP(N_INPUTS, dnn_hidden_units, N_CLASSES, neg_slope).to(device)
  
  crossEntropy = nn.CrossEntropyLoss()
    
  optimizer = optim.SGD(mlp.parameters(), lr=learning_rate)
  results.write("#GPUs : {}\n".format(torch.cuda.device_count())) #show no of available gpus
#   print("GPUs : ", torch.cuda.device_count())
  if torch.cuda.device_count() > 1:
    nn.DataParallel(mlp)
    
  results.write("#epoch batch max_steps loss train_acc test_acc test_loss\n")
  
  for batch in range(1,max_steps+1):
    
    optimizer.zero_grad()
    
    x, t = cifar10["train"].next_batch(batch_size)
    x = torch.tensor(x.reshape(batch_size, N_INPUTS), requires_grad=True).to(device)
    t_indx = torch.tensor(np.where(t==1)[1]).to(device) #shape: (batch_size,)
        
    y = mlp(x).to(device) #y predictions, t targets
    loss = crossEntropy(y, t_indx) #includes softmax
    
    #accuracy before updating
    if batch == 1:
      train_acc = accuracy(y, t) 
      y_test = mlp(x_test).to(device)
      test_loss = crossEntropy(y_test, t_test_indx)
      test_acc = accuracy(y_test, t_test)
      results.write("%d %d %d %.3f %.3f %.3f %.3f\n" % 
          (cifar10["train"]._epochs_completed, 0, max_steps, loss, train_acc, test_acc, test_loss))
#       print("Epoch: %d. Batch: %d/%d. Loss: %.3f. Train_acc: %.3f. Test_acc: %.3f" % 
#           (cifar10["train"]._epochs_completed, 0, max_steps, loss, train_acc, test_acc))
      
    
    #update weights
    loss.backward()
    optimizer.step()
        
    if batch % FLAGS.eval_freq == 0:
      train_acc = accuracy(y, t) 
      y_test = mlp(x_test).to(device)
      test_loss = crossEntropy(y_test, t_test_indx)
      test_acc = accuracy(y_test, t_test)
      results.write("%d %d %d %.3f %.3f %.3f %.3f\n" % 
          (cifar10["train"]._epochs_completed, batch, max_steps, loss, train_acc, test_acc, test_loss))
#       print("Epoch: %d. Batch: %d/%d. Loss: %.3f. Train_acc: %.3f. Test_acc: %.3f" % 
#           (cifar10["train"]._epochs_completed, batch, max_steps, loss, train_acc, test_acc))
  results.close()
    
def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  FLAGS, unparsed = parser.parse_known_args()

  main()