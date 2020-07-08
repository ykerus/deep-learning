"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

N_INPUTS = 3*32*32
N_CHANNELS = 3
N_CLASSES = 10
TEST_BATCH = 25

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
  Performs training and evaluation of ConvNet model. 

  TODO:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  learning_rate = FLAGS.learning_rate
  batch_size = FLAGS.batch_size
  max_steps = FLAGS.max_steps

  results = open("results.dat","w+")
  results.write("#convnet \n#learning_rate : {}\n#batch_size : {}\
\n#max_steps : {}\n".format(learning_rate, batch_size, max_steps))

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

  cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
  
  x_test, t_test = cifar10["test"].images, cifar10["test"].labels
  x_test = Variable(torch.tensor(x_test)).to(device)
  
  convnet = ConvNet(N_CHANNELS, N_CLASSES).to(device)
    
  crossEntropy = nn.CrossEntropyLoss()
    
    
  optimizer = optim.Adam(convnet.parameters(), lr=learning_rate)
  results.write("#GPUs : {}\n".format(torch.cuda.device_count())) #show no of available gpus
#   print("GPUs : ", torch.cuda.device_count())
  if torch.cuda.device_count() > 1:
    nn.DataParallel(mlp)
  
  results.write("#epoch batch max_steps loss train_acc test_acc test_loss\n")
    
  for batch in range(1,max_steps+1):
    
    optimizer.zero_grad()
    
    x, t = cifar10["train"].next_batch(batch_size)
    x = Variable(torch.tensor(x), requires_grad=True).to(device)
    t_indx = Variable(torch.tensor(np.where(t==1)[1])).to(device) #shape: (batch_size,)
        
    y = convnet(x)#.to(device) #y predictions, t targets
    loss = crossEntropy(y, t_indx) #includes softmax
    
    #accuracy before updating
    if batch == 1:
      train_acc = accuracy(y, t)
      test_acc = 0
      for i in range(0,x_test.shape[0],TEST_BATCH):
        y_test = convnet(x_test[i:i+TEST_BATCH,:,:,:]).to(device)
        test_acc += accuracy(y_test, t_test[i:i+TEST_BATCH])
      test_acc /= (x_test.shape[0]/TEST_BATCH)
      results.write("%d %d %d %.3f %.3f %.3f\n" % 
          (cifar10["train"]._epochs_completed, 0, max_steps, loss, train_acc, test_acc))
#       print("Epoch: %d. Batch: %d/%d. Loss: %.3f. Train_acc: %.3f. Test_acc: %.3f" % 
#           (cifar10["train"]._epochs_completed, 0, max_steps, loss, train_acc, test_acc))
      
    
    #update weights
    loss.backward()
    optimizer.step()
        
    if batch % FLAGS.eval_freq == 0:
      train_acc = accuracy(y, t)
      test_acc = 0
      for i in range(0,x_test.shape[0],TEST_BATCH):
        y_test = convnet(x_test[i:i+TEST_BATCH,:,:,:]).to(device)
        test_acc += accuracy(y_test, t_test[i:i+TEST_BATCH])
      test_acc /= (x_test.shape[0]/TEST_BATCH)
      results.write("%d %d %d %.3f %.3f %.3f\n" % 
          (cifar10["train"]._epochs_completed, batch, max_steps, loss, train_acc, test_acc))
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
  FLAGS, unparsed = parser.parse_known_args()

  main()