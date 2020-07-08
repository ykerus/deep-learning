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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader

# from part1.dataset import PalindromeDataset
# from part1.vanilla_rnn import VanillaRNN
# from part1.lstm import LSTM

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

# You may want to look into tensorboard for logging
# from torch.utils.tensorboard import SummaryWriter

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type == "RNN":
        model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden,\
                           config.num_classes, device=device)
    elif config.model_type == "LSTM":
        model = LSTM(config.input_length, config.input_dim, config.num_hidden,\
                           config.num_classes, device=device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    test_loader = iter(DataLoader(dataset, config.test_size, num_workers=1))

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    
    results = open(config.out_file,"w+")
    results.write("#model_type   : {}\n#input_length : {}\n#input_dim    : {}\n#num_classes  : {}\n#num_hidden   : {}\n#batch_size   : {}\n#learn_rate   : {}\n#train_steps  : {}\n#max_norm     : {}\n".format(
                      config.model_type, config.input_length, config.input_dim, config.num_classes, config.num_hidden,
                          config.batch_size, config.learning_rate, config.train_steps, config.max_norm))
    results.write("#train_step accuracy loss\n")

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = torch.nn.functional.one_hot(batch_inputs.type(torch.LongTensor)).type(torch.FloatTensor).to(device)
        batch_targets = batch_targets.to(device)
        
        # Only for time measurement of step through network
        t1 = time.time()

        optimizer.zero_grad()
        
#         #for calculating gradients
#         for timestep in range(config.input_length):
#             model.zero_grad()
#             batch_y, hGrad = model(batch_inputs, timestep) #without softmax
#             #prevent gradients from exploding
#             torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
#             loss = criterion(batch_y, batch_targets)
#             loss.backward()
#             results.write("{} {}\n".format(timestep,hGrad.grad.norm()))
#         print("Done calculating gradients.")
#         results.close()
#         return
        
        batch_y = model(batch_inputs) #without softmax

        #prevent gradients from exploding
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)

        loss = criterion(batch_y, batch_targets)

        loss.backward()
        
        optimizer.step() if step > 0 else 0 #to be able to test initial model
        
        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.eval_freq == 0:
#             predictions = torch.argmax(torch.abs(batch_y),1) #training: smaller batch size than test
#             accuracy = torch.sum(predictions == batch_targets).type(torch.FloatTensor)/config.batch_size
            with torch.no_grad():
                test_inputs, test_targets = next(test_loader)
                test_inputs = torch.nn.functional.one_hot(test_inputs.type(torch.LongTensor),
                                                          config.input_dim).type(torch.FloatTensor).to(device)
                test_targets = test_targets.to(device)
                test_y = model(test_inputs)
                test_loss = criterion(test_y, test_targets)
                test_predictions = torch.argmax(test_y,1)
                test_accuracy = torch.sum(test_predictions == test_targets).type(torch.FloatTensor)/config.test_size
                
#                #uncomment for printing
#                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
#                      "Accuracy = {:.2f}, Loss = {:.3f}".format(
#                        datetime.now().strftime("%Y-%m-%d %H:%M"), step,
#                        config.train_steps, config.batch_size, examples_per_second,
#                        test_accuracy, test_loss))
                
                results.write("%d %.3f %.3f\n" % (step, test_accuracy, test_loss))
            
        optimizer.step() if step == 0 else 0
            
        if np.round(test_accuracy,2) == 1.00:
            print("Achieved >99.95% accuracy.")
            break

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    results.close()


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=10, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--out_file', type=str, default="results.dat", help="Output filename") #added
    parser.add_argument('--test_size', type=int, default=4000, help="Number of test samples") #added
    parser.add_argument('--eval_freq', type=int, default=10, help="Evaluation frequency") #added

    config = parser.parse_args()

    # Train the model
    train(config)