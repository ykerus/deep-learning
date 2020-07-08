# MIT License
#
# Copyright (c) 2019 Tom Runia
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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# from part2.dataset import TextDataset
# from part2.model import TextGenerationModel

from dataset import TextDataset
from model import TextGenerationModel

################################################################################

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)  # fixme
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)
    test_loader = DataLoader(dataset, config.test_size, num_workers=1)
    
    results = open(config.out_file,"w+")
    results.write("#model_type   : {}-layer LSTM\n#seq_length   : {}\n#input_dim    : {}\n#num_classes  : {}\n#num_hidden   :\
 {}\n#batch_size   : {}\n#learn_rate   : {}\n#train_steps  : {}\n#max_norm     : {}\n#lr_decay     : {}\n#lr_step      :\
 {}\n".format( config.lstm_num_layers, config.seq_length, dataset.vocab_size, dataset.vocab_size,
                   config.lstm_num_hidden, config.batch_size, config.learning_rate, config.train_steps, config.max_norm,
                   config.learning_rate_decay, config.learning_rate_step))
    results.write("#train_step accuracy loss\n")
    gen_text = open(config.out_file[:-4]+".txt", 'w+', encoding="utf-8")
    
    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size,
                                lstm_num_hidden=config.lstm_num_hidden, lstm_num_layers=config.lstm_num_layers,
                                device=device).to(device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    
    #train
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    
    prevstep=0
    while True: #otherwise it stop after 1 epoch
      for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        step=prevstep+step
        batch_inputs = torch.nn.functional.one_hot(batch_inputs.type(torch.LongTensor), dataset.vocab_size).type(
            torch.FloatTensor).to(device)
        batch_targets = batch_targets.to(device)
    
        optimizer.zero_grad()
    
        batch_y = model(batch_inputs) #without softmax, dim: B x T x C
    
        #prevent gradients from exploding, not sure if still necessary
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
    
        # Only for time measurement of step through network
        t1 = time.time()
       
        loss = criterion(batch_y.transpose(1,2), batch_targets)
         
        loss.backward()
        optimizer.step() if step > 0 else 0 #to be able to test initial model
        
        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)
                
        if step % config.print_every == 0:
                   
            predictions = torch.argmax(torch.softmax(batch_y,2),2)
            accuracy = torch.sum(predictions==batch_targets).type(torch.FloatTensor)/config.batch_size/config.seq_length
            
#            #uncomment for printing
#            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
#               "Accuracy = {:.2f}, Loss = {:.3f}".format(
#                 datetime.now().strftime("%Y-%m-%d %H:%M"), step,
#                 int(config.train_steps), config.batch_size, examples_per_second,
#                 accuracy, loss))

            #writing results
            results.write("%d %.3f %.3f\n" % (step, accuracy, loss))
        
        optimizer.step() if step == 0 else 0
    
        if np.round(accuracy,2) == 1.00:
            print("Achieved >99.95% accuracy.")
            break
      
        if step % config.sample_every == 0:
        
            gen_text.write("--- Step: {} ---\n".format(step))
            with torch.no_grad():
                #get random char from alphabet
                rnd_char = np.random.choice(list(map(chr, range(97, 123)))).upper()
                prev = torch.zeros(dataset.vocab_size).to(device)
                prev[dataset._chars.index(rnd_char)] = 1
                prev = prev.view(1,1,-1) #dim: B x T x D
                #feed to network, maybe a bit redundant
                for i in range(config.out_seq-1):
                    gen_y = model(prev) #dim: B x T x C
                    char = torch.zeros(dataset.vocab_size).to(device)
                    softm = torch.softmax(config.temp*gen_y[0,-1,:],0).squeeze() #temperature included
#                       char[np.random.choice(np.arange(dataset.vocab_size),p=np.array(softm.cpu()))] = 1
                    char[torch.argmax(softm)] = 1 #greedy, uncomment prev line for random
                    prev = torch.cat([prev, char.view(1,1,-1)],1)
                txt = dataset.convert_to_string(torch.argmax(prev,2).squeeze().cpu())
                gen_text.write(txt+"\n\n")
            
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break
      prevstep = step
      if np.round(accuracy,2) == 1.00 or step == config.train_steps:
        break
    
    print('Done training.')
        

    #Saving model doesn't work
    #hard-coding temperatures as solution
    with torch.no_grad():  
        length = 500
        gen_text.write("--- Greedy ---\n")
        #get random char from alphabet
        rnd_char = np.random.choice(list(map(chr, range(97, 123)))).upper() 
        prev = torch.zeros(dataset.vocab_size).to(device)
        prev[dataset._chars.index(rnd_char)] = 1
        prev = prev.view(1,1,-1) #dim: B x T x D
        #feed to network, maybe a bit redundant
        for i in range(length-1):
            gen_y = model(prev) #dim: B x T x C
            char = torch.zeros(dataset.vocab_size).to(device)
            softm = torch.softmax(config.temp*gen_y[0,-1,:],0).squeeze() #temperature included
#             char[np.random.choice(np.arange(dataset.vocab_size),p=np.array(softm.cpu()))] = 1
            char[torch.argmax(softm)] = 1 #greedy
            prev = torch.cat([prev, char.view(1,1,-1)],1)
        txt = dataset.convert_to_string(torch.argmax(prev,2).squeeze().cpu())
        gen_text.write(txt+"\n\n")
        for t in [0.5,1.0,2.0]:
            gen_text.write("--- Temperature: {} ---\n".format(t))
            #get random char from alphabet
            rnd_char = np.random.choice(list(map(chr, range(97, 123)))).upper()
            prev = torch.zeros(dataset.vocab_size).to(device)
            prev[dataset._chars.index(rnd_char)] = 1
            prev = prev.view(1,1,-1) #dim: B x T x D
            #feed to network, maybe a bit redundant
            for i in range(length-1):
                gen_y = model(prev) #dim: B x T x C
                char = torch.zeros(dataset.vocab_size).to(device)
                softm = torch.softmax(t*gen_y[0,-1,:],0).squeeze() #temperature included
                char[np.random.choice(np.arange(dataset.vocab_size),p=np.array(softm.cpu()))] = 1
#                 char[torch.argmax(softm)] = 1 #greedy
                prev = torch.cat([prev, char.view(1,1,-1)],1)
            txt = dataset.convert_to_string(torch.argmax(prev,2).squeeze().cpu())
            gen_text.write(txt+"\n\n")
            
            gen_text.write("--- Temperature: {}. Finish ---\n".format(t))
            finish = "Sleeping beauty is "
            prev = torch.zeros(1,len(finish),dataset.vocab_size).to(device)
            for i, s in enumerate(finish):
                prev[0,i,dataset._chars.index(s)] = 1
            for i in range(length-len(finish)):
                gen_y = model(prev) #dim: B x T x C
                char = torch.zeros(dataset.vocab_size).to(device)
                softm = torch.softmax(t*gen_y[0,-1,:],0).squeeze() #temperature included
                char[np.random.choice(np.arange(dataset.vocab_size),p=np.array(softm.cpu()))] = 1
#                 char[torch.argmax(softm)] = 1 #greedy
                prev = torch.cat([prev, char.view(1,1,-1)],1)
            txt = dataset.convert_to_string(torch.argmax(prev,2).squeeze().cpu())
            gen_text.write(txt+"\n\n")
        
    results.close()
    gen_text.close()


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    
    parser.add_argument('--device', type=str, default="cuda:0", help='Device') #added
    parser.add_argument('--test_size', type=int, default=4000, help="Number of test samples") #added
    parser.add_argument('--out_file', type=str, default="results.dat", help="Output filename") #added
    parser.add_argument('--out_seq', type=int, default=30, help="Output sequence length") #added
    parser.add_argument('--temp', type=float, default=1, help="Temperature") #added
    parser.add_argument('--eval', type=int, default=0, help="If 1 load model and evaluate, else train") #added
    

    config = parser.parse_args()

    # Train the model
    train(config)
    
