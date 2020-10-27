#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alpaca_market_data --> ML_trainer
  (do only once)

               
"""


#This script will take the data I processed before with the algorithmic approach
#and will attempt to throw ML at the problem

#The problem is formulated as follows:
#   -Gather data from some subset of tickers using alpaca_market_data
#   -using random seed, take random splices of x days long of the tickers from data

#   -Link the splices of data to the following data or average of data points
#   -Assing a score to each splice based on how "troughy" the following minute will be

#   -Construct a set of affine layers to output a troughy score from input data
#   -Use with a loss functiont that punishes deviation from the true "troughiness"



import config
   
import alpaca_trade_api as tradeapi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

from heuristics import StockPredictor
from moneyManager import BankAccount

import os, datetime
import time

import random

#~~~Load in the data before splicing ~~~
f = open('test_data/stockdata_9_19.pckl', 'rb')
fullData = pickle.load(f)
f.close()

dTime = fullData['SPY'][:,np.r_[0:2, -1]][:,2]
sData = fullData['SPY'][:,np.r_[0:2, -1]][:,1]
dLen = len(dTime)



#~~~Splice up the full data into N chunks~~~
random.seed(11)
N_chunks = 10000 #start with a decent-sized number

#TODO: extend the length of time covered with decimation 
#(ie 1 day of minute data, followed by 6 previous days of every 30 min data, etc)
#For now, just use past day, which is ~500 points (just leave it at that)
trainingLength = 500
actionPeriod = 5
halfConfirmWindow = 750 #look this much on each side of the anchor point to confirm local trough

bounds = max(trainingLength, halfConfirmWindow) #define choosable bounds


#Make lists of "anchor points" and training data for said anchor points
#These should be N_chunks long
anchorPoints_list = []
anchorTime_list = []
anchorData_list = []
confirmData_list = []
chooseLength = len(dTime) - 2*bounds
for i in range(N_chunks):
    tempI = random.randint(bounds, dLen - bounds)
    
    anchorPoints_list.append(tempI)
    anchorTime_list.append(dTime[tempI - trainingLength : tempI])
    anchorData_list.append(sData[tempI - trainingLength : tempI])
    confirmData_list.append(sData[tempI - halfConfirmWindow : tempI + halfConfirmWindow])
    

#Now apply our heuristic for defining "troughiness"
#Defined as:
#   how low the current anchor point is relative to the confirm window's range
#   = (max(confirmWindow) - anchorPoint) / (max(confirmWindow) - min(confirmWindow))

#   this essentially translates to a "confidence" of whether to buy right now or not
troughiness_list = []
for i, curr_anchor in enumerate(anchorPoints_list):
    confirmArr = confirmData_list[i]
    confRange = max(confirmArr) - min(confirmArr)
    temp_trough = (max(confirmArr) - sData[curr_anchor]) / confRange
    
    troughiness_list.append(temp_trough)
    

#Plot histogram of troughiness values list to confirm we have a good value spread
plt.hist(troughiness_list, bins = 'auto')





#~~~Do the above splicing again for some amount of testing points~~~


random.seed(100)
test_N_chunks = 1000 #start with a decent-sized number

#Make lists of "anchor points" and training data for said anchor points
#These should be N_chunks long
test_anchorPoints_list = []
test_anchorTime_list = []
test_anchorData_list = []
test_confirmData_list = []
for i in range(test_N_chunks):
    tempI = random.randint(bounds, dLen - bounds)
    
    test_anchorPoints_list.append(tempI)
    test_anchorTime_list.append(dTime[tempI - trainingLength : tempI])
    test_anchorData_list.append(sData[tempI - trainingLength : tempI])
    test_confirmData_list.append(sData[tempI - halfConfirmWindow : tempI + halfConfirmWindow])
    

#Now apply our heuristic for defining "troughiness"
test_troughiness_list = []
for i, curr_anchor in enumerate(test_anchorPoints_list):
    confirmArr = test_confirmData_list[i]
    confRange = max(confirmArr) - min(confirmArr)
    temp_trough = (max(confirmArr) - sData[curr_anchor]) / confRange
    
    test_troughiness_list.append(temp_trough)
    


    




#~~~Construct our PyTorch model~~~

#Start with a simple, 2 layer Neural Net
#500 point vector --> 500xN affine layer --> ReLU --> Nx1 affine layer --> output
#loss (for now) is simply L1 absolute distance, or:
#   abs(true_troughiness - pred_troughiness)


from torch import nn
import torch
import copy

input_size = trainingLength
hidden_sizes = [256, 128]
output_size = 1

N = 100 #For now, do 1000 batches of 100 for minibatches


#define custom loss function
def my_loss(output, target):
    loss = torch.mean(torch.abs(output - target))
#    loss = torch.mean(torch.square(output - target))
    return loss



#x = torch.randn(N, input_size)
#y = torch.randn(N, output_size)

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      )
#                      nn.Softmax(dim=1))
#print(model)

lr = 1e-8
lossGoal = 0.3
loss_record = []
loss_test_record = []
minLoss = inf

#set up testing arrays
x_test = torch.from_numpy(np.array(test_anchorData_list)).float()
y_test_np = np.array(test_troughiness_list)
y_test_np.resize(len(test_troughiness_list),1)
y_test = torch.from_numpy(y_test_np).float()

for i in range(50000):
    miniBatch_idx = np.random.choice(N_chunks, N, replace = False)
    x_np = np.take(anchorData_list, miniBatch_idx, axis=0) #gives NxtrainingLength minibatch
    y_np = np.take(troughiness_list, miniBatch_idx, axis=0)
    y_np.resize(N,1) #So that there is no mistaken broadcasting in the loss calc...
    
    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).float()
    
    y_pred = model(x)
    
    
    loss = my_loss(y_pred, y)
    
    if i % 100 == 0:
        print(i, loss.item())
        
    model.zero_grad()

    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param.data -= lr * param.grad
        
        loss_record.append(loss.item())
        
        if loss.item() < minLoss:
            minLoss = loss.item()
            print('New Loss Record! = ' + str(minLoss))
            bestModel = copy.deepcopy(model)
            
        if loss.item() < lossGoal:
            lr = lr/10
            lossGoal *= 0.7
            
            print(i, 'New lr: ' + str(lr) + ' New lossGoal: ' + str(lossGoal))
            
        
        #test accuracy every testCycles 
        testCycles = 1
        if i % testCycles == 0:
            y_test_pred = model(x_test)
            loss_test = my_loss(y_test_pred, y_test)
            
            loss_test_record.append(loss_test.item())
            


plt.figure()
plt.plot(loss_record)
plt.plot(loss_test_record)
plt.ylim([0,0.5])
#




