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


trainingLength = 250
actionPeriod = 5
halfConfirmWindow = 500 #look this much on each side of the anchor point to confirm local trough
decimateBy = 1


trainingLength = 1000
actionPeriod = 5
halfConfirmWindow = 1000 #look this much on each side of the anchor point to confirm local trough
decimateBy = 5




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
    anchorTime_list.append(dTime[tempI - trainingLength : tempI : decimateBy])
    anchorData_list.append(sData[tempI - trainingLength : tempI : decimateBy])
#    confirmData_list.append(sData[tempI - halfConfirmWindow : tempI + halfConfirmWindow])
    confirmData_list.append(sData[tempI : tempI + halfConfirmWindow : decimateBy])
     

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
    test_anchorTime_list.append(dTime[tempI - trainingLength : tempI : decimateBy])
    test_anchorData_list.append(sData[tempI - trainingLength : tempI : decimateBy])
#    test_confirmData_list.append(sData[tempI - halfConfirmWindow : tempI + halfConfirmWindow])
    test_confirmData_list.append(sData[tempI : tempI + halfConfirmWindow : decimateBy])
    

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
import torch.optim as optim
from matplotlib import pyplot

input_size = len(anchorData_list[0])
hidden_sizes = [256, 128]
output_size = 1

N = 100 #For now, do 1000 batches of 100 for minibatches


#define custom loss function
def my_loss(output, target):
    loss = torch.mean(torch.abs(output - target))
#    loss = torch.mean(torch.square(output - target))
    return loss


# initialization function, first checks the module type,
# then applies the desired changes to the weights
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)


#x = torch.randn(N, input_size)
#y = torch.randn(N, output_size)
        
        



#~~~Create Param Sweep~~~
        
psweep_lr = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]



lr = 1e-8
lossGoal = 0.3
loss_record = []
loss_test_record = []

scatter_test = []
scatter_test_pred = []
scatter_test_diff = []
minLoss = inf

#set up testing arrays
x_test = torch.from_numpy(np.array(test_anchorData_list)).float()
y_test_np = np.array(test_troughiness_list)
y_test_np.resize(len(test_troughiness_list),1)
y_test = torch.from_numpy(y_test_np).float()




# Set up feed forward network and optimizer
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.LeakyReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.LeakyReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      )

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



for i in range(int(51e3)):
    miniBatch_idx = np.random.choice(N_chunks, N, replace = False)
    x_np = np.take(anchorData_list, miniBatch_idx, axis=0) #gives NxtrainingLength minibatch
    y_np = np.take(troughiness_list, miniBatch_idx, axis=0)
    y_np.resize(N,1) #So that there is no mistaken broadcasting in the loss calc...
    
    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).float()
    
    model.zero_grad()
    
    y_pred = model(x)
    
    
    loss = my_loss(y_pred, y)
    
    if i % 100 == 0:
        print(i, loss.item())
        
    

    loss.backward()
    optimizer.step()
    
    
    with torch.no_grad():
#        for param in model.parameters():
#            param.data -= lr * param.grad
        
        loss_record.append(loss.item())
        
        if loss.item() < minLoss:
            minLoss = loss.item()
            print('New Loss Record! = ' + str(minLoss))
            bestModel = copy.deepcopy(model)
            
#        if loss.item() < lossGoal:
#            lr = lr/10
#            lossGoal *= 0.8

#            print(i, 'New lr: ' + str(lr) + ' New lossGoal: ' + str(lossGoal))
            
        
        #test accuracy every testCycles 
        testCycles = 1
        if i % testCycles == 0:
            y_test_pred = model(x_test)
            loss_test = my_loss(y_test_pred, y_test)
            
            loss_test_record.append(loss_test.item())
            
        
        #record a scatter of y_test and y_test_pred every half epoch
        if i % 5000 == 0:
            scatter_test.append(y_test)
            y_best_test = bestModel(x_test)
            
            scatter_test_pred.append(y_best_test)
            scatter_test_diff.append(y_test - y_best_test)
            
            
            

#Use the best model
with torch.no_grad(): 
    y_best_test = bestModel(x_test)
    
    #Convert all data to numpy arrays and reshape:
    y_best_test = np.array(y_best_test.reshape(len(y_best_test)))
    y_test = np.array(y_test.reshape(len(y_test)))
    for currList in [scatter_test, scatter_test_diff, scatter_test_pred]:
        for i in range(len(currList)):
            currList[i] = np.array(currList[i].reshape(len(currList[i])))
            
    
#~~~Plot loss curve~~~
plt.figure()
plt.plot(loss_record)
plt.plot(loss_test_record)
plt.ylim([.2,0.4])

plt.xlabel('Iteration')
plt.ylabel('Loss')

#~~~Plot scatter plots of predictions changing over epoochs~~~
plt.figure()
for i in range(len(scatter_test)):
    if i == 0:
        continue
    plt.scatter(scatter_test[i], scatter_test_diff[i])
plt.scatter(y_test, y_test - y_best_test)
plt.title('Scatter y_test Result Difference')
plt.xlabel('y_test Troughiness')
plt.ylabel('Troughiness Difference: real - pred')
plt.legend(['try 1', 'try 2', 'try 3', 'try 4'])


#~~~Same as above, but as historgrams (to visualize the shifting mass)~~~
plt.figure()
histBins = np.linspace(0,1,81)
for i in range(len(scatter_test)-2):
#    if i == 0:
#        continue
    pyplot.hist(scatter_test_pred[i], bins = histBins, alpha = 0.5, label = str(i))
pyplot.hist(scatter_test[0], bins = histBins, alpha = 0.5, label = 'True Test')
pyplot.legend(loc='upper right')
plt.title('Evolving Histograms of Best Models')
plt.xlabel('y_pred_test Result')
plt.ylabel('Count number in bin')



#~~~Same plots as above, but as linear regressions instead of scatters~~~
plt.figure()
for i in range(len(scatter_test)):
    if i == 0:
        continue
    x = scatter_test[i]
    y = scatter_test_diff[i]
    m, b = np.polyfit(x.resize(len(x)), y.resize(len(y)), 1)
    
    plt.plot(x, m*x + b)

plt.title('Linear Regression y_test Result Difference')
plt.xlabel('y_test Troughiness')
plt.ylabel('Troughiness Difference: real - pred')
plt.legend(['try 1', 'try 2', 'try 3', 'try 4'])
plt.xlim([0,0.1])
plt.ylim([-.55,-0.45])

#




