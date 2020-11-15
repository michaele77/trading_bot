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
import time




#define custom loss function
def my_loss(output, target):
    loss = torch.mean(torch.abs(output - target))
    loss.requires_grad = True
#    loss = torch.mean(torch.square(output - target))
    return loss


# initialization function, first checks the module type,
# then applies the desired changes to the weights
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight)
        
        
        
class two_affine_layer_net(nn.Module):
    def __init__(self, insize, h1, h2):
        super(two_affine_layer_net, self).__init__()
        self.linear1   = nn.Linear(insize, h1)
        self.linear2   = nn.Linear(h1, h2)
        self.linear3   = nn.Linear(h2, 1)
        
		

    def forward(self, x):
        
#        self.l1 = self.linear1(x)
#        self.nl1 = self.nonlin1(self.l1)
#        self.l2 = self.linear2(self.nl1)
#        self.nl2 = self.nonlin1(self.l2)
#        self.l3 = self.linear3(self.nl2)
        
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = self.linear3(x)
        
#        with torch.no_grad():
#            self.l1 = l1
#            self.nl1 = nl1
#            self.l2 = l2
#            self.nl2 = nl2
#            self.l3 = l3
        
        return x
    
        
    def custom_loss(self,output, target):
        loss = torch.mean(torch.abs(output - target))
        return loss
            
            
        
		

#Setup one-shot parameter trackers
lr = 1e-8
input_size = len(anchorData_list[0])
hidden_sizes = [256, 128]
output_size = 1

N = 100 #For now, do 1000 batches of 100 for minibatches


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



#~~~Create Param Sweep~~~
        
#TODO: add different activation functions, add Adam optimizer extras, 
psweep_lr = [1e-9, 1e-11]
psweep_H1 = [128, 256]
psweep_H2 = [128, 256]
psweep_iterations = 15e3 
totalIters = len(psweep_lr) * len(psweep_H1) * len(psweep_H2)


psresult_y_test = []
psresult_y_best_test = []
psresult_time = []


for ps_lr in psweep_lr:
    for ps_h1 in psweep_H1:
        for ps_h2 in psweep_H2:
            
            time0 = time.time()
            # Set up feed forward network and optimizer
#            model = nn.Sequential(nn.Linear(input_size, ps_h1),
#                                  nn.LeakyReLU(),
#                                  nn.Linear(ps_h1, ps_h2),
#                                  nn.LeakyReLU(),
#                                  nn.Linear(ps_h2, output_size),
#                                  )
            model = two_affine_layer_net(input_size, ps_h1, ps_h2)
            
            criteria = torch.nn.L1Loss(reduction = 'mean')

            optimizer = torch.optim.Adam(model.parameters(), lr=ps_lr)
            
            
            
            
            
            for i in range(int(psweep_iterations)):
                
                miniBatch_idx = np.random.choice(N_chunks, N, replace = False)
                x_np = np.take(anchorData_list, miniBatch_idx, axis=0) #gives NxtrainingLength minibatch
                y_np = np.take(troughiness_list, miniBatch_idx, axis=0)
                y_np.resize(N,1) #So that there is no mistaken broadcasting in the loss calc...
                
#                x = torch.from_numpy(x_np).float()
#                y = torch.from_numpy(y_np).float()
                x = torch.tensor(x, requires_grad = True)
                y = torch.tensor(y, requires_grad = True)
                
                
                optimizer.zero_grad()
                
#                y_pred = model(x)
                y_pred = model.forward(x)
#                y_pred.requires_grad = True
#                
                
#                loss = model.custom_loss(y_pred, y)
                #For loss, just use torch loss function (otherwise branches autograd graph --> problems)
                #reduction is how the loss is output during minibatches (sum or mean output)
                loss = criteria(y_pred, y)
                
                
                
                   
                    
                
            
                loss.backward()
                optimizer.step()
                
                
                
                with torch.no_grad():
            #        for param in model.parameters():
            #            param.data -= lr * param.grad
                    
#                    loss_record.append(loss.item())
            
                    if i % 100 == 0:
                        print(i, loss.item())
                    
                    if loss.item() < minLoss:
                        minLoss = loss.item()
                        print('New Loss Record! = ' + str(minLoss))
                        bestModel = copy.deepcopy(model)
                        
                    
#                    #test accuracy every testCycles 
#                    testCycles = 1
#                    if i % testCycles == 0:
#                        y_test_pred = model(x_test)
#                        loss_test = my_loss(y_test_pred, y_test)
#                        
#                        loss_test_record.append(loss_test.item())
                        
                    
#                    #record a scatter of y_test and y_test_pred every half epoch
#                    if i % 5000 == 0:
#                        scatter_test.append(y_test)
#                        y_best_test = bestModel(x_test)
#                        
#                        scatter_test_pred.append(y_best_test)
#                        scatter_test_diff.append(y_test - y_best_test)
            
            with torch.no_grad():
                psresult_time.append(time.time() - time0)
                
                y_best_test = bestModel(x_test)
                
                #Convert all data to numpy arrays and reshape:
                y_best_test = np.array(y_best_test.reshape(len(y_best_test)))
                y_test = np.array(y_test.reshape(len(y_test)))
                
                psresult_y_best_test.append(y_best_test)
                psresult_y_test.append(y_test)
                
                
psresult_y_test_std = []
psresult_y_test_mean = []
psresult_abs_diff = []

for currY in psresult_y_best_test:
    psresult_y_test_std.append(np.std(currY))
    psresult_y_test_mean.append(np.mean(currY))
    psresult_abs_diff.append(np.mean(np.abs(currY - psresult_y_test[0])))
        
        
                
                
   

#~~~Visualization for one-shot computation~~~
       
#
##Use the best model
#with torch.no_grad(): 
#    y_best_test = bestModel(x_test)
#    
#    #Convert all data to numpy arrays and reshape:
#    y_best_test = np.array(y_best_test.reshape(len(y_best_test)))
#    y_test = np.array(y_test.reshape(len(y_test)))
#    for currList in [scatter_test, scatter_test_diff, scatter_test_pred]:
#        for i in range(len(currList)):
#            currList[i] = np.array(currList[i].reshape(len(currList[i])))
#            
#            
#    
##~~~Plot loss curve~~~
#plt.figure()
#plt.plot(loss_record)
#plt.plot(loss_test_record)
#plt.ylim([.2,0.4])
#
#plt.xlabel('Iteration')
#plt.ylabel('Loss')
#
##~~~Plot scatter plots of predictions changing over epoochs~~~
#plt.figure()
#for i in range(len(scatter_test)):
#    if i == 0:
#        continue
#    plt.scatter(scatter_test[i], scatter_test_diff[i])
#plt.scatter(y_test, y_test - y_best_test)
#plt.title('Scatter y_test Result Difference')
#plt.xlabel('y_test Troughiness')
#plt.ylabel('Troughiness Difference: real - pred')
#plt.legend(['try 1', 'try 2', 'try 3', 'try 4'])
#
#
##~~~Same as above, but as historgrams (to visualize the shifting mass)~~~
#plt.figure()
#histBins = np.linspace(0,1,81)
#for i in range(len(scatter_test)-2):
##    if i == 0:
##        continue
#    pyplot.hist(scatter_test_pred[i], bins = histBins, alpha = 0.5, label = str(i))
#pyplot.hist(scatter_test[0], bins = histBins, alpha = 0.5, label = 'True Test')
#pyplot.legend(loc='upper right')
#plt.title('Evolving Histograms of Best Models')
#plt.xlabel('y_pred_test Result')
#plt.ylabel('Count number in bin')
#
#
#
##~~~Same plots as above, but as linear regressions instead of scatters~~~
#plt.figure()
#for i in range(len(scatter_test)):
#    if i == 0:
#        continue
#    x = scatter_test[i]
#    y = scatter_test_diff[i]
#    m, b = np.polyfit(x.resize(len(x)), y.resize(len(y)), 1)
#    
#    plt.plot(x, m*x + b)
#
#plt.title('Linear Regression y_test Result Difference')
#plt.xlabel('y_test Troughiness')
#plt.ylabel('Troughiness Difference: real - pred')
#plt.legend(['try 1', 'try 2', 'try 3', 'try 4'])
#plt.xlim([0,0.1])
#plt.ylim([-.55,-0.45])
#
##
#
#
#
#
