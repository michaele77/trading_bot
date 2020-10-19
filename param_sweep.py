#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alpaca_marker_data --> param_sweep 
  (do only once)        |       |
                        |  uses |                           
                        | these |                           
                        |classes|
                        V       V
               heuristics ---> moneyManager
               
               
               
               
"""


#This script will take my custom built class and load data into it
#Similar to data_loader, but instead of primarily being for visualization, here we do parameter sweeps
#Some general sweeps we will do:
#       -different long/recent averages combinations
#       -threshold values for buying/selling
#       -stock markers

#Anything more in depth than the above (such as approaches with regards to variable-sized stock buys)
#should really be handled by the data_loader, which allows for more visualization




import config
   
import alpaca_trade_api as tradeapi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

from heuristics import StockPredictor
from moneyManager import BankAccount

import os, datetime

#~~~Load in the data before sweep loops~~~
#    f = open('test_data/stockdata_7_27.pckl', 'rb')
f = open('test_data/stockdata_9_19.pckl', 'rb')
fullData = pickle.load(f)
f.close()


#~~~Create lists by which the sweeps will occur~~~
##Sweep 1: --> go low on thesholds and average time
#sweepList_recT = [10/460, 50/460, 100/460, 300/460]
#sweepList_avgT = [int(1), int(2), int(4), int(7)]
#sweepList_thresh = [0.05, 0.1, 0.25, 0.5, 1]

##Sweep 2:
#numMin = 331.246
#temp = list(range(10))
#sweepList_recT = [(i*8 + 5)/numMin for i in temp]
#sweepList_avgT = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6] 
#sweepList_thresh = [0.02, 0.04, 0.5, 0.06, 0.08, 0.1]

#Sweep 3:
numMin = 331.246
temp = list(range(10))
sweepList_recT = [(i*6 + 5)/numMin for i in temp]
sweepList_avgT = [2, 2.5, 3, 3.5, 4, 4.5, 5] 
sweepList_thresh = [0.05, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]

paramNum = len(sweepList_avgT)*len(sweepList_recT)*len(sweepList_thresh)



#~~~INITIALIZE STOCK PREDICTOR CLASS ~~~
mySP = StockPredictor('invmean', 'data')
#    mySP.data_loader(fullData['SPY'][:,np.r_[0:2, -1]])
#    mySP.data_loader(fullData['TSLA'][:,np.r_[0:2, -1]])
mySP.data_loader(fullData['AAPL'][:,np.r_[0:2, -1]])


#~~~INITIALIZE BANK MANAGMENT CLASS ~~~
refBank = BankAccount(cashPool = 10e3, tradStrat = 'simple_heur', \
                              heuristic = mySP)
    


#~~~Param for loop sweep time ~~~
#evertime we change parameters, reset sweet param variables for the class, and rerun the predict function
paramList = [] #append calculated parameters here
dataList = [] #store heursitic results here (big list here)
iTrack = 0
for curr_recT in sweepList_recT:
    for curr_avgT in sweepList_avgT:
        for curr_thresh in sweepList_thresh:
            iTrack += 1
            print('Currently on sweep...' + str(iTrack))
            
            #reject this combination of parameters if avgT is smaller than recT...
            if  curr_avgT <= curr_recT:
                continue
            
            mySP.sweepParam_avgT = curr_avgT
            mySP.sweepParam_recT = curr_recT
            mySP.sweepParam_thresh = curr_thresh
            
            #rerun predictors for heuristics/money manager objects
            mySP.predictor()
            simpheurT, simpheurData, debugArr, cashArr, stockArr = refBank.timeLapse()
            
            #append to the tracking lists
            paramList.append((curr_avgT, curr_recT, curr_thresh))
            dataList.append((simpheurT, simpheurData))
            
            
            
#~~~Sort the last values of the hueristic outputs, print the best 5 param combos~~~
topNum = 5
lastValList = [i[1][-1] for i in dataList]
sortedVals = sorted(lastValList)
topNList = []

for i in range(topNum):
    topIdx = lastValList.index(sortedVals[-1-i])
    topNList.append(topIdx)
    
#Print the params
for i, val in enumerate(topNList):
    print('Rank ' + str(i) + ': ')
    print('Average Time: ' + str(paramList[val][0]))
    print('Recent Time: ' + str(paramList[val][1]))
    print('Threshold: ' + str(paramList[val][2]))
    print()
    

    
    


#~~~Plot the loop results~~~
plt.figure(1)
for i in range(paramNum):
    currData = dataList[i]
    plt.plot(currData[0], currData[1])
    plt.grid()
    plt.title('Parameter Sweep Comparison Graph')
    
#plot the top N results with labels  
legendList = []          
plt.figure(2)
for i, val in enumerate(topNList):
    currData = dataList[val]
    plt.plot(currData[0], currData[1])
    
    legendStr = 'Rank=' + str(i) + ', avgT=' + str(paramList[val][0]) + \
    ', recT=' +  "{:.2f}".format(paramList[val][1]) + ', thresh=' + str(paramList[val][2])
    legendList.append(legendStr)
plt.grid()
plt.title('Best ' + str(topNum) + ' Parameter Combos')
plt.legend(legendList)
    
    
            