#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alpaca_marker_data --> data_loader 
  (do only once)        |       |
                        |  uses |                           
                        | these |                           
                        |classes|
                        V       V
               heuristics ---> moneyManager
"""

#This class expects some input data and outputs either:
#-->a certainty that the data will go up
#-->an array of data as output

#Has a few functions:
#-->initialize, where user identifies what algorithm they want done
#-->loader, where input data is loaded (for specific ticker). Calcs for prediction should be done at loading time
#-->predicter, can be function or just an internal variable
#--> update? if 2 minutes have gone by, feed in only 2 new points. Loader + vars are updated.

import config
import os

import alpaca_trade_api as tradeapi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

import math

class StockPredictor:
    def __init__(self, heuristic_type, predictor_type):
        #Heuristic_type identifies which algo is to be used
        #Predictor_type identifies type of output heuristic should produce 
        self.heuristic_type = heuristic_type
        self.predictor_type = predictor_type
        self.prediction = None
        self.toBuy = None
        self.heurData = None
        self.integratorRaw = None
        self.integratorSuper = None
        self.refTime = None
        
        self.sweepParam_avgT = None
        self.sweepParam_recT = None
        self.sweepParam_thresh = None
        
        self.debug_scaleArr = None
        
    def data_loader(self, dataInput):
        self.data = dataInput
        #Now call own prediction function AT loading time
        self.predictor()
        
        return self.prediction
    
    def predictor(self):
        #We will use self.dataInput variable and heuristic/predictor types to output appropriate data
        dataLen, dataCol = self.data.shape
        mainTime = self.data[:,-1]
        mainData = self.data[:,1] #assume main data in the second column
        #first go through heuristic type options:
        if self.heuristic_type == 'invmean':
            #Take moving average of last 3 months and compare to last 5 days
            #scale to minutes (for now...not accurate before 2015)
            #We trade 9:30am to 4:00pm on EST for normal hours

            
            #Here, we decide whether to use a function input or default params in this class
            if self.sweepParam_avgT is None:
                window_avg = 3
                window_avg = 0.5
            else:
                window_avg = self.sweepParam_avgT
            
            if self.sweepParam_recT is None: 
#                window_rec = 90/460 #Should correspond to 90 minutes
                window_rec = 90/331.246 #Should correspond to 90 minutes
                window_rec = 0.6 
            else:
                window_rec = self.sweepParam_recT
            
#            #do a combined loop, recent and long term windows should be in lockstep
#            tempIndx, = np.where(window_avg <= mainTime) #assuming we are still in days
#            firstWindow = min(tempIndx)
            
            #Don't know why the above is some array thing... a relic i think
            nonEff_flag = False #THIS SHOULD BE TRUE IF USING DAYS + MINS COMBO
#            minPerDay = 460
            minPerDay = 331.246
            
            if nonEff_flag:
                firstWindow = window_avg
            else:
                firstWindow = round(window_avg*minPerDay)
            
            avgArr = np.zeros(dataLen - firstWindow)
            recArr = np.zeros(dataLen - firstWindow)
            refTime = np.zeros(dataLen - firstWindow)
            
            
            cntr = 0
            effCounter = 0
            for i in range(firstWindow, dataLen):
                if nonEff_flag:
                    tempIndx, = np.where(mainTime[i] - window_avg >= mainTime) 
                    avgIndx = max(tempIndx)
                    tempIndx, = np.where(mainTime[i] - window_rec >= mainTime) 
                    recIndx = max(tempIndx)
                    
                    avgArr[cntr] = np.sum(mainData[avgIndx:i]) / len(mainData[avgIndx:i])
                    recArr[cntr] = np.sum(mainData[recIndx:i]) / len(mainData[recIndx:i])
                    refTime[cntr] = self.data[i-1,-1]
                    
                    if mainTime[i] - mainTime[i-1] < 0.9:
                        #Now we're in minute territory
                        effCounter += 1
                        
                        #for sure past a day if we're at 600
                        if effCounter > 500*window_avg:
                            nonEff_flag = False
                else:
                    #450 seemed like the most common mintes in a day but they vary...largest was 600
                    #just use 460 (will be slightly conservative but will be a p good estimation)
                    
                    
                    
                    avgArr[cntr] = np.sum(mainData[i - round(window_avg*minPerDay):i]) / (round(window_avg*minPerDay))
                    recArr[cntr] = np.sum(mainData[i - round(window_rec*minPerDay):i]) / (round(window_rec*minPerDay))
                    refTime[cntr] = self.data[i-1,-1]
                    
                    
#                if mainTime[i] > 1765:
#                    print('stop here ye foul beastie')
                
                if i % 10000 == 0:
                    print(i)
                    
                cntr += 1
            
            #now calculate reversion
            meanRev = avgArr - recArr
            combList = [refTime, meanRev]

            
        #~~~~~~~~~~~~~~
        ##ADDED AFTER ANALYSIS
        #Remove DC Bias in the meanRev signal
        #However, if we remove all DC bias, get's fucked by the spikes
        #Split up into chunks and remove DC bias in each chunk
        #DO NOT DO BY INDEX, do by days
        
#        chunkSize = 500
#        numOfChunks = math.ceil(len(meanRev) / chunkSize)
#        
#        
#        for i in range(0, numOfChunks-1):
#            
#            tempArr = meanRev[chunkSize*i : chunkSize*(i+1)]
#            meanRev[chunkSize*i : chunkSize*(i+1)] = \
#            tempArr - np.sum(tempArr) / chunkSize
#        
#        tempArr = meanRev[(numOfChunks-1)*chunkSize ::]
#        meanRev[(numOfChunks-1)*chunkSize ::] = \
#        tempArr - np.sum(tempArr) / len(tempArr)
#        
        
        
        #~~~~~~~~~~~~~~
        #Now, go through predictor type options:
#        if self.predictor_type == 'data':
        
                   
#        if self.predictor_type == 'prediction':
        predArr = np.zeros(len(meanRev))
#        reverseThresh = abs(np.mean(mainData)*0.02) #Trying to get thresh at about 5
        
        
        #OLD PREDICTION ARRAY:
        #Prediction array will output a 1 for buy, -1 for sell, 0 for hold
#        eps = 0.00000000000001
#        for currIdx in range(len(meanRev)):
#            currAvg = np.sum(meanRev[0:currIdx]) / (len(meanRev[0:currIdx]) + eps)
#            if meanRev[currIdx] - currAvg > reverseThresh:
#                predArr[currIdx] = 1
#            elif meanRev[currIdx] - currAvg < -reverseThresh:
#                predArr[currIdx] = -1
                
        #NEW PREDICTION ARRAY: Dont threshold, use a pressure-based integration
        #therefore, the predArr essentially gets removed (there's no thresholding anymore)
        #Simply set it equal to the meanRev to preserve the variable in case we want to revert back
        eps = 0.00000000000001
        for currIdx in range(len(meanRev)):
            predArr[currIdx] = meanRev[currIdx]
  
                
        #~~~~~~~~~~~~~~
        ##Now we construct the integrator
        #we integrate the predArr (which is 1's, 0's, and -1s)
        #Set a trip point at which the integration is offset; this is a buy or a sell
        #Should be based on how many ticks we are going up
        
        #Say: if trend has been occuring for 2 days, trip it
        
        #first create a scaling array
        scaleArr = np.zeros(len(refTime))
        for x in range(len(refTime) - 1):
            scaleArr[x] = refTime[x+1] - refTime[x]
        scaleArr[-1] = scaleArr[-2] #assume last time diff is same as one before
        
        #Now add to an integration array
        #We will set the trip point to x days
        #when it hits, buy enough to offset the integration back to 0
        #for now, buy equal quantities, so just use 1's and 0's and -1's
        
        if self.sweepParam_thresh is None:
            tripScaler = 1 #Scales the trip-point (which is counted in days) by average avg - rec difference
            tripPoint = 0.2*tripScaler #2 days
            tripPoint = 0.5
        else:
            tripPoint = self.sweepParam_thresh
        buyArr = np.zeros(len(refTime))
        
        integratorArr = np.zeros(len(refTime))
        integratorSeed = 0 #starting value, change this for an off by one issue (introduce lag into integrator)
        integratorArr[0] = integratorSeed + predArr[0]
        for x in range(1, len(refTime)):
            #make sure to scale the prediction array
            integratorArr[x] = predArr[x]*scaleArr[x] + integratorArr[x-1]
            
            if integratorArr[x] >= tripPoint:
                integratorArr[x] = integratorArr[x] - tripPoint
                buyArr[x] = 1
            elif integratorArr[x] <= -tripPoint:
                integratorArr[x] = tripPoint + integratorArr[x]
                buyArr[x] = -1
                
        
        
        
        
        self.heurData = meanRev
        self.refTime = refTime
        self.prediction = predArr
        
        self.toBuy = buyArr
        self.integratorRaw = integratorArr    
        
        #Add debugging object variables:
        self.debug_scaleArr = scaleArr
            
            
        returnData = combList
        #~~~~~~~~~~~~~~
        return returnData
        
        