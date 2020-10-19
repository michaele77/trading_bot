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

import config

import alpaca_trade_api as tradeapi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

import math

class BankAccount:
    def __init__(self, cashPool, tradStrat, heuristic, volumeScale = 1):
        self.cashPool = float(cashPool)
        self.stockOwnership = 0
        self.totalAssets = cashPool
        
        self.tradStrat = tradStrat
        self.volumeScale = volumeScale
        self.heuristic = heuristic
        
        self.account = None
        
    
    #Function to reset params
    def resetAccount(self, func_cashPool, func_Stocks, func_Assets):
        self.cashPool = func_cashPool
        self.stockOwnership = func_Stocks
        self.totalAssets = func_Assets
        
    
    #Plays out a strategy with the given conditions
    def timeLapse(self):
        
        refTime = self.heuristic.refTime
        dTime = self.heuristic.data[:,-1]
        data = self.heuristic.data[:,1]
        
        ogCash = self.cashPool
        ogStock = self.stockOwnership
        ogAssets = self.totalAssets
        
        
        
        if self.tradStrat == 'passive':
            #This means we want to passively invest money as a ref (tracking the market)
            #Assume no other income
            #So, we just invest an average of the cashPool over the time period
            #Assume over 9 years (only count trading days)
            #Invest everytime we have enough money to buy a stock
            
            
            #algo sets up offset point from moving average; line up everything with that
            dayArr = np.r_[refTime[0]:math.floor(refTime[-1]) + 1]
            timeLapse = np.zeros(len(dayArr))
#            daysElapsed = dTime[-1] - offset
            moneyPerDay = self.cashPool / len(dayArr)
            
            moneyBuffer = 0 #Buffer for money to buy stock
            
            #Fill out timeLapse array with info
            for i in range(len(dayArr)):
                moneyBuffer += moneyPerDay
                currDay_idx = np.where(dTime == math.floor(dayArr[i]))
                
#                try:
#                    moneyBuffer >= data[currDay_idx]
#                except:
#                    print('strop here')
                funCond = moneyBuffer >= data[currDay_idx]
                if len(funCond) > 1:
                    #For the condition that there are two equal stock prices
                    print('Repeat days for day ' + str(dTime[currDay_idx[0][0]]))
                    print('At indices: ' + str(currDay_idx[0]))
                    funCond = funCond[0]
                    currDay_idx = currDay_idx[0][0]
                if funCond:
                        moneyBuffer -= data[currDay_idx]
                        self.cashPool -= data[currDay_idx]
                        self.stockOwnership += 1


#                try:
                self.totalAssets = self.stockOwnership*data[currDay_idx] + self.cashPool
                timeLapse[i] = self.totalAssets
#                except:
#                    print('stop ye')
            
            #Make sure to reset account to normal parameters 
            self.resetAccount(ogCash, ogStock, ogAssets)
#                
            
            return [dayArr,timeLapse]
    
            
        elif self.tradStrat == 'allin':
            #This means we want to dump all the money at the start
            #Would directly translate stock price to money
            
            #DO NOT pay attention after the fact
            #if any cash left over from initial purchase, will be dead cash
            
            #algo sets up offset point from moving average; line up everything with that
            dayArr = np.r_[refTime[0]:math.floor(refTime[-1]) + 1]
            timeLapse = np.zeros(len(dayArr))
            
            first_idx = np.where(dTime == math.floor(dayArr[0]))

            stocksToBuy = math.floor(float(self.cashPool / data[first_idx]))  
            self.stockOwnership += stocksToBuy
            self.cashPool -= self.stockOwnership*data[first_idx]
            print('Initial left over dead cash: ' + str(self.cashPool))
            
            
            #Fill out timeLapse array with info
            for i in range(len(dayArr)):
                currDay_idx = np.where(dTime == math.floor(dayArr[i]))
                if len(currDay_idx[0]) > 1:
                    print('Repeat days for day ' + str(dTime[currDay_idx[0][0]]))
                    print('At indices: ' + str(currDay_idx[0]))
                    currDay_idx = currDay_idx[0][0]
                    
                
          
                self.totalAssets = self.stockOwnership*data[currDay_idx] + self.cashPool
                
                
                
                timeLapse[i] = self.totalAssets
                
            #Make sure to reset account to normal parameters 
            self.resetAccount(ogCash, ogStock, ogAssets)
#                
            return [dayArr,timeLapse]
            
            
            
            
        elif self.tradStrat == 'simple_heur':
            #this means a heuristic has been applied
            
            #Go with very simple heuristic model
            #if heuristic says to buy and we have enough cash, buy
            #if heuristic says to sell and we have stocks to sell, sell
            
            #use the buyArr from heuristic 
            tOffset = len(dTime) - len(refTime) #start at offset
            timeLapse = np.zeros(len(refTime))
            boughtArr = timeLapse.copy()
            
            cashArr = timeLapse.copy()
            stockArr = timeLapse.copy()
            
            debugVal_pos = 30000 #Showing when stock was bought
            debugVal_neg = 25000 #Showing when stock was sold
            
            for i in range(len(refTime)):
                currStock = data[i + tOffset]
                
                if self.heuristic.toBuy[i] > 0 and currStock < self.cashPool:
                    self.stockOwnership += 1
                    self.cashPool -= currStock
                    boughtArr[i] = debugVal_pos
                elif self.heuristic.toBuy[i] < 0 and self.stockOwnership > 0:
                    self.stockOwnership -= 1
                    self.cashPool += currStock
                    boughtArr[i] = debugVal_neg
                    
                
                self.totalAssets = self.stockOwnership*currStock + self.cashPool
                timeLapse[i] = self.totalAssets
                cashArr[i] = self.cashPool
                stockArr[i] = self.stockOwnership*currStock
                
            
            #Make sure to reset account to normal parameters 
            self.resetAccount(ogCash, ogStock, ogAssets)
            
            return [refTime, timeLapse, boughtArr, cashArr, stockArr]
                
                
            
            