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


#This script will take my custom built class and load data into it
#meant to visualize + compare multiple strategies
#also meant to just view different heurstics



import config
   
import alpaca_trade_api as tradeapi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

from heuristics import StockPredictor
from moneyManager import BankAccount

import os, datetime


userIn = input('Rerun?')
if userIn == '1':
    f = open('test_data/stockdata_7_27.pckl', 'rb')
    fullData = pickle.load(f)
    f.close()
    
    mySP = StockPredictor('invmean', 'data')
    
    mySP.data_loader(fullData['SPY'][:,np.r_[0:2, -1]])
    


mydir = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.makedirs(mydir)


plt.figure(8)
plt.plot(fullData['SPY'][:,-1], fullData['SPY'][:,1])

plt.plot(mySP.refTime, mySP.heurData)
plt.plot(mySP.refTime, 70*mySP.prediction)
plt.plot(mySP.refTime[0:-1], 100*np.diff(mySP.prediction))

diffPred = np.diff(mySP.prediction)

print(sum(abs(diffPred)))
print(sum(diffPred[np.where(diffPred > 0)]))
print(sum(abs(diffPred[np.where(diffPred < 0)])))

plt.grid()
plt.xlim([0,500])
plt.ylim([-10,10])


endPoint = 3150
plt.xlim([.7,3002])
#plt.savefig('heuristic_plots/invmean_SPY_2months.eps', format='eps')



plt.figure(1)
plt.plot(mySP.refTime, mySP.toBuy, 'o')
plt.plot(mySP.refTime, mySP.integratorRaw, 'o')
#plt.plot(mySP.refTime[0:-1], np.diff(mySP.refTime), 'o')
plt.legend(['Buy?', 'raw'])

plt.xlim([2020,2050])
#plt.ylim([-10,10])

plt.grid()



#Now we load in the heuristic data into a bank account class
#This class manages the money given to the heuristic at a high level
#The heuristic gives buy and sell info at a high level, the bank account scales this to a dollar amount
#The bank account also keeps track of how much money it has 
#Usually, a bank account is tied to a heuristic


#Let's plot a passive investing strategy vs the actual stock data
refBank = BankAccount(cashPool = 10e3, tradStrat = 'passive', \
                              heuristic = mySP)

passiveT, passiveData = refBank.timeLapse()

refBank.tradStrat = 'allin'
allinT, allinData = refBank.timeLapse()

refBank.tradStrat = 'simple_heur'
simpheurT, simpheurData, debugArr, cashArr, stockArr = refBank.timeLapse()

plt.figure(2)
plt.plot(passiveT, passiveData)
plt.plot(allinT, allinData)
plt.plot(simpheurT, simpheurData)
plt.plot(simpheurT, 50*debugArr, 'o')

#Scale stock price to match first passive investing point
firstIndx = np.where(mySP.data[:,-1] == passiveT[0])
scaler = float(passiveData[0] / mySP.data[firstIndx,1])

plt.plot(mySP.data[:,-1], scaler * mySP.data[:,1])
plt.legend(['Passive Investing', 'All in', 'Simple Heuristic', 'Simp debug', 'Stock Price'])
plt.title('Stock Price Scaled by ' + str(scaler))
#plt.xlim([500,1000])
plt.grid()

plt.savefig(mydir + '/GrowthOverTime.eps', format='eps')


#Figure to debug simple heuristic
plt.figure(3)
plt.plot(mySP.data[:,-1], scaler * mySP.data[:,1])
plt.plot(simpheurT, 50*debugArr, 'o')
plt.plot(simpheurT, cashArr)
plt.plot(simpheurT, stockArr)
plt.grid()
plt.title('Simple Heuristic Strategy Breakdown')
plt.legend(['Stock Price', 'Simp Debug', 'Cash', 'Stock Value'])

plt.savefig(mydir + '/StockCashBreakdown.eps', format='eps')

