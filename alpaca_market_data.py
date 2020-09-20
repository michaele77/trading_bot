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


def get_15Min_bars(_symbol,_number):
    _df=pd.DataFrame()
    _temp=pd.DataFrame()
    if _number<=1000:
        _df = api.get_barset(_symbol, '15Min', limit=_number, start=None, end=None, after=None, until=None).df
    else:
        _num_cycles, _residual = divmod(_number, 1000)
        _df = api.get_barset(_symbol, '15Min', limit=_residual, start=None, end=None, after=None, until=None).df
        for i in range(1,_num_cycles+1):
            _temp = api.get_barset(_symbol, '15Min', limit=1000, start=None, end=None, after=None, until=_df.first_valid_index().isoformat()).df
            print(_df.first_valid_index().isoformat())
            _df= pd.concat([_temp,_df,])
    return _df


#Extracts data from the pandas dataframe into subsequent lists, then the numpy array
def extractData(fnc_symbol, fnc_endTime, fnc_timeFrame, fnc_limit):
    fnc_data = api.get_barset(fnc_symbol, fnc_timeFrame, limit = fnc_limit, end=fnc_endTime).df

    fnc_apDat = fnc_data[fnc_symbol]
    
    #Seperate into subconstituants
    f_apDat_c = [i for i in fnc_apDat['close']]
    f_apDat_h = [i for i in fnc_apDat['high']]
    f_apDat_l = [i for i in fnc_apDat['low']]
    f_apDat_o = [i for i in fnc_apDat['open']]
    f_apDat_v = [i for i in fnc_apDat['volume']]
    f_apDat_t = fnc_apDat['close'].index

    #Get epoch seconds, list comprehension
    f_apDat_epochsec = [i.value//10**9 for i in f_apDat_t]
    f_apDat_year = [i.date().year for i in f_apDat_t]
    f_apDat_month = [i.date().month for i in f_apDat_t]
    f_apDat_day = [i.date().day for i in f_apDat_t]
    f_apDat_hour = [i.time().hour for i in f_apDat_t]
    f_apDat_minute = [i.time().minute for i in f_apDat_t]
    f_apDat_second = [i.time().second for i in f_apDat_t]
    
    #create extracted numpy array
    #Will be following format:
    #[epochSec, close, open, high, low, volume, \\
    #year, month, day, hour, minute, second, ~TO BE ADDED LATER~ day indexing]
    temp = [f_apDat_epochsec, f_apDat_c, f_apDat_o, f_apDat_h, \
                    f_apDat_l, f_apDat_v, f_apDat_year, f_apDat_month, \
                    f_apDat_day, f_apDat_hour, f_apDat_minute, f_apDat_second]
    extractedMat = np.array(temp).T
    
    
    return extractedMat

#Print the plain time and day
def timeAndDay(idArray):
    print(str(int(idArray[9])) + ':' + str(int(idArray[10])) + '.' + \
          str(int(idArray[11])) + ' on ' + \
          str(int(idArray[7])) + '/' + str(int(idArray[8])) + '/' + \
          str(int(idArray[6])))
    
    

def mergeTimes(func_Data):
    data = func_Data.copy()
    
    time = data[:,0]
    tSec = data[:,-1]
    tMin = data[:,-2]
    tHour = data[:,-3]
    tDay = data[:,-4]
    tMonth = data[:,-5]
    tYear = data[:,-6]
    
    #goal is to both cut out repeating data AND add to the array a global day counter (basically active days of trading)
    #algo: -->if day changes, increment day BY ONE
    #      -->if minute changes, increment minute by exact amount
    #keep track of max minutes for final division
    
    #Only the date of 1/2/2015 overlaps, should be 1764
    #use np.delete method to remove
    prevMin = tMin[0]
    prevDay = tDay[0]
    maxMin = 0
    
    dayArr = []
    minArr = []
    dayTracker = 0
    minTracker = 0
    
    firstFlag = False
    stopFlag = False
    
    for i in range(len(time)):
        if i%20000 == 0:
            print(i)
            
            
        currMin = tMin[i]
        currDay = tDay[i]
        
#        if not int(currDay - prevDay) == 0:
#            dayTracker += 1  
#            minTracker = 0
#        dayArr.append(dayTracker)
#        
#        
#        if int(currMin - prevMin) > 0: 
#            minTracker += 1
#            firstFlag = True
#        elif int(currMin - prevMin) < 0:
#            minTracker = 0
#            firstFlag = True
#        minArr.append(minTracker)
        
        if not int(currDay - prevDay) == 0:
            dayTracker += 1  
            minTracker = 0
        
        elif abs(int(currMin - prevMin)) > 0: 
            if int(currMin - prevMin) > 0:
                
                minTracker += currMin - prevMin
                if firstFlag == False:
                    minuteStart_index = i
                    firstFlag = True
            elif int(currMin - prevMin) < 0:
                minTracker += 60 - prevMin + currMin
            
        dayArr.append(dayTracker)
        minArr.append(minTracker)
        
        
        prevMin = tMin[i]
        prevDay = tDay[i]
        if minTracker > maxMin:
            maxMin = minTracker
            print(maxMin)
        

##Stuff below worked before I actually made the day indexing work correctly...
##Now it doesnt, so commented it out lol
        #finally, delete method for numpy array
        #entering this if statement --> minute changed for the first time
        #Delete previous line (it is a repition)
#        if firstFlag == True and stopFlag == False:
#            remIndx = i - 1
#            stopFlag = True
#    
#        
#    #now remove index
#    data = np.delete(data,remIndx, axis = 0)
#    dayArr.pop(remIndx)
#    minArr.pop(remIndx)
    
    #now create dayindexing
    scaleMinArr = [dd/maxMin for dd in minArr]
    dayArr = np.array(dayArr)
    scaleMinArr = np.array(scaleMinArr)
    
    dayIndex = dayArr + scaleMinArr
    data = np.concatenate((data, dayIndex.reshape(len(dayIndex),1)), axis = 1)
    
    #Now, cut out all data that isnt in minutes (new addition)
    data = data[minuteStart_index ::,:]
    
    return data
       
    


#function for consolidating the full data into one data structure
#returns numpy matrix with rows being the entries and columns being info
#Will sweep as far as minutes will take me, then switch to 5min, then to 15min, then to days
def get_full_data(fnc_symbol):
    
    currEndTime = pd.Timestamp('2020-07-10', tz='America/New_York').isoformat()
    apData = extractData(fnc_symbol, currEndTime, 'minute', 1000)
    
    currMat = apData
    
    curYear = int(currMat[0][6])
    curMonth = int(currMat[0][7])
    curDate = int(currMat[0][8])
    currEndTime = pd.Timestamp(str(curYear) + '-' + str(curMonth) + '-' + str(curDate),\
                 tz='America/New_York').isoformat()
    
    
    while len(apData) == 1000:
        apData = extractData(fnc_symbol, currEndTime, 'minute', 1000)
        if len(apData) < 1:
            break
        currMat = np.concatenate((apData, currMat), axis=0)
        
        curYear = int(apData[0][6])
        curMonth = int(apData[0][7])
        curDate = int(apData[0][8])
        currEndTime = pd.Timestamp(str(curYear) + '-' + str(curMonth) + '-' + str(curDate),\
                     tz='America/New_York').isoformat()
        
        print(len(currMat))    
    print('Minute Length Finished')
    
    apData = extractData(fnc_symbol, currEndTime, '5Min', 1000)
    while len(apData) == 1000:
        apData = extractData(fnc_symbol, currEndTime, '5Min', 1000)
        if len(apData) < 1:
            break
        currMat = np.concatenate((apData, currMat), axis=0)
        
        curYear = int(apData[0][6])
        curMonth = int(apData[0][7])
        curDate = int(apData[0][8])
        currEndTime = pd.Timestamp(str(curYear) + '-' + str(curMonth) + '-' + str(curDate),\
                     tz='America/New_York').isoformat()
        
        print(len(currMat))  
    print('5 Minute Length Finished')
    
    apData = extractData(fnc_symbol, currEndTime, '15Min', 1000)
    while len(apData) == 1000:
        apData = extractData(fnc_symbol, currEndTime, '15Min', 1000)
        if len(apData) < 1:
            break
        currMat = np.concatenate((apData, currMat), axis=0)
        
        curYear = int(apData[0][6])
        curMonth = int(apData[0][7])
        curDate = int(apData[0][8])
        currEndTime = pd.Timestamp(str(curYear) + '-' + str(curMonth) + '-' + str(curDate),\
                     tz='America/New_York').isoformat()
        
        print(len(currMat))  
    print('15 Minute Length Finished')
    
    apData = extractData(fnc_symbol, currEndTime, 'day', 1000)
    while len(apData) == 1000:
        apData = extractData(fnc_symbol, currEndTime, 'day', 1000)
        if len(apData) < 1:
            break
        currMat = np.concatenate((apData, currMat), axis=0)
        
        curYear = int(apData[0][6])
        curMonth = int(apData[0][7])
        curDate = int(apData[0][8])
        currEndTime = pd.Timestamp(str(curYear) + '-' + str(curMonth) + '-' + str(curDate),\
                     tz='America/New_York').isoformat()
        
        print(len(currMat))  
    print('day Length Finished')
    
    return currMat
        
 


api = tradeapi.REST(
    key_id=config.API_KEY,
    secret_key=config.SECRET_KEY,
    base_url='https://paper-api.alpaca.markets'
)

# Get daily price data for AAPL over the last 5 trading days.
endTime = pd.Timestamp('2020-02-01', tz='America/New_York').isoformat() 

ticker = 'AAPL'
ticker = 'GOOG'
ticker = 'TSLA'
ticker = 'IBM'
ticker = 'FB'
ticker = 'SPY'
ticker = 'QQQ'
ticker = 'ROBT'
ticker = 'MSFT'


tickerList = ['AAPL', 'GOOG', 'TSLA', 'IBM', 'FB', 'SPY', 'QQQ', 'ROBT', 'MSFT']

userInput = input('Do you want to rerun data or just merge (1 for rerun, 2 for merge)')

if userInput == '1':
    fullDataDict = {}
    
    for currTicker in tickerList:
        temp = get_full_data(currTicker)
        temp = mergeTimes(temp)
        fullDataDict[currTicker] = temp
                
        
        
    f = open('test_data/stockdata_9_19.pckl', 'wb')
    pickle.dump(fullDataDict, f)
    f.close()
    
    f = open('test_data/stockdata_9_19.pckl', 'rb')
    testDict = pickle.load(f)
    f.close()
    
    ax = plt.figure(7)
    for i in fullDataDict:
        plt.plot(fullDataDict[i][:,0], fullDataDict[i][:,1], label=i)
    plt.legend(tickerList)
        
elif userInput == '2':
    f = open('test_data/stockdata_7_11.pckl', 'rb')
    fullDataDict = pickle.load(f)
    f.close()
    
    
    for curTick in tickerList:
        tempData = fullDataDict[curTick]
        tempData = mergeTimes(tempData)
        fullDataDict[curTick] = tempData
        
    
    
    #Now do normal saving process
    f = open('test_data/stockdata_9_19.pckl', 'wb')
    pickle.dump(fullDataDict, f)
    f.close()
    
    f = open('test_data/stockdata_9_19.pckl', 'rb')
    testDict = pickle.load(f)
    f.close()
    
    ax = plt.figure(7)
    for i in fullDataDict:
        plt.plot(fullDataDict[i][:,0], fullDataDict[i][:,1], label=i)
    plt.legend(tickerList)
    
    tt = testDict['SPY']
    ax = plt.figure(10)
    plt.plot(tt[:,-1], tt[:,1])
    plt.xlim(1000,1500)
    
    
    
    
    

#timeFrame = 'minute'
#timeFrame = '5Min'
timeFrame = '15Min'
#timeFrame = 'day'




#data = api.get_barset(ticker, timeFrame, limit = 1000).df
##data = api.get_barset('AAPL', 'day', limit = 1000).df
#
##aapl_bars = barset['AAPL']
#
#
#
#
#
#fullTickerData = get_full_data(ticker)



#Might get back to this....

##Get easy total number of days 
#runningDays = []
#prevDay = fullTickerData[0,7]
#runningTot = 0
#divider = 460
##454, 456... just say its out of 460
#for i in range(len(fullTickerData)):
#    runningTot = 1*((fullTickerData[i,7] - prevDay) > 0) + \
#    
#    prevDay = fullTickerData[i,7]
#    
#    
#    runningTot = fullTickerData[i,]
#    runningDays.append(fullTickerData[i,])













##Now get AAPL:
#apDat = data[ticker]
#
##Seperate into subconstituants
#apDat_c = [i for i in apDat['close']]
#apDat_h = [i for i in apDat['high']]
#apDat_l = [i for i in apDat['low']]
#apDat_o = [i for i in apDat['open']]
#apDat_v = [i for i in apDat['volume']]
#
#apDat_t = apDat['close'].index
#
##Make lists for all of these
##apDat_time = apDat_t.time()
##apDat_date = apDat_t.date()
#
##Get epoch seconds, list comprehension
#apDat_epochsec = [i.value//10**9 for i in apDat_t]
#apDat_year = [i.date().year for i in apDat_t]
#apDat_month = [i.date().month for i in apDat_t]
#apDat_day = [i.date().day for i in apDat_t]
#apDat_hour = [i.time().hour for i in apDat_t]
#apDat_minute = [i.time().minute for i in apDat_t]
#apDat_second = [i.time().second for i in apDat_t]



## See how much AAPL moved in that timeframe.
#week_open = aapl_bars[0].o
#week_close = aapl_bars[-1].c
#percent_change = 100*(week_close - week_open) / week_open
#print('AAPL moved {}% over the last 5 days'.format(percent_change))
