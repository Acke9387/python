import sys

import theano 

import tensorflow as tf 

import keras 

from keras import layers, models

from keras.models import Sequential

from keras.layers import Dense, Dropout 

import PYMC3_Machine_Learning_Models

import pymc3 as pm

import math

from math import floor

import pandas as pd

import numpy as np

import scipy

from scipy import signal, fftpack

from scipy.stats import linregress

from scipy.fftpack import fft, ifft, fftfreq

from scipy.signal import butter 

from scipy import special

import datetime as dt

from datetime import *

from timeit import default_timer as timer

import csv, os, io

import warnings

import pandas as pd

import time

import pdb as bp

import numba 

from numba import jit

import matplotlib.pyplot as plt

from datetime import timedelta





# -- | Data Processing | --

sp_500 = ['AAPL', 'MSFT', 'ABBV','ACN','ATVI','AYI','ADBE','AAP','AMD','AES', 'AET', 'AMG', 'AFL','A','APD','AKAM','ALK','ALB', 'ARE','ALXN','ALGN','ALLE','AGN','ADS','LNT','ALL','GOOGL','GOOG','MO','AMZN','AEE','AAL','AEP','AXP','AIG','AMT','AWK','AMP','ABC','AME','AMGN','APH','APC','ADI','ANDV','ANSS','ANTM','AON','APA','AIV','AAPL','AMAT','APTV','ADM','ARNC','AJG','AIZ','T','ADSK','ADP','AZO','AVB','AVY','BHGE','BLL','BAC','BAX','BBT','BDX','BBY','BIIB','BLK','HRB','BA','BKNG','BWA','BXP','BSX','BHF','BMY','AVGO','CHRW','CA','COG','CDNS','CPB','COF','CCL','CAT','CBOE','CBS','CELG','CNC','CNP','CTL','CERN','CF','SCHW','CHTR','CVX','CMG','CB','CHD','CI','XEC','CINF','CTAS','CSCO','C','CFG','CTXS','CME','CMS','KO','CTSH','CL','CMCSA','CMA','CAG','CXO','COP','ED','STZ','GLW','COST','COTY','CCI','CSRA','CSX','CMI','CVS','DHI','DHR','DRI','DVA','DE','DAL','XRAY','DVN','DLR','DFS','DISCA','DISCK','DISH','DG','DLTR','D','DOV','DWDP','DPS','DTE','DUK','DRE','DXC','ETFC','EMN','ETN','EBAY','ECL','EIX','EW','EA','EMR','ETR','EVHC','EOG','EQT','EFX','EQIX','EQR','ESS','EL','RE','ES','EXC','EXPE','EXPD','ESRX','EXR','XOM','FFIV','FB','FAST','FRT','FDX','FIS','FITB','FE','FISV','FLIR','FLS','FLR','FMC','FL','F','FTV','FBHS','BEN','FCX','GPS','GRMN','IT','GD','GE','GGP','GIS','GM','GPC','GILD','GPN','GS','GT','GWW','HAL','HBI','HOG','HRS','HIG','HAS','HCA','HCP','HP','HSIC','HES','HPE','HLT','HOLX','HD','HON','HRL','HST','HPQ','HUM','HBAN','HII','IDXX','INFO','ITW','ILMN','INCY','IR','INTC','ICE','IBM','IP','IPG','IFF','INTU','ISRG','IVZ','IPGP','IQV','IRM','JBHT','JEC','SJM','JNJ','JCI','JPM','JNPR','KSU','K','KEY','KMB','KIM','KMI','KLAC','KSS','KHC','KR','LB','LLL','LH','LRCX','LEG','LEN','LUK','LLY','LNC','LKQ','LMT','L','LOW','LYB','MTB','MAC','M','MRO','MPC','MAR','MMC','MLM','MAS','MA','MAT','MKC','MCD','MCK','MDT','MRK','MET','MTD','MGM','KORS','MCHP','MU','MSFT','MAA','MHK','TAP','MDLZ','MON','MNST','MCO','MS','MSI','MYL','NDAQ','NOV','NAVI','NKTR','NTAP','NFLX','NWL','NFX','NEM','NWSA','NWS','NEE','NLSN','NKE','NI','NBL','JWN','NSC','NTRS','NOC','NCLH','NRG','NUE','NVDA','ORLY','OXY','OMC','OKE','ORCL','PCAR','PKG','PH','PAYX','PYPL','PNR','PBCT','PEP','PKI','PRGO','PFE','PCG','PM','PSX','PNW','PXD','PNC','RL','PPG','PPL','PX','PFG','PG','PGR','PLD','PRU','PEG','PSA','PHM','PVH','QRVO','QCOM','PWR','DGX','RRC','RJF','RTN','O','RHT','REG','REGN','RF','RSG','RMD','RHI','ROK','COL','ROP','ROST','RCL','SPGI','CRM','SBAC','SCG','SLB','STX','SEE','SRE','SHW','SPG','SWKS','SLG','SNA','SO','LUV','SWK','SBUX','STT','SRCL','SYK','STI','SIVB','SYMC','SYF','SNPS','SYY','TROW','TTWO','TPR','TGT','TEL','FTI','TXN','TXT','BK','CLX','COO','HSY','MOS','TRV','DIS','TMO','TIF','TWX','TJX','TMK','TSS','TSCO','TDG','TRIP','FOXA','FOX','TSN','USB','UDR','ULTA','UAA','UA','UNP','UAL','UNH','UPS','URI','UTX','UHS','UNM','VFC','VLO','VAR','VTR','VRSN','VRSK','VZ','VRTX','VIAB','V','VNO','VMC','WMT','WBA','WM','WAT','WEC','WFC','WDC','WU','WRK','WY','WHR','WMB','WLTW','WYN','WYNN','XEL','XRX','XLNX','XL','XYL','YUM','ZBH','ZION','ZTS']


def DataFrame(ticker):

    pd.set_option('display.max_rows', 5000)

    avg_std = []

    check_dir = os.chdir('C:\\Users\jtmar\SP500 Data\SP500 2018')

    new_cwd = os.getcwd()

    directory = os.listdir(new_cwd)

    directory.sort()

    d = []

    for filename in directory:

        if filename[0] == '2':

            frame = pd.read_csv(filename, index_col= False)

            frame = frame[frame['Ticker'] == ticker]

            frame['Timestamp'] = pd.to_datetime(frame['Timestamp'])

            frame = frame.set_index(pd.DatetimeIndex(frame['Timestamp']))

            frame = frame.between_time("9:30:00", "16:00:00")

            frame['TimeCondition'] = np.where(
            
                np.logical_and(frame['Timestamp'] > "11:00:00", frame['Timestamp'] < "15:58:00", where= True), True, False)
                
            d.append(frame)

    df = pd.concat(d, ignore_index= True)

    df = df.reset_index(drop= True)

    stock = df['Ticker']

    close = df['ClosePrice']

    high = df['HighPrice']

    low = df['LowPrice']

    open = df['OpenPrice']

    volume = df['TotalVolume']

    timestamp = df['Timestamp']

    return(open, high, low, close, volume)








def Input_Features(open, high, low, close, volume):

    df = pd.DataFrame({"Close":close})

    df['Sin'] = np.sin(df['Close'] - df['Close'].shift(5))

    df['Cos'] = np.cos(df['Close'] - df['Close'].shift(5))

    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

    df['Exp-G'] = np.log(np.divide(df['Close'], df['Close'].shift(1))) * np.exp(df['ROC'])

    df['EMA'] = df['Close'].ewm(span=12).mean()

    df['Independent'] = np.where(df['Close'] > df['Close'].shift(1), 1, 0)

    df = df.dropna().reset_index(drop=True)

    return(df)




from sklearn.preprocessing import StandardScaler


def Data_Processing(df, split_n, neurons):

    if neurons % 16 != 0:

        neurons=math.floor(neurons/16)

    X = df.iloc[:, 1:-1]

    Y = df.iloc[:, -1]

    split = np.int(len(df)*split_n)

    X_train, X_test, Y_train, Y_test = X[:split], X[split:], Y[:split], Y[split:]

    Scaler = StandardScaler()

    X_train = Scaler.fit_transform(X_train)

    X_test = Scaler.transform(X_test)

    Classifier = Sequential()

    Classifier.add(Dense(units=neurons, kernel_initializer='uniform', activation='relu', input_dim=X.shape[1]))

    Classifier.add(Dense(units=neurons, kernel_initializer='uniform', activation='relu'))

    Classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    Classifier.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    Classifier.fit(X_train, Y_train, batch_size=10, epochs=100)

    Y_predict = Classifier.predict(X_test)

    Y_predict = (Y_predict > 0.5)

    df['Y-Predict'] = np.NaN

    df.iloc[(len(df) - len(Y_predict)):,-1:] = Y_predict

    trade_data = df.dropna()

    trade_data['Future-Returns'] = 0.

    trade_data['Future-Returns'] = np.log(np.divide(trade_data['Close'], trade_data['Close'].shift(1)))

    trade_data['Future-Returns'] = trade_data['Future-Returns'].shift(-1)


    trade_data['Strat-Returns'] = 0.

    trade_data['Strat-Returns'] = np.where(trade_data['Y-Predict'] == True, trade_data['Future-Returns'], -trade_data['Future-Returns'])

    trade_data['Market-Return'] = np.cumsum(trade_data['Future-Returns'])

    trade_data['Cum-Return'] = np.cumsum(trade_data['Strat-Returns'])


    plt.figure(figsize=(10,5))

    plt.plot(trade_data['Market-Return'], color='r', label='Market Returns')

    plt.plot(trade_data['Cum-Return'], color='g', label='Strategy Returns')

    plt.legend()

    plt.show()










def Run():

    frame = DataFrame('AMG')

    open = frame[0]

    high = frame[1]

    low = frame[2]

    close = frame[3]

    volume = frame[4]

    features = Input_Features(open, high, low, close, volume)

    Data_Processing(features, 0.8, 128)



Run()





