#!/usr/bin/env python
# coding: utf-8

# # Introduction
# State notebook purpose here

# ### Imports
# Import libraries and write settings here.

# In[ ]:


# Data manipulation
import pandas as pd
import numpy as np

# Options for pandas
pd.options.display.max_columns = 50
pd.options.display.max_rows = 30

# Display all cell outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

from IPython import get_ipython
ipython = get_ipython()

# autoreload extension
if 'autoreload' not in ipython.extension_manager.loaded:
    get_ipython().run_line_magic('load_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')

# Visualizations
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)

import cufflinks as cf
cf.go_offline(connected=True)
cf.set_config_file(theme='white')


# In[ ]:


import pickle
import pandas as pd
from datetime import datetime

#from util.benchmarks import buy_and_hodl, rsi_divergence, sma_crossover
import ta
from enum import Enum


class SIGNALS(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2


def trade_strategy(prices, initial_balance, commission, signal_fn):
    net_worths = [initial_balance]
    balance = initial_balance
    amount_held = 0
    comm_paid = 0
    trade_count = 0
    
    for i in range(1, len(prices)):
        if amount_held > 0:
            net_worths.append(balance + amount_held * prices[i])
        else:
            net_worths.append(balance)

        signal = signal_fn(i)

        if signal == SIGNALS.SELL and amount_held > 0:
            balance = amount_held * prices[i] - commission
            amount_held = 0
            comm_paid += commission
            trade_count += 1
        elif signal == SIGNALS.BUY and amount_held == 0:
            balance -= commission
            amount_held = balance / prices[i] 
            balance = 0
            comm_paid += commission 
            trade_count += 1
    
    print('xxxx total comm_paid: {:.2f}, comm_per_trade:{:.2f}, trade_count:{:d}, position:{:.2f}, net_worths:{:.2f}, total return:{:.2f}% '
          .format(comm_paid, comm_paid/trade_count, trade_count, amount_held, net_worths[-1], (net_worths[-1]/initial_balance-1)*100) )
    
    return net_worths


def buy_and_hodl(prices, initial_balance, commission):
    print('xxxx buy_and_hodl->')    
    def signal_fn(i):
        return SIGNALS.BUY

    return trade_strategy(prices, initial_balance, commission, signal_fn)


def rsi_divergence(prices, initial_balance, commission, period=3):
    print('xxxx rsi_divergence->')
    rsi = ta.rsi(prices)

    def signal_fn(i):
        if i >= period:
            rsiSum = sum(rsi[i - period:i + 1].diff().cumsum().fillna(0))
            priceSum = sum(prices[i - period:i + 1].diff().cumsum().fillna(0))

            if rsiSum < 0 and priceSum >= 0:
                return SIGNALS.SELL
            elif rsiSum > 0 and priceSum <= 0:
                return SIGNALS.BUY

        return SIGNALS.HOLD

    return trade_strategy(prices, initial_balance, commission, signal_fn)


def sma_crossover(prices, initial_balance, commission):
    print('xxxx sma_crossover->')    
    macd = ta.macd(prices)

    def signal_fn(i):
        if macd[i] > 0 and macd[i - 1] <= 0:
            return SIGNALS.SELL
        elif macd[i] < 0 and macd[i - 1] >= 0:
            return SIGNALS.BUY

        return SIGNALS.HOLD

    return trade_strategy(prices, initial_balance, commission, signal_fn)


# fratcal signals
def fractal_weekly_momentum(df, initial_balance, commission):
    #macd = ta.macd(prices)
    barType = df['barType']
    prices = df['price']
    
    print('xxxx fractal_weekly_momentum: ')    
    
    def signal_fn(i):        
        if barType[i] == 'close' and barType[i-1] == 'high' and prices[i] > prices[i-3]:
            return SIGNALS.BUY
        elif barType[i] == 'close' and barType[i-1] == 'low' and prices[i] < prices[i-3]:
            return SIGNALS.SELL

        return SIGNALS.HOLD

    return trade_strategy(prices, initial_balance, commission, signal_fn)


def fractal_weekly_meanreversion(df, initial_balance, commission):
    #macd = ta.macd(prices)
    barType = df['barType']
    prices = df['price']
    
    print('xxxx fractal_weekly_meanreversion: ')    
    
    def signal_fn(i):
        if barType[i] == 'close' and barType[i-1] == 'high' and prices[i] > 0.5*(prices[i-3] + prices[i-1]):
            return SIGNALS.SELL
        elif barType[i] == 'close' and barType[i-1] == 'low' and prices[i] < 0.5*(prices[i-3] + prices[i-1]):
            return SIGNALS.BUY

        return SIGNALS.HOLD

    return trade_strategy(prices, initial_balance, commission, signal_fn)

# fractal data 
def get_open_date(x):
    return [x.index[0], x[0], 'open']
    
def find_high_date(x):
    for i in  range(len(x)):
        if x[i] == max(x):
            #print('{},{},{}'.format(i, x[i], x.index[i]))
            return [x.index[i], x[i], 'high']
    return []

def find_low_date(x):
    for i in  range(len(x)):
        if x[i] == max(x):
            return [x.index[i], x[i], 'low']
    return []

def get_close_date(x):
    #return [x.index[-1], x[-1], 'Close']
    return [x.index[-1] + pd.DateOffset(1), x[-1], 'close']


# In[ ]:


pd.datetime.today().date()


# In[ ]:


import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from hurst import compute_Hc, random_walk

#start = datetime(2019, 1, 1)
start = datetime(2018, 1, 1)
end = pd.datetime.today().date()

s = 'SPY'
#df = web.DataReader(s, 'iex', start, end)
df = web.DataReader(s, 'yahoo', start, end)
#f = web.DataReader('F', 'robinhood')
df['Sym'] = s
df.tail()

# convert to fractal data
w = df.resample('W')
#[w['Low'].apply(find_low_date), w['High'].apply(find_high_date)]
#[w['Open'].apply(get_open_date), w['Close'].apply(get_close_date)]
pricesBar = pd.concat([w['Open'].apply(get_open_date), w['Low'].apply(find_low_date), w['High'].apply(find_high_date), w['Close'].apply(get_close_date) ])
orderedBar = np.sort(pricesBar.values)
#orderedBar

#Date	price	barType
data = pd.DataFrame(orderedBar.tolist(), columns=['Date','price','barType'])
data['Sym'] = s
data.head()

data.set_index('Date')['price'].iplot(title='{}, ({};{})'.format(data['Sym'].unique(), df.index[0], df.index[-1]))
hurst(data['price'])

# Evaluate Hurst equation
H, c, hdata = compute_Hc(data['price'], kind='price', simplified=True)

# Plot
f, ax = plt.subplots()
ax.plot(hdata[0], c*hdata[0]**H, color="deepskyblue")
ax.scatter(hdata[0], hdata[1], color="purple")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time interval')
ax.set_ylabel('R/S ratio')
ax.grid(True)
plt.show()

print("H={:.4f}, c={:.4f}".format(H,c))


# In[ ]:


# back test fractal signals 
initial_balance = 6000
commission = 4.95

buy_and_hodl_net_worths = buy_and_hodl(data['price'], initial_balance, commission)
rsi_divergence_net_worths = rsi_divergence(data['price'], initial_balance, commission)
sma_crossover_net_worths = sma_crossover(data['price'], initial_balance, commission)

fractal_weekly_mom_worths = fractal_weekly_momentum(data, initial_balance, commission)
fractal_weekly_mr_worths = fractal_weekly_meanreversion(data, initial_balance, commission)

data['buy_and_hodl_net_worths'] = buy_and_hodl_net_worths
data['rsi_divergence_net_worths'] = rsi_divergence_net_worths
data['sma_crossover_net_worths'] = sma_crossover_net_worths

data['fractal_weekly_mom_worths'] = fractal_weekly_mom_worths
data['fractal_weekly_mr_worths'] = fractal_weekly_mr_worths

ta_portfolios = ['buy_and_hodl_net_worths', 'rsi_divergence_net_worths', 'sma_crossover_net_worths', 'fractal_weekly_mom_worths', 'fractal_weekly_mr_worths']

data.set_index('Date')[ta_portfolios].iplot(
    title='{} hurst index: {:.4f} - buy_and_hodl_net_worths ${:,.2f}, market ret:{:.2f}%'
    .format(data['Sym'][0], hurst(data['price'])
            , buy_and_hodl_net_worths[-1]
            , (buy_and_hodl_net_worths[-1]/initial_balance-1)*100 ) )

# summary
data.describe()


# In[ ]:


#


# # Analysis/Modeling
# Do work here

# # Results
# Show graphs and stats here

# # Conclusions and Next Steps
# Summarize findings here

# In[ ]:


from ta import *
import sys
print(sys.executable)

def dropna(df):
    """Drop rows with "Nans" values
    """
    df = df[df < math.exp(709)]  # big number
    df = df[df != 0.0]
    df = df.dropna()
    return df


def ema(series, periods, fillna=False):
    if fillna:
        return series.ewm(span=periods, min_periods=0).mean()
    return series.ewm(span=periods, min_periods=periods).mean()


def get_min_max(x1, x2, f='min'):
    if not np.isnan(x1) and not np.isnan(x2):
        if f == 'max':
            max(x1, x2)
        elif f == 'min':
            min(x1, x2)
        else:
            raise ValueError('"f" variable value should be "min" or "max"')
    else:
        return np.nan

# Clean nan values
df = utils.dropna(data)

# Add all ta features filling nans values
#df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume", fillna=True)

print(df.columns)
print(len(df.columns))
df.head()


# In[ ]:


from datetime import datetime
#from pandas.io.data import DataReader
import pandas_datareader.data as web
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn

def hurst(ts):

    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    # Here it calculates the variances, but why it uses 
    # standard deviation and then make a root of it?
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0


# Download the stock prices series from Yahoo
#aapl = DataReader("AAPL", "yahoo", datetime(2012,1,1), datetime(2015,9,18))
aapl = web.DataReader(s, 'yahoo', datetime(2018,1,1), datetime(2019,6,24))

# Call the function
hurst(aapl['Adj Close'])


# In[ ]:


lags = range(2,100)
def hurst_ernie_chan(p):

    variancetau = []; tau = []

    for lag in lags: 

        #  Write the different lags into a vector to compute a set of tau or lags
        tau.append(lag)

        # Compute the log returns on all days, then compute the variance on the difference in log returns
        # call this pp or the price difference
        pp = subtract(p[lag:], p[:-lag])
        variancetau.append(np.var(pp))

    # we now have a set of tau or lags and a corresponding set of variances.
    #print tau
    #print variancetau

    # plot the log of those variance against the log of tau and get the slope
    m = polyfit(np.log(tau),np.log(variancetau),1)

    hurst = m[0] / 2

    return hurst

hurst_ernie_chan(aapl['Adj Close'])


# In[ ]:


# Evaluate Hurst equation
H, c, hdata = compute_Hc(aapl['Adj Close'], kind='price', simplified=True)

# Plot
f, ax = plt.subplots()
ax.plot(hdata[0], c*hdata[0]**H, color="deepskyblue")
ax.scatter(hdata[0], hdata[1], color="purple")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time interval')
ax.set_ylabel('R/S ratio')
ax.grid(True)
plt.show()

print("H={:.4f}, c={:.4f}".format(H,c))


# In[ ]:


hdata
data.describe()

