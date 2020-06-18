import numpy as np
import pandas as pd

#need to figure out how to substring file name for Ticker
df = pd.read_csv('MSFT.csv')
df['Ticker'] = 'MSFT'

SPY = pd.read_csv('SPY.csv')
SPY['Ticker'] = 'SPY'

#calculating derived variables
df['Intraday Range'] = abs(df['Open'] - df['High'])
df['IntChange'] = df['Close'].diff()
df['IntPctChange'] = df['Close'].pct_change()*100
df['200SMA'] = df['Close'].rolling(window = 200).mean()
SPY['IntPctChange'] = SPY['Close'].pct_change()*100
df['SPY_PctDiff'] =  df['IntPctChange'] - SPY['IntPctChange']

#removing missing values
df = df.dropna()

df.to_csv('MSFT_updated.csv', sep=',', index = False)
