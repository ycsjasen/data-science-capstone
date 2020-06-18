import numpy as np
import pandas as pd

#need to figure out how to substring file name for Ticker
df = pd.read_csv('../MSFT.csv')
df['Ticker'] = 'MSFT'

IXIC = pd.read_csv('../^IXIC.csv')
IXIC['Ticker'] = 'IXIC'

#calculating derived variables
df['Intraday Range'] = abs(df['Open'] - df['High'])
df['IntChange'] = df['Close'].diff()
df['IntPctChange'] = df['Close'].pct_change()*100
df['200SMA'] = df['Close'].rolling(window = 200).mean()
IXIC['IntPctChange'] = IXIC['Close'].pct_change()*100
df['IXIC_PctDiff'] =  df['IntPctChange'] - IXIC['IntPctChange']

#removing missing values
df = df.dropna()

print(df)
df.to_csv('MSFT_updated.csv', sep=',', index = False)
