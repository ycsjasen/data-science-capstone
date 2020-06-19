import pandas as pd

#need to figure out how to substring file name for Ticker
df = pd.read_csv('../MSFT.csv')
df['Ticker'] = 'MSFT'

SPY = pd.read_csv('../SPY.csv')
SPY['Ticker'] = 'SPY'

#calculating derived variables
df['Intraday Range'] = abs(df['Open'] - df['High'])
df['IntChange'] = df['Adj Close'].diff()
df['IntPctChange'] = df['Adj Close'].pct_change()*100
df['200SMA'] = df['Adj Close'].rolling(window = 200).mean()
SPY['IntPctChange'] = SPY['Adj Close'].pct_change()*100
df['SPY_PctDiff'] =  df['IntPctChange'] - SPY['IntPctChange']

#removing missing values
df = df.dropna()

#To export dataframe df to .csv file
#df.to_csv('MSFT_updated.csv', sep=',', index = False)
