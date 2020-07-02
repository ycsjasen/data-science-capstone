import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# setting dates and timeframe
startdate = datetime(2016, 1, 1)
enddate = datetime(2018, 10, 1)
train_end = datetime(2018, 6, 22)
test_end = datetime(2018, 10, 1)

# pulling stock data
MSFTdf = yf.download('MSFT', period='1d', start=startdate, end=enddate, auto_adjust = False)
MSFTdf.index = pd.to_datetime(MSFTdf.index)
MSFTdf.insert(0, 'Date', MSFTdf.index)
MSFTdf['Ticker'] = 'MSFT'

SPYdf = yf.download('SPY', period='1d', start=startdate, end=enddate, auto_adjust = False)
SPYdf.index = pd.to_datetime(SPYdf.index)
SPYdf.insert(0, 'Date', SPYdf.index)
SPYdf['Ticker'] = 'SPY'

# calculate 200 day Simple Moving Average
#MSFTdf['200SMA'] = MSFTdf['Close'].rolling(window = 200).mean()

# filling missing weekday data
MSFTdf.set_index('Date', inplace=True)
MSFTdf = MSFTdf.resample('D').ffill().reset_index()
SPYdf.set_index('Date', inplace=True)
SPYdf = SPYdf.resample('D').ffill().reset_index()

# calculating derived variables
MSFTdf['Intraday Range'] = abs(MSFTdf['Open'] - MSFTdf['High'])
MSFTdf['IntChange'] = MSFTdf['Close'].diff()
MSFTdf['Returns'] = MSFTdf['Close'].pct_change() * 100
SPYdf['Returns'] = SPYdf['Close'].pct_change() * 100
MSFTdf['SPY_PctDiff'] = MSFTdf['Returns'] - SPYdf['Returns']

# removing missing values
MSFTdf = MSFTdf.dropna()

# setting date as index
MSFTdf.set_index('Date', inplace=True)

# building Training set
train_data = MSFTdf.Close[:train_end]
test_data = MSFTdf.Close[train_end + timedelta(days=1):test_end]

# To export dataframe to .csv file
#MSFTdf.to_csv('MSFT_updated.csv', sep=',', index = True)
