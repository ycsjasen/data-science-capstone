import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# setting dates and timeframe
startdate = datetime(2016, 1, 1)
enddate = datetime(2018, 10, 1)
train_end = datetime(2018, 6, 22)
test_end = datetime(2018, 10, 1)

# pulling stock data
MSFTdf = yf.download('MSFT', period='1d', start=startdate, end=enddate, auto_adjust=False)
MSFTdf.index = pd.to_datetime(MSFTdf.index)
MSFTdf.insert(0, 'Date', MSFTdf.index)
MSFTdf['Ticker'] = 'MSFT'

# calculating derived variables
MSFTdf['IntChange'] = MSFTdf['Close'].diff()
MSFTdf['Returns'] = MSFTdf['Close'].pct_change() * 100
MSFTdf['Volatility'] = MSFTdf['IntChange'].rolling(window=2).std() * np.sqrt(len(MSFTdf)) * 0.01

# removing missing values
MSFTdf = MSFTdf.dropna()

# setting date as index
MSFTdf.set_index('Date', inplace=True)

# building Training set
train_data = MSFTdf.Returns[:train_end]
test_data = MSFTdf.Returns[train_end + timedelta(days=1):test_end]

# Setting test size
test_size = len(test_data)
data_vol = MSFTdf.Volatility[-test_size:]
