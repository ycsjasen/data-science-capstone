# data-science-capstone
Capstone project for Ryerson University CKME 136

# Data_prep
  # \Data_preparation.py
  - Fetches dataset from Yahoo Finance
  - Derives additional variables:
    - 'Intraday Range' = Intraday Range; difference between intraday High and intraday Low
    - 'IntChange' = Interday Change; difference in closing price from previous day (first differences in closing prices)
    - 'Returns' = Percentage Returns; percentage difference in closing price from previous day
  - Directly sourced by other modules
  # \Data_plot.py
  - Generates plot of the daily closing prices
  - Generates plot of the Interday Change 
  
# Dickey-Fuller
  # \Dickey_Fuller.py
  - Performs Augmented Dickey Fuller test on daily closing prices and Interday Change to determine stationarity

# ACF_PACF
  # \
