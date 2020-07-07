# data-science-capstone
Capstone project for Ryerson University CKME 136

# Packages used
  - numpy, pandas, matplotlib, sklearn, statsmodels, keras, tensorflow (tensorflow-gpu for gpu processing), arch, yfinance (pandas_datareader as alternative)

# Data_prep
  # ~\Data_preparation.py
  - Fetches dataset from Yahoo Finance
  - Derives additional variables:
    - 'Intraday Range' = Intraday Range; difference between intraday High and intraday Low
    - 'IntChange' = Interday Change; difference in closing price from previous day (first differences in closing prices)
    - 'Returns' = Percentage Returns; percentage difference in closing price from previous day
  - Interpolates missing data due to weekends/days when stock market is closed
  - Directly sourced by other modules
  # ~\Data_plot.py
  - Generates plot of the daily closing prices
  - Generates plot of the Interday Change 
  
# Dickey-Fuller
  # ~\Dickey_Fuller.py
  - Performs Augmented Dickey Fuller test on daily closing prices and Interday Change to determine stationarity

# ETS
  # ~\ETS.py
  - Generates both Additive and Multiplicative Error, Tread, Seasonal models for closing prices
  
# ACF_PACF
  # ~\ACT_PACF.py
  - Generates ACF and PACF plots of Interday Change data

# Auto-Arima
  # ~\Auto_Arima.py
  - returns the best model based on AIC via. stepwise procedure, for both seasonal and non-seasonal models
  - results found in ~\auto_arima.txt

# ARIMA
  # ~\ARIMA.py
  - Generates ARIMA (0, 1, 1) model, with orders from Auto_Arima.py
  - Generates plot of the residuals and the predicted values from the model
  - results found in ~\ARIMA_Summary.txt
  
# SARIMA
  # ~\SARIMA.py
  - Generates SARIMA (2, 1, 2)(0, 1, 1)12 model, with orders from Auto_Arima.py
  - Generates plot of the residuals and the predicted values from the model
  - results found in ~\SARIMA_Summary.txt
  
# GARCH
  # ~\GARCH_data.py
  - Interpolated data from days where the stock market was closed causes complications when trying to predict volatility
  # ~\GARCH.py
  - Generates GARCH (1,1) model
  - Evaluates model using both the entire testing set, as well as rolling origin forecast
  - Generates plot of predicted values from the model, using both evaluation methods
  
# Exp_Smoothing
  # ~\exp_smooth.py
  - Generates Additive and Multiplicative Exponential Smoothing Models that include Trend and Seasonal smoothing
  - Generates plots of the models as well
  
# LSTM
  # ~\LSTM.py
  - Generates LSTM model 
  - Generates plot of predicted values from LSTM model
