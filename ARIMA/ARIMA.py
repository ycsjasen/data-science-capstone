from Data_prep.Data_preparation import MSFTdf
import numpy as np
from datetime import datetime
from datetime import timedelta
from time import time
from statsmodels.tsa.arima_model import ARIMA

# Getting Training set
train_end = datetime(2018,1,1)
test_end = datetime(2018,9,30)
train_data = MSFTdf.Close[:train_end]
test_data = MSFTdf.Close[train_end + timedelta(days=1):test_end]

# Building ARIMA(3,1,3) model
model = ARIMA(train_data, order=(3,1,3))
start = time()
model_fit = model.fit()
end = time()
print('Model Fitting Time:', end - start)
print(model_fit.summary())

# making prediction model
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]
predictions = model_fit.predict(start=pred_start_date, end=pred_end_date)
residuals = test_data - predictions

# printing results
print('Root Mean Squared Error:', np.sqrt(np.mean(residuals**2)))
print('Mean Absolute Percent Error:', round(np.mean(abs(residuals/test_data)),4))
with open('ARIMA_Summary.txt', 'w') as wfile:
    wfile.write(model_fit.summary().as_text())
    print('\nRoot Mean Squared Error:', np.sqrt(np.mean(residuals**2)), file=wfile)
    print('Mean Absolute Percent Error:', round(np.mean(abs(residuals/test_data)),4), file=wfile)
