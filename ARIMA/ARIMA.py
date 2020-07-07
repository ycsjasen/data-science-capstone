from Data_prep.Data_preparation import test_data, train_data
import numpy as np
from time import time
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot as plt


# Building ARIMA(0,1,1) model
arima_order = (0, 1, 1)
model = ARIMA(train_data, order=arima_order)
timer_start = time()
model_fit = model.fit()
timer_end = time()

# making prediction model
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]
arima_predictions = model_fit.predict(start=pred_start_date, end=pred_end_date)
arima_residuals = test_data - arima_predictions

# printing results
print(model_fit.summary())
print('ARIMA Root Mean Squared Error:', np.sqrt(np.mean(arima_residuals ** 2)))
print('ARIMA Mean Absolute Percent Error:', round(np.mean(abs(arima_residuals / test_data)) * 100, 4), '%')
print('ARIMA Model Fitting Time:', timer_end - timer_start)
with open('ARIMA_Summary.txt', 'w') as wfile:
    wfile.write(model_fit.summary().as_text())
    print('\nARIMA Root Mean Squared Error:', np.sqrt(np.mean(arima_residuals ** 2)), file=wfile)
    print('ARIMA Mean Absolute Percent Error:', round(np.mean(abs(arima_residuals / test_data)) * 100, 4), '%',
          file=wfile)

# plotting ARIMA residuals
plt.figure(figsize=(10, 4))
plt.plot(arima_residuals)
plt.title('Residuals from ARIMA Model', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.savefig('ARIMA Residuals')

# plotting prediction and test data
plt.figure(figsize=(10, 4))
plt.plot(test_data)
plt.plot(arima_predictions)
plt.legend(('Data', 'Predictions'), fontsize=16)
plt.title('MSFT closing prices over time (ARIMA)')
plt.ylabel('Price', fontsize=16)
plt.savefig('ARIMA Predicted')
plt.show()
