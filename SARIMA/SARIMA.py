from Data_prep.Data_preparation import train_data, test_data
import numpy as np
from time import time
from statsmodels.tsa.statespace.sarimax import SARIMAX
from matplotlib import pyplot as plt


# Building SARIMA(1,1,1)(1,1,1)7 model
sarima_order = (1, 1, 1)
season_order = (1, 1, 1, 7)
sarima_model = SARIMAX(train_data, order=sarima_order, seasonal_order=season_order)
timer_start = time()
sarima_model_fit = sarima_model.fit()
print(sarima_model_fit.summary())
timer_end = time()

# making prediction model
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]
sarima_predictions = sarima_model_fit.predict(start=pred_start_date, end=pred_end_date)
sarima_residuals = test_data - sarima_predictions
sarima_observed = test_data

# printing results
print('SARIMA Root Mean Squared Error:', np.sqrt(np.mean(sarima_residuals**2)))
print('SARIMA Normal Root Mean Squared Error:', np.sqrt(np.mean(sarima_residuals**2))/abs(np.mean(test_data)))
print('SARIMA Mean Absolute Percent Error:', round(np.mean(abs(sarima_residuals/test_data))*100, 4), '%')
print('SARIMA Model Fitting Time:', timer_end - timer_start)
with open('SARIMA_Summary.txt', 'w') as wfile:
    wfile.write(sarima_model_fit.summary().as_text())
    print('\nSARIMA Root Mean Squared Error:', np.sqrt(np.mean(sarima_residuals**2)), file=wfile)
    print('SARIMA Normal Root Mean Squared Error:', np.sqrt(np.mean(sarima_residuals ** 2)) / abs(np.mean(test_data)),
          file=wfile)
    print('SARIMA Mean Absolute Percent Error:', round(np.mean(abs(sarima_residuals/test_data))*100, 4),
          '%', file=wfile)

# plotting SARIMA residuals
plt.figure(figsize=(10, 4))
plt.plot(sarima_residuals)
plt.title('Residuals from SARIMA Model', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.savefig('SARIMA Residuals')

# plotting prediction and test data
plt.figure(figsize=(10, 4))
plt.plot(test_data)
plt.plot(sarima_predictions)
plt.legend(('Data', 'Predictions'), fontsize=16)
plt.title('SARIMA - MSFT closing prices over time')
plt.ylabel('Price', fontsize=16)
plt.savefig('SARIMA Predicted')
plt.show()
