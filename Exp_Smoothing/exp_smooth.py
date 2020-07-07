from statsmodels.tsa.holtwinters import ExponentialSmoothing
from Data_prep.Data_preparation import train_data, test_data
import numpy as np
from time import time
from matplotlib import pyplot as plt

# building Additive ES model
test_size = len(test_data)
add_es_model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12)

# fitting model
timer_start = time()
add_es_model_fit = add_es_model.fit()
timer_end = time()

# generating predictions
add_es_pred = add_es_model_fit.forecast(test_size)
add_es_residuals = add_es_pred - test_data

# plotting predictions
plt.figure(figsize=(10, 4))
plt.plot(test_data)
plt.plot(add_es_pred)
plt.title('Additive Exponential Smoothing - MSFT closing prices')
plt.ylabel('Price', fontsize=16)
plt.savefig('Additive ES model')
plt.show()

# Generating summary statistics
print('Add ES Root Mean Squared Error:', np.sqrt(np.mean(add_es_residuals ** 2)))
print('Add ES Mean Absolute Percent Error:', round(np.mean(abs(add_es_residuals / test_data)) * 100, 4), '%')
print('Add ES Model Fitting Time:', timer_end - timer_start)

# Building multiplicative ES model
mul_es_model = ExponentialSmoothing(train_data, trend='mul', seasonal='mul', seasonal_periods=12)

# fitting model
timer_start = time()
mul_es_model_fit = mul_es_model.fit()
timer_end = time()

# generating predictions
mul_es_pred = mul_es_model_fit.forecast(test_size)
mul_es_residuals = mul_es_pred - test_data

# plotting predictions
plt.figure(figsize=(10, 4))
plt.plot(test_data)
plt.plot(mul_es_pred)
plt.title('Multiplicative Exponential Smoothing - MSFT closing prices')
plt.ylabel('Price', fontsize=16)
plt.savefig('Multiplicative ES model')
plt.show()

# Generating summary statistics
print('Mul ES Root Mean Squared Error:', np.sqrt(np.mean(mul_es_residuals ** 2)))
print('Mul ES Mean Absolute Percent Error:', round(np.mean(abs(mul_es_residuals / test_data)) * 100, 4), '%')
print('Mul ES Model Fitting Time:', timer_end - timer_start)
with open('LSTM_Summary.txt', 'w') as wfile:
    print('Add ES Root Mean Squared Error:', np.sqrt(np.mean(add_es_residuals ** 2)), file=wfile)
    print('Add ES Mean Absolute Percent Error:', round(np.mean(abs(add_es_residuals / test_data)) * 100, 4),
          '%', file=wfile)
    print('Mul ES Root Mean Squared Error:', np.sqrt(np.mean(mul_es_residuals ** 2)), file=wfile)
    print('Mul ES Mean Absolute Percent Error:', round(np.mean(abs(mul_es_residuals / test_data)) * 100, 4),
          '%', file=wfile)
