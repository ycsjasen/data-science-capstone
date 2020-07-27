from statsmodels.tsa.holtwinters import ExponentialSmoothing
from Data_prep.Data_preparation import train_data, test_data
import numpy as np
from time import time
from matplotlib import pyplot as plt

# building Additive ES model
test_size = len(test_data)
add_es_model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=7)

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
plt.title('Holt-Winters Exponential Smoothing - MSFT closing prices')
plt.ylabel('Price', fontsize=16)
plt.savefig('Additive ES model')
plt.show()

# Generating summary statistics
print('Add ES Root Mean Squared Error:', np.sqrt(np.mean(add_es_residuals ** 2)))
print('Add ES Normal Root Mean Squared Error:', np.sqrt(np.mean(add_es_residuals ** 2))/abs(np.mean(add_es_pred)))
print('Add ES Mean Absolute Percent Error:', round(np.mean(abs(add_es_residuals / test_data)) * 100, 4), '%')
print('Add ES Model Fitting Time:', timer_end - timer_start)

with open('HWES_Summary.txt', 'w') as wfile:
    print('Add ES Root Mean Squared Error:', np.sqrt(np.mean(add_es_residuals ** 2)), file=wfile)
    print('Add ES Normal Root Mean Squared Error:', np.sqrt(np.mean(add_es_residuals ** 2)) / abs(np.mean(add_es_pred)),
          file=wfile)
    print('Add ES Mean Absolute Percent Error:', round(np.mean(abs(add_es_residuals / test_data)) * 100, 4),
          '%', file=wfile)
