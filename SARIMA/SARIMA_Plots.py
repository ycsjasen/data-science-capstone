from SARIMA import sarima_residuals, sarima_predictions
from Data_prep.Data_preparation import test_data
import pandas as pd
from matplotlib import pyplot as plt

# plotting SARIMA residuals
plt.figure(figsize=(10,4))
plt.plot(sarima_residuals)
plt.title('Residuals from SARIMA Model', fontsize = 20)
plt.ylabel('Error', fontsize=16)
plt.axhline(sarima_residuals.mean(), color = 'r', linestyle='--', alpha = 0.2)
for year in range(2019,2019):
    plt.axvline(pd.to_datetime(str(year) + '-01-01'), color='k', linestyle='--', alpha=0.2)
plt.savefig('Residuals SARIMA')

# plotting prediction and test data
plt.figure(figsize=(10,4))
plt.plot(test_data)
plt.plot(sarima_predictions)
plt.legend(('Data', 'Predictions'), fontsize=20)
plt.title('MSFT closing prices over time')
plt.ylabel('Price', fontsize = 16)
for year in range(2019,2019):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'),color='k', linestyle= '--', alpha=0.2)
plt.axhline(test_data.mean(), color='r', alpha =0.2, linestyle='--')
plt.savefig('Predicted SARIMA')
plt.show()
