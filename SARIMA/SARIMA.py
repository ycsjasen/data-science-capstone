from Data_prep.Data_preparation import train_data, test_data
import numpy as np
from time import time
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Building SARIMA(1,1,1)(0,1,2)12 model
sarima_order = (1,1,1)
season_order = (0,1,2,12)
sarima_model = SARIMAX(train_data, order = sarima_order, seasonal_order=season_order)
start = time()
sarima_model_fit = sarima_model.fit()
end = time()
print('Model Fitting Time:', end - start)
print(sarima_model_fit.summary())

# making prediction model
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]
sarima_predictions = sarima_model_fit.predict(start=pred_start_date, end=pred_end_date)
sarima_residuals = test_data - sarima_predictions

# printing results
print('SARIMA Root Mean Squared Error:', np.sqrt(np.mean(sarima_residuals**2)))
print('SARIMA Mean Absolute Percent Error:', round(np.mean(abs(sarima_residuals/test_data)),4))
with open('SARIMA_Summary.txt', 'w') as wfile:
    wfile.write(sarima_model_fit.summary().as_text())
    print('\nRoot Mean Squared Error:', np.sqrt(np.mean(sarima_residuals**2)), file=wfile)
    print('Mean Absolute Percent Error:', round(np.mean(abs(sarima_residuals/test_data)),4), file=wfile)
