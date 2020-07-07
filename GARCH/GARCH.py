import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from time import time
import numpy as np
from GARCH_data import MSFTdf, train_data, test_data, test_size, data_vol
from statsmodels.graphics.tsaplots import plot_acf

# Generating PACF of squared returns
garch_pacf = plot_acf(MSFTdf.Returns**2)
plt.show()

# GARCH(1,1) model
garchmodel = arch_model(train_data, mean='AR', vol='GARCH', p=1, q=1)
timer_start = time()
garch_model_fit = garchmodel.fit()
timer_end = time()

# Predicting GARCH
garch_predictions = garch_model_fit.forecast(horizon=test_size)
garch_pred = pd.Series(garch_predictions.variance.values[-1, :], index=MSFTdf.IntChange.index[-test_size:])

# Calculating Summary Statistics
garch_residuals = garch_pred - data_vol
print(garch_model_fit)
print('GARCH Root Mean Squared Error:', np.sqrt(np.mean(garch_residuals**2)))
print('GARCH Mean Absolute Percent Error:', round(np.mean(abs(garch_residuals/data_vol))*100, 4), '%')
print('GARCH Model Fitting Time:', timer_end - timer_start)

# Plotting GARCH
plt.figure(figsize=(10, 4))
plt.plot(data_vol)
plt.plot(garch_pred)
plt.legend(('Data', 'Prediction'), fontsize=16)
plt.title('Volatility in price first differences over time')
plt.ylabel('Volatility', fontsize=16)
plt.savefig('Predicted GARCH')
plt.show()

# Rolling Origin
roll = []
timer_start = time()
for i in range(test_size):
    train = MSFTdf.Returns[:-(test_size-i)]
    garchmodel = arch_model(train, mean='AR', vol='GARCH', p=1, q=1)
    garch_model_fit = garchmodel.fit(disp='off')
    pred = garch_model_fit.forecast(horizon=1)
    roll.append(np.sqrt(pred.variance.values[-1, :][0]))
timer_end = time()
roll = pd.Series(roll, index=MSFTdf.IntChange.index[-test_size:])


# Calculating summary statistics for rolling origin
roll_residuals = roll - data_vol
print('GARCH (rolling origin) Root Mean Squared Error:', np.sqrt(np.mean(roll_residuals**2)))
print('GARCH (rolling origin) Mean Absolute Percent Error:', round(np.mean(abs(roll_residuals/data_vol))*100, 4), '%')
print('GARCH (rolling origin) Model Fitting Time:', timer_end - timer_start)

# Plotting GARCH with Rolling Origin
plt.figure(figsize=(10, 4))
plt.plot(data_vol)
plt.plot(roll)
plt.legend(('Data', 'Prediction'), fontsize=16)
plt.title('Volatility in price first differences over time (rolling origin)')
plt.ylabel('Volatility', fontsize=16)
plt.savefig('Predicted GARCH (Rolling Origin Forecasting)')
plt.show()

with open('GARCH_Summary.txt', 'w') as wfile:
    wfile.write(garch_model_fit.summary().as_text())
    print('\nGARCH Root Mean Squared Error:', np.sqrt(np.mean(garch_residuals**2)), file=wfile)
    print('GARCH Mean Absolute Percent Error:', round(np.mean(abs(garch_residuals/data_vol))*100, 4), '%', file=wfile)
    print('GARCH (rolling origin) Root Mean Squared Error:', np.sqrt(np.mean(roll_residuals**2)), file=wfile)
    print('GARCH (rolling origin) Mean Absolute Percent Error:', round(np.mean(abs(roll_residuals/data_vol))*100, 4),
          '%', file=wfile)
