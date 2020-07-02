import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import numpy as np
from GARCH_data import MSFTdf, train_data, test_data

test_size = len(test_data)

# GARCH(1,1) model
model = arch_model(train_data, p=1, q=1)
model_fit = model.fit()
print(model_fit)

# Predicting GARCH
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]
predictions = model_fit.forecast(horizon=test_size)
sqrt_pred = np.sqrt(predictions.variance.values[-1,:])
model_pred = pd.Series(sqrt_pred, index=MSFTdf.IntChange.index[-test_size:])

# Plotting GARCH
plt.figure(figsize=(10,4))
data_vol = plt.plot(MSFTdf.Volatility[-test_size:])
pred_vol = plt.plot(model_pred)
plt.legend(('Data', 'Prediction'), fontsize=16)
plt.title('Volatility in price first differences over time')
plt.ylabel('Volatility', fontsize = 16)
plt.savefig('Predicted GARCH')
plt.show()

# Rolling Origin
roll = []
for i in range(test_size):
    train = MSFTdf.IntChange[:-(test_size-i)]
    model = arch_model(train, p=1, q=1)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    roll.append(np.sqrt(pred.variance.values[-1,:][0]))
roll = pd.Series(roll, index=MSFTdf.IntChange.index[-test_size:])

# Plotting GARCH with Rolling Origin
plt.figure(figsize=(10,4))
data_vol = plt.plot(MSFTdf.Volatility[-test_size:])
pred_vol = plt.plot(roll)
plt.legend(('Data', 'Prediction'), fontsize=16)
plt.title('Volatility in price first differences over time')
plt.ylabel('Volatility', fontsize = 16)
plt.savefig('Predicted GARCH (Rolling Origin Forecasting)')
plt.show()
