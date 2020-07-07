import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from Data_prep.Data_preparation import MSFTdf, train_data
from time import time
import warnings
warnings.filterwarnings("ignore")

train_len = len(train_data)

# Scaling data
data = MSFTdf.filter(['Close'])
data_values = data.values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data_values)

# Creating training sets
train = scaled_data[0:train_len, :]
x_train = []
y_train = []

for i in range(60,len(train)):
    x_train.append(train[i-60:i, 0])
    y_train.append(train[i, 0])

# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshaping x_train to 3 dimensions
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))

# building LSTM
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
lstm_model.add(LSTM(50, return_sequences=False))
lstm_model.add(Dense(25))
lstm_model.add(Dense(1))

# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

#train the model
timer_start = time()
lstm_model.fit(x_train, y_train, batch_size=1, epochs=5)
timer_end = time()

# testing dataset
# creating new array, containing scaled values
test = scaled_data[train_len - 60:, :]

# creating datasets x_test and y_test
x_test = []
y_test = data_values[train_len:, :]

for i in range(60, len(test)):
    x_test.append(test[i - 60:i, 0])

# convert data to a numpy array
x_test = np.array(x_test)

# reshape data
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1], 1))

# get models predicted price values
lstm_predictions = lstm_model.predict(x_test)

# un-scaling values
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# Calculating Residuals
lstm_residuals = lstm_predictions - y_test

# Plot data
train_plot = data[:train_len]
lstm_final = data[train_len:]
lstm_final['Predictions'] = lstm_predictions

plt.figure(figsize=(10, 4))
plt.plot(lstm_final['Close'])
plt.plot(lstm_final['Predictions'])
plt.legend(('Data', 'Predictions'), fontsize=16)
plt.title('MSFT closing prices over time')
plt.ylabel('Price', fontsize = 16)
plt.savefig('Predicted LSTM')
plt.show()

print('LSTM Root Mean Squared Error:', np.sqrt(np.mean(lstm_residuals ** 2)))
print('LSTM Mean Absolute Percent Error:', round(np.mean(abs(lstm_residuals / y_test)) * 100, 4), '%')
print('LSTM Model Fitting Time:', timer_end - timer_start)
with open('LSTM_Summary.txt', 'w') as wfile:
    print('LSTM Root Mean Squared Error:', np.sqrt(np.mean(lstm_residuals ** 2)), file=wfile)
    print('LSTM Mean Absolute Percent Error:', round(np.mean(abs(lstm_residuals / y_test)) * 100, 4), '%', file=wfile)
