from Data_prep.Data_preparation import MSFTdf
import pandas as pd
from matplotlib import pyplot as plt

# plotting closing price data
plt.figure(figsize=(10, 4))
plt.plot(MSFTdf.Close)
plt.title('MSFT closing prices over time')
plt.ylabel('Price', fontsize=16)
for year in range(2017, 2019):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
plt.axhline(MSFTdf.Close.mean(), color='r', alpha=0.2, linestyle='--')
plt.savefig('Closing price MSFT')

# plotting first difference data
plt.figure(figsize=(10, 4))
plt.plot(MSFTdf.IntChange)
plt.title('MSFT first difference closing prices over time')
plt.ylabel('Price', fontsize=16)
for year in range(2017, 2019):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
plt.axhline(MSFTdf.IntChange.mean(), color='r', alpha=0.2, linestyle='--')
plt.savefig('First Differences MSFT')

plt.show()
