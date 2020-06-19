#importing matplotlib and statsmodels
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#sourcing df from Data Preparation
from Data_prep.Data_preparation import df

#exporting ACF and PACF
acf_plot = plot_acf(df.IntChange)
plt.savefig('MSFT_ACF')

pacf_plot = plot_pacf(df.IntChange)
plt.savefig('MSFT_PACF')
