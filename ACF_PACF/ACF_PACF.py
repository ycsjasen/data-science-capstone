#importing matplotlib and statsmodels
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#sourcing df from Data Preparation
from Data_prep.Data_preparation import MSFTdf

#exporting ACF and PACF
acf_plot = plot_acf(MSFTdf.IntChange)
#plt.savefig('MSFT_ACF')

pacf_plot = plot_pacf(MSFTdf.IntChange)
#plt.savefig('MSFT_PACF')
