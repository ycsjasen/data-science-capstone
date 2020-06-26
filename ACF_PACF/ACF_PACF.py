import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf

# sourcing df from Data Preparation
from Data_prep.Data_preparation import MSFTdf

# exporting ACF and PACF
acf_plot = plot_acf(MSFTdf.IntChange)
acf_vals = acf(MSFTdf.IntChange)
plt.bar(range(24), acf_vals[:24])
#plt.savefig('MSFT_ACF')

pacf_plot = plot_pacf(MSFTdf.IntChange)
pacf_vals = pacf(MSFTdf.IntChange)
plt.bar(range(24), pacf_vals[:24])
#plt.savefig('MSFT_PACF')

plt.show()
