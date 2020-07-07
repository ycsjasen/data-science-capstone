import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf, pacf

# sourcing df from Data Preparation
from Data_prep.Data_preparation import MSFTdf

# exporting ACF and PACF
acf_plot = plot_acf(MSFTdf.IntChange)
acf_vals = acf(MSFTdf.IntChange)
plt.bar(range(31), acf_vals[:31])
#plt.savefig('MSFT_ACF')

pacf_plot = plot_pacf(MSFTdf.IntChange)
pacf_vals = pacf(MSFTdf.IntChange)
plt.bar(range(31), pacf_vals[:31])
#plt.savefig('MSFT_PACF')

plt.show()
