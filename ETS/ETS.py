from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt
from Data_prep.Data_preparation import MSFTdf

# additive decomposition
result_add = seasonal_decompose(MSFTdf.Close[-70:], model='additive')
fig = result_add.plot()
plt.savefig('ETS Additive Decomposition')
plt.show()

# multiplicative decomposition
result_mult = seasonal_decompose(MSFTdf.Close[-70:], model='multiplicative')
fig = result_mult.plot()
plt.savefig('ETS Multiplicative Decomposition')
plt.show()

# additive decomposition
result_add = seasonal_decompose(MSFTdf.Close[-14:], model='additive')
fig = result_add.plot()
plt.savefig('14 day ETS Additive Decomposition')
plt.show()