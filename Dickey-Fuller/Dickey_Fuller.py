from Data_prep.Data_preparation import MSFTdf
from statsmodels.tsa.stattools import adfuller

# Augmented Dickey Fuller Test
ADFresult = adfuller(MSFTdf['Close'])

# stating null hypothesis
print('\nNull Hypothesis (H0) = Data is non-stationary')

# displaying results using Closing prices
print('\nDickey Fuller Test using Closing prices:')
print('ADF Statistic: %f' %ADFresult[0])
print('p-value: %f' %ADFresult[1])
print('Critical Values:')
for key, value in ADFresult[4].items():
    print('\t%s: %.3f' %(key, value))
if ADFresult[0] < ADFresult[4]['1%']:
    print('Reject the Null Hypothesis; Data is stationary')
else:
    print('Fail to reject null hypothesis; Data is non-stationary')

# displaying results using first differences in closing prices
ADFresult = adfuller(MSFTdf['IntChange'])
print('\nDickey Fuller Test using First Differences of Closing prices: ')
print('ADF Statistic: %f' %ADFresult[0])
print('p-value: %f' %ADFresult[1])
print('Critical Values:')
for key, value in ADFresult[4].items():
    print('\t%s: %.3f' %(key, value))
if ADFresult[0] < ADFresult[4]['1%']:
    print('Reject the Null Hypothesis; Data is stationary')
else:
    print('Fail to reject null hypothesis; Data is non-stationary')
