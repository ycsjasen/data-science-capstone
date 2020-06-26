from Data_prep.Data_preparation import MSFTdf
from pmdarima.arima import auto_arima

stepwise_model = auto_arima(MSFTdf.Close, start_p=1, start_q=1,
                           max_p=7, max_q=7, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=True)

# print lowest AIC
print(stepwise_model.aic())
