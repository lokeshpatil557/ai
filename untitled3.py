# -*- coding: utf-8 -*-
"""Untitled3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1irUa-_0X9QMPimXwmFYUsrbKZwORFXWA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm

df=pd.read_csv("TimeSeries.csv",parse_dates=['Date'],index_col='Date')
print(df)

from statsmodels.tsa.stattools import adfuller
test_result = adfuller(df['Value'])
print(test_result)

df['Seasonal_Difference']=df['Value'].shift(1)
test_result=adfuller(df['Seasonal_Difference'].dropna())
print(test_result)

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Value'], lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Value'], lags=40, ax=ax2)
plt.show()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(df['Seasonal_Difference'].dropna(), lags=8, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(df['Seasonal_Difference'].dropna(), lags=8, ax=ax2)
plt.show()

from statsmodels.tsa.arima_model import ARMA
#statsmodels.tsa.arima.model.ARIMA
#import statsmodels.api as sm
ARMAmodel = ARMA(df['Value'],order=(1,1))
ARmodel_fit = ARMAmodel.fit(disp=False)
actuals = df['Value'][200:204]
print(actuals)

ypredicted = ARmodel_fit.predict(200,203)
print(ypredicted)

from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(actuals,ypredicted)
print('MAE: %f'%mae)
#print(ARmodel_fit.aic)

import itertools
i = j = range(0, 4)
ij = itertools.product(i,j)
for parameters in ij:
    try:
        mod = ARMA(df['Value'],order=parameters)
        results = mod.fit()
        ypredicted = results.predict(200,203)  # end point included
        mae = mean_absolute_error(actuals, ypredicted)
        print('ARMA{} - MAE:{}'.format(parameters, mae))
        #print('ARMA{} - AIC:{}'.format(parameters, results.aic))
    except:
        continue

ARMAmodel = ARMA(df['Value'], order=(1, 2))
ARmodel_fit = ARMAmodel.fit()
ypredicted = ARmodel_fit.predict(200,203)  # end point included
print(ypredicted)
mae = mean_absolute_error(actuals, ypredicted)
print('MAE: %f' % mae)
print(ARmodel_fit.aic)

import statsmodels.api as sm
print(sm.__version__)

# make prediction
ypredicted = ARmodel_fit.predict(len(df), len(df)+2)
print(ypredicted)

import itertools
i = j = range(0, 4)
ij = itertools.product(i,j)
for parameters in ij:
    try:
        mod = ARMA(df['Seasonal_Difference'].dropna(),order=parameters)
        results = mod.fit()
        ypredicted = results.predict(210,213)  # end point included
        mae = mean_absolute_error(actuals, ypredicted)
        print('ARMA{} - MAE:{}'.format(parameters, mae))
        #print('ARMA{} - AIC:{}'.format(parameters, results.aic))
    except:
        continue

from statsmodels.tsa.arima_model import ARIMA

# fit model
ARIMAmodel = ARIMA(df['Value'], order=(1, 1, 1)) #notice p,d and q value here
ARIMA_model_fit = ARIMAmodel.fit(disp=False)

# make prediction
ypredicted = ARIMA_model_fit.predict(len(df), len(df)+2, typ='levels')
print(ypredicted)