import pandas as pd
from pandas import read_csv
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from matplotlib import pyplot as plt


def parse_date(date):
    day, hour = date.split(' ')
    return day


series = read_csv('austin_weather.csv', header=0)
new_df = pd.DataFrame()
new_df['Date'] = series['Date']
new_df['Price'] = list(series['TempAvgF'])
# new_df['Price'] = list(series['DewPointAvgF'])
# new_df['Price'] = list(series['WindAvgMPH'])
print(new_df["Price"].head())
new_df.set_index('Date', inplace=True)
new_df.index = pd.to_datetime(new_df.index)
tmp = new_df[(new_df.index >= '2013-12-21')]
test = tmp[(tmp.index >= '2017-07-01')]
train = tmp[(tmp.index < '2017-07-01')]
model = ExponentialSmoothing(
    train,seasonal_periods=365,trend='additive',seasonal='additive').fit()
alpha = model.params['smoothing_level']
beta = model.params['smoothing_trend']
gamma = model.params['smoothing_seasonal']
period = 365
print('alpha: {}'.format(alpha), 'beta: {}'.format(beta), 'gamma: {}'.format(gamma),sep='\n')
print('aic: ', model.aic)
L = model.level.to_frame('level')
S = model.season.to_frame('season')
P = model.trend.to_frame('slope')
R = model.resid.to_frame('resid')
hw_fit_param = S.join(L).join(P).join(R)
fitted_train = hw_fit_param.sum(axis=1).to_frame('prediction')
sale_train = train.join(fitted_train)
prediction = model.forecast(2*len(test))
sale_test = test.join(prediction.to_frame('prediction'),how='outer')
sale = pd.concat([sale_train,sale_test])
sale['dt'] = gamma*abs(sale['Price']-sale['prediction'])
sale['shifted_dt'] = sale['dt'].shift(period)
sale['dt'] = (sale['dt'] + (1-gamma)*sale['shifted_dt'])
sale['dt'] = sale['dt'].shift(period)
sale_test = sale_test.join(sale['dt'],how='left')
m = 2
sale_test['ub'] = sale_test['prediction'] + sale_test['dt']*m
sale_test['lb'] = sale_test['prediction'] - sale_test['dt']*m
sale_test_outliners = sale_test[((sale_test['Price'] > sale_test['ub'])|(sale_test['Price'] < sale_test['lb']))]
plt.figure(figsize=(18,10))
plt.plot(sale_test['prediction'],color='green',linestyle='dashdot',marker='o',markerfacecolor='green',
         label='Holt-winter Forecast')
plt.plot(sale_test['Price'],color='black',linestyle='solid',
         marker='o',markerfacecolor='black',label='Testing Data')
plt.plot(sale.loc[:sale_test.index[0]]['Price'],color='dimgray',
         linestyle='solid',marker='o',markerfacecolor='black',label='Training Data')
plt.plot(sale_test_outliners['Price'],'o',markerfacecolor='red',
         label='Anomalies',markeredgecolor='red')
plt.plot(sale_test['ub'],linestyle='dashdot',color='blue',label='Brutlag UpperBound',alpha=0.75
         ,solid_capstyle='round',solid_joinstyle='round',linewidth=3)
plt.plot(sale_test['lb'],linestyle='dashdot',color='blue',label='Brutlag LowerBound',alpha=0.75
        ,solid_capstyle='round',solid_joinstyle='round',linewidth=3)
plt.vlines(sale_test.index[0],ymin=0,ymax=sale['Price'].max(),linewidth=3,alpha=0.5)
plt.text(x=sale_train.index[-3],y=120000,s='End of training period', fontsize=12)
plt.text(x=sale.index[-5], y=1000, s=f'Scaling Factor (m) = {m}', fontsize=12)
plt.title(label='Anomalies detection with Brutlag Confidence Band and Holt-Winter ES (Whole Picture)')
plt.xlabel('Time')
plt.ylabel('Sales')
plt.legend(loc='upper left')
plt.savefig('marocha.png')