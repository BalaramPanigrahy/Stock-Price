
from darts.metrics import rmse
from sklearn.ensemble import RandomForestRegressor
from darts.models.forecasting.regression_model import RegressionModel, _LikelihoodMixin
from darts.timeseries import TimeSeries
import pandas as pd
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
#Read the csv file
df = pd.read_excel('Input_files2\Direktvermarktung_Prognose_Daten.xlsx', sheet_name='IST-Werte',  header=1)
df = df.rename(columns={df.columns[0]: 'Datum/Zeit'})
#df = df.fillna(0)
#print(df)
df = df.drop_duplicates()

df2 = pd.read_excel('Input_files2\Direktvermarktung_Prognose_Daten.xlsx', sheet_name='Globalstrahlung', header=2)
df2 = df2.rename(columns={df2.columns[1]: 'Globalstrahlung_(Ist)',df2.columns[2]: 'Globalstrahlung_(Prognose)'})
df2[["Globalstrahlung_(Prognose)"]] = df2[["Globalstrahlung_(Prognose)"]].fillna(0)
df2 = df2.drop_duplicates()

df3 = pd.read_excel('Input_files2\SWT_Prognoseparameter_DV_Anlagen_20220711.xlsx')

df4 = pd.read_excel('Input_files2\Wetterstation_Trier_Petrisberg_Temperatur_IST_PROG.xlsx', header=2, parse_dates=True)
df4 = df4.rename(columns={df4.columns[1]: 'Temperatur_(Ist)',df4.columns[2]: 'Temperatur_(Prognose)'})
df4[["Temperatur_(Prognose)"]] = df4[["Temperatur_(Prognose)"]].fillna(0)

df4_x = df4.set_index(pd.DatetimeIndex(df4['Datum/Zeit']))
df4_x.loc[df4_x.index[-1] + pd.Timedelta(1, unit='h')] = df4_x.iloc[-1]

df5 = df4_x.drop_duplicates()

df42 = df5.resample('15Min').ffill()
df42 = df42.rename_axis('Datum/Zeitnew').reset_index()

df6 = pd.merge(df2,df42, left_on='Datum/Zeit', right_on='Datum/Zeitnew', how='left')

df7 = pd.merge(df,df6, left_on='Datum/Zeit', right_on='Datum/Zeit_x', how='right')
#df7["DV_PV0007","Globalstrahlung_(Ist)","Globalstrahlung_(Prognose)","Temperatur_(Ist)","Temperatur_(Prognose)"] = pd.to_numeric(df7["DV_PV0007","Globalstrahlung_(Ist)","Globalstrahlung_(Prognose)","Temperatur_(Ist)","Temperatur_(Prognose)"])
df7.drop(['Datum/Zeit_x','Datum/Zeitnew','Datum/Zeit_y'], inplace=True, axis=1)

df8 = df7.iloc[: , [0,1,13,14,15,16]].copy()

#df9 = df8.drop['DV_PV0207','Globalstrahlung_(Ist)','Temperatur_(Ist)'](df8['Datum/Zeit' > '01-06-2022 00:00'])
df8 = df8.set_index(pd.DatetimeIndex(df8['Datum/Zeit']))
df8['month'] = df8.index.strftime('%m')
df8['Year'] = df8.index.strftime('%Y')
df8['average']=pd.Series(df8['DV_PV0007']).rolling(window=365).mean()
df8['average_std']=pd.Series(df8['DV_PV0007']).rolling(window=365).std()
#print(df8)


#series = TimeSeries.from_dataframe(df8, 'Datum/Zeit', 'DV_PV0207')
#train, val = series[:-360], series[-360:]
#model = ExponentialSmoothing()
#model.fit(train)
#prediction = model.predict(len(val), num_samples=1000)
#series.plot()
#prediction.plot(label='forecast',low_quantile=0.05,high_quantile=0.95)
#plt.legend()
#y = TimeSeries.from_series(df8['DV_PV0207'])
#features = ['Datum/Zeit','Globalstrahlung_(Prognose)','Temperatur_(Prognose)']
#future_cov = TimeSeries.from_dataframe[df[features]]
#future_cov_all = TimeSeries.from_group_dataframe(df8,group_cols='',time_col='Datum/Zeit',value_cols=['Datum/Zeit]','Globalstrahlung_(Prognose)','Temperatur_(Prognose)'])

y = TimeSeries.from_series(df8['DV_PV0007'])
features = ['Globalstrahlung_(Ist)','Temperatur_(Ist)']
features2= ['Globalstrahlung_(Prognose)','Temperatur_(Prognose)','month','Year','average','average_std']
past_cov = TimeSeries.from_dataframe(df8[features])
future_cov = TimeSeries.from_dataframe(df8[features2])
y_train = y[:-2880]

model = RegressionModel(lags = [-1,-50000], lags_future_covariates=[1,100],lags_past_covariates=[-1,-50000], model = RandomForestRegressor())
model.fit(y_train,past_covariates=past_cov, future_covariates=future_cov)
y_pred = model.predict(n=200,series=y_train,past_covariates=past_cov,future_covariates=future_cov)


'''
# We first set aside the first 80% as training series:
flow_train, _ = model.split_before(0.8)

def eval_model(model, past_covariates=None, future_covariates=None):
    # Past and future covariates are optional because they won't always be used in our tests

    # We backtest the model on the last 20% of the flow series, with a horizon of 10 steps:
    backtest = model.historical_forecasts(series=model,
                                          past_covariates=past_covariates,
                                          future_covariates=future_covariates,
                                          start=0.8,
                                          retrain=False,
                                          verbose=True,
                                          forecast_horizon=10)

    model[-len(backtest) - 100:].plot()
    backtest.plot(label='backtest (n=10)')
    print('Backtest RMSE = {}'.format(rmse(model, backtest)))
'''

#print(y_pred)
#y_pred.frame(Y=as.matrix(y_pred), date=time(y_pred))
#for item in y_pred:
#    print(item)
#list1 = y_pred.values.tolist()
#print(list1)
#df_y=DataFrame(eval(y_pred))
#print('datatype: ',type(y_pred))
df_new=y_pred.pd_dataframe()
#print(df_new)
#df_new.plot()
#plt.show()
df_measured = pd.read_excel('Input_files\Direktvermarktung_Prognose_Daten.xlsx', sheet_name='IST-Werte',  header=1)
df_measured = df_measured.rename(columns={df_measured.columns[0]: 'Datum/Zeit'})
df_measured = df_measured[['Datum/Zeit','DV_PV0007']]
#print(df_measured)
df_measured2 = pd.read_excel('Input_files\Direktvermarktung_Prognose_Daten.xlsx', sheet_name='Prognose-Werte',  header=1)
df_measured2 = df_measured2.rename(columns={df_measured2.columns[0]: 'Datum/Zeit'})
df_measured2 = df_measured2[['Datum/Zeit','DV_PV0007']]
#df_merge3=pd.merge(df_measured,df_measured2, left_on='Datum/Zeit',right_on='Datum/Zeit', how='left')
#print(df_merge3)
#len1=len(df_merge3)
#df_merge4=pd.merge(df_merge3,df_new, left_on='Datum/Zeit', right_index=True, how='left').set_index('Datum/Zeit')
#print(df_merge4)
#plt.plot(range(len(df_merge3)), df_new)
#print(df_measured2)
df_merge=  pd.merge(df_measured,df_new, left_on='Datum/Zeit', right_index=True, how='right').set_index('Datum/Zeit')
#print(df_merge)
#df_merge.plot()
#plt.show()
df_merge2= pd.merge(df_merge,df_measured2, left_index=True, right_on='Datum/Zeit', how='left').set_index('Datum/Zeit')
#df_merge2=  pd.merge(df_measured2,df_new, left_on='Datum/Zeit', right_index=True, how='right').set_index('Datum/Zeit')
#df_merge.plot()
#print(df_merge2)
#df_merge2.plot()
#plt.show()
df_merge2.plot(kind='line',figsize=(15,8))
plt.show()

#ax = df_merge2.plot(kind='line')
#i=[1,2,3]
#for i, line in enumerate(ax.get_lines()):
#    line.i
#line.set_marker(markers[i])
# adding legend
#ax.legend(ax.get_lines(), df_merge2.columns, loc='best')
#plt.show()

#data.frame()
#DataArray.to_dataframe(name=None, dim_order=None)
#DataArray.to_series(y_pred)
# create valid markers from mpl.markers
#valid_markers = ([item[0] for item in mpl.markers.MarkerStyle.markers.items() if
#                  item[1] is not 'nothing' and not item[1].startswith('tick') and not item[1].startswith('caret')])

# valid_markers = mpl.markers.MarkerStyle.filled_markers

#markers = np.random.choice(valid_markers, df_merge2.shape[1], replace=False)

#ax = df_merge2.plot(kind='line')
#for i, line in enumerate(ax.get_lines()):
#    line.set_marker(markers[i])

# adding legend
#ax.legend(ax.get_lines(), df_merge2.columns, loc='best')
#plt.show()