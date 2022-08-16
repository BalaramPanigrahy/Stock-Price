import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np

pd.set_option('display.max_columns', None)
df = pd.read_excel('Input_files\Direktvermarktung_Prognose_Daten.xlsx', sheet_name='IST-Werte', header=1)
df = df.rename(columns={df.columns[0]: 'Datum/Zeit'})
df = df.fillna(0)
df = df.drop_duplicates()
df['Datum/Zeit'] = pd.to_datetime(df['Datum/Zeit'])  # convert date column to DateTime

ax = df.plot(x='Datum/Zeit', y='DV_PV0007', figsize=(12, 6))
plt.show()
xcoords = ['01-01-2020', '01-01-2021', '01-01-2022']
for xc in xcoords:
    plt.axvline(x=xc, color='black', linestyle='--')
    plt.show()

df.set_index('Datum/Zeit', inplace=True)
#print(df)
analysis = df[['DV_PV0007']].copy()

period = int(len(df)/8)
print(period)
decompose_result = seasonal_decompose(analysis, model="additive", period=period)

trend = decompose_result.trend
seasonal = decompose_result.seasonal
residual = decompose_result.resid

decompose_result.plot();
plt.show()


def analyze_stationarity(timeseries, title):
    fig, ax = plt.subplots(2, 1, figsize=(16, 8))

    rolmean = pd.Series(timeseries).rolling(window=30).mean()
    rolstd = pd.Series(timeseries).rolling(window=30).std()
    ax[0].plot(timeseries, label=title)
    ax[0].plot(rolmean, label='rolling mean');
    ax[0].plot(rolstd, label='rolling std (x10)');
    ax[0].set_title('30-day window')
    ax[0].legend()

    rolmean = pd.Series(timeseries).rolling(window=365).mean()
    rolstd = pd.Series(timeseries).rolling(window=365).std()
    ax[1].plot(timeseries, label=title)
    ax[1].plot(rolmean, label='rolling mean');
    ax[1].plot(rolstd, label='rolling std (x10)');
    ax[1].set_title('365-day window')
    ax[1].legend()
    plt.show()


pd.options.display.float_format = '{:.8f}'.format
analyze_stationarity(df['DV_PV0007'], 'raw data')


df['average']=pd.Series(df['DV_PV0007']).rolling(window=365).mean()
df['average_std']=pd.Series(df['DV_PV0007']).rolling(window=365).std()
df['month'] = df.index.strftime('%m')
print(df)
