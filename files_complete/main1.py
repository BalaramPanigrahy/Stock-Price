import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from datetime import datetime

#Read the csv file
df = pd.read_excel(r'Input_files\Direktvermerktung_Prognose_Daten.xlsx', sheet_name='IST-Werte',  header=1)
df = df.rename(columns={df.columns[0]: 'Datum/Zeit'})
#df = df.loc[:, df.columns != 'Datum/Zeit'].fillna(0)
df = df.fillna(0)
df = df.drop_duplicates()
pd.set_option('display.max_columns', None)


df2 = pd.read_excel('Input_files\Direktvermarktung_Prognose_Daten.xlsx', sheet_name='Globalstrahlung', header=2)
df2 = df2.rename(columns={df2.columns[1]: 'Globalstrahlung_(Ist)',df2.columns[2]: 'Globalstrahlung_(Prognose)'})
df2[["Globalstrahlung_(Ist)", "Globalstrahlung_(Prognose)"]] = df2[["Globalstrahlung_(Ist)", "Globalstrahlung_(Prognose)"]].fillna(0)
df2 = df2.drop_duplicates()

df3 = pd.read_excel('Input_files\SWT_Prognoseparameter_DV_Anlagen_20220711.xlsx')


df4 = pd.read_excel('Input_files\Wetterstation_Trier_Petrisberg_Temperatur_IST_PROG.xlsx', header=2, parse_dates=True)
df4 = df4.rename(columns={df4.columns[1]: 'Temperatur_(Ist)',df4.columns[2]: 'Temperatur_(Prognose)'})
df4[["Temperatur_(Ist)", "Temperatur_(Prognose)"]] = df4[["Temperatur_(Ist)", "Temperatur_(Prognose)"]].fillna(0)


df4_x = df4.set_index(pd.DatetimeIndex(df4['Datum/Zeit']))
df4_x.loc[df4_x.index[-1] + pd.Timedelta(1, unit='h')] = df4_x.iloc[-1]

df5 = df4_x.drop_duplicates()

df42 = df5.resample('15Min').ffill()
df42 = df42.rename_axis('Datum/Zeitnew').reset_index()

df6 = pd.merge(df2,df42, left_on='Datum/Zeit', right_on='Datum/Zeitnew', how='left')

df7 = pd.merge(df,df6, left_on='Datum/Zeit', right_on='Datum/Zeit_x', how='right')

#df7["DV_PV0007","Globalstrahlung_(Ist)","Globalstrahlung_(Prognose)","Temperatur_(Ist)","Temperatur_(Prognose)"] = pd.to_numeric(df7["DV_PV0007","Globalstrahlung_(Ist)","Globalstrahlung_(Prognose)","Temperatur_(Ist)","Temperatur_(Prognose)"])
df7.drop(['Datum/Zeit_x','Datum/Zeitnew','Datum/Zeit_y'], inplace=True, axis=1)

df7['Average']=pd.Series(df7['DV_PV0207']).rolling(window=365).mean()
df['Average_std']=pd.Series(df['DV_PV0207']).rolling(window=365).std()
df7 = df7.fillna(0)
print(df7)

#Separate dates for future plotting
train_dates = pd.to_datetime(df7['Datum/Zeit'])
#print(train_dates.tail(3)) #Check last few dates.
cols = list(df7)[12:17]

#Date column is not used in training.
print('Columns used for training are:')
print(cols) #['DV_PV0207', 'Globalstrahlung_(Ist)', 'Globalstrahlung_(Prognose)', 'Temperatur_(Ist)', 'Temperatur_(Prognose)','Average']

#New dataframe with only training data - 5 columns
df_for_training = df7[cols].astype(float)
#print(df_for_training)


# df_for_plot=df_for_training.tail(5000)
# df_for_plot.plot.line()

# LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
#scaler = StandardScaler()
scaler = MinMaxScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

#df_for_training[cols] = scaler.fit_transform(df_for_training[cols])
#df_for_training_scaled = df_for_training.copy()
#print(df_for_training)
#print(df_for_training['DV_PV0207'].sort_values(ascending=True))

#check the values

# As required for LSTM networks, we require to reshape an input data into n_samples x timestamps x n_features.
# Here, the n_features is 5. We will make timestamps = 81792 (past timestamp data used for training).

# Empty lists to be populated using formatted training data
trainX = []
trainY = []

n_future = 90  # Number of days we want to look into the future based on the past days.
n_past = 10000  # Number of past days we want to use to predict the future.

# Reformat input data into a shape: (n_samples x timestamps x n_features)
# In our case, our df_for_training_scaled has a shape (81792, 5)
# 81792 refers to the number of data points and 5 refers to the columns (multi-variables).
for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])

print(df_for_training_scaled.shape[1])
print(df_for_training.shape[1])

trainX, trainY = np.array(trainX), np.array(trainY)
#x_train, x_test, y_train, y_test = train_test_split(trainX,trainY,test_size=0.20,random_state=4)
#print(trainX.shape)
print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))
#print(trainX)
#print(trainY)

#print(trainY.head(20))
# In our case, trainX has a shape (41732, 40000, 5).
# 41732 because we are looking back 40000 days (81732 - 40000 = 41732).
# Remember that we cannot look back 14 days until we get to the 15th day.
# Also, trainY has a shape (41732, 1). Our model only predicts a single value, but
# it needs multiple variables (5 in my example) to make this prediction.
# This is why we can only predict a single day after our training, the day after where our data ends.
# To predict more days in future, we need all the 5 variables which we do not have.
# We need to predict all variables if we want to do that.

# define the Autoencoder model
#print(trainX.shape[1])
#print(trainX.shape[2])
#print(trainY.shape[1])

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

# fit the model
history = model.fit(trainX, trainY, epochs=5, batch_size=250, validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

#forecasting...
#start with last day in training date and predict future
#n_future = 90
#forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()
forecast = model.predict(trainX[-n_future:])#forecast
#print(forecast)

#perform inverse transformation to rescale bacl to original range
#since we used 5 variables for transform, the inverse expects same dimensions
#therefore, let us copy our values 5 times and discard them after inverse transform
forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]
#print(y_pred_future)

#convert timestamp to date
#forecast_dates = []
#for time_i in forecast_period_dates:
    #forecast_dates.append(time_i.date())

#df_forecast = pd.DataFrame({'Date': np.array(forecast_dates),'pred':y_pred_future})
df_forecast = pd.DataFrame({'pred':y_pred_future})
#df_forecast['Datum/Zeit'] = pd.to_datetime(df_forecast['Datum/Zeit'])
print(df_forecast)

# creating excel writer object
writer = pd.ExcelWriter('Output_files/converted-to-excel7.xlsx')
# write dataframe to excel
df_forecast.to_excel(writer)
# save the excel
writer.save()

#original = df[['Datum/Zeit','DV_PV0207']]
#original['Datum/Zeit'] = pd.to_datetime(original['Datum/Zeit'])
#original = original.loc[original['Datum/Zeit'] >= '01-03-2020']
#print(original)


#sns.lineplot(original['Datum/Zeit', original['DV_PV0207']])
#sns.lineplot(original['Datum/Zeit'],df_forecast['pred'])



# creating excel writer object
#writer = pd.ExcelWriter('output_files/converted-to-excel3.xlsx')
# write dataframe to excel
#df7.to_excel(writer)
# save the excel
#writer.save()



##df4= df4.set_index(['Datum/Zeit'])
#df5=df4.resample('15T', label = 'right', closed='right')
#print(df5)
#df5= pd.merge(df2, df4, on='Datum/Zeit',  how='outer')
#print(df5.tail(115))