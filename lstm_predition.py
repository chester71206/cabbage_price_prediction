# -*- coding: utf-8 -*-
"""
Created on Mon May 13 00:41:15 2024

@author: chester
"""

import pandas as pd

# 讀取六個CSV檔案
set1 = pd.read_excel(r"C:\Users\chester\Desktop\cabbage\github_repo\Nantou_cabbage-108-112.xls")
weather_csv2019 = pd.read_csv(r"C:\Users\chester\Desktop\cabbage\github_repo\csv_file\42HA10-2019-Precipitation-day.csv")
weather_csv2020 = pd.read_csv(r"C:\Users\chester\Desktop\cabbage\github_repo\csv_file\42HA10-2020-Precipitation-day.csv")
weather_csv2021 = pd.read_csv(r"C:\Users\chester\Desktop\cabbage\github_repo\csv_file\42HA10-2021-Precipitation-day.csv")
weather_csv2022 = pd.read_csv(r"C:\Users\chester\Desktop\cabbage\github_repo\csv_file\42HA10-2022-Precipitation-day.csv")
weather_csv2023 = pd.read_csv(r"C:\Users\chester\Desktop\cabbage\github_repo\csv_file\42HA10-2023-Precipitation-day.csv")

temperature_csv2019 = pd.read_csv(r"C:\Users\chester\Desktop\cabbage\github_repo\csv_file\42HA10-2019-AirTemperature-day.csv")
temperature_csv2020 = pd.read_csv(r"C:\Users\chester\Desktop\cabbage\github_repo\csv_file\42HA10-2020-AirTemperature-day.csv")
temperature_csv2021 = pd.read_csv(r"C:\Users\chester\Desktop\cabbage\github_repo\csv_file\42HA10-2021-AirTemperature-day.csv")
temperature_csv2022 = pd.read_csv(r"C:\Users\chester\Desktop\cabbage\github_repo\csv_file\42HA10-2022-AirTemperature-day.csv")
temperature_csv2023 = pd.read_csv(r"C:\Users\chester\Desktop\cabbage\github_repo\csv_file\42HA10-2023-AirTemperature-day.csv")
rain_dictionary = {}

def create_rain_dic(year,df_csv):
  for i in range(0,12):
    for j in range(0,31):
      month=str(i+1)
      day=str(j+1)
      if i+1<10:
        month='0'+month
      if j+1<10:
        day='0'+day
      if(df_csv.iloc[j,i+1]=='--'):
        rain_dictionary[str(year)+'/'+month+'/'+day]=0.0
      else:
        rain_dictionary[str(year)+'/'+month+'/'+day]=float(df_csv.iloc[j,i+1])

create_rain_dic(108,weather_csv2019)
create_rain_dic(109,weather_csv2020)
create_rain_dic(110,weather_csv2021)
create_rain_dic(111,weather_csv2022)
create_rain_dic(112,weather_csv2023)


temparature_dictionary = {}

def create_temparature_dic(year,df_csv):
  for i in range(0,12):
    for j in range(0,31):
      month=str(i+1)
      day=str(j+1)
      if i+1<10:
        month='0'+month
      if j+1<10:
        day='0'+day
      if(df_csv.iloc[j,i+1]=='--'):
        temparature_dictionary[str(year)+'/'+month+'/'+day]=df_csv.iloc[31,i+1]
      else:
        temparature_dictionary[str(year)+'/'+month+'/'+day]=float(df_csv.iloc[j,i+1])

create_temparature_dic(108,temperature_csv2019)
create_temparature_dic(109,temperature_csv2020)
create_temparature_dic(110,temperature_csv2021)
create_temparature_dic(111,temperature_csv2022)
create_temparature_dic(112,temperature_csv2023)



set1=set1.drop(set1.index[:4])
set1=set1.drop(set1.index[-1])
import numpy as np

data_value = set1.values
DATA=[]
for i in range (0,len(data_value)):
  if rain_dictionary[data_value[i][0]]!='--':
    temp=np.append(data_value[i],rain_dictionary[data_value[i][0]])
  else:
    temp=np.append(data_value[i],0)
  DATA.append(temp)
data_value=DATA

DATA=[]
for i in range (0,len(data_value)):
  temp=np.append(data_value[i],temparature_dictionary[data_value[i][0]])
  DATA.append(temp)
data_value=DATA

data=[]
for i in range(0,len(data_value)):
  temp=[data_value[i][6],data_value[i][11],data_value[i][12]]
  data.append(temp)

original_data=data #用來驗證答案

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)
  
X=[]
Y=[]
LENGTH=10
for i in range(0,len(data)-31):
  X.append(data[i:10+i])
  Y.append(data[i+11])
  
X=np.array(X)
Y=np.array(Y)

X= X.astype(float)
Y= Y.astype(float)



X_train=X[290:837+290]
X_val=X[837+290:837+580]
X_test=X[0:290]

Y_train=Y[290:837+290]
Y_val=Y[837+290:837+580]
Y_test=Y[0:290]


import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from tensorflow.keras import layers
from keras.utils import to_categorical

model = Sequential()
model.add(layers.LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(units=32, return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(units=32))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(3, activation="linear"))
optimizer = Adam(learning_rate=1e-05)

model.compile(loss='mse', optimizer=optimizer)


model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=300, batch_size=32)


predictions = model.predict(X_test)
predictions=scaler.inverse_transform(predictions)
test_MSE=0
for i in range(0,len(Y_test)):
    print("pre:",predictions[i][0],"origin:",original_data[i][0])
    test_MSE=test_MSE+(predictions[i][0]-original_data[i][0])*(predictions[i][0]-original_data[i][0])

test_MSE=test_MSE/len(Y_test)    
ori=[]
pre=[]
for i in range(len(Y_test)):
    ori.append(original_data[i][0])
    pre.append(predictions[i][0])
print("test_MSE:",test_MSE)
    
import matplotlib.pyplot as plt
plt.plot(ori, label='TEST')
plt.plot(pre, label='PREDICTION')
plt.legend()
plt.show()
