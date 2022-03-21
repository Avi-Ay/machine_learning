# -*- coding: utf-8 -*-
"""Linear_Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/180tLfdDA9F4d8qzcIggDPQgiWFotfsZE
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

elections = pd.read_csv('/content/dataB.csv')
election = elections.iloc[1:,1:]
cities = pd.read_csv('/content/D.csv')
#df2 = pd.read_csv('/content/dataB.csv')
features = pd.read_csv('/content/dataC.csv')
names = features[["yshuv","shem_yshuv"]]
feature = features[["yshuv","Totalpop16","AhuzOlim1990","Bagrut_1516","Eshkol_15","Shetah_shiput_13","DiabRate14_16"]]

city = cities.iloc[:,1:3]
city.head()

city.columns

# full_data = df.fillna({
#     'Outlook' : df['Outlook'].mode().loc[0],
#     'Tempreture' : df['Tempreture'].mean(),
#     'Humidity' : df['Humidity'].mean(),
#     'Wind' : df['Wind'].mode().loc[0]
# })
# full_data

feature.columns

election.columns

city.rename(columns={'סמל יישוב': 'yshuv_symbol'}, inplace=True)
city.rename(columns={'סך אוכלוסייה ': 'Totalpop'}, inplace=True)

feature.rename(columns={'yshuv': 'yshuv_symbol'}, inplace=True)

election.rename(columns={'To': 'to'}, inplace=True)
election.rename(columns={'To.1': 'To'}, inplace=True)

election.rename(columns={'Symbol of a settlement': 'yshuv_symbol'}, inplace=True)

full_feature = pd.merge(city,feature, on ='yshuv_symbol',how='left')
del full_feature['Totalpop16']
full_feature.columns

feature.head()

data = pd.merge(feature,election.iloc[:,:5], on ='yshuv_symbol',how='left')

data['The Arabs Bloc'] = election['And the same'] 
data['The_Left_Bloc'] = election['Truth'] + election['Meretz'] + data['The Arabs Bloc']
data['The_Center_Bloc'] = election['Here'] 
data['The Ultra-Orthodox Bloc'] = election['third'] + election['Shot']
data['The_Right_Bloc'] =  election['To'] + election['Tab'] + election['about'] + election['forgave'] + data['The Ultra-Orthodox Bloc']

data.columns

f,ax = plt.subplots(1,1,figsize=(15,7))
ax.plot(data['yshuv_symbol'],data['The_Left_Bloc'],'.r')
ax.plot(data['yshuv_symbol'],data['The_Center_Bloc'],'.y')
ax.plot(data['yshuv_symbol'],data['The_Right_Bloc'],'.b')
ax.set_xlabel("Yshuv Symbol")
ax.set_ylabel("Kosher Votes")
ax.set_title('The 20th Election Map')

data.isna().sum() #count the number of NAN in each column

data.head()

# for i in range(1,1220):
#   if feature['yshuv_symbol'] == i :
#     feature['Totalpop16'] =

# election['Totalpop'] = 0
# value = city[city['yshuv_symbol'] == 175]]
# election.loc[election['yshuv_symbol'] == 175,'Totalpop'] = value

feature.loc[feature['yshuv_symbol']==473]

city.loc[city['yshuv_symbol']==473]

data.columns

data.head()

data.iloc[:,:6]

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(data.iloc[:,6:])
data.iloc[:,6:] = imputer.transform(data.iloc[:,6:])

f,ax = plt.subplots(1,1,figsize=(15,7))
ax.plot(data['Bagrut_1516'],data['The_Left_Bloc'],'*r')
# ax.plot(data['Bagrut_1516'],data['The_Center_Bloc'],'.y')
# plt.plot(kind = 'scatter', x= data['Bagrut_1516'] , y = data['The_Right_Bloc'])
# plt.show()
ax.set_xlabel("Yshuv Symbol")
ax.set_ylabel("Kosher Votes")
ax.set_title('The 20th Election Bagrut Map')

plt.plot(data['yshuv_symbol'],data['The_Left_Bloc'],'.r')
plt.plot(data['yshuv_symbol'],data['The_Right_Bloc'],'.b')
plt.plot(data['yshuv_symbol'],data['The_Center_Bloc'],'.y')
plt.xlabel("Yshuv Symbol")
plt.ylabel("Kosher Votes")

plt.plot(data['yshuv_symbol'],data['Kosher'])
plt.plot(data['yshuv_symbol'],data['Kosher'],'.r')
plt.xlabel("Yshuv Symbol")
plt.ylabel("Kosher Votes")

data[['yshuv_symbol']].boxplot()

data[['The_Right_Bloc']].boxplot()

scaler =  MinMaxScaler()
data_scaled = scaler.fit(data)
data.describe()

data.describe()

data_scaled