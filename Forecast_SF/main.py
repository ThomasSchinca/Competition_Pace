# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 13:48:45 2023

@author: thoma
"""

import pandas as pd 
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
np.random.seed(1)
from shapefinder import Shape,finder
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
import math 
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from pmdarima.arima import auto_arima

# =============================================================================
# Forecast : ShapeFinder vs Bencharks
# =============================================================================

############## Conflict

df_tot_m=pd.read_csv('data.csv',index_col=0)
bench1 = pd.read_csv('bench1.csv',index_col=0)
bench2 = pd.read_csv('bench2.csv',index_col=0)
sf_err=[]
m_sf=[]
for h in [5,6,7,8,9,10,12,15]:
    
    
    train_df = df_tot_m.iloc[:-15-h,:]
    test_df = df_tot_m.iloc[-15-h:,:]
    
    dtw_sel=2
    tot_df_sf=pd.DataFrame()
    tot_df_ar=pd.DataFrame()
    tot_df_ar2=pd.DataFrame()
    tot_df_obs=pd.DataFrame()
    for row in range(len(test_df.columns)):
        ts=test_df.iloc[:,row]
        df_sf=pd.DataFrame()
        df_ar=pd.DataFrame()
        df_ar2=pd.DataFrame()
        df_obs=pd.DataFrame()
        seq=ts.iloc[:h]
        for i in range(math.ceil(15/h)):
            if (seq==0).all()==False:
                ### Shape 
                sh = Shape()
                sh.set_shape(np.array(seq))
                scaler = MinMaxScaler((0,1))
                df_scaler=pd.concat([seq,seq,seq],axis=1)
                df_scaler.index=range(len(df_scaler))
                df_scaler.columns = ['Prediction', 'CI lower', 'CI upper']
                fitted_scaler = scaler.fit(df_scaler)
                model = finder(train_df,Shape=sh)
                min_d=0.3
                pred = model.predict(horizon=h,plot=False,metric='dtw',min_d=min_d,dtw_sel=dtw_sel,select=True)
                while pred is None:
                    min_d=min_d+0.2
                    pred = model.predict(horizon=h,plot=False,metric='dtw',min_d=min_d,dtw_sel=dtw_sel,select=True)
                pred=fitted_scaler.inverse_transform(pred)
                pred=pd.DataFrame(pred)
            else:
                pred=pd.concat([ts.iloc[i*h:(i+1)*h],ts.iloc[i*h:(i+1)*h],ts.iloc[i*h:(i+1)*h]],axis=1)
                pred.columns=[0,1,2]
            
            seq=pred.iloc[:,0]
            df_sf=pd.concat([df_sf,pred],axis=0)
        df_sf=df_sf.iloc[3:15,:]
        ### Bench 
        ben1=bench1.iloc[-12:,row]
        ben2=bench2.iloc[-12:,row]
        
        obs = ts.iloc[-12:]
        
        df_ar=pd.concat([df_ar,ben1],axis=0)
        df_ar2=pd.concat([df_ar2,ben2],axis=0)
        df_obs=pd.concat([df_obs,obs],axis=0)
    
        df_sf=df_sf.reset_index(drop=True)   
        df_ar=df_ar.reset_index(drop=True) 
        df_ar2=df_ar2.reset_index(drop=True) 
        df_obs=df_obs.reset_index(drop=True)  
        if (ts.iloc[-12:]==0).all()==False:
            plt.figure(figsize=(10, 6))
            plt.plot(df_sf.iloc[:,0], label='ShapeFinder', marker='o',color='r')
            plt.fill_between(range(len(df_sf)), df_sf.iloc[:,1], df_sf.iloc[:,2],color='r', alpha=0.1)
            plt.plot(df_ar.iloc[:,0], label='Bench1', marker='o',color='g')
            plt.plot(df_ar2.iloc[:,0], label='Bench2', marker='o',color='b')
            plt.plot(df_obs.iloc[:,0],label='Obs',marker='o',linewidth=5)
            plt.legend()
            plt.grid(True)
            plt.title(test_df.columns[row])
            plt.show()
        
        tot_df_sf=pd.concat([tot_df_sf,df_sf],axis=0)
        tot_df_ar=pd.concat([tot_df_ar,df_ar],axis=0)
        tot_df_ar2=pd.concat([tot_df_ar2,df_ar2],axis=0)
        tot_df_obs=pd.concat([tot_df_obs,df_obs],axis=0)
    
    m_bench = mean_squared_error(tot_df_obs,tot_df_ar)
    m_bench2 = mean_squared_error(tot_df_obs,tot_df_ar2)
    m_sf.append(mean_squared_error(tot_df_obs,tot_df_sf.iloc[:,0]))
    
    
    data = tot_df_sf.iloc[:,0].tolist()
    num_columns = len(data) // 12 
    new_columns = [f'Column_{i+1}' for i in range(num_columns)]
    df_sf = pd.DataFrame(columns=new_columns)
    for i, column in enumerate(new_columns):
        start_idx = i * 12
        end_idx = (i + 1) * 12
        df_sf[column] = data[start_idx:end_idx]
    df_sf.columns=df_tot_m.columns
    df_err = abs(df_sf-df_tot_m.iloc[-12:,:].reset_index(drop=True))
    sf_err.append(df_err.mean(axis=1))

# =============================================================================
# Forecast : LSTM vs Bencharks
# =============================================================================

tf.random.set_seed(0)
df_tot_m=pd.read_csv('data.csv',index_col=0)
train_df = df_tot_m.iloc[:-15,:]
scaler = MinMaxScaler((0,1))
train_df = scaler.fit_transform(train_df)
train_df=pd.DataFrame(train_df)
m_lstm=[]
ar=10
t_d=pd.DataFrame()
for i in range(len(train_df.columns)):
    ser= train_df.iloc[:,i]
    df_ser=pd.DataFrame(ser)
    for aut in [*range(1,ar+1)]: 
        df_ser = pd.concat([ser.shift(aut),df_ser],axis=1)
    df_ser=df_ser.dropna()   
    df_ser=df_ser.reset_index(drop=True)
    df_ser.columns=[*range(len(df_ser.columns))]
    t_d= pd.concat([t_d,df_ser],axis=0)
t_d=np.array(t_d) 
nonzero_rows = np.any(t_d != 0, axis=1)
t_d = t_d[nonzero_rows]

x_train = t_d[:,:-1]
y_train = t_d[:,-1]
x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1]))

config = {'activation':['relu'],
          'n_layer' : [1],
          'drop' : [0.1,0.5],
          'l_r' : [0.001,0.0001]}

model = keras.Sequential()
nb_hidden = int(len(t_d[:,1])/(10*len(t_d[1,:])))
model.add(LSTM(nb_hidden, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(nb_hidden, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
n_epochs = 500
es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=30)
model.fit(x_train, y_train,validation_split=0.3,callbacks=[es], epochs=n_epochs, batch_size=32, verbose=1, shuffle=True)

tot_df_nn=pd.DataFrame()
tot_df_ar=pd.DataFrame()
tot_df_ar2=pd.DataFrame()
tot_df_obs=pd.DataFrame()
for row in range(len(train_df.columns)):
    if (train_df.iloc[-ar:,row]==0).all()==False:
        x_test = np.array(train_df.iloc[-ar:,row])
        x_test = x_test.reshape((1,1,x_test.shape[0]))
        pred=[]
        for pr in range(15):
            p=model.predict(x_test,verbose=0)[0]
            pred.append(p[0])
            x_test=np.concatenate([x_test[0][0][1:],np.array(p)])
            x_test = x_test.reshape((1,1,x_test.shape[0]))
        df_nn=pd.Series(pred)
        df_nn=df_nn*df_tot_m.iloc[:-15,row].max()+df_tot_m.iloc[:-15,row].min()
    else:
        df_nn=pd.Series([0]*15)
    df_nn=df_nn.iloc[-12:]    
    
    ### Bench 
    ben1=bench1.iloc[-12:,row]
    ben2=bench2.iloc[-12:,row]
    
    obs = df_tot_m.iloc[-12:,row]

    df_nn=df_nn.reset_index(drop=True)   
    df_ar=ben1.reset_index(drop=True) 
    df_ar2=ben2.reset_index(drop=True) 
    obs=obs.reset_index(drop=True)  
    if (obs==0).all()==False:
        plt.figure(figsize=(10, 6))
        plt.plot(df_nn, label='LSTM', marker='o',color='r')
        plt.plot(df_ar, label='Bench1', marker='o',color='g')
        plt.plot(df_ar2, label='Bench2', marker='o',color='b')
        plt.plot(obs,label='Obs',marker='o',linewidth=5)
        plt.legend()
        plt.grid(True)
        plt.title(test_df.columns[row])
        plt.show()
    
    tot_df_nn=pd.concat([tot_df_nn,df_nn],axis=0)
    tot_df_ar=pd.concat([tot_df_ar,df_ar],axis=0)
    tot_df_ar2=pd.concat([tot_df_ar2,df_ar2],axis=0)
    tot_df_obs=pd.concat([tot_df_obs,obs],axis=0)
    
    
m_bench = mean_squared_error(tot_df_obs,tot_df_ar)
m_bench2 = mean_squared_error(tot_df_obs,tot_df_ar2)
m_lstm=mean_squared_error(tot_df_obs,tot_df_nn.iloc[:,0])

# =============================================================================
# Forecast : ARIMA vs Bencharks
# =============================================================================


df_tot_m=pd.read_csv('data.csv',index_col=0)
train_df = df_tot_m.iloc[:-15,:]
tot_df_ar=pd.DataFrame()
tot_df_ar1=pd.DataFrame()
tot_df_ar2=pd.DataFrame()
tot_df_obs=pd.DataFrame()
for row in range(len(train_df.columns)):
    if (train_df.iloc[-15:,row]==0).all()==False:
        arima = auto_arima(train_df.iloc[:,row].dropna())
        pred_ar=arima.predict(15)
        df_ar=pd.Series(pred)
        df_ar=df_ar*df_tot_m.iloc[:-15,row].max()+df_tot_m.iloc[:-15,row].min()
    else:
        df_ar=pd.Series([0]*15)
    df_ar=df_ar.iloc[-12:]   
    ### Bench 
    ben1=bench1.iloc[-12:,row]
    ben2=bench2.iloc[-12:,row]
    
    obs = df_tot_m.iloc[-12:,row]

    df_ar=df_ar.reset_index(drop=True)   
    df_ar1=ben1.reset_index(drop=True) 
    df_ar2=ben2.reset_index(drop=True) 
    obs=obs.reset_index(drop=True)  
    if (obs==0).all()==False:
        plt.figure(figsize=(10, 6))
        plt.plot(df_ar, label='ARIMA', marker='o',color='r')
        plt.plot(df_ar1, label='Bench1', marker='o',color='g')
        plt.plot(df_ar2, label='Bench2', marker='o',color='b')
        plt.plot(obs,label='Obs',marker='o',linewidth=5)
        plt.legend()
        plt.grid(True)
        plt.title(test_df.columns[row])
        plt.show()
    
    tot_df_ar=pd.concat([tot_df_ar,df_ar],axis=0)
    tot_df_ar1=pd.concat([tot_df_ar1,df_ar1],axis=0)
    tot_df_ar2=pd.concat([tot_df_ar2,df_ar2],axis=0)
    tot_df_obs=pd.concat([tot_df_obs,obs],axis=0)
    
    
m_bench = mean_squared_error(tot_df_obs,tot_df_ar1)
m_bench2 = mean_squared_error(tot_df_obs,tot_df_ar2)
m_ar=mean_squared_error(tot_df_obs,tot_df_ar.iloc[:,0])







