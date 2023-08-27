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


df_tot_tot=pd.read_csv('data.csv',index_col=0)
bench1_tot = pd.read_csv('bench1.csv',index_col=0)
bench2_tot = pd.read_csv('bench2.csv',index_col=0)

l_sf=[]
l_b=[]
l_b2=[]
for h in [5]:
    for min_d in [0.1,0.5]:
        for min_d_e in [0.1]:
            
            tot_df_sf=pd.DataFrame()
            tot_df_ar=pd.DataFrame()
            tot_df_ar2=pd.DataFrame()
            tot_df_obs=pd.DataFrame()
            for num in range(4):
                
                if num == 0:
                    df_tot_m=df_tot_tot
                    bench1=bench1_tot
                    bench2=bench2_tot
                else:
                    df_tot_m=df_tot_tot.iloc[:(-12*num),:]
                    bench1=bench1_tot.iloc[:(-12*num),:]
                    bench2=bench2_tot.iloc[:(-12*num),:]
                
            # =============================================================================
            # Forecast : ShapeFinder vs Bencharks
            # =============================================================================
            
            ############## Conflict
                
                train_df = df_tot_m.iloc[:-15-h,:]
                test_df = df_tot_m.iloc[-15-h:,:]
                
                dtw_sel=2
                df_sf_tot=pd.DataFrame()
                df_ar=pd.DataFrame()
                df_ar2=pd.DataFrame()
                df_obs=pd.DataFrame()
                for row in range(len(test_df.columns)):
                    ts=test_df.iloc[:,row]
                    df_sf=pd.DataFrame()
                    seq=ts.iloc[:h]
                    for i in range(math.ceil(15/h)):
                        min_d_2=min_d
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
                            pred = model.predict(horizon=h,plot=False,metric='dtw',min_d=min_d_2,dtw_sel=dtw_sel,select=True)
                            while pred is None:
                                min_d_2=min_d_2+min_d_e
                                pred = model.predict(horizon=h,plot=False,metric='dtw',min_d=min_d_2,dtw_sel=dtw_sel,select=True)
                            pred=fitted_scaler.inverse_transform(pred)
                            pred[pred<0]=0
                            pred=pd.DataFrame(pred)
                        else:
                            pred=pd.concat([seq,seq,seq],axis=1)
                            pred.columns=[0,1,2]
                            
                        df_sf=pd.concat([df_sf,pred],axis=0)
                        seq=pred.iloc[:,0]
                    
                    df_sf=df_sf.iloc[3:15,:]
                    df_sf=df_sf.reset_index(drop=True)
                    
                    df_sf_tot = pd.concat([df_sf_tot,df_sf.iloc[:,0]],axis=1)
                    ### Bench 
                    ben1=bench1.iloc[-12:,row]
                    ben2=bench2.iloc[-12:,row]
                    obs = ts.iloc[-12:]
                     
                    ben1=ben1.reset_index(drop=True) 
                    ben2=ben2.reset_index(drop=True) 
                    obs=obs.reset_index(drop=True) 
                    
                    df_ar=pd.concat([df_ar,ben1],axis=1)
                    df_ar2=pd.concat([df_ar2,ben2],axis=1)
                    df_obs=pd.concat([df_obs,obs],axis=1)
                     
                    # if (obs==0).all()==False:
                    #     plt.figure(figsize=(10, 6))
                    #     plt.plot(df_sf_tot.iloc[:,-1], label='ShapeFinder', marker='o',color='r')
                    #     plt.plot(df_ar.iloc[:,-1], label='Bench1', marker='o',color='g')
                    #     plt.plot(df_ar2.iloc[:,-1], label='Bench2', marker='o',color='b')
                    #     plt.plot(obs,label='Obs',marker='o',linewidth=5)
                    #     plt.legend()
                    #     plt.grid(True)
                    #     plt.title(test_df.columns[row])
                    #     plt.show()
                    
                tot_df_sf=pd.concat([tot_df_sf,df_sf_tot],axis=0)
                tot_df_ar=pd.concat([tot_df_ar,df_ar],axis=0)
                tot_df_ar2=pd.concat([tot_df_ar2,df_ar2],axis=0)
                tot_df_obs=pd.concat([tot_df_obs,df_obs],axis=0)
            
            tot_df_sf.columns = tot_df_obs.columns
            tot_df_ar.columns = tot_df_obs.columns
            tot_df_ar2.columns = tot_df_obs.columns
            
            err_sf=[]
            err_b=[]
            err_b2=[]
            for i in range(len(tot_df_sf.columns)):
                for y in range(4):
                    err_sf.append(mean_squared_error(tot_df_obs.iloc[y*12:(y+1)*12,i], tot_df_sf.iloc[y*12:(y+1)*12,i]))
                    err_b.append(mean_squared_error(tot_df_obs.iloc[y*12:(y+1)*12,i], tot_df_ar.iloc[y*12:(y+1)*12,i]))
                    err_b2.append(mean_squared_error(tot_df_obs.iloc[y*12:(y+1)*12,i], tot_df_ar2.iloc[y*12:(y+1)*12,i]))
                err_sf.append(mean_squared_error(tot_df_obs.iloc[:,i], tot_df_sf.iloc[:,i]))
                err_b.append(mean_squared_error(tot_df_obs.iloc[:,i], tot_df_ar.iloc[:,i]))
                err_b2.append(mean_squared_error(tot_df_obs.iloc[:,i], tot_df_ar2.iloc[:,i]))
            
            err_sf = np.array(err_sf).reshape((5,191),order='F')
            err_b = np.array(err_b).reshape((5,191),order='F')
            err_b2 = np.array(err_b2).reshape((5,191),order='F')
            
            x_labels = ['ShapeFinder', 'Bench1', 'Bench2']
            means = [err_sf[4, :].mean(), err_b[4, :].mean(), err_b2[4, :].mean()]
            plt.bar(x_labels, means)
            plt.xlabel('Models')
            plt.ylabel('MSE') 
            plt.title('Mean Squared Error Comparison')
            plt.show()
            
            l_sf.append(err_sf)
            l_b.append(err_b)
            l_b2.append(err_b2)
     

# =============================================================================
# Forecast : LSTM vs Bencharks
# =============================================================================

tot_df_nn=pd.DataFrame()
tot_df_ar=pd.DataFrame()
tot_df_ar2=pd.DataFrame()
tot_df_obs=pd.DataFrame()
for num in range(4):
    if num == 0:
        df_tot_m=df_tot_tot
        bench1=bench1_tot
        bench2=bench2_tot
    else:
        df_tot_m=df_tot_tot.iloc[:(-12*num),:]
        bench1=bench1_tot.iloc[:(-12*num),:]
        bench2=bench2_tot.iloc[:(-12*num),:]
    np.random.seed(0)
    keras.utils.set_random_seed(0)
    
    df_nn=pd.DataFrame()
    df_ar=pd.DataFrame()
    df_ar2=pd.DataFrame()
    df_obs=pd.DataFrame()
    
    train_df = df_tot_m.iloc[:-15,:]
    scaler = MinMaxScaler((0,1))
    train_df = scaler.fit_transform(train_df)
    train_df=pd.DataFrame(train_df)
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
    model.fit(x_train, y_train,validation_split=0.3,callbacks=[es], epochs=n_epochs, batch_size=32, verbose=1)

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
            pred=pd.Series(pred)
            pred=pred*df_tot_m.iloc[:-15,row].max()+df_tot_m.iloc[:-15,row].min()
        else:
            pred=pd.Series([0]*15)
        pred=pred.iloc[-12:]    
        pred=pred.reset_index(drop=True)
        df_nn=pd.concat([df_nn,pred],axis=1)
    
        ben1=bench1.iloc[-12:,row]
        ben2=bench2.iloc[-12:,row]
        obs = df_tot_m.iloc[-12:,row]
        ben1=ben1.reset_index(drop=True)
        ben2=ben2.reset_index(drop=True)
        obs=obs.reset_index(drop=True)
        
        df_ar=pd.concat([df_ar,ben1],axis=1)
        df_ar2=pd.concat([df_ar2,ben2],axis=1)
        df_obs=pd.concat([df_obs,obs],axis=1)
    
        df_nn=df_nn.reset_index(drop=True)   
        df_ar=df_ar.reset_index(drop=True) 
        df_ar2=df_ar2.reset_index(drop=True) 
        df_obs=df_obs.reset_index(drop=True) 
        
        if (obs==0).all()==False:
            plt.figure(figsize=(10, 6))
            plt.plot(df_nn.iloc[:,-1], label='LSTM', marker='o',color='r')
            plt.plot(df_ar.iloc[:,-1], label='Bench1', marker='o',color='g')
            plt.plot(df_ar2.iloc[:,-1], label='Bench2', marker='o',color='b')
            plt.plot(obs,label='Obs',marker='o',linewidth=5)
            plt.legend()
            plt.grid(True)
            plt.title(test_df.columns[row])
            plt.show()
        
    tot_df_nn=pd.concat([tot_df_nn,df_nn],axis=0)
    tot_df_ar=pd.concat([tot_df_ar,df_ar],axis=0)
    tot_df_ar2=pd.concat([tot_df_ar2,df_ar2],axis=0)
    tot_df_obs=pd.concat([tot_df_obs,df_obs],axis=0)

tot_df_nn.columns = tot_df_obs.columns
tot_df_ar.columns = tot_df_obs.columns
tot_df_ar2.columns = tot_df_obs.columns

err_nn=[]
err_b=[]
err_b2=[]
for i in range(len(tot_df_nn.columns)):
    for y in range(4):
        err_nn.append(mean_squared_error(tot_df_obs.iloc[y*12:(y+1)*12,i], tot_df_nn.iloc[y*12:(y+1)*12,i]))
        err_b.append(mean_squared_error(tot_df_obs.iloc[y*12:(y+1)*12,i], tot_df_ar.iloc[y*12:(y+1)*12,i]))
        err_b2.append(mean_squared_error(tot_df_obs.iloc[y*12:(y+1)*12,i], tot_df_ar2.iloc[y*12:(y+1)*12,i]))
    err_nn.append(mean_squared_error(tot_df_obs.iloc[:,i], tot_df_nn.iloc[:,i]))
    err_b.append(mean_squared_error(tot_df_obs.iloc[:,i], tot_df_ar.iloc[:,i]))
    err_b2.append(mean_squared_error(tot_df_obs.iloc[:,i], tot_df_ar2.iloc[:,i]))

err_nn = np.array(err_nn).reshape((5,191),order='F')
err_b = np.array(err_b).reshape((5,191),order='F')
err_b2 = np.array(err_b2).reshape((5,191),order='F')

x_labels = ['LSTM', 'Bench1', 'Bench2']
means = [err_nn[4, :].mean(), err_b[4, :].mean(), err_b2[4, :].mean()]
plt.bar(x_labels, means)
plt.xlabel('Models')
plt.ylabel('MSE')  # You might want to add a y-label too
plt.title('Mean Squared Error Comparison')  # You can add a title if needed
plt.show()

# =============================================================================
# Forecast : ARIMA vs Bencharks
# =============================================================================
m_ar=[]
for num in range(4):
    if num == 0:
        df_tot_m=df_tot_tot
        bench1=bench1_tot
        bench2=bench2_tot
    else:
        df_tot_m=df_tot_tot.iloc[:(-12*num),:]
        bench1=bench1_tot.iloc[:(-12*num),:]
        bench2=bench2_tot.iloc[:(-12*num),:]
        
    train_df = df_tot_m.iloc[:-15,:]
    tot_df_ar=pd.DataFrame()
    for row in range(len(train_df.columns)):
        if (train_df.iloc[-15:,row]==0).all()==False:
            arima = auto_arima(train_df.iloc[:,row].dropna())
            pred_ar=arima.predict(15)
            df_ar=pd.Series(pred_ar)
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
    m_ar.append(mean_squared_error(tot_df_obs,tot_df_ar.iloc[:,0]))







