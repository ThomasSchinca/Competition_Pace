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
from dtaidistance import dtw
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from pmdarima.arima import auto_arima
from scipy.stats import ttest_1samp

df_tot_tot=pd.read_csv('data.csv',index_col=0)
bench1_tot = pd.read_csv('bench1.csv',index_col=0)
bench2_tot = pd.read_csv('bench2.csv',index_col=0)

l_sf=[]
l_b=[]
l_b2=[]
l_sf_d=[]
l_b_d=[]
l_b2_d=[]
df_fin=[]
df_fin2=[]
df_fin3=[]
df_fin4=[]
df_fin5=[]
df_fin6=[]
df_fin7=[]
df_fin8=[]
h=15
for min_d in [0.3]:
    for min_d_e in [0.1]:
        tot_df_sf=pd.DataFrame()
        tot_df_ar=pd.DataFrame()
        tot_df_ar2=pd.DataFrame()
        tot_df_obs=pd.DataFrame()
        tot_df_sf_diff=pd.DataFrame()
        tot_df_ar_diff=pd.DataFrame()
        tot_df_ar2_diff=pd.DataFrame()
        tot_df_obs_diff=pd.DataFrame()
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
            df_sf_tot=pd.DataFrame()
            df_ar_d=pd.DataFrame()
            df_ar2_d=pd.DataFrame()
            df_obs_d=pd.DataFrame()
            df_sf_diff_tot=pd.DataFrame()
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
                
                df_sf_diff=df_sf.iloc[2:15,:]
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
                 
                if (obs==0).all()==False:
                    plt.figure(figsize=(10, 6))
                    plt.plot(df_sf_tot.iloc[:,-1], label='ShapeFinder', marker='o',color='r')
                    plt.plot(df_ar.iloc[:,-1], label='Bench1', marker='o',color='g')
                    plt.plot(df_ar2.iloc[:,-1], label='Bench2', marker='o',color='b')
                    plt.plot(obs,label='Obs',marker='o',linewidth=5)
                    plt.legend()
                    plt.grid(True)
                    plt.title(test_df.columns[row])
                    plt.show()
                
                
                df_sf_diff=df_sf_diff.reset_index(drop=True)
                df_sf_diff=df_sf_diff.diff()
                df_sf_diff=df_sf_diff.iloc[1:,:]
                df_sf_diff_tot = pd.concat([df_sf_diff_tot,df_sf_diff.iloc[:,0]],axis=1)
                
                ### Bench 
                ben1_diff=bench1.iloc[-13:,row]
                ben1_diff=ben1_diff.diff()
                ben1_diff=ben1_diff.iloc[1:]
                ben2_diff=bench2.iloc[-13:,row]
                ben2_diff=ben2_diff.diff()
                ben2_diff=ben2_diff.iloc[1:]
                obs_diff = ts.iloc[-13:]
                obs_diff=obs_diff.diff()
                obs_diff=obs_diff.iloc[1:]
                 
                ben1_diff=ben1_diff.reset_index(drop=True) 
                ben2_diff=ben2_diff.reset_index(drop=True) 
                obs_diff=obs_diff.reset_index(drop=True) 
                ben1=ben1.reset_index(drop=True) 
                ben2=ben2.reset_index(drop=True) 
                obs=obs.reset_index(drop=True) 
                
                df_ar_d=pd.concat([df_ar_d,ben1_diff],axis=1)
                df_ar2_d=pd.concat([df_ar2_d,ben2_diff],axis=1)
                df_obs_d=pd.concat([df_obs_d,obs_diff],axis=1)
                
                
                
            tot_df_sf=pd.concat([tot_df_sf,df_sf_tot],axis=0)
            tot_df_ar=pd.concat([tot_df_ar,df_ar],axis=0)
            tot_df_ar2=pd.concat([tot_df_ar2,df_ar2],axis=0)
            tot_df_obs=pd.concat([tot_df_obs,df_obs],axis=0)
            tot_df_sf_diff=pd.concat([tot_df_sf_diff,df_sf_diff_tot],axis=0)
            tot_df_ar_diff=pd.concat([tot_df_ar_diff,df_ar_d],axis=0)
            tot_df_ar2_diff=pd.concat([tot_df_ar2_diff,df_ar2_d],axis=0)
            tot_df_obs_diff=pd.concat([tot_df_obs_diff,df_obs_d],axis=0)
        
        tot_df_sf.columns = tot_df_obs.columns
        tot_df_ar.columns = tot_df_obs.columns
        tot_df_ar2.columns = tot_df_obs.columns
        tot_df_sf_diff.columns = tot_df_obs_diff.columns
        tot_df_ar_diff.columns = tot_df_obs_diff.columns
        tot_df_ar2_diff.columns = tot_df_obs_diff.columns
        
        
        df_fin.append(tot_df_sf)
        df_fin2.append(tot_df_ar)
        df_fin3.append(tot_df_ar2)
        df_fin4.append(tot_df_obs)
        df_fin5.append(tot_df_sf_diff)
        df_fin6.append(tot_df_ar_diff)
        df_fin7.append(tot_df_ar2_diff)
        df_fin8.append(tot_df_obs_diff)
        
        
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
        
        tot_df_sf_diff.index = tot_df_sf.index
        err_sf_d=[]
        err_b_d=[]
        err_b2_d=[]
        for i in range(len(tot_df_sf_diff.columns)):
            for y in range(4):
                if (tot_df_obs_diff.iloc[y*12:(y+1)*12,i]==0).all() and (tot_df_sf_diff.iloc[y*12:(y+1)*12,i]==0).all():
                    err_sf_d.append(0.0)
                else :
                    err_sf_d.append(dtw.distance(tot_df_obs_diff.iloc[y*12:(y+1)*12,i], tot_df_sf_diff.iloc[y*12:(y+1)*12,i]))
                if (tot_df_obs_diff.iloc[y*12:(y+1)*12,i]==0).all() and  (tot_df_ar_diff.iloc[y*12:(y+1)*12,i]==0).all():    
                    err_b_d.append(0.0)
                else:
                    err_b_d.append(dtw.distance(tot_df_obs_diff.iloc[y*12:(y+1)*12,i], tot_df_ar_diff.iloc[y*12:(y+1)*12,i]))
                if (tot_df_obs_diff.iloc[y*12:(y+1)*12,i]==0).all() and  (tot_df_ar2_diff.iloc[y*12:(y+1)*12,i]==0).all():    
                    err_b2_d.append(0.0)
                else:
                    err_b2_d.append(dtw.distance(tot_df_obs_diff.iloc[y*12:(y+1)*12,i], tot_df_ar2_diff.iloc[y*12:(y+1)*12,i]))

        err_sf_d = np.array(err_sf_d).reshape((4,191),order='F')
        err_b_d = np.array(err_b_d).reshape((4,191),order='F')
        err_b2_d = np.array(err_b2_d).reshape((4,191),order='F')
        
        x_labels = ['ShapeFinder', 'Bench1', 'Bench2']
        means = [err_sf_d.mean(), err_b_d.mean(), err_b2_d.mean()]
        plt.bar(x_labels, means)
        plt.xlabel('Models')
        plt.ylabel('DTW') 
        plt.title('DTW of diff')
        plt.show()
        
        l_sf_d.append(err_sf_d)
        l_b_d.append(err_b_d)
        l_b2_d.append(err_b2_d)
     

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



# =============================================================================
# Normal
# =============================================================================

# for k in range(len(df_fin)):
#     tot_df_nn=df_fin[k]
#     tot_df_ar=df_fin2[0]
#     tot_df_ar2=df_fin3[0]
#     tot_df_obs=df_fin4[0]
#     err_nn=[]
#     err_b=[]
#     err_b2=[]
#     for i in range(12):
#         err_nn.append(mean_squared_error(tot_df_obs.loc[i].to_numpy().flatten(), tot_df_nn.loc[i].to_numpy().flatten()))
#         err_b.append(mean_squared_error(tot_df_obs.loc[i].to_numpy().flatten(), tot_df_ar.loc[i].to_numpy().flatten()))
#         err_b2.append(mean_squared_error(tot_df_obs.loc[i].to_numpy().flatten(), tot_df_ar2.loc[i].to_numpy().flatten()))
#     plt.plot(err_nn,label=k)
# plt.plot(err_b,label='ar1')    
# plt.plot(err_b2,label='ar2')   
# plt.legend()
# plt.show()



# tot_df_nn=df_fin[0]
# for i in range(3,8):
#     tot_df_nn.loc[i]=df_fin[1].loc[i]
# for i in range(5,12):    
#     tot_df_nn.loc[i]=df_fin[2].loc[i]    

# tot_df_ar=df_fin2[0]
# tot_df_ar2=df_fin3[0]
# tot_df_obs=df_fin4[0]
# err_nn=[]
# err_b=[]
# err_b2=[]
# for i in range(len(tot_df_nn.columns)):
#     for y in range(4):
#         err_nn.append(mean_squared_error(tot_df_obs.iloc[y*12:(y+1)*12,i], tot_df_nn.iloc[y*12:(y+1)*12,i]))
#         err_b.append(mean_squared_error(tot_df_obs.iloc[y*12:(y+1)*12,i], tot_df_ar.iloc[y*12:(y+1)*12,i]))
#         err_b2.append(mean_squared_error(tot_df_obs.iloc[y*12:(y+1)*12,i], tot_df_ar2.iloc[y*12:(y+1)*12,i]))
#     err_nn.append(mean_squared_error(tot_df_obs.iloc[:,i], tot_df_nn.iloc[:,i]))
#     err_b.append(mean_squared_error(tot_df_obs.iloc[:,i], tot_df_ar.iloc[:,i]))
#     err_b2.append(mean_squared_error(tot_df_obs.iloc[:,i], tot_df_ar2.iloc[:,i]))

# err_nn = np.array(err_nn).reshape((5,191),order='F')
# err_b = np.array(err_b).reshape((5,191),order='F')
# err_b2 = np.array(err_b2).reshape((5,191),order='F')

# x_labels = ['ShapeFinder', 'Bench1', 'Bench2']
# means = [err_nn[4, :].mean(), err_b[4, :].mean(), err_b2[4, :].mean()]
# plt.bar(x_labels, means)
# plt.xlabel('Models')
# plt.ylabel('MSE')  # You might want to add a y-label too
# plt.title('Mean Squared Error Comparison')  # You can add a title if needed
# plt.show()

# x_labels = ['ShapeFinder', 'Bench1', 'Bench2']
# means = [err_nn[0, :].mean(), err_b[0, :].mean(), err_b2[0, :].mean()]
# plt.bar(x_labels, means)
# plt.xlabel('Models')
# plt.ylabel('MSE')  # You might want to add a y-label too
# plt.title('Mean Squared Error Comparison - 2019')  # You can add a title if needed
# plt.show()

# x_labels = ['ShapeFinder', 'Bench1', 'Bench2']
# means = [err_nn[1, :].mean(), err_b[1, :].mean(), err_b2[1, :].mean()]
# plt.bar(x_labels, means)
# plt.xlabel('Models')
# plt.ylabel('MSE')  # You might want to add a y-label too
# plt.title('Mean Squared Error Comparison - 2020')  # You can add a title if needed
# plt.show()

# x_labels = ['ShapeFinder', 'Bench1', 'Bench2']
# means = [err_nn[2, :].mean(), err_b[2, :].mean(), err_b2[2, :].mean()]
# plt.bar(x_labels, means)
# plt.xlabel('Models')
# plt.ylabel('MSE')  # You might want to add a y-label too
# plt.title('Mean Squared Error Comparison - 2021')  # You can add a title if needed
# plt.show()

# x_labels = ['ShapeFinder', 'Bench1', 'Bench2']
# means = [err_nn[3, :].mean(), err_b[3, :].mean(), err_b2[3, :].mean()]
# plt.bar(x_labels, means)
# plt.xlabel('Models')
# plt.ylabel('MSE')  # You might want to add a y-label too
# plt.title('Mean Squared Error Comparison - 2022')  # You can add a title if needed
# plt.show()

# =============================================================================
# Normalized
# =============================================================================

scaler=MinMaxScaler((0,1))
df = scaler.fit_transform(df_tot_tot) 
df=pd.DataFrame(df)

for k in range(len(df_fin)):
    tot_df_nn=df_fin[k]
    tot_df_nn=scaler.transform(tot_df_nn)
    tot_df_nn=pd.DataFrame(tot_df_nn)
    tot_df_ar=df_fin2[0]
    tot_df_ar=scaler.transform(tot_df_ar)
    tot_df_ar=pd.DataFrame(tot_df_ar)
    tot_df_ar2=df_fin3[0]
    tot_df_ar2=scaler.transform(tot_df_ar2)
    tot_df_ar2=pd.DataFrame(tot_df_ar2)
    tot_df_obs=df_fin4[0]
    tot_df_obs=scaler.transform(tot_df_obs)
    tot_df_obs=pd.DataFrame(tot_df_obs)
    err_nn=[]
    err_b=[]
    err_b2=[]
    for i in range(12):
        err_nn.append(mean_squared_error(tot_df_obs.loc[i].to_numpy().flatten(), tot_df_nn.loc[i].to_numpy().flatten()))
        err_b.append(mean_squared_error(tot_df_obs.loc[i].to_numpy().flatten(), tot_df_ar.loc[i].to_numpy().flatten()))
        err_b2.append(mean_squared_error(tot_df_obs.loc[i].to_numpy().flatten(), tot_df_ar2.loc[i].to_numpy().flatten()))
    plt.plot(err_nn,label=k)
#plt.plot(err_b,label='ar1')    
plt.plot(err_b2,label='ar2')   
plt.legend()
plt.show()

tot_df_nn=df_fin[0]
tot_df_nn=scaler.transform(tot_df_nn)
tot_df_nn=pd.DataFrame(tot_df_nn)
tot_df_nn=tot_df_nn.fillna(0)
# for i in range(5,12):    
#     tot_df_nn.loc[i]=df_fin[2].loc[i]    

tot_df_ar=df_fin2[0]
tot_df_ar=scaler.transform(tot_df_ar)
tot_df_ar=pd.DataFrame(tot_df_ar)
tot_df_ar2=df_fin3[0]
tot_df_ar2=scaler.transform(tot_df_ar2)
tot_df_ar2=pd.DataFrame(tot_df_ar2)
tot_df_obs=df_fin4[0]
tot_df_obs=scaler.transform(tot_df_obs)
tot_df_obs=pd.DataFrame(tot_df_obs)
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

x_labels = ['Bench1', 'Bench2']
mea_2=(err_b2[:4, :].flatten()-err_nn[:4, :].flatten())
mea_1=(err_b[:4, :].flatten()-err_nn[:4, :].flatten())
means = [mea_2.mean(),mea_1.mean()]
plt.bar(x_labels, means, capsize=5)
plt.xlabel('Models')
plt.ylabel('MSE')  # You might want to add a y-label too
plt.title('Mean Squared Error Normalized Comparison')  # You can add a title if needed
#plt.yscale('log')
plt.show()

#### No zeros

err_nn=[]
err_b=[]
err_b2=[]
go=0
for i in range(len(tot_df_nn.columns)):
    if (tot_df_obs.iloc[:,i]==0).all()==False:
        for y in range(4):
            err_nn.append(mean_squared_error(tot_df_obs.iloc[y*12:(y+1)*12,i], tot_df_nn.iloc[y*12:(y+1)*12,i]))
            err_b.append(mean_squared_error(tot_df_obs.iloc[y*12:(y+1)*12,i], tot_df_ar.iloc[y*12:(y+1)*12,i]))
            err_b2.append(mean_squared_error(tot_df_obs.iloc[y*12:(y+1)*12,i], tot_df_ar2.iloc[y*12:(y+1)*12,i]))
        err_nn.append(mean_squared_error(tot_df_obs.iloc[:,i], tot_df_nn.iloc[:,i]))
        err_b.append(mean_squared_error(tot_df_obs.iloc[:,i], tot_df_ar.iloc[:,i]))
        err_b2.append(mean_squared_error(tot_df_obs.iloc[:,i], tot_df_ar2.iloc[:,i]))
        go=go+1
        
err_nn = np.array(err_nn).reshape((5,go),order='F')
err_b = np.array(err_b).reshape((5,go),order='F')
err_b2 = np.array(err_b2).reshape((5,go),order='F')

x_labels = ['Bench2', 'Bench1']
mea_2=(err_b2[:4, :].flatten()-err_nn[:4, :].flatten())
mea_1=(err_b[:4, :].flatten()-err_nn[:4, :].flatten())
means = [mea_2.mean(),mea_1.mean()]
confidence_intervals = [1.96 * np.std(mea_2) / np.sqrt(len(mea_2)),
                        1.96 * np.std(mea_1) / np.sqrt(len(mea_1))]
plt.bar(x_labels, means, yerr=confidence_intervals, capsize=5)
plt.xlabel('Models')
plt.ylabel('MSE')  # You might want to add a y-label too
plt.title('Mean Squared Error Normalized Comparison')  # You can add a title if needed
plt.yscale('log')
plt.show()

x_labels = ['Bench2']
means = [mea_2.mean()]
confidence_intervals = [1.96 * np.std(mea_2) / np.sqrt(len(mea_2))]
plt.bar(x_labels, means, yerr=confidence_intervals, capsize=5)
plt.xlabel('Models')
plt.ylabel('MSE')  # You might want to add a y-label too
plt.title('Mean Squared Error Normalized Comparison')  # You can add a title if needed
plt.show()

# =============================================================================
# Diff
# =============================================================================

tot_df_obs=df_fin8[0]
# scaler=MinMaxScaler((0,1))
# tot_df_obs = scaler.fit_transform(tot_df_obs) 
# tot_df_obs=pd.DataFrame(tot_df_obs)

# for k in range(len(df_fin)):
#     tot_df_nn=df_fin5[k]
#     tot_df_nn=scaler.transform(tot_df_nn)
#     tot_df_nn=pd.DataFrame(tot_df_nn)
#     tot_df_ar=df_fin6[0]
#     tot_df_ar=scaler.transform(tot_df_ar)
#     tot_df_ar=pd.DataFrame(tot_df_ar)
#     tot_df_ar2=df_fin7[0]
#     tot_df_ar2=scaler.transform(tot_df_ar2)
#     tot_df_ar2=pd.DataFrame(tot_df_ar2)
#     err_nn=[]
#     err_b=[]
#     err_b2=[]
#     for i in range(12):
#         err_nn.append(mean_squared_error(tot_df_obs.loc[i].to_numpy().flatten(), tot_df_nn.loc[i].to_numpy().flatten()))
#         err_b.append(mean_squared_error(tot_df_obs.loc[i].to_numpy().flatten(), tot_df_ar.loc[i].to_numpy().flatten()))
#         err_b2.append(mean_squared_error(tot_df_obs.loc[i].to_numpy().flatten(), tot_df_ar2.loc[i].to_numpy().flatten()))
#     plt.plot(err_nn,label=k)
# #plt.plot(err_b,label='ar1')    
# plt.plot(err_b2,label='ar2')   
# plt.legend()
# plt.show()

tot_df_nn=df_fin5[0].iloc[:-1]
tot_df_ar=df_fin6[0]
tot_df_ar2=df_fin7[0]
# tot_df_nn=scaler.transform(tot_df_nn)
# tot_df_nn=pd.DataFrame(tot_df_nn)
cg_nn=[]
cg_b=[]
cg_b2=[]
count=0
for i in range(len(tot_df_nn.columns)):
    for y in range(len(tot_df_nn)):
        if abs(tot_df_obs.iloc[y,i])>20:
            if tot_df_obs.iloc[y,i]>0:
                if (tot_df_nn.iloc[y,i]>0.7*tot_df_obs.iloc[y,i]) and (tot_df_nn.iloc[y,i]<1.3*tot_df_obs.iloc[y,i]):
                    cg_nn.append((y,i))
                if (tot_df_ar.iloc[y,i]>0.7*tot_df_obs.iloc[y,i]) and (tot_df_ar.iloc[y,i]<1.3*tot_df_obs.iloc[y,i]):
                    cg_b.append((y,i))
                if (tot_df_ar2.iloc[y,i]>0.7*tot_df_obs.iloc[y,i]) and (tot_df_ar2.iloc[y,i]<1.3*tot_df_obs.iloc[y,i]):
                    cg_b2.append((y,i))
            if tot_df_obs.iloc[y,i]<0:     
                if (tot_df_nn.iloc[y,i]<0.7*tot_df_obs.iloc[y,i]) and (tot_df_nn.iloc[y,i]>1.3*tot_df_obs.iloc[y,i]):
                    cg_nn.append((y,i))
                if (tot_df_ar.iloc[y,i]<0.7*tot_df_obs.iloc[y,i]) and (tot_df_ar.iloc[y,i]>1.3*tot_df_obs.iloc[y,i]):
                    cg_b.append((y,i))
                if (tot_df_ar2.iloc[y,i]<0.7*tot_df_obs.iloc[y,i]) and (tot_df_ar2.iloc[y,i]>1.3*tot_df_obs.iloc[y,i]):
                    cg_b2.append((y,i))
            count=count+1    

first_elements = [tup[0] for tup in cg_nn]
second_elements = [tup[1] for tup in cg_nn]
df_cgn = pd.DataFrame({'1': first_elements, '2': second_elements})


plot_nn=df_fin[0] 
plot_ar=df_fin2[0]
plot_ar2=df_fin3[0]
plot_obs=df_fin4[0]
year=['2019','2020','2021','2022']
for i in range(191):
    if (plot_obs.iloc[:,i]==0).all()==False:
        for y in range(4):
            plt.plot(plot_nn.iloc[y*12:(y+1)*12,i].reset_index(drop=True),label='ShapeFinder')
            plt.plot(plot_ar.iloc[y*12:(y+1)*12,i].reset_index(drop=True),label='B1')
            plt.plot(plot_ar2.iloc[y*12:(y+1)*12,i].reset_index(drop=True),label='B2')
            plt.plot(plot_obs.iloc[y*12:(y+1)*12,i].reset_index(drop=True),label='Obs',marker='o',linewidth=5)
            plt.legend()
            plt.title(plot_nn.columns[i]+year[y])
            plt.show()

err_nn = np.array(err_nn).reshape((5,191),order='F')
err_b = np.array(err_b).reshape((5,191),order='F')
err_b2 = np.array(err_b2).reshape((5,191),order='F')

x_labels = ['Bench1', 'Bench2']
mea_2=(err_b2[:4, :].flatten()-err_nn[:4, :].flatten())
mea_1=(err_b[:4, :].flatten()-err_nn[:4, :].flatten())
means = [mea_2.mean(),mea_1.mean()]
plt.bar(x_labels, means, capsize=5)
plt.xlabel('Models')
plt.ylabel('MSE')  # You might want to add a y-label too
plt.title('Mean Squared Error Normalized Comparison')  # You can add a title if needed
plt.yscale('log')
plt.show()






























































l_sf=[]
l_b=[]
l_b2=[]
l_sf_d=[]
l_b_d=[]
l_b2_d=[]
df_fin=[]
df_fin2=[]
df_fin3=[]
df_fin4=[]
df_fin5=[]
df_fin6=[]
df_fin7=[]
df_fin8=[]
h=15
h_train=10
for min_d in [0.1,0.3]:
    for min_d_e in [0.1]:
        tot_df_sf=pd.DataFrame()
        tot_df_ar=pd.DataFrame()
        tot_df_ar2=pd.DataFrame()
        tot_df_obs=pd.DataFrame()
        tot_df_sf_diff=pd.DataFrame()
        tot_df_ar_diff=pd.DataFrame()
        tot_df_ar2_diff=pd.DataFrame()
        tot_df_obs_diff=pd.DataFrame()
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
            
            train_df = df_tot_m.iloc[:-15-h_train,:]
            test_df = df_tot_m.iloc[-15-h_train:,:]
            
            dtw_sel=2
            df_sf_tot=pd.DataFrame()
            df_ar=pd.DataFrame()
            df_ar2=pd.DataFrame()
            df_obs=pd.DataFrame()
            df_sf_tot=pd.DataFrame()
            df_ar_d=pd.DataFrame()
            df_ar2_d=pd.DataFrame()
            df_obs_d=pd.DataFrame()
            df_sf_diff_tot=pd.DataFrame()
            for row in range(len(test_df.columns)):
                ts=test_df.iloc[:,row]
                df_sf=pd.DataFrame()
                seq=ts.iloc[:h_train]
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
                
                df_sf_diff=df_sf.iloc[2:15,:]
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
                 
                if (obs==0).all()==False:
                    plt.figure(figsize=(10, 6))
                    plt.plot(df_sf_tot.iloc[:,-1], label='ShapeFinder', marker='o',color='r')
                    plt.plot(df_ar.iloc[:,-1], label='Bench1', marker='o',color='g')
                    plt.plot(df_ar2.iloc[:,-1], label='Bench2', marker='o',color='b')
                    plt.plot(obs,label='Obs',marker='o',linewidth=5)
                    plt.legend()
                    plt.grid(True)
                    plt.title(test_df.columns[row])
                    plt.show()
                
                
                df_sf_diff=df_sf_diff.reset_index(drop=True)
                df_sf_diff=df_sf_diff.diff()
                df_sf_diff=df_sf_diff.iloc[1:,:]
                df_sf_diff_tot = pd.concat([df_sf_diff_tot,df_sf_diff.iloc[:,0]],axis=1)
                
                ### Bench 
                ben1_diff=bench1.iloc[-13:,row]
                ben1_diff=ben1_diff.diff()
                ben1_diff=ben1_diff.iloc[1:]
                ben2_diff=bench2.iloc[-13:,row]
                ben2_diff=ben2_diff.diff()
                ben2_diff=ben2_diff.iloc[1:]
                obs_diff = ts.iloc[-13:]
                obs_diff=obs_diff.diff()
                obs_diff=obs_diff.iloc[1:]
                 
                ben1_diff=ben1_diff.reset_index(drop=True) 
                ben2_diff=ben2_diff.reset_index(drop=True) 
                obs_diff=obs_diff.reset_index(drop=True) 
                ben1=ben1.reset_index(drop=True) 
                ben2=ben2.reset_index(drop=True) 
                obs=obs.reset_index(drop=True) 
                
                df_ar_d=pd.concat([df_ar_d,ben1_diff],axis=1)
                df_ar2_d=pd.concat([df_ar2_d,ben2_diff],axis=1)
                df_obs_d=pd.concat([df_obs_d,obs_diff],axis=1)
                
                
                
            tot_df_sf=pd.concat([tot_df_sf,df_sf_tot],axis=0)
            tot_df_ar=pd.concat([tot_df_ar,df_ar],axis=0)
            tot_df_ar2=pd.concat([tot_df_ar2,df_ar2],axis=0)
            tot_df_obs=pd.concat([tot_df_obs,df_obs],axis=0)
            tot_df_sf_diff=pd.concat([tot_df_sf_diff,df_sf_diff_tot],axis=0)
            tot_df_ar_diff=pd.concat([tot_df_ar_diff,df_ar_d],axis=0)
            tot_df_ar2_diff=pd.concat([tot_df_ar2_diff,df_ar2_d],axis=0)
            tot_df_obs_diff=pd.concat([tot_df_obs_diff,df_obs_d],axis=0)
        
        tot_df_sf.columns = tot_df_obs.columns
        tot_df_ar.columns = tot_df_obs.columns
        tot_df_ar2.columns = tot_df_obs.columns
        tot_df_sf_diff.columns = tot_df_obs_diff.columns
        tot_df_ar_diff.columns = tot_df_obs_diff.columns
        tot_df_ar2_diff.columns = tot_df_obs_diff.columns
        
        tot_df_sf=tot_df_sf.fillna(0)
        tot_df_sf_diff=tot_df_sf_diff.fillna(0)
        
        df_fin.append(tot_df_sf)
        df_fin2.append(tot_df_ar)
        df_fin3.append(tot_df_ar2)
        df_fin4.append(tot_df_obs)
        df_fin5.append(tot_df_sf_diff)
        df_fin6.append(tot_df_ar_diff)
        df_fin7.append(tot_df_ar2_diff)
        df_fin8.append(tot_df_obs_diff)
        
        
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
        
        tot_df_sf_diff.index = tot_df_sf.index
        err_sf_d=[]
        err_b_d=[]
        err_b2_d=[]
        for i in range(len(tot_df_sf_diff.columns)):
            for y in range(4):
                if (tot_df_obs_diff.iloc[y*12:(y+1)*12,i]==0).all() and (tot_df_sf_diff.iloc[y*12:(y+1)*12,i]==0).all():
                    err_sf_d.append(0.0)
                else :
                    err_sf_d.append(dtw.distance(tot_df_obs_diff.iloc[y*12:(y+1)*12,i], tot_df_sf_diff.iloc[y*12:(y+1)*12,i]))
                if (tot_df_obs_diff.iloc[y*12:(y+1)*12,i]==0).all() and  (tot_df_ar_diff.iloc[y*12:(y+1)*12,i]==0).all():    
                    err_b_d.append(0.0)
                else:
                    err_b_d.append(dtw.distance(tot_df_obs_diff.iloc[y*12:(y+1)*12,i], tot_df_ar_diff.iloc[y*12:(y+1)*12,i]))
                if (tot_df_obs_diff.iloc[y*12:(y+1)*12,i]==0).all() and  (tot_df_ar2_diff.iloc[y*12:(y+1)*12,i]==0).all():    
                    err_b2_d.append(0.0)
                else:
                    err_b2_d.append(dtw.distance(tot_df_obs_diff.iloc[y*12:(y+1)*12,i], tot_df_ar2_diff.iloc[y*12:(y+1)*12,i]))

        err_sf_d = np.array(err_sf_d).reshape((4,191),order='F')
        err_b_d = np.array(err_b_d).reshape((4,191),order='F')
        err_b2_d = np.array(err_b2_d).reshape((4,191),order='F')
        
        x_labels = ['ShapeFinder', 'Bench1', 'Bench2']
        means = [err_sf_d.mean(), err_b_d.mean(), err_b2_d.mean()]
        plt.bar(x_labels, means)
        plt.xlabel('Models')
        plt.ylabel('DTW') 
        plt.title('DTW of diff')
        plt.show()
        
        l_sf_d.append(err_sf_d)
        l_b_d.append(err_b_d)
        l_b2_d.append(err_b2_d)
        
        
# =============================================================================
# Normalized
# =============================================================================

scaler=MinMaxScaler((0,1))
df = scaler.fit_transform(df_tot_tot) 
df=pd.DataFrame(df)

for k in range(len(df_fin)):
    tot_df_nn=df_fin[k]
    tot_df_nn=tot_df_nn.fillna(0)
    tot_df_nn=scaler.transform(tot_df_nn)
    tot_df_nn=pd.DataFrame(tot_df_nn)
    tot_df_nn.index = df_fin[k].index
    tot_df_ar=df_fin2[0]
    tot_df_ar=scaler.transform(tot_df_ar)
    tot_df_ar=pd.DataFrame(tot_df_ar)
    tot_df_ar.index = df_fin[k].index
    tot_df_ar2=df_fin3[0]
    tot_df_ar2=scaler.transform(tot_df_ar2)
    tot_df_ar2=pd.DataFrame(tot_df_ar2)
    tot_df_obs=df_fin4[0]
    tot_df_obs=scaler.transform(tot_df_obs)
    tot_df_obs=pd.DataFrame(tot_df_obs)
    tot_df_ar2.index = tot_df_nn.index
    tot_df_obs.index = df_fin[k].index
    err_nn=[]
    err_b=[]
    err_b2=[]
    for i in range(12):
        err_nn.append(mean_squared_error(tot_df_obs.loc[i].to_numpy().flatten(), tot_df_nn.loc[i].to_numpy().flatten()))
        err_b.append(mean_squared_error(tot_df_obs.loc[i].to_numpy().flatten(), tot_df_ar.loc[i].to_numpy().flatten()))
        err_b2.append(mean_squared_error(tot_df_obs.loc[i].to_numpy().flatten(), tot_df_ar2.loc[i].to_numpy().flatten()))
    plt.plot(err_nn,label=k)
#plt.plot(err_b,label='ar1')    
plt.plot(err_b2,label='ar2')   
plt.legend()
plt.show()

tot_df_nn=df_fin[1]
tot_df_nn=tot_df_nn.fillna(0)
tot_df_nn=scaler.transform(tot_df_nn)
tot_df_nn=pd.DataFrame(tot_df_nn)

# tot_df_nn2=df_fin[1]
# tot_df_nn2=tot_df_nn.fillna(0)
# tot_df_nn2=scaler.transform(tot_df_nn2)
# tot_df_nn2=pd.DataFrame(tot_df_nn2)

# tot_df_nn.loc[:6] = tot_df_nn2.loc[:6] 

tot_df_ar=df_fin2[0]
tot_df_ar=scaler.transform(tot_df_ar)
tot_df_ar=pd.DataFrame(tot_df_ar)
tot_df_ar2=df_fin3[0]
tot_df_ar2=scaler.transform(tot_df_ar2)
tot_df_ar2=pd.DataFrame(tot_df_ar2)
tot_df_obs=df_fin4[0]
tot_df_obs=scaler.transform(tot_df_obs)
tot_df_obs=pd.DataFrame(tot_df_obs)
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

x_labels = ['Bench2']
mea_2=(err_b2[:4, :].flatten()-err_nn[:4, :].flatten())
mea_1=(err_b[:4, :].flatten()-err_nn[:4, :].flatten())
means = [mea_2.mean()]
plt.bar(x_labels, means, capsize=5)
plt.xlabel('Models')
plt.ylabel('MSE')  # You might want to add a y-label too
plt.title('Mean Squared Error Normalized Comparison')  # You can add a title if needed
#plt.yscale('log')
plt.show()


d_nn=[]
d_b=[]
d_b2=[]
win=2
for i in range(len(tot_df_nn.columns)):
    for y in range(4):
        real = tot_df_obs.iloc[y*12:(y+1)*12,i]
        sf=tot_df_nn.iloc[y*12:(y+1)*12,i]
        b1=tot_df_ar.iloc[y*12:(y+1)*12,i]
        b2=tot_df_ar2.iloc[y*12:(y+1)*12,i]
        max_s=0
        max_b1=0
        max_b2=0
        for wi in range(12-win):
            real_win = real.iloc[wi:wi+win+1]
            sf_win= sf.iloc[wi:wi+win+1]
            b1_win=b1.iloc[wi:wi+win+1]
            b2_win=b2.iloc[wi:wi+win+1]
            t_v=[]
            for value in real_win[1:].index:
                if ((real_win[value]<1.05*real_win[value-1]) and (real_win[value]>0.95*real_win[value-1])) or (real_win[value-1]==0 and real_win[value]==0) :
                    t_v.append(1)
                elif (real_win[value]>1.05*real_win[value-1]):
                    t_v.append(2)
                elif (real_win[value]<0.95*real_win[value-1]):
                    t_v.append(0)
            if (pd.Series(t_v)==1).all()==False:
                sf_v=[]
                for value in sf_win[1:].index:
                    if ((sf_win[value]<1.05*sf_win[value-1]) and (sf_win[value]>0.95*sf_win[value-1])) or (sf_win[value-1]==0 and sf_win[value]==0) :
                        sf_v.append(1)
                    elif (sf_win[value]>1.05*sf_win[value-1]):
                        sf_v.append(2)
                    elif (sf_win[value]<0.95*sf_win[value-1]):
                        sf_v.append(0)
                mat=0        
                for index in range(len(sf_v)):
                    if t_v[index] ==sf_v[index]:
                        mat += 1  
                max_s=max_s+mat/len(sf_v)
                
                b1_v=[]
                for value in b1_win[1:].index:
                    if ((b1_win[value]<1.05*b1_win[value-1]) and (b1_win[value]>0.95*b1_win[value-1])) or (b1_win[value-1]==0 and b1_win[value]==0) :
                        b1_v.append(1)
                    elif (b1_win[value]>1.05*b1_win[value-1]):
                        b1_v.append(2)
                    elif (b1_win[value]<0.95*b1_win[value-1]):
                        b1_v.append(0)
                mat=0        
                for index in range(len(b1_v)):
                    if t_v[index] ==b1_v[index]:
                        mat += 1  
                max_b1=max_b1+mat/len(b1_v)
                
                b2_v=[]
                for value in b2_win[1:].index:
                    if ((b2_win[value]<1.05*b2_win[value-1]) and (b2_win[value]>0.95*b2_win[value-1])) or (b2_win[value-1]==0 and b2_win[value]==0) :
                        b2_v.append(1)
                    elif (b2_win[value]>1.05*b2_win[value-1]):
                        b2_v.append(2)
                    elif (b2_win[value]<0.95*b2_win[value-1]):
                        b2_v.append(0)
                mat=0        
                for index in range(len(b2_v)):
                    if t_v[index] ==b2_v[index]:
                        mat += 1  
                max_b2=max_b2+mat/len(b2_v)
        d_nn.append(max_s)
        d_b.append(max_b1)
        d_b2.append(max_b2)

d_nn = np.array(d_nn).reshape((4,191),order='F')
d_b = np.array(d_b).reshape((4,191),order='F')
d_b2 = np.array(d_b2).reshape((4,191),order='F')

d_comp = (d_nn - d_b2)
d_comp = pd.DataFrame(d_comp)
plt.plot(d_comp.mean())


year=['2022','2021','2020','2019']
for i in d_comp.mean().sort_values(ascending=False)[5:10].index:
    for y in range(4):
        real = tot_df_obs.iloc[y*12:(y+1)*12,i]
        sf=tot_df_nn.iloc[y*12:(y+1)*12,i]
        b1=tot_df_ar.iloc[y*12:(y+1)*12,i]
        b2=tot_df_ar2.iloc[y*12:(y+1)*12,i]
        plt.figure(figsize=(10, 6))
        plt.plot(sf, label='ShapeFinder', marker='o',color='r')
        #plt.plot(b1, label='Bench1', marker='o',color='g')
        plt.plot(b2, label='Bench2', marker='o',color='b')
        plt.plot(real,label='Obs',marker='o',linewidth=5)
        plt.legend()
        plt.grid(True)
        plt.title(test_df.columns[i]+year[y])
        plt.show()
        
        print(test_df.columns[i]+year[y])
        print(mean_squared_error(real,sf))
        print(mean_squared_error(real,b2))
        print('      ')
        
        

#         if (tot_df_obs.iloc[y*12:(y+1)*12,i].diff()[1:]==0).all() and (tot_df_nn.iloc[y*12:(y+1)*12,i].diff()[1:]==0).all():
#             d_nn.append(0)
#         else : 
#             d_nn.append(dtw.distance(tot_df_obs.iloc[y*12:(y+1)*12,i].diff()[1:].tolist(), tot_df_nn.iloc[y*12:(y+1)*12,i].diff()[1:].tolist()))
#         if (tot_df_obs.iloc[y*12:(y+1)*12,i].diff()[1:]==0).all() and (tot_df_ar.iloc[y*12:(y+1)*12,i].diff()[1:]==0).all():
#             d_b.append(0)
#         else : 
#             d_b.append(dtw.distance(tot_df_obs.iloc[y*12:(y+1)*12,i].diff()[1:].tolist(), tot_df_ar.iloc[y*12:(y+1)*12,i].diff()[1:].tolist()))
#         if (tot_df_obs.iloc[y*12:(y+1)*12,i].diff()[1:]==0).all() and (tot_df_ar2.iloc[y*12:(y+1)*12,i].diff()[1:]==0).all():
#             d_b2.append(0)
#         else : 
#             d_b2.append(dtw.distance(tot_df_obs.iloc[y*12:(y+1)*12,i].diff()[1:].tolist(), tot_df_ar2.iloc[y*12:(y+1)*12,i].diff()[1:].tolist()))

# d_nn = np.array(d_nn).reshape((4,191),order='F')
# d_b = np.array(d_b).reshape((4,191),order='F')
# d_b2 = np.array(d_b2).reshape((4,191),order='F')

# d_comp = (d_b2 - d_nn)
# d_comp = pd.DataFrame(d_comp)
# d_comp.median().mean()
# plt.plot(d_comp.median(axis=1).to_numpy().flatten())



