# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans
import numpy as np
from sklearn.metrics import mean_squared_error
from functions import extract_b_clu,Compare_RF_exo_h,Compare_pred_exo_h,Compare_RF_exo_only_h,Compare_pred_exo_only_h
from functions_deep_learning import Compare_nn_exo_h
import matplotlib.pyplot as plt
import tensorflow as tf
np.random.seed(1)
tf.random.set_seed(1)

df_c = pd.read_csv('data.csv',index_col=0)
name_c=df_c.columns
index_m=df_c.index
df_c = df_c.fillna(0)

scaler = MinMaxScaler(feature_range=(0,1))
df_c = scaler.fit_transform(df_c)
df_c=pd.DataFrame(df_c)


bench1 = pd.read_csv('bench1.csv',index_col=0)
bench1.columns = name_c
bench1 = scaler.transform(bench1)
bench2 = pd.read_csv('bench2.csv',index_col=0)
bench2.columns = name_c
bench2 = scaler.transform(bench2)

df_sum = df_c.sum(axis=1)
df_ar=df_sum
for i in [1,2,3,12]:
    h=df_sum.iloc[:-i].reset_index(drop=True)
    h.index=df_ar.index[i:]
    df_ar = pd.concat([df_ar,h],axis=1)  
df_ar=df_ar.iloc[:,1:]
#df_ar=df_ar.fillna(0)

df_c.columns = name_c
df_c.index = index_m

df_resu_ar = pd.DataFrame()
df_resu_arx = pd.DataFrame()
df_resu_rf = pd.DataFrame()
df_resu_rfx =pd.DataFrame()
df_resu_nn = pd.DataFrame()
df_resu_nnx =pd.DataFrame()
df_resu_obs=pd.DataFrame()
df_resu_tot=pd.DataFrame()

### for year = 2018 
df_c = df_c.iloc[:-35,:]
val_len=-16

for row in [*range(len(df_c.columns))]:
    ts = df_c.iloc[:,row]
    if (ts.iloc[-31:-16]==0).all()==False:
        ts_resu_ar=[]
        ts_resu_arx=[]
        ts_resu_obs_ar=[]
        ts_resu_rf=[]
        ts_resu_rfx=[]
        ts_resu_nn=[]
        ts_resu_nnx=[]
        ts_resu_obs_nn=[]
        ts_resu_tot=[]
        t_spt=0.85
        # Extract patterns from the time series 
        X1 = extract_b_clu(ts,[3,5,7],[3,5,7,9],train_test_split=t_spt,top=10)
        X1=X1.iloc[:-1,:]
        X = df_ar.iloc[-len(X1):,:].reset_index(drop=True)
        X.index = X.index
        for h in range(3,15):
            # Initialize minimum error for ARIMA model
            min_ar_m=np.inf
            inclu =1
            # Finding the best ARIMA and ARIMA with patterns models with the minimum validation MSE
            res_temp=Compare_pred_exo_h(ts.iloc[4:],np.array(X.iloc[:,:inclu]),np.array(X1),h=h,train_test_split=t_spt)
            if mean_squared_error(res_temp['Darima_pred'][:val_len],res_temp['Obs'][:val_len])<min_ar_m:
                res=res_temp
                min_ar_m = mean_squared_error(res_temp['Darima_pred'][:val_len],res_temp['Obs'][:val_len])
                res_ar=res_temp
            for inclu in [3,5,7,10]:
                res_temp=Compare_pred_exo_only_h(ts.iloc[4:],np.array(X.iloc[:,:inclu]),np.array(X1),h,train_test_split=t_spt)
                if mean_squared_error(res_temp['Darima_pred'][:val_len],res_ar['Obs'][:val_len])<min_ar_m:
                    res=res_temp
                    min_ar_m = mean_squared_error(res_temp['Darima_pred'][:val_len],res_ar['Obs'][:val_len])
            
            # Same for RF and LSTM models
            random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 2000, num = 10)],
                   'max_features': ['auto', 'sqrt'],
                   'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)]+[None],
                   'min_samples_split': [2,5,10],
                   'min_samples_leaf': [1,2,4],
                   'bootstrap': [True, False]}
            config = {'activation':['relu'],
                      'n_layer' : [1],
                      'drop' : [0.1,0.5],
                      'l_r' : [0.001,0.0001]}
            
            min_rf_m=np.inf
            min_rfx_m=np.inf
            inclu=1
            for ar in [1,3,5,10]:
                res_rf_temp = Compare_RF_exo_h(ts,X.iloc[:,:inclu],X1,h,ar=ar,number_s=3,train_test_split=t_spt,opti_grid=random_grid)
                obs_val = res_rf_temp['obs'][:val_len]
                if mean_squared_error(res_rf_temp['rf_pred'][:val_len],res_rf_temp['obs'][:val_len])<min_rf_m:
                    res_rf=res_rf_temp
                    min_rf_m = mean_squared_error(res_rf_temp['rf_pred'][:val_len],res_rf_temp['obs'][:val_len])
                if mean_squared_error(res_rf_temp['rfx_pred'][:val_len],res_rf_temp['obs'][:val_len])<min_rfx_m:
                    res_rfx=res_rf_temp
                    min_rfx_m = mean_squared_error(res_rf_temp['rfx_pred'][:val_len],res_rf_temp['obs'][:val_len])  
                for inclu in [3,5,7,10]:
                    res_rf_temp = Compare_RF_exo_only_h(ts,X.iloc[:,:inclu],X1,h,ar=ar,number_s=3,train_test_split=t_spt,opti_grid=random_grid)
                    if mean_squared_error(res_rf_temp['rfx_pred'][:val_len],obs_val)<min_rfx_m:
                        res_rfx=res_rf_temp
                        min_rfx_m = mean_squared_error(res_rf_temp['rfx_pred'][:val_len],obs_val)
            min_nn_m=np.inf
            for ar in [1,3,5,10]:
                res_nn_t = Compare_nn_exo_h(ts,X,X1,h,ar=ar,number_s=3,train_test_split=t_spt,opti_grid=config)
                if mean_squared_error(res_nn_t['rfx_pred'][:val_len],obs_val)<min_nn_m:
                    res_nn=res_nn_t
                    min_nn_m = mean_squared_error(res_nn_t['rfx_pred'][:val_len],obs_val)
            
            tot_val = pd.Series([mean_squared_error(res_ar['arima_pred'][:val_len],res_ar['Obs'][:val_len]),mean_squared_error(res['Darima_pred'][:val_len],res_ar['Obs'][:val_len]),mean_squared_error(res_rf['rf_pred'][:val_len],obs_val),mean_squared_error(res_rfx['rfx_pred'][:val_len],obs_val),mean_squared_error(res_nn['rf_pred'][:val_len],obs_val),mean_squared_error(res_nn['rfx_pred'][:val_len],obs_val)])
            
            ts_resu_ar.append(res_ar['arima_pred'][-16+h])
            ts_resu_arx.append(res['Darima_pred'][-16+h])
            ts_resu_rf.append(res_rf['rf_pred'][-15+h])
            ts_resu_rfx.append(res_rfx['rfx_pred'][-15+h])
            ts_resu_nn.append(res_nn['rf_pred'][-15+h][0])
            ts_resu_nnx.append(res_nn['rfx_pred'][-15+h][0])
            ts_resu_obs_nn.append(res_rf['obs'][-15+h])
            
            ts_tot = [res_ar['arima_pred'][-16+h],res['Darima_pred'][-16+h],res_rf['rf_pred'][-15+h],res_rfx['rfx_pred'][-15+h],res_nn['rf_pred'][-15+h][0],res_nn['rfx_pred'][-15+h][0]]
            ts_resu_tot.append(ts_tot[tot_val.idxmin()])
        df_resu_ar=pd.concat([df_resu_ar,pd.Series(ts_resu_ar)],axis=1)
        df_resu_arx=pd.concat([df_resu_arx,pd.Series(ts_resu_arx)],axis=1)
        df_resu_rf=pd.concat([df_resu_rf,pd.Series(ts_resu_rf)],axis=1)
        df_resu_rfx=pd.concat([df_resu_rfx,pd.Series(ts_resu_rfx)],axis=1)
        df_resu_nn=pd.concat([df_resu_nn,pd.Series(ts_resu_nn)],axis=1)
        df_resu_nnx=pd.concat([df_resu_nnx,pd.Series(ts_resu_nnx)],axis=1)
        df_resu_obs=pd.concat([df_resu_obs,pd.Series(ts_resu_obs_nn)],axis=1)
        df_resu_tot=pd.concat([df_resu_tot,pd.Series(ts_resu_tot)],axis=1)
        
        df_resu_ar.to_csv('Results/resu_ar.csv')
        df_resu_arx.to_csv('Results/resu_arx.csv')
        df_resu_rf.to_csv('Results/resu_rf.csv')
        df_resu_rfx.to_csv('Results/resu_rfx.csv')
        df_resu_nn.to_csv('Results/resu_nn.csv')
        df_resu_nnx.to_csv('Results/resu_nnx.csv')
        df_resu_obs.to_csv('Results/resu_obs.csv')
        df_resu_tot.to_csv('Results/resu_tot.csv')

    else:
        ts_resu_ar=[0]*12
        ts_resu_arx=[0]*12
        ts_resu_rf=[0]*12
        ts_resu_rfx=[0]*12
        ts_resu_nn=[0]*12
        ts_resu_nnx=[0]*12
        ts_resu_obs_nn=ts.iloc[-13:-1].tolist()
        ts_resu_tot=[0]*12
        
        df_resu_ar=pd.concat([df_resu_ar,pd.Series(ts_resu_ar)],axis=1)
        df_resu_arx=pd.concat([df_resu_arx,pd.Series(ts_resu_arx)],axis=1)
        df_resu_rf=pd.concat([df_resu_rf,pd.Series(ts_resu_rf)],axis=1)
        df_resu_rfx=pd.concat([df_resu_rfx,pd.Series(ts_resu_rfx)],axis=1)
        df_resu_nn=pd.concat([df_resu_nn,pd.Series(ts_resu_nn)],axis=1)
        df_resu_nnx=pd.concat([df_resu_nnx,pd.Series(ts_resu_nnx)],axis=1)
        df_resu_obs=pd.concat([df_resu_obs,pd.Series(ts_resu_obs_nn)],axis=1)
        df_resu_tot=pd.concat([df_resu_tot,pd.Series(ts_resu_tot)],axis=1)
        df_resu_ar.to_csv('Results/resu_ar.csv')
        df_resu_arx.to_csv('Results/resu_arx.csv')
        df_resu_rf.to_csv('Results/resu_rf.csv')
        df_resu_rfx.to_csv('Results/resu_rfx.csv')
        df_resu_nn.to_csv('Results/resu_nn.csv')
        df_resu_nnx.to_csv('Results/resu_nnx.csv')
        df_resu_obs.to_csv('Results/resu_obs.csv')
        df_resu_tot.to_csv('Results/resu_tot.csv')
        
    plt.figure(figsize=(15,8))
    plt.plot(ts_resu_tot,label = 'Pred - TOT',marker='o')   
    plt.plot(ts_resu_obs_nn,label = 'Obs',marker='o')  
    plt.legend()
    plt.title(str(df_c.columns[row]))
    plt.show()
