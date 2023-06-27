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

df = pd.read_parquet('cm_actuals_allyears.parquet')
df=df.reset_index() 

df_country = pd.read_csv('country_list.csv',index_col=0)

df_x=pd.DataFrame()
for date in ['2017','2018','2019','2020']:
    df_re = pd.read_parquet('cm_features_to_oct'+str(date)+'.parquet')
    df_re=df_re.reset_index()
    df_x=pd.concat([df_x,df_re])    
df_x =df_x[df_x['country_id'].isin(df_country['id'].tolist())]

df_y = df_x.loc[:,['month_id','country_id','ged_sb']]
df_y = pd.concat([df_y,df])
df_y = df_y.sort_values(['country_id','month_id'])
df_y =df_y.drop_duplicates()
df_y = df_y.reset_index(drop=True)

bench =  pd.read_parquet('bm_cm_bootstrap_expanded_2021.parquet')
bench = bench.reset_index(drop=False)
bench = bench.groupby(['month_id','country_id']).mean()['outcome']
bench_2 = pd.read_parquet('bm_cm_last_historical_poisson_expanded_2021.parquet')
bench_2 = bench_2.reset_index(drop=False)
bench_2 = bench_2.groupby(['month_id','country_id']).mean()['outcome']


df_c=pd.DataFrame(index=range(121,505))
for i in df_y.country_id.unique():
    df_sub = df_y[df_y.country_id==i]['ged_sb']
    df_sub.name=df_country[df_country.id==i]['name'].iloc[0]
    df_sub.index=df_y[df_y.country_id==i]['month_id']
    df_sub = df_sub[~df_sub.index.duplicated(keep='first')]
    df_c = pd.concat([df_c,df_sub],axis=1)





scaler = MinMaxScaler(feature_range=(0,1))
df_c = scaler.fit_transform(df_c)
df_c=pd.DataFrame(df_c)
df_sum = df_c.sum(axis=1)
df_ar=df_sum
for i in [1,2,3,12]:
    h=df_sum.iloc[:-i].reset_index(drop=True)
    h.index=df_ar.index[i:]
    df_ar = pd.concat([df_ar,h],axis=1)  
df_ar=df_ar.iloc[:,1:]
df_ar=df_ar.fillna(0)

h=1
names=['AR','ARX','RF','RFX','LSTM','LSTMX']
t_spt=0.9

resu_tot=pd.DataFrame()
resu_ar=pd.DataFrame()
resu_rf=pd.DataFrame()
resu_nn=pd.DataFrame()

ts_resu_ar = []
ts_resu_arx = []
ts_resu_rf = []
ts_resu_rfx =[]
ts_resu_nn = []
ts_resu_nnx =[]
ts_resu_obs_nn=[]
ts_resu_obs_ar=[]

zero_col=[]
for row in [*range(len(df_c.columns))]:
    ts = df_c.iloc[:,row]
    try:
        ts=ts.dropna()
        val_len= -12
        
        # Extract patterns from the time series 
        X1 = extract_b_clu(ts,[3,5,7],[3,5,7,9],train_test_split=t_spt,top=10)
        X1=X1.iloc[:-1,:]
        X = df_ar.iloc[-len(X1):,:].reset_index(drop=True)
        X.index = X.index
        #X = pd.concat([X_1,X],axis=1)
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
        tot_res = pd.Series([mean_squared_error(res_ar['arima_pred'][val_len:],res_ar['Obs'][val_len:]),mean_squared_error(res['Darima_pred'][val_len:],res_ar['Obs'][val_len:]),mean_squared_error(res_rf['rf_pred'][val_len:],res_rf['obs'][val_len:]),mean_squared_error(res_rfx['rfx_pred'][val_len:],res_rf['obs'][val_len:]),mean_squared_error(res_nn['rf_pred'][val_len:],res_rf['obs'][val_len:]),mean_squared_error(res_nn['rfx_pred'][val_len:],res_rf['obs'][val_len:])])
        tot_res[len(tot_res)]=tot_res.iloc[tot_val.idxmin()]
        tot_ar = tot_res.iloc[:2]
        tot_ar[2] = tot_res.iloc[tot_val.iloc[:2].idxmin()]
        tot_rf = tot_res.iloc[2:4]
        tot_rf[4] = tot_res.iloc[tot_val.iloc[2:4].idxmin()]
        tot_nn = tot_res.iloc[4:6]
        tot_nn[6] = tot_res.iloc[tot_val.iloc[4:6].idxmin()]
        
        resu_tot=pd.concat([resu_tot,tot_res],axis=1)
        resu_rf=pd.concat([resu_rf,tot_rf],axis=1)
        resu_nn=pd.concat([resu_nn,tot_nn],axis=1)
        resu_ar=pd.concat([resu_ar,tot_ar],axis=1)
        print('The best model is : '+str(names[tot_val.idxmin()]))
        ts_resu_ar.append(res_ar['arima_pred'])
        ts_resu_arx.append(res['Darima_pred'])
        ts_resu_obs_ar.append(res_ar['Obs'])
        ts_resu_rf.append(res_rf['rf_pred'])
        ts_resu_rfx.append(res_rfx['rfx_pred'])
        ts_resu_nn.append(res_nn['rf_pred'])
        ts_resu_nnx.append(res_nn['rfx_pred'])
        ts_resu_obs_nn.append(res_rf['obs'])
        
        resu_tot.to_csv('Results/resu_tot.csv')
        resu_ar.to_csv('Results/resu_ar.csv')
        resu_rf.to_csv('Results/resu_rf.csv')
        resu_nn.to_csv('Results/resu_nn.csv')
        
        plt.figure(figsize=(15,8))
        plt.plot(ts_resu_ar[row],label = 'AR',marker='o')    
        plt.plot(ts_resu_arx[row],label = 'ARX',marker='o')  
        plt.plot(ts_resu_obs_ar[row].reset_index(drop=True),label = 'Obs',marker='o')   
        plt.vlines(len(ts_resu_ar[row])+val_len,0,ts_resu_obs_ar[row].max(),color='r',linestyles='--')
        plt.legend()
        plt.title('ARIMA '+str(df.columns[row]))
        plt.show()
        
        plt.figure(figsize=(15,10))
        plt.plot(ts_resu_rf[row],label = 'RF',marker='o')  
        plt.plot(ts_resu_rfx[row],label = 'RFX',marker='o')      
        plt.plot(ts_resu_nn[row],label = 'LSTM',marker='o')      
        plt.plot(ts_resu_nnx[row],label = 'LSTMX',marker='o')      
        plt.plot(ts_resu_obs_nn[row],label = 'Obs',marker='o')   
        plt.vlines(len(ts_resu_rf[row])+val_len,0,ts_resu_obs_nn[row].max(),color='r',linestyles='--')
        plt.legend()
        plt.title('RF+LSTM '+str(df.columns[row]))
        plt.show()
    except:
        zero_col.append(row)
    
    
resu_tot = pd.read_csv('Results/resu_tot_week.csv',index_col=0)

# Load the total result data, drop missing values, and concatenate with other dataframes
resu_tot=resu_tot.iloc[:,:7]
resu_tot=resu_tot.T
resu_tot = resu_tot.dropna(axis=0)
resu_tot.columns = ['AR','ARX','RF','RFX','LSTM','LSTMX','C_All']

# Calculate the mean of each column and sort in descending order
means= resu_tot.mean().sort_values(ascending=False)
means=means.iloc[2:]
# Create a bar plot for the means
fig, ax = plt.subplots(figsize=(8, 6))
means.plot(kind='bar', ax=ax)
ax.set_xlabel('Models')
ax.set_ylabel('Mean')
ax.set_title('Mean of MSE')
plt.ylim(0.021, 0.028)
plt.show()