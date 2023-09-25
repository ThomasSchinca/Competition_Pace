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
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
import random 

df_tot_tot=pd.read_csv('data.csv',index_col=0)
bench1_tot = pd.read_csv('bench1.csv',index_col=0)
bench2_tot = pd.read_csv('bench2.csv',index_col=0)

h=15
h_train=10
for min_d in [0.1]:
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
                    pred = model.predict(horizon=h,plot=False,metric='dtw',min_d=min_d_2,dtw_sel=dtw_sel,select=True,return_seq=False)
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
        
        
tot_df_sf.to_csv('10_h15.csv')
tot_df_ar.to_csv('ar.csv')
tot_df_ar2.to_csv('ar2.csv')
tot_df_obs.to_csv('obs.csv')


# =============================================================================
# Bootstrap
# =============================================================================
tot=pd.DataFrame(columns= ['year','week','country','boot','value'])
h=15
h_train=10
for min_d in [0.1]:
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
                    pred = model.predict(horizon=h,plot=False,metric='dtw',min_d=min_d_2,dtw_sel=dtw_sel,select=True,return_seq=True)
                    while pred is None:
                        min_d_2=min_d_2+min_d_e
                        pred = model.predict(horizon=h,plot=False,metric='dtw',min_d=min_d_2,dtw_sel=dtw_sel,select=True,return_seq=True)
                    for time in range(500):
                        pred_b = pred.iloc[random.choices(range(len(pred)),k=len(pred)),:]
                        pred_b = pd.concat([pred_b.mean(),pred_b.mean(),pred_b.mean()],axis=1)
                        pred_b=fitted_scaler.inverse_transform(pred_b)
                        pred_b[pred_b<0]=0
                        pred_b=pd.DataFrame(pred_b)
                        tot_b=pd.DataFrame([[num]*12,[*range(12)],[row]*12,[time]*12,pred_b.iloc[3:,0].tolist()]).T
                        tot_b.columns=tot.columns
                        tot=pd.concat([tot,tot_b])
                else:
                    pred=pd.concat([seq,seq,seq],axis=1)
                    pred.columns=[0,1,2]
                    pred=pred.iloc[3:,0].tolist()+[0]*5
                    for time in range(500):
                        tot_b=pd.DataFrame([[num]*12,[*range(12)],[row]*12,[time]*12,pred]).T
                        tot_b.columns=tot.columns
                        tot=pd.concat([tot,tot_b])
                tot.to_csv('boot_100.csv')


tot = pd.read_csv('boot.csv')
tot_mean= tot.groupby(['year','week','country']).mean()
tot_mean = tot_mean.reset_index()
df_boot=pd.DataFrame()
for i in range(191):
    df_c = tot_mean[tot_mean.country==i]['value']
    df_boot=pd.concat([df_boot,df_c.reset_index(drop=True)],axis=1)
    
