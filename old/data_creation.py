# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 17:19:30 2023

@author: thoma
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt 

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


df_c=pd.DataFrame(index=range(121,505))
for i in df_y.country_id.unique():
    df_sub = df_y[df_y.country_id==i]['ged_sb']
    df_sub.name=df_country[df_country.id==i]['name'].iloc[0]
    df_sub.index=df_y[df_y.country_id==i]['month_id']
    df_sub = df_sub[~df_sub.index.duplicated(keep='first')]
    df_c = pd.concat([df_c,df_sub],axis=1)

#df_c.to_csv('data.csv')

tot_b=pd.DataFrame()
tot_b_2 = pd.DataFrame()
for date in ['2018','2019','2020','2021']:
    bench =  pd.read_parquet('bm_cm_bootstrap_expanded_'+str(date)+'.parquet')
    bench = bench.reset_index(drop=False)
    bench = bench.groupby(['month_id','country_id']).mean()['outcome']
    bench_2 = pd.read_parquet('bm_cm_last_historical_poisson_expanded_'+str(date)+'.parquet')
    bench_2 = bench_2.reset_index(drop=False)
    bench_2 = bench_2.groupby(['month_id','country_id']).mean()['outcome']
    
    bench = bench.reset_index()
    bench = pd.DataFrame(bench).pivot(columns='country_id',index='month_id', values='outcome')
    tot_b=pd.concat([tot_b,bench],axis=0)
    bench_2 = bench_2.reset_index()
    bench_2 = pd.DataFrame(bench_2).pivot(columns='country_id',index='month_id', values='outcome')
    tot_b_2=pd.concat([tot_b_2,bench_2],axis=0)
    
#tot_b.to_csv('bench1.csv')
#tot_b_2.to_csv('bench2.csv')


tot_boot=pd.DataFrame()
for date in ['2018','2019','2020','2021']:
    bench =  pd.read_parquet('pred_'+str(date)+'_15bootstrap.parquet')
    bench = bench.reset_index(drop=False)
    bench = bench.groupby(['month_id','country_id']).mean()['outcome']
    bench = bench.reset_index()
    bench = pd.DataFrame(bench).pivot(columns='country_id',index='month_id', values='outcome')
    tot_boot=pd.concat([tot_boot,bench],axis=0)
   
bench1_tot = pd.read_csv('bench1.csv',index_col=0)
bench2_tot = pd.read_csv('bench2.csv',index_col=0)    
    
scaler=MinMaxScaler((0,1))
df = scaler.fit_transform(df_c) 
df=pd.DataFrame(df)
df=df.iloc[-48:,:]
tot_boot=tot_boot.fillna(0)
tot_boot=scaler.transform(tot_boot)
tot_boot=pd.DataFrame(tot_boot)
bench1_tot=scaler.transform(bench1_tot)
bench1_tot=pd.DataFrame(bench1_tot)
bench2_tot=scaler.transform(bench2_tot)
bench2_tot=pd.DataFrame(bench2_tot)

err_nn=[]
err_b=[]
err_b2=[]
for i in range(len(tot_boot.columns)):
    for y in range(4):
        err_nn.append(mean_squared_error(df.iloc[y*12:(y+1)*12,i], tot_boot.iloc[y*12:(y+1)*12,i]))
        err_b.append(mean_squared_error(df.iloc[y*12:(y+1)*12,i], bench1_tot.iloc[y*12:(y+1)*12,i]))
        err_b2.append(mean_squared_error(df.iloc[y*12:(y+1)*12,i], bench2_tot.iloc[y*12:(y+1)*12,i]))
    err_nn.append(mean_squared_error(df.iloc[:,i], tot_boot.iloc[:,i]))
    err_b.append(mean_squared_error(df.iloc[:,i], bench1_tot.iloc[:,i]))
    err_b2.append(mean_squared_error(df.iloc[:,i], bench2_tot.iloc[:,i]))

err_nn = np.array(err_nn).reshape((5,191),order='F')
err_b = np.array(err_b).reshape((5,191),order='F')
err_b2 = np.array(err_b2).reshape((5,191),order='F')

d_mse = err_nn/err_b2

x_labels = ['Bench1','Bench2']
# mea_2=(err_b2[:4, :].flatten()-err_nn[:4, :].flatten())
# mea_1=(err_b[:4, :].flatten()-err_nn[:4, :].flatten())

mea_1=(err_b[:4, :].flatten()-err_nn[:4, :].flatten())
mea_2=(err_b2[:4, :].flatten()-err_nn[:4, :].flatten())

means = [mea_1.mean(),mea_2.mean()]
std = [2*mea_1.std()/np.sqrt(len(mea_1)),2*mea_2.std()/np.sqrt(len(mea_2))]
mean2_data = pd.DataFrame({
    'mean': means,
    'std': std
})

blue_color = '#404040'  # Dark grey shade
orange_color = '#A0A0A0'  # Light grey shade
def mse_formatter(x, pos):
    return '{:.1e}'.format(x)
def improvement_formatter(x, pos):
    return '{:.1e}'.format(x)

plt.figure(figsize=(12,8))
marker_size = 150
linewidth = 3
fonts=25
plt.rc('font', size=24)
plt.scatter(mean2_data.index, mean2_data['mean'], color=blue_color, marker='o', s=marker_size)
plt.errorbar(mean2_data.index, mean2_data['mean'], yerr=mean2_data['std'], fmt='none', color=blue_color, linewidth=linewidth)
plt.grid(False)
plt.hlines(0,-0.5,1.5,linestyles='--',color='r')
plt.xticks([0,1],['Benchmark 1','Benchmark 2'])
plt.ylabel('MSE difference',color=blue_color)
plt.tick_params(axis='y', colors=blue_color)
plt.show()

x_labels = ['Bench2']
mea_2=(err_b2[:4, :].flatten()-err_nn[:4, :].flatten())
means = [mea_2.mean()]
std = [2*mea_2.std()/np.sqrt(len(mea_2))]
mean2_data = pd.DataFrame({
    'mean': means,
    'std': std
})

blue_color = '#404040'  # Dark grey shade
orange_color = '#A0A0A0'  # Light grey shade
def mse_formatter(x, pos):
    return '{:.1e}'.format(x)


plt.figure(figsize=(12,8))
marker_size = 150
linewidth = 3
fonts=25
plt.rc('font', size=24)
plt.ylabel('MSE difference',color=blue_color)
plt.tick_params(axis='y', colors=blue_color)
plt.scatter(mean2_data.index, mean2_data['mean'], color=blue_color, marker='o', s=marker_size)
plt.errorbar(mean2_data.index, mean2_data['mean'], yerr=mean2_data['std'], fmt='none', color=blue_color, linewidth=linewidth)
plt.grid(False)
plt.hlines(0,-0.5,0.5,linestyles='--',color='r')
plt.xticks([0],['Benchmark 2'])
plt.show()


