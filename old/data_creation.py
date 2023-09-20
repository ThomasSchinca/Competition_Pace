# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 17:19:30 2023

@author: thoma
"""

import pandas as pd

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

