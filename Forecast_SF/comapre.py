# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 11:38:50 2024

@author: thoma
"""

import requests
import pandas as pd 
import numpy as np 
import warnings
import pickle
warnings.filterwarnings("ignore")
np.random.seed(1)
from shape import Shape,finder
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
from cycler import cycler
from scipy.cluster.hierarchy import linkage, fcluster
import seaborn as sns
import geopandas as gpd
from dtaidistance import ed
from scipy.stats import ttest_1samp

plot_params = {"text.usetex":True,"font.family":"serif","font.size":20,"xtick.labelsize":20,"ytick.labelsize":20,"axes.labelsize":20,"figure.titlesize":20,"figure.figsize":(8,5),"axes.prop_cycle":cycler(color=['black','rosybrown','gray','indianred','red','maroon','silver',])}
plt.rcParams.update(plot_params)
df_list_preds={f"fatalities001_2022_00_t01/cm?page={i}":i for i in range(1,8)}

df_all=pd.DataFrame()
for i in range(len(df_list_preds)):
    response = requests.get(f'https://api.viewsforecasting.org/{list(df_list_preds.keys())[i]}')
    json_data = response.json()
    df=pd.DataFrame(json_data["data"])
    df=df[['country_id','month_id','month','sc_cm_sb_main']]
    #df=df.loc[df["month"]==list(df_list_preds.values())[i]]
    df_all = pd.concat([df_all, df])
    df_all=df_all.reset_index(drop=True)
cc_sort=df_all.country_id.unique()
cc_sort.sort()
df_preds_test_1 = df_all.pivot(index="month_id",columns='country_id', values='sc_cm_sb_main')


df_list_preds={f"fatalities001_2023_00_t01/cm?page={i}":i for i in range(1,8)}

df_all=pd.DataFrame()
for i in range(len(df_list_preds)):
    response = requests.get(f'https://api.viewsforecasting.org/{list(df_list_preds.keys())[i]}')
    json_data = response.json()
    df=pd.DataFrame(json_data["data"])
    df=df[['country_id','month_id','month','sc_cm_sb_main']]
    #df=df.loc[df["month"]==list(df_list_preds.values())[i]]
    df_all = pd.concat([df_all, df])
    df_all=df_all.reset_index(drop=True)
cc_sort=df_all.country_id.unique()
cc_sort.sort()
df_preds_test_2 = df_all.pivot(index="month_id",columns='country_id', values='sc_cm_sb_main')



df_list_input={f"predictors_fatalities002_0000_00/cm?page={i}":i for i in range(1,78)}

df_input_t=pd.DataFrame()
for i in range(len(df_list_input)):
    #print(f'https://api.viewsforecasting.org/{list(df_list_input.keys())[i]}')
    response = requests.get(f'https://api.viewsforecasting.org/{list(df_list_input.keys())[i]}')
    json_data = response.json()
    df=pd.DataFrame(json_data["data"])
    df=df[["country_id","month_id","ucdp_ged_sb_best_sum"]]
    df_input_t = pd.concat([df_input_t, df])
    df_input_t=df_input_t.reset_index(drop=True)

country_list = pd.read_csv('country_list.csv',index_col=0)
df_conf=pd.read_csv('reg_coun.csv',index_col=0,squeeze=True)
replace_c = {'Cambodia (Kampuchea)': 'Cambodia','DR Congo (Zaire)':'Congo, DRC',
             'Ivory Coast':'Cote d\'Ivoire', 'Kingdom of eSwatini (Swaziland)':'Swaziland',
             'Myanmar (Burma)':'Myanmar','Russia (Soviet Union)':'Russia',
             'Serbia (Yugoslavia)':'Serbia','Madagascar (Malagasy)':'Madagascar',
             'Macedonia, FYR':'Macedonia','Vietnam (North Vietnam)':'Vietnam',
             'Yemen (North Yemen)':'Yemen','Zimbabwe (Rhodesia)':'Zimbabwe',
             'United States of America':'United States','Solomon Islands':'Solomon Is.',
             'Bosnia-Herzegovina':'Bosnia and Herzegovina'}
df_conf.rename(index=replace_c, inplace=True)
df_conf['Sao Tome and Principe']='Africa'
df_input_s = df_input_t[df_input_t['country_id'].isin(cc_sort)]
df_input = df_input_s.pivot(index="month_id",columns='country_id',values='ucdp_ged_sb_best_sum')
df_input.index = pd.date_range('01/01/1989',periods=len(df_input),freq='M')
df_input = df_input.iloc[:-1,:]
df_input.columns = country_list['name']

#df_input.to_csv('df_input.csv')

df_tot_m = df_input.copy()
df_tot_m.replace(0, np.nan, inplace=True)
df_tot_m = df_tot_m.dropna(axis=1, how='all')
df_tot_m = df_tot_m.fillna(0)
#df_tot_m.to_csv('df_tot_m.csv')
h_train=10
h=12
dict_m={i :[] for i in df_input.columns}
df_input_sub=df_input.iloc[:-24]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        shape = Shape()
        shape.set_shape(df_input_sub.iloc[-h_train:,coun]) 
        find = finder(df_tot_m.iloc[:-24],shape)
        find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2)
        min_d_d=0.1
        while len(find.sequences)<5:
            min_d_d += 0.05
            find.find_patterns(min_d=min_d_d,select=True,metric='dtw',dtw_sel=2)
        dict_m[df_input.columns[coun]]=find.sequences
    else :
        pass
# with open('test1.pkl', 'wb') as f:
#     pickle.dump(dict_m, f)
    
h_train=10
h=12
dict_m={i :[] for i in df_input.columns}
df_input_sub=df_input.iloc[:-12]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        shape = Shape()
        shape.set_shape(df_input_sub.iloc[-h_train:,coun]) 
        find = finder(df_tot_m.iloc[:-12],shape)
        find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2)
        min_d_d=0.1
        while len(find.sequences)<5:
            min_d_d += 0.05
            find.find_patterns(min_d=min_d_d,select=True,metric='dtw',dtw_sel=2)
        dict_m[df_input.columns[coun]]=find.sequences
    else :
        pass
# with open('test2.pkl', 'wb') as f:
#     pickle.dump(dict_m, f)
    
    
    
with open('test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)

pred_tot_min=[]
pred_tot_pr=[]
len_mat=[]
horizon=12
df_input_sub=df_input.iloc[:-24]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-24]):
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/2, criterion='distance')
    
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        pred_ori=val_sce.loc[pr.idxmax(),:]
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        len_mat.append(len(tot_seq))
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
        
df_sf_1 = pd.concat(pred_tot_pr,axis=1)
df_sf_1.columns=country_list['name']
df_sf_1.to_csv('sf1.csv')        

with open('test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
    
df_input_sub=df_input.iloc[:-12]
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-12]):
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/2, criterion='distance')
        # if len(pd.Series(clusters).value_counts())>7:
        #     sub_norm = tot_seq[(tot_seq > 5).any(axis=1)].index
        #     tot_seq_c = tot_seq.copy()
        #     tot_seq_c.loc[sub_norm,:] = 10
        #     linkage_matrix = linkage(tot_seq_c, method='ward')
        #     clusters = fcluster(linkage_matrix, horizon/2, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        pred_ori=val_sce.loc[pr.idxmax(),:]
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))       
        
df_obs_1 = df_input.iloc[-24:-12,:]        
df_obs_1.to_csv('obs1.csv')   
df_obs_2 = df_input.iloc[-12:,:]        
df_obs_2.to_csv('obs2.csv')   

df_v_1 = df_preds_test_1.iloc[:12,:]
df_v_1.to_csv('views1.csv')  
df_v_2 = df_preds_test_2.iloc[:12,:]
df_v_2.to_csv('views2.csv')  

df_sf_2= pd.concat(pred_tot_pr,axis=1)
df_sf_2.columns=country_list['name']
df_sf_2.to_csv('sf2.csv')  
        



































# =============================================================================
# Selection Compound
# =============================================================================

len_mat=[]      
df_sure=[]
pr_list=[]   
pr_main=[]
pr_scale=[]  
with open('test1.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
df_input_sub=df_input.iloc[:-24]
horizon=12
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-24]):
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        tot_seq_c = tot_seq.copy()
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        pr_main.append(pr.max())
        len_mat.append(len(tot_seq))
        pr_scale.append(df_input_sub.iloc[-h_train:,coun].sum())
        testu = (df_input.iloc[-24:-12,coun] - df_input_sub.iloc[-h_train:,coun].min()) / (df_input_sub.iloc[-h_train:,coun].max() - df_input_sub.iloc[-h_train:,coun].min())
        tot_seq_c = pd.concat([pd.DataFrame(tot_seq_c),pd.DataFrame(testu.reset_index(drop=True)).T],axis=0)
        linkage_matrix_2 = linkage(tot_seq_c, method='ward')
        clusters_2 = fcluster(linkage_matrix_2, horizon/3, criterion='distance')
        if len(pd.Series(clusters_2).value_counts())==1:
            df_sure.append(tot_seq_c)
        if len(pd.Series(clusters).value_counts())==len(pd.Series(clusters_2).value_counts()):
            pr_list.append(pr[clusters_2[-1]])
        else:
            pr_list.append(-1)       
    else:
        pr_list.append(None)
        
        
        
        
with open('test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
df_input_sub=df_input.iloc[:-24]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-24]):
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        # if len(pd.Series(clusters).value_counts())>7:
        #     sub_norm = tot_seq[(tot_seq > 5).any(axis=1)].index
        #     tot_seq_c = tot_seq.copy()
        #     tot_seq_c.loc[sub_norm,:] = 10
        #     linkage_matrix = linkage(tot_seq_c, method='ward')
        #     clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        pred_ori=val_sce.loc[pr.idxmax(),:]
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))

err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], pred_tot_pr[i]))
    err_views.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]))
err_sf_pr = np.array(err_sf_pr)
err_views = np.array(err_views)
mse_list=np.log((err_views+1)/(err_sf_pr+1))    
#mse_list2=np.log(err_sf_pr+1)   
#mse_list2=np.array(err_sf_pr)  


with open('test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
df_input_sub=df_input.iloc[:-12]
horizon=12
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-12]):
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        tot_seq_c = tot_seq.copy()
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        pr_main.append(pr.max())
        len_mat.append(len(tot_seq))
        pr_scale.append(df_input_sub.iloc[-h_train:,coun].sum())
        testu = (df_input.iloc[-12:,coun] - df_input_sub.iloc[-h_train:,coun].min()) / (df_input_sub.iloc[-h_train:,coun].max() - df_input_sub.iloc[-h_train:,coun].min())
        tot_seq_c = pd.concat([pd.DataFrame(tot_seq_c),pd.DataFrame(testu.reset_index(drop=True)).T],axis=0)
        linkage_matrix_2 = linkage(tot_seq_c, method='ward')
        clusters_2 = fcluster(linkage_matrix_2, horizon/3, criterion='distance')
        if len(pd.Series(clusters_2).value_counts())==1:
            df_sure.append(tot_seq_c)
        if len(pd.Series(clusters).value_counts())==len(pd.Series(clusters_2).value_counts()):
            pr_list.append(pr[clusters_2[-1]])
        else:
            pr_list.append(-1)
    else:
        pr_list.append(None)
        
with open('test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
df_input_sub=df_input.iloc[:-12]
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        l_find=dict_m[df_input.columns[coun]]
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)
            if date+horizon<len(df_tot_m.iloc[:-12]):
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)
                pred_seq.append(seq.tolist())
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        # if len(pd.Series(clusters).value_counts())>7:
        #     sub_norm = tot_seq[(tot_seq > 5).any(axis=1)].index
        #     tot_seq_c = tot_seq.copy()
        #     tot_seq_c.loc[sub_norm,:] = 10
        #     linkage_matrix = linkage(tot_seq_c, method='ward')
        #     clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        pred_ori=val_sce.loc[pr.idxmax(),:]
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))      
err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], pred_tot_pr[i]))
    err_views.append(mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]))
err_sf_pr = np.array(err_sf_pr)
err_views = np.array(err_views)
mse_list2=np.log((err_views+1)/(err_sf_pr+1))  

      
mse_list_tot=np.concatenate([mse_list,mse_list2],axis=0)
pr_list=pd.Series(pr_list)
pr_main=pd.Series(pr_main)
pr_scale=pd.Series(pr_scale)
len_mat=pd.Series(len_mat)

df_tot_res = pd.DataFrame([mse_list_tot,pr_list]).T
df_tot_res[2]=[np.nan]*len(df_tot_res)
df_tot_res[2][pr_list == (-1)]='New'
df_tot_res[2][(pr_list >= 0) & (pr_list < 0.5)]='Low'
df_tot_res[2][(pr_list >= 0.5) & (pr_list < 1)]='High'
df_tot_res[2][(pr_list == 1)]='Sure'

ind_keep = pr_list.dropna().index 
pr_list=pr_list.dropna()
nan_percentage = ((pr_list==-1).sum() / len(pr_list)) * 100
zero_to_half = ((pr_list >= 0) & (pr_list < 0.5)).sum() / len(pr_list) * 100
half_to_02 = ((pr_list >= 0.5) & (pr_list < 1)).sum() / len(pr_list) * 100
ones = (pr_list == 1).sum() / len(pr_list) * 100

# Plotting
categories = ['New scenario', 'Low probability (0 to 0.5)', 'High proba (0.5 to 1)', 'Sure 100\%']
percentages = [nan_percentage, zero_to_half,half_to_02, ones]
plt.figure(figsize=(10, 6))
plt.bar(categories, percentages, color=['lightblue', 'lightblue', 'lightblue', 'lightblue'])
plt.title('Distibution in Different Categories')
plt.xlabel('Categories')
plt.ylabel('Percentage')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

df_tot_res = df_tot_res.dropna()
plt.boxplot(df_tot_res[df_tot_res[2]=='New'][0],positions=[0])
plt.boxplot(df_tot_res[df_tot_res[2]=='Low'][0],positions=[1])
plt.boxplot(df_tot_res[df_tot_res[2]=='High'][0],positions=[2])
plt.boxplot(df_tot_res[df_tot_res[2]=='Sure'][0],positions=[3])
plt.xticks([0,1,2,3],['New', 'Low', 'High', 'Sure'])
plt.ylim(-3,3)
plt.show()

#df_tot_res.groupby(2).median()     
        
for i in range(len(df_sure)):
    for j in range(len(df_sure[i])-1):
        plt.plot(df_sure[i].iloc[j,:],color='black',alpha=0.3)
    plt.plot(df_sure[i].iloc[:-1].mean(),color='black')
    plt.plot(df_sure[i].iloc[-1,:],color='red')
    plt.title(f'Distance = {ed.distance(df_sure[i].iloc[:-1].mean(),df_sure[i].iloc[-1,:])}')
    plt.show()
    
    
df_sel = pd.concat([df_tot_res.iloc[:,0].reset_index(drop=True),pr_scale,pr_main,len_mat],axis=1)
df_sel = df_sel.dropna()
df_sel.columns=['log MSE','Scale','Main_Pr','N_Matches']
df_sel['Confidence']=df_sel.iloc[:,2]*np.log10(df_sel.iloc[:,3])
n_df_sel= df_sel[df_sel['log MSE'] < 0]
p_df_sel= df_sel[df_sel['log MSE'] > 0]

plt.scatter(n_df_sel.iloc[:,1],n_df_sel.iloc[:,2],label='Neg')
plt.scatter(p_df_sel.iloc[:,1],p_df_sel.iloc[:,2],label='Pos')
plt.hlines(0.74, 0, 260, linestyles='--',color='green')
plt.hlines(0.95, 0, 10000, linestyles='--',color='green')
plt.vlines(260, 0, 0.74, linestyles='--',color='green')
plt.vlines(10000, 0, 0.95, linestyles='--',color='green')
plt.vlines(10000, 0.95, 1.1, linestyles='--',color='red')
plt.fill_between([0, 280], [0, 0.74], color='red', alpha=0.2)
plt.fill_between([10000, 1000000],0, 1.1 ,color='red', alpha=0.2)
plt.fill_between([0, 10000],0.95, 1.1, color='red', alpha=0.2)
plt.text(1,1.03,'Not enough matches',color='red')
plt.text(20000,0.4,'Too risky',color='red')
plt.text(1,0.2,'Views doing good',color='red',fontsize=20)
plt.ylim(0,1.1)
plt.xlim(0.5,1000000)
plt.xscale('log')
plt.legend()
plt.xlabel('Scale')
plt.ylabel('Highest scenario Pr')
plt.show()


df_keep_1=df_sel[(df_sel.iloc[:,1]<10000) & (0.74<df_sel.iloc[:,2])& (df_sel.iloc[:,2]<0.95)].index
df_keep_2=df_sel[(df_sel.iloc[:,1]<10000) & (df_sel.iloc[:,1]>260) & (df_sel.iloc[:,2]<=0.74)].index
df_try = pd.Series([0]*len(df_sel.iloc[:,0]))
df_try[df_keep_2] = df_sel.loc[df_keep_2,'log MSE']
df_try[df_keep_1] = df_sel.loc[df_keep_1,'log MSE']
df_try=pd.concat([df_try,pd.Series([0]*271)])
ttest_1samp(df_try,0)

plt.figure(figsize=(10,8))
plt.scatter(n_df_sel.iloc[:,1],n_df_sel.iloc[:,2]*np.log10(n_df_sel.iloc[:,3]),label='Neg',color='blue')
plt.scatter(p_df_sel.iloc[:,1],p_df_sel.iloc[:,2]*np.log10(p_df_sel.iloc[:,3]),label='Pos',color='red')
plt.xscale('log')
plt.hlines(1.5, 0, 200, linestyles='--',color='red')
plt.vlines(200, 0, 1.5, linestyles='--',color='red')
plt.vlines(10000, 0, 2, linestyles='--',color='red')
plt.legend()
plt.text(15000,0.4,'Too risky',color='red',fontsize=22)
plt.text(1,0.1,'Views doing better',color='red',fontsize=22)
plt.xlabel('Scale')
plt.ylabel('Confidence = (Pr*log(N matches))')
plt.show()

df_keep_1=df_sel[(df_sel.iloc[:,1]<10000) & (1.5<df_sel.iloc[:,4])].index
df_keep_2=df_sel[(df_sel.iloc[:,1]<10000) & (df_sel.iloc[:,1]>200) & (df_sel.iloc[:,4]<=1.5)].index
df_try = pd.Series([0]*len(df_sel.iloc[:,0]))
df_try[df_keep_2] = df_sel.loc[df_keep_2,'log MSE']
df_try[df_keep_1] = df_sel.loc[df_keep_1,'log MSE']
df_try=pd.concat([df_try,pd.Series([0]*271)])

ttest_1samp(df_try,0)


# =============================================================================
# Evaluation
# =============================================================================

def diff_explained(df_input,pred,k=5):
    d_nn=[]
    for i in range(len(df_input.columns)):
        real = df_input.iloc[:,i]
        real=real.reset_index(drop=True)
        sf = pred.iloc[:,i]
        sf=sf.reset_index(drop=True)
        max_s=0
        if (real==0).all()==False:
            for value in real[1:].index:
                if (real[value]==real[value-1]):
                    1
                else:
                    max_exp=0
                    if (real[value]-real[value-1])/(sf[value]-sf[value-1])>0 and sf[value]-sf[value-1] != 0:
                        t=abs(((real[value]-real[value-1])-(sf[value]-sf[value-1]))/(real[value]-real[value-1]))
                        max_exp = np.exp(-(k*t))
                    else:
                        if value==horizon-1:
                            if (real[value]-real[value-1])/(sf[value-1]-sf[value-2])>0 and sf[value-1]-sf[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(sf[value-1]-sf[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                        elif value==1:
                            if (real[value]-real[value-1])/(sf[value+1]-sf[value])>0 and sf[value+1]-sf[value] != 0:
                                t=abs(((real[value]-real[value-1])-(sf[value+1]-sf[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                        else : 
                            if (real[value]-real[value-1])/(sf[value-1]-sf[value-2])>0 and sf[value-1]-sf[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(sf[value-1]-sf[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                            if (real[value]-real[value-1])/(sf[value+1]-sf[value])>0 and sf[value+1]-sf[value] != 0:
                                t=abs(((real[value]-real[value-1])-(sf[value+1]-sf[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                    max_s=max_s+max_exp 
            d_nn.append(max_s)
        else:
            d_nn.append(0) 
    return(np.array(d_nn))
    

d_nn = diff_explained(df_input.iloc[-24:-24+horizon],df_sf_1)
d_nn2 = diff_explained(df_input.iloc[-12:],df_sf_2)
d_nn = np.concatenate([d_nn,d_nn2])

d_b = diff_explained(df_input.iloc[-24:-24+horizon],df_preds_test_1.iloc[:12])
d_b2 = diff_explained(df_input.iloc[-12:],df_preds_test_2.iloc[:12])
d_b = np.concatenate([d_b,d_b2])

d_null = diff_explained(df_input.iloc[-24:-24+horizon],pd.DataFrame(np.zeros((horizon,len(df_input.columns)))))
d_null2 = diff_explained(df_input.iloc[-12:],pd.DataFrame(np.zeros((horizon,len(df_input.columns)))))
d_null = np.concatenate([d_null,d_null2])

d_t1 = diff_explained(df_input.iloc[-24:-24+horizon],df_input.iloc[-24-horizon:-24])
d_t12 = diff_explained(df_input.iloc[-12:],df_input.iloc[-24:-24+horizon])
d_t1= np.concatenate([d_t1,d_t12])



err_sf_pr=[]
err_views=[]
err_zero=[]
err_t1=[]
for i in range(len(df_input.columns)):   
    err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i]))
    err_views.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]))
    err_zero.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], pd.Series(np.zeros((horizon,)))))
    err_t1.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i],df_input.iloc[-24-horizon:-24,i]))
for i in range(len(df_input.columns)):   
    err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i]))
    err_views.append(mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]))
    err_zero.append(mean_squared_error(df_input.iloc[-12:,i], pd.Series(np.zeros((horizon,)))))
    err_t1.append(mean_squared_error(df_input.iloc[-12:,i],df_input.iloc[-24:-24+horizon,i]))
err_sf_pr = pd.Series(err_sf_pr)
err_views = pd.Series(err_views)
err_zero = pd.Series(err_zero)
err_t1 = pd.Series(err_t1)

err_mix = err_views.copy()
err_mix[ind_keep[df_keep_1]] = err_sf_pr.loc[ind_keep[df_keep_1]]
err_mix[ind_keep[df_keep_2]] = err_sf_pr.loc[ind_keep[df_keep_2]]


d_mix = d_b.copy()
d_mix[ind_keep[df_keep_1]] = d_nn[ind_keep[df_keep_1]]
d_mix[ind_keep[df_keep_2]] = d_nn[ind_keep[df_keep_2]]


d_nn = d_nn[~np.isnan(d_nn)]
d_b = d_b[~np.isnan(d_b)]
d_null = d_null[~np.isnan(d_null)]
d_t1= d_t1[~np.isnan(d_t1)]
d_mix = d_mix[~np.isnan(d_mix)]

means = [np.log((d_nn+1)/(d_mix+1)).mean(),np.log((d_b+1)/(d_mix+1)).mean(),np.log((d_null+1)/(d_mix+1)).mean(),np.log((d_t1+1)/(d_mix+1)).mean()]
std_error = [2*np.log((x+1)/(d_mix+1)).std()/np.sqrt(len((x-d_mix))) for x in [d_nn,d_b,d_null,d_t1]]
mean_de = pd.DataFrame({
    'mean': means,
    'std': std_error
})

means = [np.log((err_sf_pr+1)/(err_mix+1)).mean(),np.log((err_views+1)/(err_mix+1)).mean(),np.log((err_zero+1)/(err_mix+1)).mean(),np.log((err_t1+1)/(err_mix+1)).mean()]
std_error = [2*np.log((x+1)/(err_mix+1)).std()/np.sqrt(len((x-err_mix))) for x in [err_sf_pr,err_views,err_zero,err_t1]]
mean_mse = pd.DataFrame({
    'mean': means,
    'std': std_error
})

name=['SF','Views','Null','t-1']
fig,ax = plt.subplots(figsize=(12,8))
for i in range(4):
    plt.scatter(mean_mse["mean"][i],mean_de["mean"][i],color="black",s=150)
    plt.plot([mean_mse["mean"][i],mean_mse["mean"][i]],[mean_de["mean"][i]-mean_de["std"][i],mean_de["mean"][i]+mean_de["std"][i]],linewidth=3,color="black")
    plt.plot([mean_mse["mean"][i]-mean_mse["std"][i],mean_mse["mean"][i]+mean_mse["std"][i]],[mean_de["mean"][i],mean_de["mean"][i]],linewidth=3,color="black")
    plt.text(mean_mse["mean"][i]+0.04, mean_de["mean"][i]+0.005, name[i], size=20, color='black')
plt.scatter(0,0,color="purple",s=150)
plt.text(0.02,0.005, 'SF/Views', size=20, color='purple')
plt.xlabel("Accuracy")
plt.ylabel("Difference explained")
plt.xlim(0.5,-0.05)
plt.ylim(-0.17,0.06)
plt.xticks([])
plt.yticks([])
# Add arrows to the axes
plt.plot(-0.05,-0.17, ls="", marker=">", ms=10, color="k",clip_on=False)
plt.plot(0.5, 0.06, ls="", marker="^", ms=10, color="k", clip_on=False)
ax.spines[['right', 'top']].set_visible(False)
plt.show()



means = [np.log(d_mix+1).mean(),np.log(d_b+1).mean(),np.log(d_null+1).mean(),np.log(d_t1+1).mean(),np.log(d_nn+1).mean()]
std_error = [2*np.log(x+1).std()/np.sqrt(len(x)) for x in [d_mix,d_b,d_null,d_t1,d_nn]]
mean_de = pd.DataFrame({
    'mean': means,
    'std': std_error
})

means = [np.log(err_mix+1).mean(),np.log(err_views+1).mean(),np.log(err_zero+1).mean(),np.log(err_t1+1).mean(),np.log(err_sf_pr+1).mean()]
std_error = [2*np.log(x+1).std()/np.sqrt(len(x)) for x in [err_mix,err_views,err_zero,err_t1,err_sf_pr]]
mean_mse = pd.DataFrame({
    'mean': means,
    'std': std_error
})

name=['SF/Views','Views','Null','t-1','SF']
fig,ax = plt.subplots(figsize=(12,8))
for i in range(5):
    plt.scatter(mean_mse["mean"][i],mean_de["mean"][i],color="black",s=150)
    plt.plot([mean_mse["mean"][i],mean_mse["mean"][i]],[mean_de["mean"][i]-mean_de["std"][i],mean_de["mean"][i]+mean_de["std"][i]],linewidth=3,color="black")
    plt.plot([mean_mse["mean"][i]-mean_mse["std"][i],mean_mse["mean"][i]+mean_mse["std"][i]],[mean_de["mean"][i],mean_de["mean"][i]],linewidth=3,color="black")
    plt.text(mean_mse["mean"][i], mean_de["mean"][i], name[i], size=20, color='black')
plt.xlabel("Accuracy (log MSE)")
plt.ylabel("Difference explained (Log DE)")
plt.show()


for i in range(len(df_input.columns)): 
    if (df_input.iloc[-24:-24+horizon,i]==0).all()==False:
        if mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i])*3<mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]):
            plt.plot(df_input.iloc[-24:-24+horizon,i].reset_index(drop=True),linewidth=2,color='black')
            plt.plot(df_sf_1.iloc[:,i].reset_index(drop=True),linewidth=5,color='purple')
            plt.plot(df_preds_test_1.iloc[:12,i].reset_index(drop=True),linewidth=2,color='grey')
            plt.box(False)
            plt.xticks([])
            plt.yticks([])
            plt.title(f'{df_input.columns[i]} - 2021')
            plt.show()
    if (df_input.iloc[-12:,i]==0).all()==False:
        if mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i])*3<mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]):
            plt.plot(df_input.iloc[-12:,i].reset_index(drop=True),linewidth=2,color='black')
            plt.plot(df_sf_2.iloc[:,i].reset_index(drop=True),linewidth=5,color='purple')
            plt.plot( df_preds_test_2.iloc[:12,i].reset_index(drop=True),linewidth=2,color='grey')
            plt.box(False)
            plt.xticks([])
            plt.yticks([])
            plt.title(f'{df_input.columns[i]} - 2022')
            plt.show()
