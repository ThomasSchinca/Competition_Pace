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
df_input.columns = country_list['name']

df_tot_m = df_input.copy()
df_tot_m.replace(0, np.nan, inplace=True)
df_tot_m = df_tot_m.dropna(axis=1, how='all')
df_tot_m = df_tot_m.fillna(0)

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

#### Scenarios

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
        if len(pd.Series(clusters).value_counts())>7:
            sub_norm = tot_seq[(tot_seq > 5).any(axis=1)].index
            tot_seq_c = tot_seq.copy()
            tot_seq_c.loc[sub_norm,:] = 10
            linkage_matrix = linkage(tot_seq_c, method='ward')
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
        len_mat.append(0)
plot_res_diff(24)

err_sf_pr=[]
err_views=[]

for i in range(len(df_input.columns)):   
    err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], pred_tot_pr[i]))
    err_views.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]))

err_sf_pr = np.array(err_sf_pr)
err_views = np.array(err_views)
test = pd.DataFrame([err_views-err_sf_pr,len_mat])

# for i in range(len(df_input.columns)):
#     if (mean_squared_error(df_input.iloc[-24:-24+horizon,i], pred_tot_min[i])*2<mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:horizon,i])) & (df_input.iloc[-24:-24+horizon,i].sum()>10):
#         plt.plot(df_input.iloc[-24:-24+horizon,i].reset_index(drop=True),label='True')
#         plt.plot(df_preds_test_1.iloc[:horizon,i].reset_index(drop=True),label='Views')
#         plt.plot(pred_tot_min[i],label='SF',marker='s')
#         plt.legend()
#         plt.title(f'Best - 2022 - {df_input.columns[i]}')
#         plt.show()

fig, axes = plt.subplots(ncols=1, nrows=sum((mean_squared_error(df_input.iloc[-12:,i], pred_tot_min[i])*2 <  mean_squared_error(df_input.iloc[-12:,i], df_preds_test_1.iloc[:horizon,i])) & (df_input.iloc[-12:,i].sum() > 10) for i in range(len(df_input.columns))), figsize=(8,40))
subplot_index = 0
for i in range(len(df_input.columns)):
    if (mean_squared_error(df_input.iloc[-12:,i], pred_tot_min[i])*2 < mean_squared_error(df_input.iloc[-12:,i], df_preds_test_1.iloc[:horizon,i])) & (df_input.iloc[-12:,i].sum() > 10):
        ax = axes[subplot_index]
        ax.plot(df_input.iloc[-12:,i].reset_index(drop=True), label='True')
        ax.plot(df_preds_test_1.iloc[:horizon,i].reset_index(drop=True), label='Views')
        ax.plot(pred_tot_min[i], label='SF', marker='s')
        ax.legend()
        ax.set_title(f'Best - 2022 - {df_input.columns[i]}')
        subplot_index += 1
plt.tight_layout()
plt.show()    
        
# for i in range(len(df_input.columns)):
#     if (mean_squared_error(df_input.iloc[-24:-24+horizon,i], pred_tot_min[i])>2*mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:horizon,i])) & (df_input.iloc[-24:-24+horizon,i].sum()>10):
#         plt.plot(df_input.iloc[-24:-24+horizon,i].reset_index(drop=True),label='True')
#         plt.plot(df_preds_test_1.iloc[:horizon,i].reset_index(drop=True),label='Views')
#         plt.plot(pred_tot_min[i],label='SF',marker='s')
#         plt.legend()
#         plt.title(f'Worst - 2022 - {df_input.columns[i]}')
#         plt.show()
               

fig, axes = plt.subplots(ncols=1, nrows=sum((mean_squared_error(df_input.iloc[-12:,i], pred_tot_min[i]) >  2*mean_squared_error(df_input.iloc[-12:,i], df_preds_test_1.iloc[:horizon,i])) & (df_input.iloc[-12:,i].sum() > 10) for i in range(len(df_input.columns))), figsize=(8,40))
subplot_index = 0
for i in range(len(df_input.columns)):
    if (mean_squared_error(df_input.iloc[-12:,i], pred_tot_min[i]) >  2*mean_squared_error(df_input.iloc[-12:,i], df_preds_test_1.iloc[:horizon,i])) & (df_input.iloc[-12:,i].sum() > 10):
        ax = axes[subplot_index]
        ax.plot(df_input.iloc[-12:,i].reset_index(drop=True), label='True')
        ax.plot(df_preds_test_1.iloc[:horizon,i].reset_index(drop=True), label='Views')
        ax.plot(pred_tot_min[i], label='SF', marker='s')
        ax.legend()
        ax.set_title(f'Best - 2022 - {df_input.columns[i]}')
        subplot_index += 1
plt.tight_layout()
plt.show()    


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
        if len(pd.Series(clusters).value_counts())>7:
            sub_norm = tot_seq[(tot_seq > 5).any(axis=1)].index
            tot_seq_c = tot_seq.copy()
            tot_seq_c.loc[sub_norm,:] = 10
            linkage_matrix = linkage(tot_seq_c, method='ward')
            clusters = fcluster(linkage_matrix, horizon/2, criterion='distance')
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
plot_res_diff(12)

err_sf_pr=[]
err_views=[]

for i in range(len(df_input.columns)):   
    err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], pred_tot_pr[i]))
    err_views.append(mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]))

err_sf_pr = np.array(err_sf_pr)
err_views = np.array(err_views)
test = pd.Series(err_views-err_sf_pr)

# for i in range(len(df_input.columns)):
#     if (mean_squared_error(df_input.iloc[-12:,i], pred_tot_min[i])*2<mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:horizon,i])) & (df_input.iloc[-12:,i].sum()>10):
#         plt.plot(df_input.iloc[-12:,i].reset_index(drop=True),label='True')
#         plt.plot(df_preds_test_2.iloc[:horizon,i].reset_index(drop=True),label='Views')
#         plt.plot(pred_tot_min[i],label='SF',marker='s')
#         plt.legend()
#         plt.title(f'Best - 2023 - {df_input.columns[i]}')
#         plt.show()
        
fig, axes = plt.subplots(ncols=1, nrows=sum((mean_squared_error(df_input.iloc[-12:,i], pred_tot_min[i])*2 <  mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:horizon,i])) & (df_input.iloc[-12:,i].sum() > 10) for i in range(len(df_input.columns))), figsize=(8,40))
subplot_index = 0
for i in range(len(df_input.columns)):
    if (mean_squared_error(df_input.iloc[-12:,i], pred_tot_min[i])*2 < mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:horizon,i])) & (df_input.iloc[-12:,i].sum() > 10):
        ax = axes[subplot_index]
        ax.plot(df_input.iloc[-12:,i].reset_index(drop=True), label='True')
        ax.plot(df_preds_test_2.iloc[:horizon,i].reset_index(drop=True), label='Views')
        ax.plot(pred_tot_min[i], label='SF', marker='s')
        ax.legend()
        ax.set_title(f'Best - 2023 - {df_input.columns[i]}')
        subplot_index += 1
plt.tight_layout()
plt.show()    
        
# for i in range(len(df_input.columns)):
#     if (mean_squared_error(df_input.iloc[-12:,i], pred_tot_min[i])>2*mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:horizon,i])) & (df_input.iloc[-12:,i].sum()>10):
#         plt.plot(df_input.iloc[-12:,i].reset_index(drop=True),label='True')
#         plt.plot(df_preds_test_2.iloc[:horizon,i].reset_index(drop=True),label='Views')
#         plt.plot(pred_tot_min[i],label='SF',marker='s')
#         plt.legend()
#         plt.title(f'Worst - 2023 - {df_input.columns[i]}')
#         plt.show()

fig, axes = plt.subplots(ncols=1, nrows=sum((mean_squared_error(df_input.iloc[-12:,i], pred_tot_min[i]) > 2 * mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:horizon,i])) & (df_input.iloc[-12:,i].sum() > 10) for i in range(len(df_input.columns))), figsize=(10, 30))
subplot_index = 0
for i in range(len(df_input.columns)):
    if (mean_squared_error(df_input.iloc[-12:,i], pred_tot_min[i]) > 2 * mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:horizon,i])) & (df_input.iloc[-12:,i].sum() > 10):
        ax = axes[subplot_index]
        ax.plot(df_input.iloc[-12:,i].reset_index(drop=True), label='True')
        ax.plot(df_preds_test_2.iloc[:horizon,i].reset_index(drop=True), label='Views')
        ax.plot(pred_tot_min[i], label='SF', marker='s')
        ax.legend()
        ax.set_title(f'Worst - 2023 - {df_input.columns[i]}')
        subplot_index += 1
plt.tight_layout()
plt.show()               
    
    
def plot_res_diff(start):    
    err_sf=[]
    err_sf_pr=[]
    err_views=[]
    err_zero=[]
    err_t1=[]
    for i in range(len(df_input.columns)):
        if start==12:
            err_sf.append(mean_squared_error(df_input.iloc[-start:,i], pred_tot_min[i]))
            err_sf_pr.append(mean_squared_error(df_input.iloc[-start:,i], pred_tot_pr[i]))
            err_views.append(mean_squared_error(df_input.iloc[-start:,i], df_preds_test_2.iloc[:horizon,i]))
            err_zero.append(mean_squared_error(df_input.iloc[-start:,i], pd.Series(np.zeros((horizon,)))))
            err_t1.append(mean_squared_error(df_input.iloc[-start:,i], df_input.iloc[-start-horizon:-start,i]))
        else:
            err_sf.append(mean_squared_error(df_input.iloc[-start:-start+horizon,i], pred_tot_min[i]))
            err_sf_pr.append(mean_squared_error(df_input.iloc[-start:-start+horizon,i], pred_tot_pr[i]))
            err_views.append(mean_squared_error(df_input.iloc[-start:-start+horizon,i], df_preds_test_1.iloc[:horizon,i]))
            err_zero.append(mean_squared_error(df_input.iloc[-start:-start+horizon,i], pd.Series(np.zeros((horizon,)))))
            err_t1.append(mean_squared_error(df_input.iloc[-start:-start+horizon,i], df_input.iloc[-start-horizon:-start,i]))

    err_sf = np.log(np.array(err_sf)+1)
    err_sf_pr = np.log(np.array(err_sf_pr)+1)
    err_views = np.log(np.array(err_views)+1)
    err_zero = np.log(np.array(err_zero)+1)
    err_t1 = np.log(np.array(err_t1)+1)
    
    means = [(err_views-err_sf).mean(),(err_zero-err_sf).mean(),(err_t1-err_sf).mean()]
    std_error = [2*(x-err_sf).std()/np.sqrt(len((x-err_sf))) for x in [err_views,err_zero,err_t1]]
    mean_mse = pd.DataFrame({
        'mean': means,
        'std': std_error
    })
    
    means = [(err_views-err_sf_pr).mean(),(err_zero-err_sf_pr).mean(),(err_t1-err_sf_pr).mean()]
    std_error = [2*(x-err_sf_pr).std()/np.sqrt(len((x-err_sf_pr))) for x in [err_views,err_zero,err_t1]]
    mean_mse1 = pd.DataFrame({
        'mean': means,
        'std': std_error
    })
         
    # Difference explained
    d_nn=[]
    d_nn1=[]
    d_b=[]
    d_null=[]
    d_t1=[]
    
    k=5
    for i in range(len(df_input.columns)):
        if start==12:
            real = df_input.iloc[-start:,i]
        else:
            real = df_input.iloc[-start:-start+horizon,i]
        real=real.reset_index(drop=True)
        sf=pred_tot_min[i]
        sf=sf.reset_index(drop=True)
        sf1=pred_tot_pr[i]
        sf1=sf1.reset_index(drop=True)
        if start==12:
            b1=df_preds_test_2.iloc[:horizon,i]
        else:
            b1=df_preds_test_1.iloc[:horizon,i]
        b1=b1.reset_index(drop=True)
        null=pd.Series(np.zeros((horizon,)))
        null=null.reset_index(drop=True)
        t1=df_input.iloc[-start-horizon:-start,i]
        t1=t1.reset_index(drop=True)       
         
        max_s=0
        max_s1=0
        max_b1=0
        max_null=0
        max_t1=0
    
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
                    
                    
                    
                    max_exp=0
                    if (real[value]-real[value-1])/(sf1[value]-sf1[value-1])>0 and sf1[value]-sf1[value-1] != 0:
                        t=abs(((real[value]-real[value-1])-(sf1[value]-sf1[value-1]))/(real[value]-real[value-1]))
                        max_exp = np.exp(-(k*t))
                    else:
                        if value==horizon-1:
                            if (real[value]-real[value-1])/(sf1[value-1]-sf1[value-2])>0 and sf1[value-1]-sf1[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(sf1[value-1]-sf1[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                        elif value==1:
                            if (real[value]-real[value-1])/(sf1[value+1]-sf1[value])>0 and sf1[value+1]-sf1[value] != 0:
                                t=abs(((real[value]-real[value-1])-(sf1[value+1]-sf1[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                        else : 
                            if (real[value]-real[value-1])/(sf1[value-1]-sf1[value-2])>0 and sf1[value-1]-sf1[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(sf1[value-1]-sf1[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                            if (real[value]-real[value-1])/(sf1[value+1]-sf1[value])>0 and sf1[value+1]-sf1[value] != 0:
                                t=abs(((real[value]-real[value-1])-(sf1[value+1]-sf1[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                    max_s1=max_s1+max_exp 
    
    
    
                    
                    
                    max_exp=0
                    if (real[value]-real[value-1])/(b1[value]-b1[value-1])>0 and b1[value]-b1[value-1] != 0:
                        t=abs(((real[value]-real[value-1])-(b1[value]-b1[value-1]))/(real[value]-real[value-1]))
                        max_exp = np.exp(-(k*t))
                    else:
                        if value==horizon-1:
                            if (real[value]-real[value-1])/(b1[value-1]-b1[value-2])>0 and b1[value-1]-b1[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(b1[value-1]-b1[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                     max_exp = np.exp(-(k*t))
                        elif value==1:
                            if (real[value]-real[value-1])/(b1[value+1]-b1[value])>0 and b1[value+1]-b1[value] != 0:
                                t=abs(((real[value]-real[value-1])-(b1[value+1]-b1[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                        else : 
                            if (real[value]-real[value-1])/(b1[value-1]-b1[value-2])>0 and b1[value-1]-b1[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(b1[value-1]-b1[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                            if (real[value]-real[value-1])/(b1[value+1]-b1[value])>0 and b1[value+1]-b1[value] != 0:
                                t=abs(((real[value]-real[value-1])-(b1[value+1]-b1[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                    max_b1=max_b1+max_exp 
    
    
    
                    
    
                    max_exp=0
                    if (real[value]-real[value-1])/(null[value]-null[value-1])>0 and null[value]-null[value-1] != 0:
                        t=abs(((real[value]-real[value-1])-(null[value]-null[value-1]))/(real[value]-real[value-1]))
                        max_exp = np.exp(-(k*t))
                    else:
                        if value==horizon-1:
                            if (real[value]-real[value-1])/(null[value-1]-null[value-2])>0 and null[value-1]-null[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(null[value-1]-null[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                     max_exp = np.exp(-(k*t))
                        elif value==1:
                            if (real[value]-real[value-1])/(null[value+1]-null[value])>0 and null[value+1]-null[value] != 0:
                                t=abs(((real[value]-real[value-1])-(null[value+1]-null[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                        else : 
                            if (real[value]-real[value-1])/(null[value-1]-null[value-2])>0 and null[value-1]-null[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(null[value-1]-null[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                            if (real[value]-real[value-1])/(null[value+1]-null[value])>0 and null[value+1]-null[value] != 0:
                                t=abs(((real[value]-real[value-1])-(null[value+1]-null[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                    max_null=max_null+max_exp 
    
    
    
                    max_exp=0
                    if (real[value]-real[value-1])/(t1[value]-t1[value-1])>0 and t1[value]-t1[value-1] != 0:
                        t=abs(((real[value]-real[value-1])-(t1[value]-t1[value-1]))/(real[value]-real[value-1]))
                        max_exp = np.exp(-(k*t))
                    else:
                        if value==horizon-1:
                            if (real[value]-real[value-1])/(t1[value-1]-t1[value-2])>0 and t1[value-1]-t1[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(t1[value-1]-t1[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                     max_exp = np.exp(-(k*t))
                        elif value==1:
                            if (real[value]-real[value-1])/(t1[value+1]-t1[value])>0 and t1[value+1]-t1[value] != 0:
                                t=abs(((real[value]-real[value-1])-(t1[value+1]-t1[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                        else : 
                            if (real[value]-real[value-1])/(t1[value-1]-t1[value-2])>0 and t1[value-1]-t1[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(t1[value-1]-t1[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                            if (real[value]-real[value-1])/(t1[value+1]-t1[value])>0 and t1[value+1]-t1[value] != 0:
                                t=abs(((real[value]-real[value-1])-(t1[value+1]-t1[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                    max_t1=max_t1+max_exp     
        
        d_nn.append(max_s)
        d_nn1.append(max_s1)
        d_b.append(max_b1)
        d_null.append(max_null)
        d_t1.append(max_t1)
    d_nn = np.log(np.array(d_nn)+1)
    d_nn1 = np.log(np.array(d_nn1)+1)
    d_b = np.log(np.array(d_b)+1)
    d_null = np.log(np.array(d_null)+1)
    d_t1 = np.log(np.array(d_t1)+1)
    
    means = [(d_b-d_nn).mean(),(d_null-d_nn).mean(),(d_t1-d_nn).mean()]
    std_error = [2*(x-d_nn).std()/np.sqrt(len((x-d_nn))) for x in [d_b,d_null,d_t1]]
    mean_de = pd.DataFrame({
        'mean': means,
        'std': std_error
    })
    
    means = [(d_b-d_nn1).mean(),(d_null-d_nn1).mean(),(d_t1-d_nn1).mean()]
    std_error = [2*(x-d_nn1).std()/np.sqrt(len((x-d_nn1))) for x in [d_b,d_null,d_t1]]
    mean_de1 = pd.DataFrame({
        'mean': means,
        'std': std_error
    })
    
    name=['Views','Null','t-1']
    fig,ax = plt.subplots(figsize=(12,8))
    for i in range(3):
        plt.scatter(mean_mse["mean"][i],mean_de["mean"][i],color="black",s=150)
        plt.plot([mean_mse["mean"][i],mean_mse["mean"][i]],[mean_de["mean"][i]-mean_de["std"][i],mean_de["mean"][i]+mean_de["std"][i]],linewidth=3,color="black")
        plt.plot([mean_mse["mean"][i]-mean_mse["std"][i],mean_mse["mean"][i]+mean_mse["std"][i]],[mean_de["mean"][i],mean_de["mean"][i]],linewidth=3,color="black")
        plt.text(mean_mse["mean"][i], mean_de["mean"][i], name[i], size=20, color='black')
        plt.scatter(mean_mse1["mean"][i],mean_de1["mean"][i],color="blue",s=150)
        plt.plot([mean_mse1["mean"][i],mean_mse1["mean"][i]],[mean_de1["mean"][i]-mean_de1["std"][i],mean_de1["mean"][i]+mean_de1["std"][i]],linewidth=3,color="blue")
        plt.plot([mean_mse1["mean"][i]-mean_mse1["std"][i],mean_mse1["mean"][i]+mean_mse1["std"][i]],[mean_de1["mean"][i],mean_de1["mean"][i]],linewidth=3,color="blue")
        plt.text(mean_mse1["mean"][i], mean_de1["mean"][i], name[i], size=20, color='blue')
    plt.xlabel("Accuracy (log ratio MSE)")
    plt.ylabel("Difference explained (Log ratio DE)")
    if start==12:
        plt.title('Test 2023')
    else:
        plt.title('Test 2022')
    plt.hlines(0,-0.3,0.3,linestyle='--')
    plt.vlines(0,-0.2,0.05,linestyle='--')
    plt.show()
    
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        