# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 13:20:40 2024

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
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from cycler import cycler
from sklearn.metrics import mean_absolute_percentage_error

plot_params = {"text.usetex":True,"font.family":"serif","font.size":20,"xtick.labelsize":20,"ytick.labelsize":20,"axes.labelsize":20,"figure.titlesize":20,"figure.figsize":(8,5),"axes.prop_cycle":cycler(color=['black','rosybrown','gray','indianred','red','maroon','silver',])}
plt.rcParams.update(plot_params)

def diff_explained(df_input,pred,k=5,horizon=12):
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
    

df_input=pd.read_csv('df_input.csv',index_col=0,parse_dates=True)
df_tot_m = df_input.copy()
df_tot_m.replace(0, np.nan, inplace=True)
df_tot_m = df_tot_m.dropna(axis=1, how='all')
df_tot_m = df_tot_m.fillna(0)
country_list = pd.read_csv('country_list.csv',index_col=0)
df_conf=pd.read_csv('reg_coun.csv',index_col=0)
df_conf=pd.Series(df_conf.region)
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
df_obs_1 = df_input.iloc[-24:-12,:] 
df_obs_2 = df_input.iloc[-12:,:]  

# =============================================================================
# Test kNN like model
# =============================================================================

with open('test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
h_train=10
df_input_sub=df_input.iloc[:-24]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
        l_find=dict_m[df_input.columns[coun]]          
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)            
            if date+horizon<len(df_tot_m.iloc[:-24]):                              
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)               
                seq = (seq - mi) / (ma - mi)                                
                pred_seq.append(seq.tolist())        
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()   
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
        
df_sf_1 = pd.concat(pred_tot_pr,axis=1)
df_sf_1.columns=country_list['name']

df_sf_1_tot=[]
for k_min in [1,3,5]:
    pred_tot_min=[]
    pred_tot_pr=[]
    df_input_sub=df_input.iloc[:-24]
    for coun in range(len(df_input_sub.columns)):
        if not (df_input_sub.iloc[-h_train:,coun]==0).all():
            inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
            l_find=dict_m[df_input.columns[coun]]          
            tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
            pred_seq=[]
            if (df_input_sub.columns[coun] in ['Peru','Burkina Faso','Cameroon','Tunisia','Ukraine','Somalia','Armenia','Iran','Afghanistan','Kyrgyzstan','Myanmar','Thailand','Mozambique','Congo, RDC','Indonesia','Israel','Syria','India','Kenya','Sudan']) & k_min==1:
                k_min=2
            for col,last_date,mi,ma,somme in tot_seq[:k_min]:
                date=df_tot_m.iloc[:-24].index.get_loc(last_date)            
                if date+horizon<len(df_tot_m.iloc[:-24]):                              
                    seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)               
                    seq = (seq - mi) / (ma - mi)                                
                    pred_seq.append(seq.tolist())        
            tot_seq=pd.DataFrame(pred_seq)
            pred_ori=tot_seq.mean(axis=0)
            preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
            pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        else:
            pred_tot_min.append(pd.Series(np.zeros((horizon,))))
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
            
    df_sf_1_sub = pd.concat(pred_tot_pr,axis=1)
    df_sf_1_sub.columns=country_list['name']
    df_sf_1_tot.append(df_sf_1_sub)
 
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
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)       
            if date+horizon<len(df_tot_m.iloc[:-12]):               
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)              
                pred_seq.append(seq.tolist())                
        tot_seq=pd.DataFrame(pred_seq)       
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:     
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))  
    
df_sf_2= pd.concat(pred_tot_pr,axis=1)
df_sf_2.columns=country_list['name']    
 
df_sf_2_tot=[]
for k_min in [1,3,5]:
    pred_tot_min=[]
    pred_tot_pr=[]
    df_input_sub=df_input.iloc[:-12]
    for coun in range(len(df_input_sub.columns)):
        if not (df_input_sub.iloc[-h_train:,coun]==0).all():
            inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
            l_find=dict_m[df_input.columns[coun]]          
            tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
            pred_seq=[]
            if (df_input_sub.columns[coun] in ['Brazil','Peru','Mali','Iraq','Burkina Faso','Cameroon','Turkey','Uzbekistan','Yemen','Azerbaijan','Iran','Kyrgyzstan','Tajikistan','Burundi','South Africa','Mozambique','Papua New Guinea','Libya','Israel','Syria','Egypt','Morocco','South Sudan']) & k_min==1:
                k_min=2
            for col,last_date,mi,ma,somme in tot_seq[:k_min]:
                date=df_tot_m.iloc[:-12].index.get_loc(last_date)            
                if date+horizon<len(df_tot_m.iloc[:-12]):                              
                    seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)               
                    seq = (seq - mi) / (ma - mi)                                
                    pred_seq.append(seq.tolist())        
            tot_seq=pd.DataFrame(pred_seq)
            pred_ori=tot_seq.mean(axis=0)
            preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
            pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        else:
            pred_tot_min.append(pd.Series(np.zeros((horizon,))))
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
            
    df_sf_2_sub = pd.concat(pred_tot_pr,axis=1)
    df_sf_2_sub.columns=country_list['name']
    df_sf_2_tot.append(df_sf_2_sub)  
 
   
mse_list_sub=[]
for n_k,k_min in enumerate([1,3,5]):
    err_sf_pr=[]
    err_sub=[]
    for i in range(len(df_input.columns)):   
        err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i]))
        err_sub.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1_tot[n_k].iloc[:,i]))
        err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i]))
        err_sub.append(mean_squared_error(df_input.iloc[-12:,i], df_sf_2_tot[n_k].iloc[:,i]))
    err_sf_pr = np.array(err_sf_pr)
    err_sub = np.array(err_sub)
    mse_list_sub.append(np.log((err_sub+1)/(err_sf_pr+1)))
    
d_nn = diff_explained(df_input.iloc[-24:-24+horizon],df_sf_1)
d_nn2 = diff_explained(df_input.iloc[-12:],df_sf_2)
d_nn = np.concatenate([d_nn,d_nn2])    
    
de_list_sub=[]
for n_k,k_min in enumerate([1,3,5]):
    d_sub = diff_explained(df_input.iloc[-24:-24+horizon],df_sf_1_tot[n_k])
    d_sub2 = diff_explained(df_input.iloc[-12:],df_sf_2_tot[n_k])
    d_sub = np.concatenate([d_sub,d_sub2])
    de_list_sub.append(np.log((d_nn+1)/(d_sub+1)))
 
# means = [arr.mean() for arr in mse_list_sub]
# cis = [stats.sem(arr, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(arr)-1) for arr in mse_list_sub]
# x_ticks = ["Closest Pattern", "kNN-3", "kNN-5"]
# plt.figure(figsize=(8, 5))
# plt.errorbar(x=x_ticks, y=means, yerr=np.squeeze(cis), fmt='o', markersize=8, color='black', ecolor='black')
# plt.xlabel("Arrays")
# plt.ylabel("Mean Values with 95% CI")
# plt.title("Mean and 95% Confidence Interval for Different Configuration")
# plt.axhline(0,linestyle='--')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# means = [arr.mean() for arr in de_list_sub]
# cis = [stats.sem(arr, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(arr)-1) for arr in de_list_sub]
# x_ticks = ["Closest Pattern", "kNN-3", "kNN-5"]
# plt.figure(figsize=(8, 5))
# plt.errorbar(x=x_ticks, y=means, yerr=np.squeeze(cis), fmt='o', markersize=8, color='black', ecolor='black')
# plt.xlabel("Arrays")
# plt.ylabel("Mean Values with 95% CI")
# plt.title("Mean and 95% Confidence Interval for Different Configuration - DE")
# plt.axhline(0,linestyle='--')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# =============================================================================
# Different cut-off for clusterings
# =============================================================================

with open('test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)

horizon=12
h_train=10

for cut in [2,4,5]:
    pred_tot_min=[]
    pred_tot_pr=[]
    df_input_sub=df_input.iloc[:-24]
    for coun in range(len(df_input_sub.columns)):
        if not (df_input_sub.iloc[-h_train:,coun]==0).all():
            inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
            l_find=dict_m[df_input.columns[coun]]          
            tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
            pred_seq=[]
            for col,last_date,mi,ma,somme in tot_seq:
                date=df_tot_m.iloc[:-24].index.get_loc(last_date)            
                if date+horizon<len(df_tot_m.iloc[:-24]):                              
                    seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)               
                    seq = (seq - mi) / (ma - mi)                                
                    pred_seq.append(seq.tolist())        
            tot_seq=pd.DataFrame(pred_seq)
            linkage_matrix = linkage(tot_seq, method='ward')
            clusters = fcluster(linkage_matrix, horizon/cut, criterion='distance')
            tot_seq['Cluster'] = clusters
            val_sce = tot_seq.groupby('Cluster').mean()   
            pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
            pred_ori=val_sce.loc[pr==pr.max(),:]
            pred_ori=pred_ori.mean(axis=0)
            preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
            pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        else:
            pred_tot_min.append(pd.Series(np.zeros((horizon,))))
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
            
    df_sf_1_sub = pd.concat(pred_tot_pr,axis=1)
    df_sf_1_sub.columns=country_list['name']
    df_sf_1_tot.append(df_sf_1_sub)           
            
            
with open('test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 

horizon=12
for cut in [2,4,5]:
    df_input_sub=df_input.iloc[:-12]
    pred_tot_min=[]
    pred_tot_pr=[]

    for coun in range(len(df_input_sub.columns)):
        if not (df_input_sub.iloc[-h_train:,coun]==0).all():
            inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
            l_find=dict_m[df_input.columns[coun]]
            tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]           
            pred_seq=[]                     
            for col,last_date,mi,ma,somme in tot_seq:
                date=df_tot_m.iloc[:-12].index.get_loc(last_date)       
                if date+horizon<len(df_tot_m.iloc[:-12]):               
                    seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                    seq = (seq - mi) / (ma - mi)              
                    pred_seq.append(seq.tolist())                
            tot_seq=pd.DataFrame(pred_seq)       
            linkage_matrix = linkage(tot_seq, method='ward')
            clusters = fcluster(linkage_matrix, horizon/cut, criterion='distance')
            tot_seq['Cluster'] = clusters
            val_sce = tot_seq.groupby('Cluster').mean()
            pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
            pred_ori=val_sce.loc[pr==pr.max(),:]
            pred_ori=pred_ori.mean(axis=0)
            preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
            pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        else:     
            pred_tot_min.append(pd.Series(np.zeros((horizon,))))
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))  


    df_sf_2_sub = pd.concat(pred_tot_pr,axis=1)
    df_sf_2_sub.columns=country_list['name']
    df_sf_2_tot.append(df_sf_2_sub)  

    
# =============================================================================
# Diff thres 
# =============================================================================

# dict_m={i :[] for i in df_input.columns} 
# df_input_sub=df_input.iloc[:-24]
# for coun in range(len(df_input_sub.columns)):
#     if not (df_input_sub.iloc[-h_train:,coun]==0).all():
#         shape = Shape()
#         shape.set_shape(df_input_sub.iloc[-h_train:,coun]) 
#         find = finder(df_tot_m.iloc[:-24],shape)
#         find.find_patterns_keep_all(select=True,metric='dtw',dtw_sel=2)
#         dict_m[df_input.columns[coun]]=find.sequences
#     else :
#         pass
# with open('test1_keep_all.pkl', 'wb') as f:
#     pickle.dump(dict_m, f) 
    
# dict_m={i :[] for i in df_input.columns} 
# df_input_sub=df_input.iloc[:-12]
# for coun in range(len(df_input_sub.columns)):
#     if not (df_input_sub.iloc[-h_train:,coun]==0).all():
#         shape = Shape()
#         shape.set_shape(df_input_sub.iloc[-h_train:,coun]) 
#         find = finder(df_tot_m.iloc[:-12],shape)
#         find.find_patterns_keep_all(select=True,metric='dtw',dtw_sel=2)
#         dict_m[df_input.columns[coun]]=find.sequences
#     else :
#         pass
# with open('test2_keep_all.pkl', 'wb') as f:
#     pickle.dump(dict_m, f) 

with open('test1_keep_all.pkl', 'rb') as f:
    dict_m = pickle.load(f)
for thres in [0.2,0.3,0.4,0.5,0.75,1]:
    pred_tot_min=[]
    pred_tot_pr=[]
    df_input_sub=df_input.iloc[:-24]
    for coun in range(len(df_input_sub.columns)):
        if not (df_input_sub.iloc[-h_train:,coun]==0).all():
            inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
            l_find=dict_m[df_input.columns[coun]]     
            tot_seq = []
            for series, weight in l_find:
                if weight<thres:
                    tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum()])
            if len(tot_seq)<5:
                tot_seq = []
                for series, weight in l_find[:5]:
                    tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum()])
            pred_seq=[]
            for col,last_date,mi,ma,somme in tot_seq:
                date=df_tot_m.iloc[:-24].index.get_loc(last_date)            
                if date+horizon<len(df_tot_m.iloc[:-24]):                              
                    seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)               
                    seq = (seq - mi) / (ma - mi)                                
                    pred_seq.append(seq.tolist())        
            tot_seq=pd.DataFrame(pred_seq)
            tot_seq = tot_seq.dropna()
            linkage_matrix = linkage(tot_seq, method='ward')
            clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
            tot_seq['Cluster'] = clusters
            val_sce = tot_seq.groupby('Cluster').mean()   
            pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
            pred_ori=val_sce.loc[pr==pr.max(),:]
            pred_ori=pred_ori.mean(axis=0)
            preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
            pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        else:
            pred_tot_min.append(pd.Series(np.zeros((horizon,))))
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
            
    df_sf_1_sub = pd.concat(pred_tot_pr,axis=1)
    df_sf_1_sub.columns=country_list['name']
    df_sf_1_tot.append(df_sf_1_sub)
    
with open('test2_keep_all.pkl', 'rb') as f:
    dict_m = pickle.load(f)
for thres in [0.2,0.3,0.4,0.5,0.75,1]:
    pred_tot_min=[]
    pred_tot_pr=[]
    df_input_sub=df_input.iloc[:-12]
    for coun in range(len(df_input_sub.columns)):
        if not (df_input_sub.iloc[-h_train:,coun]==0).all():
            inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
            l_find=dict_m[df_input.columns[coun]]     
            tot_seq = []
            for series, weight in l_find:
                if weight<thres:
                    tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum()])
            if len(tot_seq)<5:
                tot_seq = []
                for series, weight in l_find[:5]:
                    tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum()])
            pred_seq=[]
            for col,last_date,mi,ma,somme in tot_seq:
                date=df_tot_m.iloc[:-12].index.get_loc(last_date)            
                if date+horizon<len(df_tot_m.iloc[:-12]):                              
                    seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)               
                    seq = (seq - mi) / (ma - mi)                                
                    pred_seq.append(seq.tolist())        
            tot_seq=pd.DataFrame(pred_seq)
            tot_seq = tot_seq.dropna()
            linkage_matrix = linkage(tot_seq, method='ward')
            clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
            tot_seq['Cluster'] = clusters
            val_sce = tot_seq.groupby('Cluster').mean()   
            pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
            pred_ori=val_sce.loc[pr==pr.max(),:]
            pred_ori=pred_ori.mean(axis=0)
            preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
            pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        else:
            pred_tot_min.append(pd.Series(np.zeros((horizon,))))
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
            
    df_sf_2_sub = pd.concat(pred_tot_pr,axis=1)
    df_sf_2_sub.columns=country_list['name']
    df_sf_2_tot.append(df_sf_2_sub)
    
    
# mse_list_sub=[]
# for n_k in range(3,len(df_sf_2_tot)):
#     err_sf_pr=[]
#     err_sub=[]
#     for i in range(len(df_input.columns)):   
#         err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i]))
#         err_sub.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1_tot[n_k].iloc[:,i]))
#         err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i]))
#         err_sub.append(mean_squared_error(df_input.iloc[-12:,i], df_sf_2_tot[n_k].iloc[:,i]))
#     err_sf_pr = np.array(err_sf_pr)
#     err_sub = np.array(err_sub)
#     mse_list_sub.append(np.log((err_sub+1)/(err_sf_pr+1)))

# means = [arr.mean() for arr in mse_list_sub]
# cis = [stats.sem(arr, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(arr)-1) for arr in mse_list_sub]
# x_ticks = [0.2,0.3,0.4,0.5,0.75,1]
# plt.figure(figsize=(8, 5))
# plt.errorbar(x=x_ticks, y=means, yerr=np.squeeze(cis), fmt='o', markersize=8, color='black', ecolor='black')
# plt.xlabel("Arrays")
# plt.ylabel("Mean Values with 95% CI")
# plt.title("Mean and 95% Confidence Interval for Different Configuration - MSE")
# plt.axhline(0,linestyle='--')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# de_list_sub=[]
# for n_k in range(3,len(df_sf_2_tot)):
#     d_sub = diff_explained(df_input.iloc[-24:-24+horizon],df_sf_1_tot[n_k])
#     d_sub2 = diff_explained(df_input.iloc[-12:],df_sf_2_tot[n_k])
#     d_sub = np.concatenate([d_sub,d_sub2])
#     de_list_sub.append(np.log((d_nn+1)/(d_sub+1)))
    
# means = [arr.mean() for arr in de_list_sub]
# cis = [stats.sem(arr, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(arr)-1) for arr in de_list_sub]
# x_ticks = [0.2,0.3,0.4,0.5,0.75,1]
# plt.figure(figsize=(8, 5))
# plt.errorbar(x=x_ticks, y=means, yerr=np.squeeze(cis), fmt='o', markersize=8, color='black', ecolor='black')
# plt.xlabel("Arrays")
# plt.ylabel("Mean Values with 95% CI")
# plt.title("Mean and 95% Confidence Interval for Different Configuration - DE")
# plt.axhline(0,linestyle='--')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()


with open('test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
h_train=10
df_input_sub=df_input.iloc[:-24]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
        l_find=dict_m[df_input.columns[coun]]          
        tot_seq = []
        for series, weight in l_find:
            if weight==0:
                tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum(),1/0.0001])
            else:
                tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum(),1/weight])        
        pred_seq=[]
        w_list=[]
        for col,last_date,mi,ma,somme,wei in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)            
            if date+horizon<len(df_tot_m.iloc[:-24]):                              
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)               
                seq = (seq - mi) / (ma - mi)                                
                pred_seq.append(seq.tolist())     
                w_list.append(wei)                
        tot_seq=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        tot_seq['weights'] = w_list
        val_sce = tot_seq.groupby('Cluster').apply(lambda x: (x.iloc[:, :12].T * x['weights']).sum(axis=1) / x['weights'].sum())
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))

df_sf_1_w_clu = pd.concat(pred_tot_pr,axis=1)
df_sf_1_w_clu.columns=country_list['name']
df_sf_1_w_clu = df_sf_1_w_clu.fillna(0)


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
        tot_seq = []
        for series, weight in l_find:
            if weight==0:
                tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum(),1/0.0001])
            else:
                tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum(),1/weight])        
        pred_seq=[]
        w_list=[]        
        for col,last_date,mi,ma,somme,wei in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)       
            if date+horizon<len(df_tot_m.iloc[:-12]):               
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)              
                pred_seq.append(seq.tolist()) 
                w_list.append(wei)                
        tot_seq=pd.DataFrame(pred_seq)       
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        tot_seq['weights'] = w_list
        val_sce = tot_seq.groupby('Cluster').apply(lambda x: (x.iloc[:, :12].T * x['weights']).sum(axis=1) / x['weights'].sum())
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:     
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))  
    
df_sf_2_w_clu = pd.concat(pred_tot_pr,axis=1)
df_sf_2_w_clu.columns=country_list['name']
df_sf_2_w_clu = df_sf_2_w_clu.fillna(0)

# =============================================================================
# Chadefaux comp
# =============================================================================

with open('test1_keep_all.pkl', 'rb') as f:
    dict_m = pickle.load(f)

pred_tot_min=[]
pred_tot_pr=[]
df_input_sub=df_input.iloc[:-24]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
        l_find=dict_m[df_input.columns[coun]]     
        tot_seq = []
        for series, weight in l_find:
            if weight==0:
                tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum(),1/0.0001])
            else:
                tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum(),1/weight])
        pred_seq=[]
        w_list=[]
        for col,last_date,mi,ma,somme,wei in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)            
            if date+horizon<len(df_tot_m.iloc[:-24]):                              
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)               
                seq = (seq - mi) / (ma - mi)                                
                pred_seq.append(seq.tolist())  
                w_list.append(wei)
        tot_seq=pd.DataFrame(pred_seq)
        weights = np.array(w_list).reshape(-1, 1)
        weights/weights.sum()
        pred_ori = np.average(tot_seq, axis=0, weights=weights.flatten())
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pd.Series(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()))
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
        
df_sf_1_chad = pd.concat(pred_tot_pr,axis=1)
df_sf_1_chad.columns=country_list['name']
df_sf_1_chad = df_sf_1_chad.fillna(0)
    
with open('test2_keep_all.pkl', 'rb') as f:
    dict_m = pickle.load(f)
pred_tot_min=[]
pred_tot_pr=[]
df_input_sub=df_input.iloc[:-12]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
        l_find=dict_m[df_input.columns[coun]]     
        tot_seq = []
        for series, weight in l_find:
            if weight==0:
                tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum(),1/0.0001])
            else:
                tot_seq.append([series.name, series.index[-1],series.min(),series.max(),series.sum(),1/weight])
        pred_seq=[]
        w_list=[]
        for col,last_date,mi,ma,somme,wei in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)            
            if date+horizon<len(df_tot_m.iloc[:-12]):                              
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)               
                seq = (seq - mi) / (ma - mi)                                
                pred_seq.append(seq.tolist())        
                w_list.append(wei)
        tot_seq=pd.DataFrame(pred_seq)
        weights = np.array(w_list).reshape(-1, 1)
        weights/weights.sum()
        pred_ori = np.average(tot_seq, axis=0, weights=weights.flatten())
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pd.Series(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()))
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
        
df_sf_2_chad = pd.concat(pred_tot_pr,axis=1)
df_sf_2_chad.columns=country_list['name']
df_sf_2_chad = df_sf_2_chad.fillna(0)


# err_sf_pr=[]
# err_sub=[]
# for i in range(len(df_input.columns)):   
#     err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i]))
#     err_sub.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1_chad.iloc[:,i]))
#     err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i]))
#     err_sub.append(mean_squared_error(df_input.iloc[-12:,i], df_sf_2_chad.iloc[:,i]))
# err_sf_pr = np.array(err_sf_pr)
# err_sub = np.array(err_sub)
# mse_chad = np.log((err_sub+1)/(err_sf_pr+1))

# means = [mse_chad.mean()]
# cis = [stats.sem(arr, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(arr)-1) for arr in [mse_chad]]
# x_ticks = ['Chadefaux-2022']
# plt.figure(figsize=(8, 5))
# plt.errorbar(x=x_ticks, y=means, yerr=np.squeeze(cis), fmt='o', markersize=8, color='black', ecolor='black')
# plt.xlabel("Arrays")
# plt.ylabel("Mean Values with 95% CI")
# plt.title("Mean and 95% Confidence Interval for Different Configuration - MSE")
# plt.axhline(0,linestyle='--')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# d_nn = diff_explained(df_input.iloc[-24:-24+horizon],df_sf_1)
# d_nn2 = diff_explained(df_input.iloc[-12:],df_sf_2)
# d_nn = np.concatenate([d_nn,d_nn2])

# d_sub = diff_explained(df_input.iloc[-24:-24+horizon],df_sf_1_chad)
# d_sub2 = diff_explained(df_input.iloc[-12:],df_sf_2_chad)
# d_sub = np.concatenate([d_sub,d_sub2])
# de_list_sub = np.array(np.log((d_nn+1)/(d_sub+1)))
    
# means = [de_list_sub.mean()]
# cis = [stats.sem(arr, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(arr)-1) for arr in [de_list_sub]]
# x_ticks = ['Chadefaux-2022']
# plt.figure(figsize=(8, 5))
# plt.errorbar(x=x_ticks, y=means, yerr=np.squeeze(cis), fmt='o', markersize=8, color='black', ecolor='black')
# plt.xlabel("Arrays")
# plt.ylabel("Mean Values with 95% CI")
# plt.title("Mean and 95% Confidence Interval for Different Configuration - DE")
# plt.axhline(0,linestyle='--')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# =============================================================================
# Diff windows
# =============================================================================

# for wind in [0,1,3]:
#     dict_m={i :[] for i in df_input.columns} 
#     df_input_sub=df_input.iloc[:-24]
#     for coun in range(len(df_input_sub.columns)):
#         if not (df_input_sub.iloc[-h_train:,coun]==0).all():
#             shape = Shape()
#             shape.set_shape(df_input_sub.iloc[-h_train:,coun]) 
#             find = finder(df_tot_m.iloc[:-24],shape)
#             find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2)
#             min_d_d=0.1
#             while len(find.sequences)<5:
#                 min_d_d += 0.05
#                 find.find_patterns(min_d=min_d_d,select=True,metric='dtw',dtw_sel=wind)                
#             dict_m[df_input.columns[coun]]=find.sequences
#         else :
#             pass
#     with open(f'test1_wind{wind}.pkl', 'wb') as f:
#         pickle.dump(dict_m, f) 
    
#     dict_m={i :[] for i in df_input.columns} 
#     df_input_sub=df_input.iloc[:-12]
#     for coun in range(len(df_input_sub.columns)):
#         if not (df_input_sub.iloc[-h_train:,coun]==0).all():
#             shape = Shape()
#             shape.set_shape(df_input_sub.iloc[-h_train:,coun]) 
#             find = finder(df_tot_m.iloc[:-24],shape)
#             find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2)
#             min_d_d=0.1
#             while len(find.sequences)<5:
#                 min_d_d += 0.05
#                 find.find_patterns(min_d=min_d_d,select=True,metric='dtw',dtw_sel=wind)                
#             dict_m[df_input.columns[coun]]=find.sequences
#         else :
#             pass
#     with open(f'test2_wind{wind}.pkl', 'wb') as f:
#         pickle.dump(dict_m, f) 
        
        
for k_win,wind in enumerate([0,1,3]):
    with open(f'test1_wind{wind}.pkl', 'rb') as f:
        dict_m = pickle.load(f)
    pred_tot_min=[]
    pred_tot_pr=[]
    horizon=12
    h_train=10
    df_input_sub=df_input.iloc[:-24]
    for coun in range(len(df_input_sub.columns)):
        if not (df_input_sub.iloc[-h_train:,coun]==0).all():
            inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
            l_find=dict_m[df_input.columns[coun]]          
            tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
            pred_seq=[]
            for col,last_date,mi,ma,somme in tot_seq:
                date=df_tot_m.iloc[:-24].index.get_loc(last_date)            
                if date+horizon<len(df_tot_m.iloc[:-24]):                              
                    seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)               
                    seq = (seq - mi) / (ma - mi)                                
                    pred_seq.append(seq.tolist())        
            tot_seq=pd.DataFrame(pred_seq)
            linkage_matrix = linkage(tot_seq, method='ward')
            clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
            tot_seq['Cluster'] = clusters
            val_sce = tot_seq.groupby('Cluster').mean()   
            pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
            pred_ori=val_sce.loc[pr==pr.max(),:]
            pred_ori=pred_ori.mean(axis=0)
            preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
            pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        else:
            pred_tot_min.append(pd.Series(np.zeros((horizon,))))
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
            
    df_sf_1_sub = pd.concat(pred_tot_pr,axis=1)
    df_sf_1_sub.columns=country_list['name']
    df_sf_1_tot.append(df_sf_1_sub)
    
    with open(f'test2_wind{wind}.pkl', 'rb') as f:
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
            for col,last_date,mi,ma,somme in tot_seq:
                date=df_tot_m.iloc[:-12].index.get_loc(last_date)       
                if date+horizon<len(df_tot_m.iloc[:-12]):               
                    seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                    seq = (seq - mi) / (ma - mi)              
                    pred_seq.append(seq.tolist())                
            tot_seq=pd.DataFrame(pred_seq)       
            linkage_matrix = linkage(tot_seq, method='ward')
            clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
            tot_seq['Cluster'] = clusters
            val_sce = tot_seq.groupby('Cluster').mean()
            pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
            pred_ori=val_sce.loc[pr==pr.max(),:]
            pred_ori=pred_ori.mean(axis=0)
            preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
            pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        else:     
            pred_tot_min.append(pd.Series(np.zeros((horizon,))))
            pred_tot_pr.append(pd.Series(np.zeros((horizon,))))  
        
    df_sf_2_sub= pd.concat(pred_tot_pr,axis=1)
    df_sf_2_sub.columns=country_list['name']    
    df_sf_2_tot.append(df_sf_2_sub) 
    
# mse_list_sub=[]
# for n_k in range(9,len(df_sf_2_tot)):
#     err_sf_pr=[]
#     err_sub=[]
#     for i in range(len(df_input.columns)):   
#         err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i]))
#         err_sub.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1_tot[n_k].iloc[:,i]))
#         err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i]))
#         err_sub.append(mean_squared_error(df_input.iloc[-12:,i], df_sf_2_tot[n_k].iloc[:,i]))
#     err_sf_pr = np.array(err_sf_pr)
#     err_sub = np.array(err_sub)
#     mse_list_sub.append(np.log((err_sub+1)/(err_sf_pr+1)))

# means = [arr.mean() for arr in mse_list_sub]
# cis = [stats.sem(arr, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(arr)-1) for arr in mse_list_sub]
# x_ticks = [0,1,3]
# plt.figure(figsize=(8, 5))
# plt.errorbar(x=x_ticks, y=means, yerr=np.squeeze(cis), fmt='o', markersize=8, color='black', ecolor='black')
# plt.xlabel("Arrays")
# plt.ylabel("Mean Values with 95% CI")
# plt.title("Mean and 95% Confidence Interval for Different Configuration - MSE")
# plt.axhline(0,linestyle='--')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# de_list_sub=[]
# for n_k in range(9,len(df_sf_2_tot)):
#     d_sub = diff_explained(df_input.iloc[-24:-24+horizon],df_sf_1_tot[n_k])
#     d_sub2 = diff_explained(df_input.iloc[-12:],df_sf_2_tot[n_k])
#     d_sub = np.concatenate([d_sub,d_sub2])
#     de_list_sub.append(np.log((d_nn+1)/(d_sub+1)))
    
# means = [arr.mean() for arr in de_list_sub]
# cis = [stats.sem(arr, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(arr)-1) for arr in de_list_sub]
# x_ticks = [0,1,3]
# plt.figure(figsize=(8, 5))
# plt.errorbar(x=x_ticks, y=means, yerr=np.squeeze(cis), fmt='o', markersize=8, color='black', ecolor='black')
# plt.xlabel("Arrays")
# plt.ylabel("Mean Values with 95% CI")
# plt.title("Mean and 95% Confidence Interval for Different Configuration - DE")
# plt.axhline(0,linestyle='--')
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# =============================================================================
# Random match
# =============================================================================

dict_m={i :[] for i in df_input.columns} 
df_input_sub=df_input.iloc[:-24]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        shape = Shape()
        shape.set_shape(df_input_sub.iloc[-h_train:,coun]) 
        find = finder(df_tot_m.iloc[:-24],shape)
        find.find_patterns_random(select=True,metric='dtw',dtw_sel=2,k=5)
        dict_m[df_input.columns[coun]]=find.sequences
    else :
        pass
with open('test1_random.pkl', 'wb') as f:
    pickle.dump(dict_m, f) 
    
dict_m={i :[] for i in df_input.columns} 
df_input_sub=df_input.iloc[:-12]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        shape = Shape()
        shape.set_shape(df_input_sub.iloc[-h_train:,coun]) 
        find = finder(df_tot_m.iloc[:-12],shape)
        find.find_patterns_random(select=True,metric='dtw',dtw_sel=2,k=5)
        dict_m[df_input.columns[coun]]=find.sequences
    else :
        pass
with open('test2_random.pkl', 'wb') as f:
    pickle.dump(dict_m, f) 



with open('test1_random.pkl', 'rb') as f:
    dict_m = pickle.load(f)
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
h_train=10
df_input_sub=df_input.iloc[:-24]
for coun in range(len(df_input_sub.columns)):
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]            
        l_find=dict_m[df_input.columns[coun]]          
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        pred_seq=[]
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)            
            if date+horizon<len(df_tot_m.iloc[:-24]):                              
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)               
                seq = (seq - mi) / (ma - mi)                                
                pred_seq.append(seq.tolist())        
        tot_seq=pd.DataFrame(pred_seq)
        tot_seq=tot_seq.dropna()
        tot_seq = tot_seq[np.isfinite(tot_seq).all(1)]
        tot_seq=tot_seq.iloc[:5]
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()   
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
        
df_sf_1_random = pd.concat(pred_tot_pr,axis=1)
df_sf_1_random.columns=country_list['name']


with open('test2_random.pkl', 'rb') as f:
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
        for col,last_date,mi,ma,somme in tot_seq:
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)       
            if date+horizon<len(df_tot_m.iloc[:-12]):               
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                seq = (seq - mi) / (ma - mi)              
                pred_seq.append(seq.tolist())                
        tot_seq=pd.DataFrame(pred_seq)   
        tot_seq=tot_seq.dropna()
        tot_seq = tot_seq[np.isfinite(tot_seq).all(1)]
        tot_seq=tot_seq.iloc[:5]
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        val_sce = tot_seq.groupby('Cluster').mean()
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:     
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))  
    
df_sf_2_random= pd.concat(pred_tot_pr,axis=1)
df_sf_2_random.columns=country_list['name']    


# =============================================================================
# Plot total
# =============================================================================

#df_sf_1_tot = df_sf_1_tot[:-2]
#df_sf_2_tot = df_sf_2_tot[:-2]

df_sf_1_tot.append(df_sf_1_chad)
df_sf_2_tot.append(df_sf_2_chad)

df_sf_1_tot.append(df_sf_1_random)
df_sf_2_tot.append(df_sf_2_random)

df_sf_1_tot.append(df_sf_1_w_clu)
df_sf_2_tot.append(df_sf_2_w_clu)

mse_list_sub=[]
for n_k in range(len(df_sf_2_tot)):
    err_sf_pr=[]
    err_sub=[]
    for i in range(len(df_input.columns)):   
        err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i]))
        err_sub.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1_tot[n_k].iloc[:,i]))
        err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i]))
        err_sub.append(mean_squared_error(df_input.iloc[-12:,i], df_sf_2_tot[n_k].iloc[:,i]))
    err_sf_pr = np.array(err_sf_pr)
    err_sub = np.array(err_sub)
    mse_list_sub.append(np.log((err_sub+1)/(err_sf_pr+1)))
de_list_sub=[]
for n_k in range(len(df_sf_2_tot)):
    d_sub = diff_explained(df_input.iloc[-24:-24+horizon],df_sf_1_tot[n_k])
    d_sub2 = diff_explained(df_input.iloc[-12:],df_sf_2_tot[n_k])
    d_sub = np.concatenate([d_sub,d_sub2])
    de_list_sub.append(np.log((d_nn+1)/(d_sub+1)))
    
    
means = [arr.mean() for arr in mse_list_sub]
cis = [stats.sem(arr, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(arr)-1) for arr in mse_list_sub]
de_means = [arr.mean() for arr in de_list_sub]
de_cis = [stats.sem(arr, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(arr)-1) for arr in de_list_sub]


#x_ticks = ["Closest Pattern", "kNN-3", "kNN-5", "Thres - 0.2", "Thres - 0.3", "Thres - 0.4", "Thres - 0.5", 
#           "Thres - 0.75", "Thres - 1", "Wind-0", "Wind-1", "Wind-3", "Old Model",'Random']

x_ticks = ["(a) Top match", "Top 3 matches", "Top 5 matches","(b) cut=2","cut=4","cut=5","(c) dist=0.2","dist=0.3","dist=0.4","dist=0.5","dist=0.75","dist=1","(d) win=0","win=1","win=3","(e) Chadefaux (2022)","(f) Random","(g) Weighted cluster"]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
ax1.errorbar(x=x_ticks, y=means, yerr=np.squeeze(cis), fmt='o', markersize=8, color='black', ecolor='black')
ax1.set_ylabel("Mean squared error (log-ratio)",labelpad=24)
#ax1.set_title("MSE",fontsize=15)
ax1.axhline(0, linestyle='--', color='gray')
#ax1.grid(True, linestyle='--', alpha=0.6)
ax2.errorbar(x=x_ticks, y=de_means, yerr=np.squeeze(de_cis), fmt='o', markersize=8, color='black', ecolor='black')
ax2.set_ylabel("Difference explained (log-ratio)")
#ax2.set_title("DE",fontsize=15)
ax2.axhline(0, linestyle='--', color='gray')
#ax2.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', which='major')
ax2.tick_params(axis='both', which='major')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("out/robust.jpeg",dpi=400,bbox_inches="tight")
plt.show()

means_s=means[:3]
cis_s=cis[:3]
de_means_s=de_means[:3]
de_cis_s=de_cis[:3]
x_ticks = ["Top match", "Top 3 matches", "Top 5 matches"]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
ax1.errorbar(x=x_ticks, y=means_s, yerr=np.squeeze(cis_s), fmt='o', markersize=8, color='black', ecolor='black')
ax1.set_ylabel("Mean squared error (log-ratio)",labelpad=19)
#ax1.set_title("MSE",fontsize=15)
ax1.axhline(0, linestyle='--', color='gray')
#ax1.grid(True, linestyle='--', alpha=0.6)
ax2.errorbar(x=x_ticks, y=de_means_s, yerr=np.squeeze(de_cis_s), fmt='o', markersize=8, color='black', ecolor='black')
ax2.set_ylabel("Difference explained (log-ratio)")
#ax2.set_title("DE",fontsize=15)
ax2.axhline(0, linestyle='--', color='gray')
#ax2.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', which='major')
ax1.set_yticks([0,0.25,0.5,0.75,1],["0.00","0.25","0.50","0.75","1.00"])
ax1.set_ylim(-0.1,1.1)
ax2.set_yticks([-0.02,0.0,0.02,0.04,0.06],["--0.02","0.00","0.02","0.04","0.06"])
ax2.set_ylim(-0.028,0.075)
ax2.tick_params(axis='both', which='major')
plt.tight_layout()
plt.savefig("out/robust_a.jpeg",dpi=400,bbox_inches="tight")
plt.show()

means_s=means[3:6]
cis_s=cis[3:6]
de_means_s=de_means[3:6]
de_cis_s=de_cis[3:6]
x_ticks = ["cut=2","cut=4","cut=5"]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
ax1.errorbar(x=x_ticks, y=means_s, yerr=np.squeeze(cis_s), fmt='o', markersize=8, color='black', ecolor='black')
ax1.set_ylabel("Mean squared error (log-ratio)",labelpad=19)
#ax1.set_title("MSE",fontsize=15)
ax1.axhline(0, linestyle='--', color='gray')
#ax1.grid(True, linestyle='--', alpha=0.6)
ax2.errorbar(x=x_ticks, y=de_means_s, yerr=np.squeeze(de_cis_s), fmt='o', markersize=8, color='black', ecolor='black')
ax2.set_ylabel("Difference explained (log-ratio)")
#ax2.set_title("DE",fontsize=15)
ax2.axhline(0, linestyle='--', color='gray')
#ax2.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', which='major')
ax1.set_yticks([0,0.25,0.5,0.75,1],["0.00","0.25","0.50","0.75","1.00"])
ax1.set_ylim(-0.1,1.1)
ax2.set_yticks([-0.02,0.0,0.02,0.04,0.06],["--0.02","0.00","0.02","0.04","0.06"])
ax2.set_ylim(-0.028,0.075)
ax2.tick_params(axis='both', which='major')
plt.tight_layout()
plt.savefig("out/robust_b.jpeg",dpi=400,bbox_inches="tight")
plt.show()

means_s=means[6:12]
cis_s=cis[6:12]
de_means_s=de_means[6:12]
de_cis_s=de_cis[6:12]
x_ticks = ["dist=0.2","dist=0.3","dist=0.4","dist=0.5","dist=0.75","dist=1"]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
ax1.errorbar(x=x_ticks, y=means_s, yerr=np.squeeze(cis_s), fmt='o', markersize=8, color='black', ecolor='black')
ax1.set_ylabel("Mean squared error (log-ratio)",labelpad=19)
#ax1.set_title("MSE",fontsize=15)
ax1.axhline(0, linestyle='--', color='gray')
#ax1.grid(True, linestyle='--', alpha=0.6)
ax2.errorbar(x=x_ticks, y=de_means_s, yerr=np.squeeze(de_cis_s), fmt='o', markersize=8, color='black', ecolor='black')
ax2.set_ylabel("Difference explained (log-ratio)")
#ax2.set_title("DE",fontsize=15)
ax2.axhline(0, linestyle='--', color='gray')
#ax2.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', which='major')
ax1.set_yticks([0,0.25,0.5,0.75,1],["0.00","0.25","0.50","0.75","1.00"])
ax1.set_ylim(-0.1,1.1)
ax2.set_yticks([-0.02,0.0,0.02,0.04,0.06],["--0.02","0.00","0.02","0.04","0.06"])
ax2.set_ylim(-0.028,0.075)
ax2.tick_params(axis='both', which='major')
plt.tight_layout()
plt.savefig("out/robust_c.jpeg",dpi=400,bbox_inches="tight")
plt.show()

means_s=means[12:15]
cis_s=cis[12:15]
de_means_s=de_means[12:15]
de_cis_s=de_cis[12:15]
x_ticks = ["win=0","win=1","win=3",]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
ax1.errorbar(x=x_ticks, y=means_s, yerr=np.squeeze(cis_s), fmt='o', markersize=8, color='black', ecolor='black')
ax1.set_ylabel("Mean squared error (log-ratio)",labelpad=19)
#ax1.set_title("MSE",fontsize=15)
ax1.axhline(0, linestyle='--', color='gray')
#ax1.grid(True, linestyle='--', alpha=0.6)
ax2.errorbar(x=x_ticks, y=de_means_s, yerr=np.squeeze(de_cis_s), fmt='o', markersize=8, color='black', ecolor='black')
ax2.set_ylabel("Difference explained (log-ratio)")
#ax2.set_title("DE",fontsize=15)
ax2.axhline(0, linestyle='--', color='gray')
#ax2.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', which='major')
ax1.set_yticks([0,0.25,0.5,0.75,1],["0.00","0.25","0.50","0.75","1.00"])
ax1.set_ylim(-0.1,1.1)
ax2.set_yticks([-0.02,0.0,0.02,0.04,0.06],["--0.02","0.00","0.02","0.04","0.06"])
ax2.set_ylim(-0.028,0.075)
ax2.tick_params(axis='both', which='major')
plt.tight_layout()
plt.savefig("out/robust_d.jpeg",dpi=400,bbox_inches="tight")
plt.show()

means_s=means[15:18]
cis_s=cis[15:18]
de_means_s=de_means[15:18]
de_cis_s=de_cis[15:18]
x_ticks = ["Chadefaux (2022)","Random","Weighted cluster"]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
ax1.errorbar(x=x_ticks, y=means_s, yerr=np.squeeze(cis_s), fmt='o', markersize=8, color='black', ecolor='black')
ax1.set_ylabel("Mean squared error (log-ratio)",labelpad=19)
#ax1.set_title("MSE",fontsize=15)
ax1.axhline(0, linestyle='--', color='gray')
#ax1.grid(True, linestyle='--', alpha=0.6)
ax2.errorbar(x=x_ticks, y=de_means_s, yerr=np.squeeze(de_cis_s), fmt='o', markersize=8, color='black', ecolor='black')
ax2.set_ylabel("Difference explained (log-ratio)")
#ax2.set_title("DE",fontsize=15)
ax2.axhline(0, linestyle='--', color='gray')
#ax2.grid(True, linestyle='--', alpha=0.6)
ax1.tick_params(axis='both', which='major')
ax1.set_yticks([0,0.25,0.5,0.75,1],["0.00","0.25","0.50","0.75","1.00"])
ax1.set_ylim(-0.1,1.1)
ax2.set_yticks([-0.02,0.0,0.02,0.04,0.06],["--0.02","0.00","0.02","0.04","0.06"])
ax2.set_ylim(-0.028,0.075)
ax2.tick_params(axis='both', which='major')
plt.tight_layout()
plt.savefig("out/robust_e.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# =============================================================================
# Analysis of the zeros
# =============================================================================

df_p_views_1 = pd.read_csv('views1.csv',index_col=0)
df_p_views_2 = pd.read_csv('views2.csv',index_col=0)

c_y=0
c_n=0
l_mse_zero_not=[]
l_mse_zero=[]
pair_pred=[]
for coun in range(len(df_input.columns)): 
    if (df_input.iloc[-34:-24,coun]==0).all():
        if not (df_input.iloc[-24:-12,coun]==0).all():
            mse_v = mean_squared_error(df_input.iloc[-24:-12,coun], df_p_views_1.iloc[:,coun])
            mse_z = mean_squared_error(df_input.iloc[-24:-12,coun], pd.Series([0]*12))
            l_mse_zero_not.append(np.log((mse_v+1)/(mse_z+1)))
            pair_pred.append([df_input.iloc[-24:-12,coun].sum(),df_p_views_1.iloc[:,coun].sum()])
            c_y+=1
        else:
            c_n+=1
            mse_v = mean_squared_error(df_input.iloc[-24:-12,coun], df_p_views_1.iloc[:,coun])
            mse_z = mean_squared_error(df_input.iloc[-24:-12,coun], pd.Series([0]*12))
            l_mse_zero.append(np.log((mse_v+1)/(mse_z+1)))
            
for coun in range(len(df_input.columns)): 
    if (df_input.iloc[-22:-12,coun]==0).all():
        if not (df_input.iloc[-12:,coun]==0).all():
            mse_v = mean_squared_error(df_input.iloc[-12:,coun], df_p_views_2.iloc[:,coun])
            mse_z = mean_squared_error(df_input.iloc[-12:,coun], pd.Series([0]*12))
            l_mse_zero_not.append(np.log((mse_v+1)/(mse_z+1)))
            pair_pred.append([df_input.iloc[-12:,coun].sum(),df_p_views_2.iloc[:,coun].sum()])
            c_y+=1
        else:
            c_n+=1
            mse_v = mean_squared_error(df_input.iloc[-24:-12,coun], df_p_views_1.iloc[:,coun])
            mse_z = mean_squared_error(df_input.iloc[-24:-12,coun], pd.Series([0]*12))
            l_mse_zero.append(np.log((mse_v+1)/(mse_z+1)))
            
means_z = [arr.mean() for arr in [np.array(l_mse_zero_not),np.array(l_mse_zero),np.array(l_mse_zero_not+l_mse_zero)]]
cis_z = [stats.sem(arr, axis=0) * stats.t.ppf((1 + 0.95) / 2., len(arr)-1) for arr in [np.array(l_mse_zero_not),np.array(l_mse_zero),np.array(l_mse_zero_not+l_mse_zero)]]
x_ticks = ['Non-flat future (14\%)', 'Flat future (86\%)','All']
plt.figure(figsize=(10, 5))
plt.errorbar(x=x_ticks, y=means_z, yerr=np.squeeze(cis_z), fmt='o', markersize=8, color='black', ecolor='black')
plt.ylabel("Mean squared error (log-ratio)")
#plt.title("MSE of Zeros",fontsize=15)
plt.axhline(0,linestyle='--')
plt.xlim(-0.5,2.5)
plt.yticks([-0.025,-0.02,-0.015,-0.01,-0.005,0,0.005],[-0.025,-0.02,-0.015,-0.01,-0.005,0,0.005])
ax1.tick_params(axis='both', which='major')
ax2.tick_params(axis='both', which='major')
#plt.xticks(rotation=45, ha='right')
#plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("out/flat.jpeg",dpi=400,bbox_inches="tight")
plt.show()

pair_pred=pd.DataFrame(pair_pred)

plt.figure(figsize=(8,8))
plt.axhline(0,linestyle='--',color='lightgrey')
plt.axvline(0,linestyle='--',color='lightgrey')
plt.scatter(pair_pred.iloc[:,0],pair_pred.iloc[:,1],marker='x')
plt.ylabel('Views Pred Sum',fontsize=15)
plt.xlabel('Observed Sum',fontsize=15)
plt.xlim(-1,20)
plt.ylim(-1,20)
plt.show()

plt.figure(figsize=(8,8))
plt.axhline(0,linestyle='--',color='lightgrey')
plt.axvline(0,linestyle='--',color='lightgrey')
plt.scatter(pair_pred.iloc[:,0],pair_pred.iloc[:,1],marker='x')
plt.ylabel('Views Pred Sum',fontsize=15)
plt.xlabel('Observed Sum',fontsize=15)
plt.xlim(-5,150)
plt.ylim(-5,150)
plt.show()


# =============================================================================
# cluster check 
# =============================================================================


with open('test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)
horizon=12
df_input_sub=df_input.iloc[:-24]
cluster_dist=[]
check_inside=[]
check_out=[]
for coun in range(len(df_input_sub.columns)):
    # If the last h_train observations of training data are not flat
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        # For each case, get region, year and magnitude as the log(total fatalities)

        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        # Get reference reporitory for case                
        l_find=dict_m[df_input.columns[coun]]
        # For each case in repository, get country name, last time point, minimum, maximum and sum of fatalities                
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        
        # For each case in reference repository                       
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            # Get views id for last month            
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)
            # If reference + horizon is in training data                        
            if date+horizon<len(df_tot_m.iloc[:-24]):
                # Extract sequence for reference, for the next 12 months                                
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
                # Scaling                
                seq = (seq - mi) / (ma - mi)
                # Add to list                                
                pred_seq.append(seq.tolist())
                # Add region, decade and magnitude                
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
                
        # Sequences for all case in the reference repository, if they belong to training data,
        # every row is one sequence                 
        tot_seq=pd.DataFrame(pred_seq)
        
        ### Apply hierachical clustering to sequences in reference repository ###        
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        pr_old = pd.Series(clusters).value_counts(normalize=True).sort_index()
        tot_seq['Cluster'] = clusters
        
        pred_seq.append((df_input.iloc[-24:-12,coun]-df_input_sub.iloc[-h_train:,coun].min())  /(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min()))
        tot_seq_s=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq_s, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        if clusters[-1] in pr[pr==pr.max()].index.tolist():
            check_inside.append(pr_old.max())
        else:
            check_out.append(pr_old.max())     
        
    else:
        pass
        


with open('test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
    
df_input_sub=df_input.iloc[:-12]
horizon=12

for coun in range(len(df_input_sub.columns)):
    # If the last h_train observations of training data are not flat
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        
        ### Extract information for cases in reference repository ###

        # For each case, get region, year and magnitude as the log(total fatalities)
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        # Get reference reporitory for case                
        l_find=dict_m[df_input.columns[coun]]
        # For each case in repository, get country name, last time point, minimum, maximum and sum of fatalities                
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        
        # For each case in reference repository               
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        
        # For each case in reference repository                       
        for col,last_date,mi,ma,somme in tot_seq:
            # Get views id for last month
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)
            # If reference + horizon is in training data            
            if date+horizon<len(df_tot_m.iloc[:-12]):
                # Extract sequence for reference, for the next 12 months                
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                # Scaling                
                seq = (seq - mi) / (ma - mi)
                # Add to list                
                pred_seq.append(seq.tolist())
                # Add region, decade and magnitude
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)

        # Sequences for all case in the reference repository, if they belong to training data,
        # every row is one sequence                 
        tot_seq=pd.DataFrame(pred_seq)
        
        ### Apply hierachical clustering to sequences in reference repository ###        
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        pr_old = pd.Series(clusters).value_counts(normalize=True).sort_index()
        tot_seq['Cluster'] = clusters
        
        
        pred_seq.append((df_input.iloc[-12:,coun]-df_input_sub.iloc[-h_train:,coun].min())  /(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min()))
        tot_seq_s=pd.DataFrame(pred_seq)
        linkage_matrix = linkage(tot_seq_s, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        if clusters[-1] in pr[pr==pr.max()].index.tolist():
            check_inside.append(pr_old.max())
        else:
            check_out.append(pr_old.max())  
    
    else:
        pass

bins = [0.2, 0.4, 0.6, 0.8, 1]
inside_hist, _ = np.histogram(check_inside, bins=bins)
out_hist, _ = np.histogram(check_out, bins=bins)
inside_percentage = inside_hist / (inside_hist + out_hist + 1e-9) * 100  # Prevent division by zero
out_percentage = out_hist / (inside_hist + out_hist + 1e-9) * 100
width = 0.1
fig,ax = plt.subplots(figsize=(12,8))
bin_centers = (np.array(bins[:-1]) + np.array(bins[1:])) / 2
plt.bar(bin_centers, inside_percentage, width=width, color='gray', alpha=0.5, label='Outcome Inside')
#plt.bar(bin_centers + width / 2, out_percentage, width=width, color='orange', alpha=0.5, label='Outcome Outside')
plt.xlabel('Size of majority cluster')
plt.ylabel('Futures (\%) assigned to majority cluster')
#plt.title('Percentage Distribution Between Bins')
plt.xticks(bins)
#plt.legend()
plt.savefig("out/hist.jpeg",dpi=400,bbox_inches="tight")
plt.show()







    
with open('test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)
    
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
h_train=10
df_input_sub=df_input.iloc[:-24]
cluster_dist=[]
for coun in range(len(df_input_sub.columns)):
    # If the last h_train observations of training data are not flat
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        # For each case, get region, year and magnitude as the log(total fatalities)

        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        # Get reference reporitory for case                
        l_find=dict_m[df_input.columns[coun]]
        # For each case in repository, get country name, last time point, minimum, maximum and sum of fatalities                
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        
        # For each case in reference repository                       
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            # Get views id for last month            
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)
            # If reference + horizon is in training data                        
            if date+horizon<len(df_tot_m.iloc[:-24]):
                # Extract sequence for reference, for the next 12 months                                
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
                # Scaling                
                seq = (seq - mi) / (ma - mi)
                # Add to list                                
                pred_seq.append(seq.tolist())
                # Add region, decade and magnitude                
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
                
        # Sequences for all case in the reference repository, if they belong to training data,
        # every row is one sequence                 
        tot_seq=pd.DataFrame(pred_seq)
        
        ### Apply hierachical clustering to sequences in reference repository ###        
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        
        # Calculate mean sequence for each cluster        
        val_sce = tot_seq.groupby('Cluster').mean()
        
        # Proportions for each cluster        
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        #cluster_dist.append(pr.max())
        cluster_dist.append(pd.Series(clusters).value_counts().max())
        
        # A. Get mean sequence with lowest intensity
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        # Adjust by range (*max-min) and add min value
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
       
        # B. Get mean sequence for cluster with highest number of observations                
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        # Adjust by range (*max-min) and add min value
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        # Append predictions
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        
    
    else:
        # Add zeros
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))

err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], pred_tot_pr[i]))
err_sf_pr = np.array(err_sf_pr)

mse_list_raw = err_sf_pr.copy()
 
### Using 2022 to make predictions in 2023 ###

with open('test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
    
df_input_sub=df_input.iloc[:-12]
pred_tot_min=[]
pred_tot_pr=[]
horizon=12

for coun in range(len(df_input_sub.columns)):
    # If the last h_train observations of training data are not flat
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        
        ### Extract information for cases in reference repository ###

        # For each case, get region, year and magnitude as the log(total fatalities)
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        # Get reference reporitory for case                
        l_find=dict_m[df_input.columns[coun]]
        # For each case in repository, get country name, last time point, minimum, maximum and sum of fatalities                
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        
        # For each case in reference repository               
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        
        # For each case in reference repository                       
        for col,last_date,mi,ma,somme in tot_seq:
            # Get views id for last month
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)
            # If reference + horizon is in training data            
            if date+horizon<len(df_tot_m.iloc[:-12]):
                # Extract sequence for reference, for the next 12 months                
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                # Scaling                
                seq = (seq - mi) / (ma - mi)
                # Add to list                
                pred_seq.append(seq.tolist())
                # Add region, decade and magnitude
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)

        # Sequences for all case in the reference repository, if they belong to training data,
        # every row is one sequence                 
        tot_seq=pd.DataFrame(pred_seq)
        
        ### Apply hierachical clustering to sequences in reference repository ###        
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        
        # Calculate mean sequence for each cluster
        val_sce = tot_seq.groupby('Cluster').mean()
        
        # Proportions for each cluster
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        #cluster_dist.append(pr.max())
        cluster_dist.append(pd.Series(clusters).value_counts().max())
        
        # A. Get mean sequence with lowest intensity
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        # Adjust by range (*max-min) and add min value
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        
        # B. Get mean sequence for cluster with highest number of observations                
        pred_ori=val_sce.loc[pr==pr.max(),:]
        pred_ori=pred_ori.mean(axis=0)
        # Adjust by range (*max-min) and add min value
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        # Append predictions
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        
        
    else:
        # Add zeros         
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))  
### Get MSE ###
err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], pred_tot_pr[i]))
err_sf_pr = np.array(err_sf_pr)

mse_list_true = np.concatenate([mse_list_raw,err_sf_pr],axis=0)



with open('test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
h_train=10
df_input_sub=df_input.iloc[:-24]
cluster_dist=[]
for coun in range(len(df_input_sub.columns)):
    # If the last h_train observations of training data are not flat
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        # For each case, get region, year and magnitude as the log(total fatalities)

        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        # Get reference reporitory for case                
        l_find=dict_m[df_input.columns[coun]]
        # For each case in repository, get country name, last time point, minimum, maximum and sum of fatalities                
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        
        # For each case in reference repository                       
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            # Get views id for last month            
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)
            # If reference + horizon is in training data                        
            if date+horizon<len(df_tot_m.iloc[:-24]):
                # Extract sequence for reference, for the next 12 months                                
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
                # Scaling                
                seq = (seq - mi) / (ma - mi)
                # Add to list                                
                pred_seq.append(seq.tolist())
                # Add region, decade and magnitude                
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
                
        # Sequences for all case in the reference repository, if they belong to training data,
        # every row is one sequence                 
        tot_seq=pd.DataFrame(pred_seq)
        
        ### Apply hierachical clustering to sequences in reference repository ###        
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        
        # Calculate mean sequence for each cluster        
        val_sce = tot_seq.groupby('Cluster').mean()
        
        # Proportions for each cluster        
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        #cluster_dist.append(pr.max())
        cluster_dist.append(pd.Series(clusters).value_counts().max())
        
        # A. Get mean sequence with lowest intensity
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        # Adjust by range (*max-min) and add min value
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
       
        # B. Get mean sequence for cluster with highest number of observations                
        pred_ori=val_sce.loc[pr==pr.median(),:]
        pred_ori=pred_ori.mean(axis=0)
        # Adjust by range (*max-min) and add min value
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        # Append predictions
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        
    
    else:
        # Add zeros
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))

err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    pred_tot_pr[i]=pred_tot_pr[i].fillna(0)
    err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], pred_tot_pr[i]))
err_sf_pr = np.array(err_sf_pr)

mse_list_raw = err_sf_pr.copy()
 
### Using 2022 to make predictions in 2023 ###

with open('test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
    
df_input_sub=df_input.iloc[:-12]
pred_tot_min=[]
pred_tot_pr=[]
horizon=12

for coun in range(len(df_input_sub.columns)):
    # If the last h_train observations of training data are not flat
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        
        ### Extract information for cases in reference repository ###

        # For each case, get region, year and magnitude as the log(total fatalities)
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        # Get reference reporitory for case                
        l_find=dict_m[df_input.columns[coun]]
        # For each case in repository, get country name, last time point, minimum, maximum and sum of fatalities                
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        
        # For each case in reference repository               
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        
        # For each case in reference repository                       
        for col,last_date,mi,ma,somme in tot_seq:
            # Get views id for last month
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)
            # If reference + horizon is in training data            
            if date+horizon<len(df_tot_m.iloc[:-12]):
                # Extract sequence for reference, for the next 12 months                
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                # Scaling                
                seq = (seq - mi) / (ma - mi)
                # Add to list                
                pred_seq.append(seq.tolist())
                # Add region, decade and magnitude
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)

        # Sequences for all case in the reference repository, if they belong to training data,
        # every row is one sequence                 
        tot_seq=pd.DataFrame(pred_seq)
        
        ### Apply hierachical clustering to sequences in reference repository ###        
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        
        # Calculate mean sequence for each cluster
        val_sce = tot_seq.groupby('Cluster').mean()
        
        # Proportions for each cluster
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        #cluster_dist.append(pr.max())
        cluster_dist.append(pd.Series(clusters).value_counts().max())
        
        # A. Get mean sequence with lowest intensity
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        # Adjust by range (*max-min) and add min value
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        
        # B. Get mean sequence for cluster with highest number of observations                
        pred_ori=val_sce.loc[pr==pr.median(),:]
        pred_ori=pred_ori.mean(axis=0)
        # Adjust by range (*max-min) and add min value
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        # Append predictions
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        # Add zeros         
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))  
### Get MSE ###
err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    pred_tot_pr[i]=pred_tot_pr[i].fillna(0)
    err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], pred_tot_pr[i]))
err_sf_pr = np.array(err_sf_pr)

mse_list_med = np.concatenate([mse_list_raw,err_sf_pr],axis=0)






with open('test1.pkl', 'rb') as f:
    dict_m = pickle.load(f)
pred_tot_min=[]
pred_tot_pr=[]
horizon=12
h_train=10
df_input_sub=df_input.iloc[:-24]
cluster_dist=[]
for coun in range(len(df_input_sub.columns)):
    # If the last h_train observations of training data are not flat
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        # For each case, get region, year and magnitude as the log(total fatalities)

        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        # Get reference reporitory for case                
        l_find=dict_m[df_input.columns[coun]]
        # For each case in repository, get country name, last time point, minimum, maximum and sum of fatalities                
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        
        # For each case in reference repository                       
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        for col,last_date,mi,ma,somme in tot_seq:
            # Get views id for last month            
            date=df_tot_m.iloc[:-24].index.get_loc(last_date)
            # If reference + horizon is in training data                        
            if date+horizon<len(df_tot_m.iloc[:-24]):
                # Extract sequence for reference, for the next 12 months                                
                seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
                # Scaling                
                seq = (seq - mi) / (ma - mi)
                # Add to list                                
                pred_seq.append(seq.tolist())
                # Add region, decade and magnitude                
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)
                
        # Sequences for all case in the reference repository, if they belong to training data,
        # every row is one sequence                 
        tot_seq=pd.DataFrame(pred_seq)
        
        ### Apply hierachical clustering to sequences in reference repository ###        
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        
        # Calculate mean sequence for each cluster        
        val_sce = tot_seq.groupby('Cluster').mean()
        
        # Proportions for each cluster        
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        #cluster_dist.append(pr.max())
        cluster_dist.append(pd.Series(clusters).value_counts().max())
        
        # A. Get mean sequence with lowest intensity
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        # Adjust by range (*max-min) and add min value
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
       
        # B. Get mean sequence for cluster with highest number of observations                
        pred_ori=val_sce.loc[pr==pr.min(),:]
        pred_ori=pred_ori.mean(axis=0)
        # Adjust by range (*max-min) and add min value
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        # Append predictions
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        
    
    else:
        # Add zeros
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))

err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    pred_tot_pr[i]=pred_tot_pr[i].fillna(0)
    err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], pred_tot_pr[i]))
err_sf_pr = np.array(err_sf_pr)

mse_list_raw = err_sf_pr.copy()
 
### Using 2022 to make predictions in 2023 ###

with open('test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
    
df_input_sub=df_input.iloc[:-12]
pred_tot_min=[]
pred_tot_pr=[]
horizon=12

for coun in range(len(df_input_sub.columns)):
    # If the last h_train observations of training data are not flat
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        
        ### Extract information for cases in reference repository ###

        # For each case, get region, year and magnitude as the log(total fatalities)
        inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
        # Get reference reporitory for case                
        l_find=dict_m[df_input.columns[coun]]
        # For each case in repository, get country name, last time point, minimum, maximum and sum of fatalities                
        tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
        
        # For each case in reference repository               
        pred_seq=[]
        co=[]
        deca=[]
        scale=[]
        
        # For each case in reference repository                       
        for col,last_date,mi,ma,somme in tot_seq:
            # Get views id for last month
            date=df_tot_m.iloc[:-12].index.get_loc(last_date)
            # If reference + horizon is in training data            
            if date+horizon<len(df_tot_m.iloc[:-12]):
                # Extract sequence for reference, for the next 12 months                
                seq=df_tot_m.iloc[:-12].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-12].columns.get_loc(col)].reset_index(drop=True)
                # Scaling                
                seq = (seq - mi) / (ma - mi)
                # Add to list                
                pred_seq.append(seq.tolist())
                # Add region, decade and magnitude
                co.append(df_conf[col])
                deca.append(last_date.year)
                scale.append(somme)

        # Sequences for all case in the reference repository, if they belong to training data,
        # every row is one sequence                 
        tot_seq=pd.DataFrame(pred_seq)
        
        ### Apply hierachical clustering to sequences in reference repository ###        
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        tot_seq['Cluster'] = clusters
        
        # Calculate mean sequence for each cluster
        val_sce = tot_seq.groupby('Cluster').mean()
        
        # Proportions for each cluster
        pr = round(pd.Series(clusters).value_counts(normalize=True).sort_index(),2)
        #cluster_dist.append(pr.max())
        cluster_dist.append(pd.Series(clusters).value_counts().max())
        
        # A. Get mean sequence with lowest intensity
        pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
        # Adjust by range (*max-min) and add min value
        pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        
        # B. Get mean sequence for cluster with highest number of observations                
        pred_ori=val_sce.loc[pr==pr.min(),:]
        pred_ori=pred_ori.mean(axis=0)
        # Adjust by range (*max-min) and add min value
        preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
        # Append predictions
        pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
    else:
        # Add zeros         
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))  
### Get MSE ###
err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    pred_tot_pr[i]=pred_tot_pr[i].fillna(0)
    err_sf_pr.append(mean_absolute_percentage_error(df_input.iloc[-12:,i], pred_tot_pr[i]))
err_sf_pr = np.array(err_sf_pr)

mse_list_min = np.concatenate([mse_list_raw,err_sf_pr],axis=0)

mse_list_true = np.log(mse_list_true+1)
mse_list_med = np.log(mse_list_med+1)
mse_list_min = np.log(mse_list_min+1)

means = [mse_list_true.mean(),mse_list_med.mean(),mse_list_min.mean()]
std_error = [1.96*mse_list_true.std()/np.sqrt(len(mse_list_true)),1.96*mse_list_med.std()/np.sqrt(len(mse_list_med)),1.96*mse_list_min.std()/np.sqrt(len(mse_list_min))]
mean_de = pd.DataFrame({
    'mean': means,
    'std': std_error
})

fig,ax = plt.subplots(figsize=(12,8))
for i in range(3):
    plt.scatter(i,mean_de["mean"][i],color="black",s=150)
    #plt.plot([mean_dtw["mean"][i],mean_dtw["mean"][i]],[mean_de["mean"][i]-mean_de["std"][i],mean_de["mean"][i]+mean_de["std"][i]],linewidth=3,color="gray")
    plt.plot([i,i],[mean_de["mean"][i]-mean_de["std"][i],mean_de["mean"][i]+mean_de["std"][i]],linewidth=3,color="black")
plt.xticks([0,1,2],['Maximum cluster size','Median cluster size','Minimum cluster size'],fontsize=20)
plt.ylabel('Mean absolute percentage error (log-ratio)',fontsize=20)
plt.xlim(-0.5,2.5)
plt.savefig("out/mape.jpeg",dpi=400,bbox_inches="tight")
plt.show()


####################
### Validation set #
####################

from shape import Shape,finder
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.metrics import mean_squared_error

df_input=pd.read_csv('df_input.csv',index_col=0,parse_dates=True)
df_tot_m = df_input.copy()
df_tot_m.replace(0, np.nan, inplace=True)
df_tot_m = df_tot_m.dropna(axis=1, how='all')
df_tot_m = df_tot_m.fillna(0)
country_list = pd.read_csv('country_list.csv',index_col=0)
df_conf=pd.read_csv('reg_coun.csv',index_col=0)
df_conf=pd.Series(df_conf.region)
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


for min_d_s in [0.1,0.5,1]:
    h_train=10
    dict_m={i :[] for i in df_input.columns} 
    df_input_sub=df_input.iloc[:-36]
    for coun in range(len(df_input_sub.columns)):
        # If the last h_train observations of training data are not flat, run Shape finder, else pass    
        if not (df_input_sub.iloc[-h_train:,coun]==0).all():
            shape = Shape()
            # Set last h_train observations of training data as shape
            shape.set_shape(df_input_sub.iloc[-h_train:,coun]) 
            # Find matches in training data        
            find = finder(df_tot_m.iloc[:-36],shape)
            find.find_patterns_fast(min_d=min_d_s,select=True,metric='dtw',dtw_sel=2,min_mat=5,d_increase= 0.05)           
            dict_m[df_input.columns[coun]]=find.sequences
        else :
            pass


err_sf_pr_tot=[]   
de_list=[]  
for file in ['test_0.1.pkl','test_0.5.pkl','test_1dist.pkl']: 
    with open(file, 'rb') as f:
        dict_m = pickle.load(f) 
    for thres_hold in [12,6,4]:
        pred_tot_min=[]
        pred_tot_pr=[]
        horizon=12
        h_train=10
        df_input_sub=df_input.iloc[:-36]
        cluster_dist=[]
        for coun in range(len(df_input_sub.columns)):
            # If the last h_train observations of training data are not flat
            if not (df_input_sub.iloc[-h_train:,coun]==0).all():
                # For each case, get region, year and magnitude as the log(total fatalities)
        
                inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
                # Get reference reporitory for case                
                l_find=dict_m[df_input.columns[coun]]
                # For each case in repository, get country name, last time point, minimum, maximum and sum of fatalities                
                tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
                
                # For each case in reference repository                       
                pred_seq=[]
                co=[]
                deca=[]
                scale=[]
                for col,last_date,mi,ma,somme in tot_seq:
                    # Get views id for last month            
                    date=df_tot_m.iloc[:-36].index.get_loc(last_date)
                    # If reference + horizon is in training data                        
                    if date+horizon<len(df_tot_m.iloc[:-36]):
                        # Extract sequence for reference, for the next 12 months                                
                        seq=df_tot_m.iloc[:-36].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
                        # Scaling                
                        seq = (seq - mi) / (ma - mi)
                        # Add to list                                
                        pred_seq.append(seq.tolist())
                        # Add region, decade and magnitude                
                        co.append(df_conf[col])
                        deca.append(last_date.year)
                        scale.append(somme)
                        
                # Sequences for all case in the reference repository, if they belong to training data,
                # every row is one sequence                 
                tot_seq=pd.DataFrame(pred_seq)
                
                ### Apply hierachical clustering to sequences in reference repository ###        
                linkage_matrix = linkage(tot_seq, method='ward')
                clusters = fcluster(linkage_matrix, thres_hold, criterion='distance')
                tot_seq['Cluster'] = clusters
                
                # Calculate mean sequence for each cluster        
                val_sce = tot_seq.groupby('Cluster').mean()
                
                # Proportions for each cluster        
                pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
                #cluster_dist.append(pr.max())
                cluster_dist.append(pd.Series(clusters).value_counts().max())
                
                # A. Get mean sequence with lowest intensity
                pred_ori=val_sce.loc[val_sce.sum(axis=1).idxmin(),:]
                # Adjust by range (*max-min) and add min value
                pred_tot_min.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
               
                # B. Get mean sequence for cluster with highest number of observations                
                pred_ori=val_sce.loc[pr==pr.min(),:]
                pred_ori=pred_ori.mean(axis=0)
                # Adjust by range (*max-min) and add min value
                preds=pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min()
                # Append predictions
                pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
        
            else:
                pred_tot_min.append(pd.Series(np.zeros((horizon,))))
                pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
        
        err_sf_pr=[]
        for i in range(len(df_input.columns)):   
            pred_tot_pr[i]=pred_tot_pr[i].fillna(0)
            err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], pred_tot_pr[i]))
        err_sf_pr = np.array(err_sf_pr)
        err_sf_pr_tot.append(err_sf_pr)
        
        df_sf = pd.concat(pred_tot_pr,axis=1)
        df_sf.columns=country_list['name']
        
        d_nn = diff_explained(df_input.iloc[-36:-24],df_sf)
        de_list.append(d_nn)

means = [x.mean() for x in de_list]
std_error = [1.993*x.std()/np.sqrt(len((x))) for x in de_list]
mean_de = pd.DataFrame({
    'mean': means,
    'std': std_error
})

# MSE
means = [x.mean() for x in err_sf_pr_tot]
std_error = [1.993*x.std()/np.sqrt(len((x))) for x in err_sf_pr_tot]
mean_mse = pd.DataFrame({
    'mean': means,
    'std': std_error
})

name = ['min_d=' + str(i) + ',thres=' + str(j) for i in ['0.1', '0.5', '1'] for j in ['12','6', '4']]
fig,ax = plt.subplots(figsize=(12,8))
for i in range(9):
    plt.scatter(mean_mse["mean"][i],mean_de["mean"][i],label=name[i],s=150)
    #plt.plot([mean_mse["mean"][i],mean_mse["mean"][i]],[mean_de["mean"][i]-mean_de["std"][i],mean_de["mean"][i]+mean_de["std"][i]],linewidth=3,color="gray")
    #plt.plot([mean_mse["mean"][i]-mean_mse["std"][i],mean_mse["mean"][i]+mean_mse["std"][i]],[mean_de["mean"][i],mean_de["mean"][i]],linewidth=3,color="gray")
plt.xlabel("Error (MSE)")
plt.ylabel("Difference explained ratio (DE)")
plt.xlim(150000000,0)
plt.legend()
plt.show()
