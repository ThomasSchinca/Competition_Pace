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
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from dtaidistance import ed
from scipy.stats import ttest_1samp
import os
import seaborn as sns


plot_params = {"text.usetex":True,"font.family":"serif","font.size":20,"xtick.labelsize":20,"ytick.labelsize":20,"axes.labelsize":20,"figure.titlesize":20,"figure.figsize":(8,5),"axes.prop_cycle":cycler(color=['black','rosybrown','gray','indianred','red','maroon','silver',])}
plt.rcParams.update(plot_params)

# ### Define out paths ------
# home = '/Users/hannahfrank/'
# if not os.path.exists(os.path.join(home,'desktop/Views_competition_out')):
#     os.makedirs(os.path.join(home,'desktop/Views_competition_out'))

# out_paths = {"plots": os.path.join(home,'desktop/Views_competition_out/plots'),
#     "data": os.path.join(home,'desktop/Views_competition_out/data'),
#     "analysis": os.path.join(home,'desktop/Views_competition_out/analysis'),}

# for key, val in out_paths.items():
#     if not os.path.exists(val):
#         os.makedirs(val)
        
        
### MSE example plot ###

fig,ax = plt.subplots(figsize=(12,8))
sf=[0.6,0.8,0.05,0.4,0.3]
real=[0,1,0.5,0.8,0]
b2=[0.45,0.45,0.45,0.45,0.45]
plt.plot(sf, label='Model I', marker='o',color='black',markersize=0,linewidth=3,linestyle="dashed")
plt.plot(b2, label='Model II', marker='o',color='black',markersize=0,linewidth=3,linestyle="dotted")
plt.plot(real,label='Actuals',marker='o',color="black",markersize=0,linewidth=3)
plt.xticks([0,1,2,3,4],[1,2,3,4,5])
plt.yticks([0,0.2,0.4,0.6,0.8,1],[0,0.2,0.4,0.6,0.8,1])
plt.title('MSE Model I $>$ MSE Model II')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=20,ncol=3)
#plt.savefig(os.path.join(out_paths["analysis"],"mse_example.jpeg"),dpi=400,bbox_inches="tight")
plt.show()

################
### Get Data ###
################

# Test data I, 2022
df_list_preds={f"fatalities001_2022_00_t01/cm?page={i}":i for i in range(1,8)}

df_all=pd.DataFrame()
for i in range(len(df_list_preds)):
    response = requests.get(f'https://api.viewsforecasting.org/{list(df_list_preds.keys())[i]}')
    json_data = response.json()
    df=pd.DataFrame(json_data["data"])
    df=df[['country_id','month_id','month','sc_cm_sb_main']]
    # Reverse log
    df['sc_cm_sb_main']=np.exp(df['sc_cm_sb_main'])-1
    df_all = pd.concat([df_all, df])
    df_all=df_all.reset_index(drop=True)
cc_sort=df_all.country_id.unique()
cc_sort.sort()
df_preds_test_1 = df_all.pivot(index="month_id",columns='country_id', values='sc_cm_sb_main')

# Test data II, 2023
df_list_preds={f"fatalities001_2023_00_t01/cm?page={i}":i for i in range(1,8)}

df_all=pd.DataFrame()
for i in range(len(df_list_preds)):
    response = requests.get(f'https://api.viewsforecasting.org/{list(df_list_preds.keys())[i]}')
    json_data = response.json()
    df=pd.DataFrame(json_data["data"])
    df=df[['country_id','month_id','month','sc_cm_sb_main']]
    # Reverse log
    df['sc_cm_sb_main']=np.exp(df['sc_cm_sb_main'])-1
    df_all = pd.concat([df_all, df])
    df_all=df_all.reset_index(drop=True)
cc_sort=df_all.country_id.unique()
cc_sort.sort()
df_preds_test_2 = df_all.pivot(index="month_id",columns='country_id', values='sc_cm_sb_main')

# Input data, 1989-2023
df_list_input={f"predictors_fatalities002_0000_00/cm?page={i}":i for i in range(1,78)}

df_input_t=pd.DataFrame()
for i in range(len(df_list_input)):
    response = requests.get(f'https://api.viewsforecasting.org/{list(df_list_input.keys())[i]}')
    json_data = response.json()
    df=pd.DataFrame(json_data["data"])
    df=df[["country_id","month_id","ucdp_ged_sb_best_sum"]]
    df_input_t = pd.concat([df_input_t, df])
    df_input_t=df_input_t.reset_index(drop=True)

# Get regions
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

# Add labels to input data
df_input_s = df_input_t[df_input_t['country_id'].isin(cc_sort)]
df_input = df_input_s.pivot(index="month_id",columns='country_id',values='ucdp_ged_sb_best_sum')
df_input.index = pd.date_range('01/01/1989',periods=len(df_input),freq='M')
df_input = df_input.iloc[:-1,:]
df_input.columns = country_list['name']

# Save
#df_input.to_csv('df_input.csv')

# Load
df_input=pd.read_csv('df_input.csv',index_col=0,parse_dates=True)

# Fix missing values
df_tot_m = df_input.copy()
df_tot_m.replace(0, np.nan, inplace=True)
df_tot_m = df_tot_m.dropna(axis=1, how='all')
df_tot_m = df_tot_m.fillna(0)
#df_tot_m.to_csv('df_tot_m.csv')

# Save
df_obs_1 = df_input.iloc[-24:-12,:]        
df_obs_1.to_csv('obs1.csv')   
df_obs_2 = df_input.iloc[-12:,:]        
df_obs_2.to_csv('obs2.csv')   

df_v_1 = df_preds_test_1.iloc[:12,:]
df_v_1.to_csv('views1.csv')  
df_v_2 = df_preds_test_2.iloc[:12,:]
df_v_2.to_csv('views2.csv')  

col_sums=df_input.sum(axis=0)
col_sums.median()
np.percentile(col_sums,[10,20,30,40,50,60,70,80,90])

### Maximum 
fig,ax = plt.subplots(figsize=(12,8))
sns.kdeplot(df_input["Syria"],color="black", shade=True)
sns.kdeplot(df_input["Afghanistan"],color="gray", shade=True)
fig.legend(labels=['Syria','Afghanistan'],loc='upper center', bbox_to_anchor=(0.7, 0.88), fontsize=20,ncol=2)
ax.set_xlabel("Number of fatalities per month")
#plt.savefig(os.path.join(out_paths["analysis"],"example_max.jpeg"),dpi=400,bbox_inches="tight")
plt.show()

### 90th percentile
fig,ax = plt.subplots(figsize=(12,8))
sns.kdeplot(df_input["Philippines"],color="gray", shade=True)
sns.kdeplot(df_input["Algeria"],color="black", shade=True)
fig.legend(labels=['Philippines','Algeria'],loc='upper center', bbox_to_anchor=(0.7, 0.88), fontsize=20,ncol=2)
ax.set_xlabel("Number of fatalities per month")
#plt.savefig(os.path.join(out_paths["analysis"],"example_90per.jpeg"),dpi=400,bbox_inches="tight")
plt.show()

### 80th percentile
fig,ax = plt.subplots(figsize=(12,8))
sns.kdeplot(df_input["Egypt"],color="gray", shade=True)
sns.kdeplot(df_input["Peru"],color="black", shade=True)
fig.legend(labels=['Egypt','Peru'],loc='upper center', bbox_to_anchor=(0.7, 0.88), fontsize=20,ncol=2)
ax.set_xlabel("Number of fatalities per month")
#plt.savefig(os.path.join(out_paths["analysis"],"example_80per.jpeg"),dpi=400,bbox_inches="tight")
plt.show()

### 70th percentile
fig,ax = plt.subplots(figsize=(12,8))
sns.kdeplot(df_input["Bangladesh"],color="gray", shade=True)
sns.kdeplot(df_input["Moldova"],color="black", shade=True)
fig.legend(labels=['Bangladesh','Moldova'],loc='upper center', bbox_to_anchor=(0.7, 0.88), fontsize=20,ncol=2)
ax.set_xlabel("Number of fatalities per month")
#plt.savefig(os.path.join(out_paths["analysis"],"example_70per.jpeg"),dpi=400,bbox_inches="tight")
plt.show()

### 60th percentile
fig,ax = plt.subplots(figsize=(12,8))
sns.kdeplot(df_input["China"],color="gray", shade=True)
sns.kdeplot(df_input["Macedonia"],color="black", shade=True)
fig.legend(labels=['China','Macedonia'],loc='upper center', bbox_to_anchor=(0.7, 0.88), fontsize=20,ncol=2)
ax.set_xlabel("Number of fatalities per month")
#plt.savefig(os.path.join(out_paths["analysis"],"example_60per.jpeg"),dpi=400,bbox_inches="tight")
plt.show()

### 50th percentile
fig,ax = plt.subplots(figsize=(12,8))
sns.kdeplot(df_input["Kosovo"],color="gray", shade=True)
sns.kdeplot(df_input["Belgium"],color="black", shade=True)
fig.legend(labels=['Kosovo','Belgium'],loc='upper center', bbox_to_anchor=(0.7, 0.88), fontsize=20,ncol=2)
ax.set_xlabel("Number of fatalities per month")
#plt.savefig(os.path.join(out_paths["analysis"],"example_50per.jpeg"),dpi=400,bbox_inches="tight")
plt.show()


####################
### Shape finder ###
####################

### Step 1. Get reference repository 

### For 2021 ###

h_train=10
dict_m={i :[] for i in df_input.columns} # until 2020
# Remove last two years in data to get training data
df_input_sub=df_input.iloc[:-24]
# For each country in df
for coun in range(len(df_input_sub.columns)):
    # If the last h_train observations of training data are not flat, run Shape finder, else pass    
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        shape = Shape()
        # Set last h_train observations of training data as shape
        shape.set_shape(df_input_sub.iloc[-h_train:,coun]) 
        # Find matches in training data        
        find = finder(df_tot_m.iloc[:-24],shape)
        find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2)
        min_d_d=0.1
        # If there are fewer than 5 observations in reference, increase max distance until 5 observations are matched
        while len(find.sequences)<5:
            min_d_d += 0.05
            find.find_patterns(min_d=min_d_d,select=True,metric='dtw',dtw_sel=2)
        # Save matches and distances                
        dict_m[df_input.columns[coun]]=find.sequences
    else :
        pass

# For saving        
# with open('test1.pkl', 'wb') as f:
#     pickle.dump(dict_m, f) 

### For 2022 ###
    
h_train=10
dict_m={i :[] for i in df_input.columns}
# Remove last year in data to get training data
df_input_sub=df_input.iloc[:-12]
# For each country in df
for coun in range(len(df_input_sub.columns)):
    # If the last h_train observations of training data are not flat, run Shape finder, else pass
    if not (df_input_sub.iloc[-h_train:,coun]==0).all():
        shape = Shape()
        # Set last h_train observations of training data as shape        
        shape.set_shape(df_input_sub.iloc[-h_train:,coun]) 
        # Find matches in training data        
        find = finder(df_tot_m.iloc[:-12],shape)
        find.find_patterns(min_d=0.1,select=True,metric='dtw',dtw_sel=2)
        min_d_d=0.1
        # If there are fewer than 5 observations in reference, increase max distance until 5 observations are matched        
        while len(find.sequences)<5:
            min_d_d += 0.05
            find.find_patterns(min_d=min_d_d,select=True,metric='dtw',dtw_sel=2)
        # Save matches and distances        
        dict_m[df_input.columns[coun]]=find.sequences
    else :
        pass
    
# For saving        
# with open('test2.pkl', 'wb') as f:
#     pickle.dump(dict_m, f)
      
### Step 2. Make predictions

### Using 2021 to make predictions in 2022 ###

with open('test1.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
    
len_mat=[]      
df_sure=[]
pr_list=[]   
pr_main=[]
pr_scale=[]  

df_input_sub=df_input.iloc[:-24]
horizon=12
h_train=10

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
        tot_seq_c = tot_seq.copy()
        
        ### Apply hierachical clustering to sequences in reference repository ###        
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        
        dn=dendrogram(linkage_matrix,color_threshold=horizon/3,above_threshold_color='black')
        # Access the lines (linkages) in the dendrogram and set their colors
        for i, d in zip(dn['icoord'], dn['dcoord']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > horizon/3:
                col = 'black'
            else:
                col = 'lightgray'
            plt.plot(i, d, color=col)
        plt.axhline(y=horizon/3, c='black', ls='--', lw=0.8)
        plt.xticks([])
        #plt.yticks([])
        #plt.savefig(os.path.join(out_paths["analysis"],f"dendogram_{coun}_2022.jpeg"),dpi=400,bbox_inches="tight")
        plt.show()
        
        # Proportion of cases assigned to each cluster
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        # Save maximum proportion
        pr_main.append(pr.max())
        # How many references are used        
        len_mat.append(len(tot_seq))
        # Save maximum value        
        pr_scale.append(df_input_sub.iloc[-h_train:,coun].sum())
        # Scale actuals
        testu = (df_input.iloc[-24:-12,coun] - df_input_sub.iloc[-h_train:,coun].min()) / (df_input_sub.iloc[-h_train:,coun].max() - df_input_sub.iloc[-h_train:,coun].min())
        # Append scaled actuals as last row in reference repository
        tot_seq_c = pd.concat([pd.DataFrame(tot_seq_c),pd.DataFrame(testu.reset_index(drop=True)).T],axis=0)
    
        ### Apply hierachical clustering to sequences in reference repository ###
        linkage_matrix_2 = linkage(tot_seq_c, method='ward')
        clusters_2 = fcluster(linkage_matrix_2, horizon/3, criterion='distance')
        
        # If number of clusters is 1 append to sure repository         
        if len(pd.Series(clusters_2).value_counts())==1:
            df_sure.append(tot_seq_c)
        # If first and second clustering have same number of clusters append 
        # proportion of input sequence            
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
        
        # Plot 
        plt.figure(figsize=(10, 6))
        plt.plot(preds, label='Shape Finder', marker='o',color='purple')
        plt.plot(df_preds_test_1.iloc[:12,coun].reset_index(drop=True), label='ViEWS', marker='o',color='darkgreen')
        plt.plot(df_input.iloc[-24:-12,coun].reset_index(drop=True),label='Actuals',marker='o',color="black")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5,fontsize=12)
        plt.title(df_input_sub.columns[coun])
        plt.grid(True)
        #plt.savefig(os.path.join(out_paths["analysis"],f"compare_preds_{coun}.jpeg"),dpi=400,bbox_inches="tight")
        plt.show()           
    
    else:
        # Add zeros
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
        
# Save        
df_sf_1 = pd.concat(pred_tot_pr,axis=1)
df_sf_1.columns=country_list['name']
df_sf_1.to_csv('sf1.csv')  

### Get MSE ###
err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], pred_tot_pr[i]))
    err_views.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]))
err_sf_pr = np.array(err_sf_pr)
err_views = np.array(err_views)
mse_list=np.log((err_views+1)/(err_sf_pr+1))    
 
### Using 2022 to make predictions in 2023 ###

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
        tot_seq_c = tot_seq.copy()
        
        ### Apply hierachical clustering to sequences in reference repository ###        
        linkage_matrix = linkage(tot_seq, method='ward')
        clusters = fcluster(linkage_matrix, horizon/3, criterion='distance')
        
        dn=dendrogram(linkage_matrix,color_threshold=horizon/3,above_threshold_color='black')
        # Access the lines (linkages) in the dendrogram and set their colors
        for i, d in zip(dn['icoord'], dn['dcoord']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > horizon/3:
                col = 'black'
            else:
                col = 'lightgray'
            plt.plot(i, d, color=col)
        plt.axhline(y=horizon/3, c='black', ls='--', lw=0.8)
        plt.xticks([])
        #plt.yticks([])
        #plt.savefig(os.path.join(out_paths["analysis"],f"dendogram_{coun}_2023.jpeg"),dpi=400,bbox_inches="tight")
        plt.show()
        
        # Proportion of cases assigned to each cluster        
        pr = pd.Series(clusters).value_counts(normalize=True).sort_index()
        # Save maximum proportion
        pr_main.append(pr.max())
        # How many references are used
        len_mat.append(len(tot_seq))
        # Save maximum value
        pr_scale.append(df_input_sub.iloc[-h_train:,coun].sum())
        # Scale actuals
        testu = (df_input.iloc[-12:,coun] - df_input_sub.iloc[-h_train:,coun].min()) / (df_input_sub.iloc[-h_train:,coun].max() - df_input_sub.iloc[-h_train:,coun].min())
        # Append scaled actuals as last row in reference repository        
        tot_seq_c = pd.concat([pd.DataFrame(tot_seq_c),pd.DataFrame(testu.reset_index(drop=True)).T],axis=0)
        
        ### Apply hierachical clustering to sequences in reference repository ###        
        linkage_matrix_2 = linkage(tot_seq_c, method='ward')
        clusters_2 = fcluster(linkage_matrix_2, horizon/3, criterion='distance')
        
        # If number of clusters is 1 append to sure repository                 
        if len(pd.Series(clusters_2).value_counts())==1:
            df_sure.append(tot_seq_c)
        # If first and second clustering have same number of clusters append 
        # proportion of input sequence               
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
        
        # Plot 
        plt.figure(figsize=(10, 6))
        #plt.plot(preds, label='Shape Finder', marker='o',color='purple')
        plt.plot(df_preds_test_1.iloc[:12,coun].reset_index(drop=True), label='ViEWS', marker='o',color='darkgreen')
        plt.plot(df_input.iloc[-24:-12,coun].reset_index(drop=True),label='Actuals',marker='o',color="black")
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5,fontsize=12)
        plt.title(df_input_sub.columns[coun])
        plt.grid(True)
        #plt.savefig(os.path.join(out_paths["analysis"],f"compare_preds_{coun}.jpeg"),dpi=400,bbox_inches="tight")
        plt.show()    
    
    else:
        # Add zeros         
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))  
     
# Save
df_sf_2= pd.concat(pred_tot_pr,axis=1)
df_sf_2.columns=country_list['name']
df_sf_2.to_csv('sf2.csv')  

### Get MSE ###
err_sf_pr=[]
err_views=[]
for i in range(len(df_input.columns)):   
    err_sf_pr.append(mean_squared_error(df_input.iloc[-12:,i], pred_tot_pr[i]))
    err_views.append(mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]))
err_sf_pr = np.array(err_sf_pr)
err_views = np.array(err_views)
mse_list2=np.log((err_views+1)/(err_sf_pr+1))  


### Plot proportion of cluster the input sequences were assigned to ###

# Total log ratios between Views and Shape finder
mse_list_tot=np.concatenate([mse_list,mse_list2],axis=0)

# Proportions
pr_list=pd.Series(pr_list)
pr_main=pd.Series(pr_main)
pr_scale=pd.Series(pr_scale)
len_mat=pd.Series(len_mat)

# Categorize proportions
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

# Plot I
categories = ['New scenario', 'Low probability (0 to 0.5)', 'High proba (0.5 to 1)', 'Sure 100\%']
percentages = [nan_percentage, zero_to_half,half_to_02, ones]
plt.figure(figsize=(10, 6))
plt.bar(categories, percentages, color=['lightblue', 'lightblue', 'lightblue', 'lightblue'])
plt.title('Proportion of cluster the input sequence was assigned to')
plt.xlabel('Categories')
plt.ylabel('Percentage')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Plot II
df_tot_res = df_tot_res.dropna()
plt.boxplot(df_tot_res[df_tot_res[2]=='New'][0],positions=[0])
plt.boxplot(df_tot_res[df_tot_res[2]=='Low'][0],positions=[1])
plt.boxplot(df_tot_res[df_tot_res[2]=='High'][0],positions=[2])
plt.boxplot(df_tot_res[df_tot_res[2]=='Sure'][0],positions=[3])
plt.xticks([0,1,2,3],['New', 'Low', 'High', 'Sure'])
plt.title('Distribution of the proportion of cluster the input sequence was assigned to')
plt.ylim(-3,3)
plt.show()

# Plots for "sure" sequences 
for i in range(len(df_sure)):
    for j in range(len(df_sure[i])-1):
        plt.plot(df_sure[i].iloc[j,:],color='black',alpha=0.3)
    plt.plot(df_sure[i].iloc[:-1].mean(),color='black',label="Mean of matches")
    plt.plot(df_sure[i].iloc[-1,:],color='red',label="Input")
    plt.title(f'{df_sure[i].iloc[-1,:].name}, Distance = {ed.distance(df_sure[i].iloc[:-1].mean(),df_sure[i].iloc[-1,:])}')
    plt.legend()
    plt.show()
    
###############################################    
### Compound between Shape Finder and ViEWS ###
###############################################    

# Selection criterion for compound   
df_sel = pd.concat([df_tot_res.iloc[:,0].reset_index(drop=True),pr_scale,pr_main,len_mat],axis=1)
df_sel = df_sel.dropna()
df_sel.columns=['log MSE','Scale','Main_Pr','N_Matches']
df_sel['Confidence']=df_sel.iloc[:,2]*np.log10(df_sel.iloc[:,3])
#n_df_sel= df_sel[df_sel['log MSE'] <= -0.2]
#p_df_sel= df_sel[df_sel['log MSE'] >= 0.2]
n_df_sel= df_sel[df_sel['log MSE'] <= 0]
p_df_sel= df_sel[df_sel['log MSE'] > 0]

fig,ax = plt.subplots(figsize=(12,8))
plt.scatter(n_df_sel.iloc[:,1],n_df_sel.iloc[:,2]*np.log10(n_df_sel.iloc[:,3]),label='Negative log-ratio',color='black',s=50)
plt.scatter(p_df_sel.iloc[:,1],p_df_sel.iloc[:,2]*np.log10(p_df_sel.iloc[:,3]),label='Positive log-ratio',color='gray',s=50)
plt.xscale('log')
x_values = np.linspace(0, 100000, 1000)
plt.plot(x_values, np.exp(5.2*np.log10(x_values) - 25) + 0.6, color='black',linestyle='--',linewidth=3)
plt.legend()
plt.xlabel('Severity, Number of fatalities',size=20)
plt.ylabel('Confidence, p*log(N)',size=20)
plt.ylim(0.2,1.7)
plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1))
#plt.savefig(os.path.join(out_paths["analysis"],"compound_select.jpeg"),dpi=400,bbox_inches="tight")
plt.show()

df_sel_s = df_sel.sort_values(['Scale'])
df_sel_s=df_sel_s[df_sel_s['Confidence']>0.6] 
df_sel_s=df_sel_s[df_sel_s['Scale']<25000]
df_keep_1 = df_sel_s.index
ttest_1samp(df_sel_s.iloc[:,0],0)

df_try=pd.concat([df_sel_s.iloc[:,0],pd.Series([0]*(111-len(df_sel_s)))])
ttest_1samp(df_try,0)

##################
### Evaluation ###
##################

# Function to get difference explained
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
    
# Calculate difference explaines
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

# Calculate MSE
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
d_mix = d_b.copy()
d_mix[ind_keep[df_keep_1]] = d_nn[ind_keep[df_keep_1]]
d_nn = d_nn[~np.isnan(d_nn)]
d_b = d_b[~np.isnan(d_b)]
d_null = d_null[~np.isnan(d_null)]
d_t1= d_t1[~np.isnan(d_t1)]
d_mix = d_mix[~np.isnan(d_mix)]


# Difference explained
means = [d_nn.mean(),d_b.mean(),d_null.mean(),d_t1.mean(),d_mix.mean()]
std_error = [2*d_nn.std()/np.sqrt(len(d_nn)),2*d_b.std()/np.sqrt(len(d_b)),2*d_null.std()/np.sqrt(len(d_null)),2*d_t1.std()/np.sqrt(len(d_t1)),2*d_mix.std()/np.sqrt(len(d_mix))]
mean_de = pd.DataFrame({
    'mean': means,
    'std': std_error
})

# MSE
means = [err_sf_pr.mean(),err_views.mean(),err_zero.mean(),err_t1.mean(),err_mix.mean()]
std_error = [2*err_sf_pr.std()/np.sqrt(len(err_sf_pr)),2*err_views.std()/np.sqrt(len(err_views)),2*err_zero.std()/np.sqrt(len(err_zero)),2*err_t1.std()/np.sqrt(len(err_t1)),2*err_mix.std()/np.sqrt(len(err_mix))]
mean_mse = pd.DataFrame({
    'mean': means,
    'std': std_error
})

fig,ax = plt.subplots(figsize=(12,8))

plt.scatter(mean_mse["mean"][0],mean_de["mean"][0],color="black",s=150)
#plt.plot([mean_mse["mean"][0],mean_mse["mean"][0]],[mean_de["mean"][0]-mean_de["std"][0],mean_de["mean"][0]+mean_de["std"][0]],linewidth=3,color="black")
#plt.plot([mean_mse["mean"][0]-mean_mse["std"][0],mean_mse["mean"][0]+mean_mse["std"][0]],[mean_de["mean"][0],mean_de["mean"][0]],linewidth=3,color="black")

plt.scatter(mean_mse["mean"][1],mean_de["mean"][1],color="black",s=150)
#plt.plot([mean_mse["mean"][1],mean_mse["mean"][1]],[mean_de["mean"][1]-mean_de["std"][1],mean_de["mean"][1]+mean_de["std"][1]],linewidth=3,color="black")
#plt.plot([mean_mse["mean"][1]-mean_mse["std"][1],mean_mse["mean"][1]+mean_mse["std"][1]],[mean_de["mean"][1],mean_de["mean"][1]],linewidth=3,color="black")

plt.scatter(mean_mse["mean"][2],mean_de["mean"][2],color="black",s=150)
#plt.plot([mean_mse["mean"][2],mean_mse["mean"][2]],[mean_de["mean"][2]-mean_de["std"][2],mean_de["mean"][2]+mean_de["std"][2]],linewidth=3,color="black")
#plt.plot([mean_mse["mean"][2]-mean_mse["std"][2],mean_mse["mean"][2]+mean_mse["std"][2]],[mean_de["mean"][2],mean_de["mean"][2]],linewidth=3,color="black")

plt.scatter(mean_mse["mean"][3],mean_de["mean"][3],color="black",s=150)
#plt.plot([mean_mse["mean"][3],mean_mse["mean"][3]],[mean_de["mean"][3]-mean_de["std"][3],mean_de["mean"][3]+mean_de["std"][3]],linewidth=3,color="black")
#plt.plot([mean_mse["mean"][3]-mean_mse["std"][3],mean_mse["mean"][3]+mean_mse["std"][3]],[mean_de["mean"][3],mean_de["mean"][3]],linewidth=3,color="black")

#plt.scatter(mean_mse["mean"][4],mean_de["mean"][4],color="gray",s=150)
#plt.plot([mean_mse["mean"][4],mean_mse["mean"][4]],[mean_de["mean"][4]-mean_de["std"][4],mean_de["mean"][4]+mean_de["std"][4]],linewidth=3,color="gray")
#plt.plot([mean_mse["mean"][4]-mean_mse["std"][4],mean_mse["mean"][4]+mean_mse["std"][4]],[mean_de["mean"][4],mean_de["mean"][4]],linewidth=3,color="gray")


plt.xlabel("Accuracy  (MSE reversed)")
plt.ylabel("Difference explained  (DE)")
ax.invert_xaxis()
plt.text(1600000, 0.265, "Shape Finder", size=20, color='black')
plt.text(1570000, 0.231, "ViEWS", size=20, color='black')
plt.text(1500000, 0.005, "Null", size=20, color='black')
plt.text(3000000, 0.29, "t-1", size=20, color='black')
ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25,0.3,0.35])
ax.set_xticks([0, 1500000,3000000,4500000,6000000,7500000])
#plt.savefig(os.path.join(out_paths["analysis"],"scatter1.jpeg"),dpi=400,bbox_inches="tight")

# Difference explained
means = [np.log((d_nn+1)/(d_mix+1)).mean(),np.log((d_b+1)/(d_mix+1)).mean(),np.log((d_null+1)/(d_mix+1)).mean(),np.log((d_t1+1)/(d_mix+1)).mean()]
std_error = [2*np.log((x+1)/(d_mix+1)).std()/np.sqrt(len((x-d_mix))) for x in [d_nn,d_b,d_null,d_t1]]
mean_de = pd.DataFrame({
    'mean': means,
    'std': std_error
})

# MSE
means = [np.log((err_sf_pr+1)/(err_mix+1)).mean(),np.log((err_views+1)/(err_mix+1)).mean(),np.log((err_zero+1)/(err_mix+1)).mean(),np.log((err_t1+1)/(err_mix+1)).mean()]
std_error = [2*np.log((x+1)/(err_mix+1)).std()/np.sqrt(len((x-err_mix))) for x in [err_sf_pr,err_views,err_zero,err_t1]]
mean_mse = pd.DataFrame({
    'mean': means,
    'std': std_error
})

name=['SF','Views','Null','t-1']

fig,ax = plt.subplots(figsize=(12,8))
for i in range(4):
    plt.scatter(mean_mse["mean"][i],mean_de["mean"][i],color="gray",s=150)
    plt.plot([mean_mse["mean"][i],mean_mse["mean"][i]],[mean_de["mean"][i]-mean_de["std"][i],mean_de["mean"][i]+mean_de["std"][i]],linewidth=3,color="gray")
    plt.plot([mean_mse["mean"][i]-mean_mse["std"][i],mean_mse["mean"][i]+mean_mse["std"][i]],[mean_de["mean"][i],mean_de["mean"][i]],linewidth=3,color="gray")
plt.scatter(0,0,color="black",s=150)
plt.xlabel("Accuracy ratio (MSE reversed)")
plt.ylabel("Difference explained ratio (DE)")

plt.xlim(0.5,-0.05)
plt.ylim(-0.2,0.06)
plt.text(0.29, 0.005, "t-1", size=20, color='dimgray')
plt.text(0.036,-0.156, "Null", size=20, color='dimgray')
plt.text(0.034, -0.03, "ViEWS", size=20, color='dimgray')
plt.text(0.16, -0.002, 'Shape Finder', size=20,color="dimgray")
plt.text(0.033, 0.008, 'Compound', size=20,color="black")
ax.set_yticks([-0.2,-0.15,-0.1,-0.05,0,0.05])
ax.set_xticks([0, 0.1,0.2,0.3,0.4,0.5])
#plt.savefig(os.path.join(out_paths["analysis"],"scatter2.jpeg"),dpi=400,bbox_inches="tight")
plt.show()

# Plot cases, where Shape finder has lower MSE then ViEWSFore 
for i in range(len(df_input.columns)): 
    if (df_input.iloc[-24:-24+horizon,i]==0).all()==False:
        if mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i])*2<mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]):
            plt.plot(df_input.iloc[-24:-24+horizon,i].reset_index(drop=True),linewidth=2,color='black')
            plt.plot(df_sf_1.iloc[:,i].reset_index(drop=True),linewidth=5,color='purple')
            plt.plot(df_preds_test_1.iloc[:12,i].reset_index(drop=True),linewidth=2,color='grey')
            plt.box(False)
            plt.xticks([])
            plt.yticks([])
            plt.title(f'{df_input.columns[i]} - 2022')
            plt.show()
    if (df_input.iloc[-12:,i]==0).all()==False:
        if mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i])*2<mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]):
            plt.plot(df_input.iloc[-12:,i].reset_index(drop=True),linewidth=2,color='black')
            plt.plot(df_sf_2.iloc[:,i].reset_index(drop=True),linewidth=5,color='purple')
            plt.plot(df_preds_test_2.iloc[:12,i].reset_index(drop=True),linewidth=2,color='grey')
            plt.box(False)
            plt.xticks([])
            plt.yticks([])
            plt.title(f'{df_input.columns[i]} - 2023')
            plt.show()
