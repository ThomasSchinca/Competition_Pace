# -*- coding: utf-8 -*-

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
import matplotlib.colors as mcolors
from scipy.stats import linregress
import matplotlib.lines as mlines
from datetime import timedelta
#import dtreeviz
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn import tree

plot_params = {"text.usetex":True,"font.family":"serif","font.size":20,"xtick.labelsize":20,"ytick.labelsize":20,"axes.labelsize":20,"figure.titlesize":20,"figure.figsize":(8,5),"axes.prop_cycle":cycler(color=['black','rosybrown','gray','indianred','red','maroon','silver',])}
plt.rcParams.update(plot_params)
      
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

# Input data, 1990-2023
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
df_input.index = pd.date_range('01/01/1990',periods=len(df_input),freq='M')
df_input = df_input.iloc[:408,:]
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
#df_obs_1.to_csv('obs1.csv')   
df_obs_2 = df_input.iloc[-12:,:]        
#df_obs_2.to_csv('obs2.csv')   

df_v_1 = df_preds_test_1.iloc[:12,:]
#df_v_1.to_csv('views1.csv')  
df_v_2 = df_preds_test_2.iloc[:12,:]
#df_v_2.to_csv('views2.csv')  

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
            #plt.plot(i, d, color=col)
        #plt.axhline(y=horizon/3, c='black', ls='--', lw=0.8)
        #plt.xticks([])
        #plt.yticks([])
        #plt.savefig(os.path.join(out_paths["analysis"],f"dendogram_{coun}_2022.jpeg"),dpi=400,bbox_inches="tight")
        #plt.show()
        
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
        plt.plot(preds, label='DTP model', linestyle="dashed",color='black',linewidth=2)
        plt.plot(df_preds_test_1.iloc[:12,coun].reset_index(drop=True), label='ViEWS ensemble',linestyle="dotted",color='black',linewidth=2)
        plt.plot(df_obs_1.iloc[:,coun].reset_index(drop=True),label='Actuals',linestyle="solid",color="black",linewidth=2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5,fontsize=20)
        #plt.title(f"{df_input_sub.columns[coun]}",size=30)
        plt.xticks([0,2,4,6,8,10],["2022-01","2022-03","2022-05","2022-07","2022-09","2022-11"],size=25)
        plt.yticks(size=25)
        plt.savefig(f"out/compare_preds_{df_input_sub.columns[coun]}_2022.jpeg",dpi=400,bbox_inches="tight")
        plt.show()    
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_preds_test_1.iloc[:12,coun].reset_index(drop=True), label='ViEWS ensemble',linestyle="dotted",color='black',linewidth=2)
        plt.plot(df_obs_1.iloc[:,coun].reset_index(drop=True),label='Actuals',linestyle="solid",color="black",linewidth=2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5,fontsize=20)
        #plt.title(f"Risk averse prediction for {df_input_sub.columns[coun]}",size=30)
        plt.xticks([0,2,4,6,8,10],["2022-01","2022-03","2022-05","2022-07","2022-09","2022-11"],size=25)
        plt.yticks(size=25)
        plt.savefig(f"out/risk_averse_{df_input_sub.columns[coun]}_2022.jpeg",dpi=400,bbox_inches="tight")
        plt.show()        
    
    else:
        # Add zeros
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(pd.Series(np.zeros((horizon,))), label='DTP model', linestyle="dashed",color='black',linewidth=2)
        plt.plot(df_preds_test_1.iloc[:12,coun].reset_index(drop=True), label='ViEWS ensemble',linestyle="dotted",color='black',linewidth=2)
        plt.plot(df_obs_1.iloc[:,coun].reset_index(drop=True),label='Actuals',linestyle="solid",color="black",linewidth=2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5,fontsize=20)
        #plt.title(f"Predictions for {df_input_sub.columns[coun]}",size=30)
        plt.xticks([0,2,4,6,8,10],["2022-01","2022-03","2022-05","2022-07","2022-09","2022-11"],size=25)
        plt.yticks(size=25)
        plt.savefig(f"out/compare_preds_{df_input_sub.columns[coun]}_2022.jpeg",dpi=400,bbox_inches="tight")
        plt.show()  
        
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

mse_list_raw = err_sf_pr.copy()
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
            #plt.plot(i, d, color=col)
        #plt.axhline(y=horizon/3, c='black', ls='--', lw=0.8)
        #plt.xticks([])
        #plt.yticks([])
        #plt.savefig(os.path.join(out_paths["analysis"],f"dendogram_{coun}_2023.jpeg"),dpi=400,bbox_inches="tight")
        #plt.show()
        
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
        
        # # Plot 
        plt.figure(figsize=(10, 6))
        plt.plot(preds, label='DTP model', linestyle="dashed",color='black',linewidth=2)
        plt.plot(df_preds_test_2.iloc[:12,coun].reset_index(drop=True), label='ViEWS ensemble',linestyle="dotted",color='black',linewidth=2)
        plt.plot(df_obs_2.iloc[:,coun].reset_index(drop=True),label='Actuals',linestyle="solid",color="black",linewidth=2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5,fontsize=20)
        #plt.title(f"Predictions for {df_input_sub.columns[coun]}",size=30)
        plt.xticks([0,2,4,6,8,10],["2023-01","2023-03","2023-05","2023-07","2023-09","2023-11"],size=25)
        plt.yticks(size=25)
        plt.savefig(f"out/compare_preds_{df_input_sub.columns[coun]}_2023.jpeg",dpi=400,bbox_inches="tight")
        plt.show()    
        
        plt.figure(figsize=(10, 6))
        plt.plot(df_preds_test_2.iloc[:12,coun].reset_index(drop=True), label='ViEWS ensemble',linestyle="dotted",color='black',linewidth=2)
        plt.plot(df_obs_2.iloc[:,coun].reset_index(drop=True),label='Actuals',linestyle="solid",color="black",linewidth=2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5,fontsize=20)
        #plt.title(f"Risk averse prediction for {df_input_sub.columns[coun]}",size=30)
        plt.xticks([0,2,4,6,8,10],["2023-01","2023-03","2023-05","2023-07","2023-09","2023-11"],size=25)
        plt.yticks(size=25)
        plt.savefig(f"out/risk_averse_{df_input_sub.columns[coun]}_2023.jpeg",dpi=400,bbox_inches="tight")
        plt.show()   
    
    else:
        # Add zeros         
        pred_tot_min.append(pd.Series(np.zeros((horizon,))))
        pred_tot_pr.append(pd.Series(np.zeros((horizon,))))  
        
        # Plot 
        plt.figure(figsize=(10, 6))
        plt.plot(pd.Series(np.zeros((horizon,))), label='DTP model', linestyle="dashed",color='black',linewidth=2)
        plt.plot(df_preds_test_2.iloc[:12,coun].reset_index(drop=True), label='ViEWS ensemble',linestyle="dotted",color='black',linewidth=2)
        plt.plot(df_obs_2.iloc[:,coun].reset_index(drop=True),label='Actuals',linestyle="solid",color="black",linewidth=2)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5,fontsize=20)
        #plt.title(f"Predictions for {df_input_sub.columns[coun]}",size=30)
        plt.xticks([0,2,4,6,8,10],["2023-01","2023-03","2023-05","2023-07","2023-09","2023-11"],size=25)
        plt.yticks(size=25)
        plt.savefig(f"out/compare_preds_{df_input_sub.columns[coun]}_2023.jpeg",dpi=400,bbox_inches="tight")
        plt.show()  
     
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
mse_list_raw = np.concatenate([mse_list_raw,err_sf_pr],axis=0)

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

# # Plot I
# categories = ['New scenario', 'Low probability (0 to 0.5)', 'High proba (0.5 to 1)', 'Sure 100\%']
# percentages = [nan_percentage, zero_to_half,half_to_02, ones]
# plt.figure(figsize=(10, 6))
# plt.bar(categories, percentages, color=['lightblue', 'lightblue', 'lightblue', 'lightblue'])
# plt.title('Proportion of cluster the input sequence was assigned to')
# plt.xlabel('Categories')
# plt.ylabel('Percentage')
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

# # Plot II
df_tot_res = df_tot_res.dropna()
# plt.boxplot(df_tot_res[df_tot_res[2]=='New'][0],positions=[0])
# plt.boxplot(df_tot_res[df_tot_res[2]=='Low'][0],positions=[1])
# plt.boxplot(df_tot_res[df_tot_res[2]=='High'][0],positions=[2])
# plt.boxplot(df_tot_res[df_tot_res[2]=='Sure'][0],positions=[3])
# plt.xticks([0,1,2,3],['New', 'Low', 'High', 'Sure'])
# plt.title('Distribution of the proportion of cluster the input sequence was assigned to')
# plt.ylim(-3,3)
# plt.show()

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
df_sel = pd.concat([df_tot_res.iloc[:,0].reset_index(drop=True),pr_scale,pr_main,len_mat],axis=1)
df_sel = df_sel.dropna()
df_sel.columns=['log MSE','Scale','Main_Pr','N_Matches']
plt.scatter(df_sel.iloc[:,2],df_sel.iloc[:,0]/df_sel.iloc[:,1])
plt.yscale('log')
plt.show()




# Selection criterion for compound   
ind_keep_mse =  df_tot_res.index
df_sel = pd.concat([df_tot_res.iloc[:,0].reset_index(drop=True),pr_scale,pr_main,len_mat],axis=1)
df_sel = df_sel.dropna()
df_sel.columns=['log MSE','Scale','Main_Pr','N_Matches']
df_sel['Confidence']=df_sel.iloc[:,2]*np.log10(df_sel.iloc[:,3])
n_df_sel= df_sel[df_sel['log MSE'] <= 0]
p_df_sel= df_sel[df_sel['log MSE'] > 0]

fig = plt.figure(figsize=(24, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(n_df_sel.iloc[:,1], n_df_sel.iloc[:,2] * np.log10(n_df_sel.iloc[:,3]), label='Negative log-ratio', color='darkgrey', s=50)
ax1.scatter(p_df_sel.iloc[:,1], p_df_sel.iloc[:,2] * np.log10(p_df_sel.iloc[:,3]), label='Positive log-ratio', color='purple', s=50)
ax1.set_xscale('log')
x_values = np.linspace(1, 100000, 1000)  # Start from 1 to avoid log(0)
ax1.plot(x_values, np.exp(5.2 * np.log10(x_values) - 25) + 0.6, color='black', linestyle='--', linewidth=3, label='Selection Function')
ax1.legend()
ax1.set_xlabel('Severity, Number of fatalities', size=20)
ax1.set_ylabel('Confidence, p*log(N)', size=20)
ax1.set_ylim(0.2, 1.7)
ax1.legend(fontsize=15)
ax1.tick_params(axis='both', labelsize=15)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
n_df_sel= df_sel[df_sel['Scale'] <= 25000]
p_df_sel= df_sel[df_sel['Scale'] > 25000]
sub_n_df_sel = n_df_sel[(n_df_sel['log MSE']>=0) & (n_df_sel['Confidence']>=0.6)]
sub_p_df_sel = n_df_sel[(n_df_sel['log MSE']<0) & (n_df_sel['Confidence']>=0.6)]
pos_under = n_df_sel[(n_df_sel['log MSE']>=0) & (n_df_sel['Confidence']<0.6)]
neg_under = n_df_sel[(n_df_sel['log MSE']<0) & (n_df_sel['Confidence']<0.6)]
pos_over = n_df_sel[(n_df_sel['log MSE']>=0) & (n_df_sel['Confidence']>=0.6)]
neg_over = n_df_sel[(n_df_sel['log MSE']<0) & (n_df_sel['Confidence']>=0.6)]
len_neg_under = len(neg_under['log MSE'])
len_pos_under = len(pos_under['log MSE'])
len_neg_over = len(neg_over['log MSE'])
len_pos_over = len(pos_over['log MSE'])
total_under = len_neg_under + len_pos_under
total_over = len_neg_over + len_pos_over
norm_len_neg_under = len_neg_under / total_under
norm_len_pos_under = len_pos_under / total_under
norm_len_neg_over = len_neg_over / total_over
norm_len_pos_over = len_pos_over / total_over
categories = ['Below Function', 'Above Function']
neg_values = [norm_len_neg_under, norm_len_neg_over]
pos_values = [norm_len_pos_under, norm_len_pos_over]
mean_under_negative = neg_under['log MSE'].mean()
mean_under_positive = pos_under['log MSE'].mean()
mean_over_negative = neg_over['log MSE'].mean()
mean_over_positive = pos_over['log MSE'].mean()
ax2 = fig.add_subplot(gs[0, 1])
bar_width = 0.5
bar_positions = range(len(categories))
bars1 = ax2.bar(bar_positions, neg_values, bar_width, label='Negative', color='lightgrey')
bars2 = ax2.bar(bar_positions, pos_values, bar_width, bottom=neg_values, label='Positive', color='#CBC3E3')
for bar in bars1:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{int(height * 100)} %', ha='center', va='center', color='black', fontsize=14)
for i, bar in enumerate(bars2):
    height = bar.get_height() + neg_values[i]
    ax2.text(bar.get_x() + bar.get_width() / 2, height - pos_values[i] / 2, f'{int(bar.get_height() * 100)} %', ha='center', va='center', color='purple', fontsize=14)
ax3 = ax2.twinx()
ax3.set_ylabel('Mean Log Ratio', fontsize=20)
ax3.set_ylim(-1, 1)
ax3.set_yticks(np.arange(0, max(mean_under_negative, mean_over_negative) + 10, 10))
ax3.plot([0, 1], [n_df_sel[n_df_sel['Confidence']<0.6]['log MSE'].mean(), n_df_sel[n_df_sel['Confidence']>=0.6]['log MSE'].mean()], marker='o', color='black', label='Mean')
ax3.axhline(0, color='black', linestyle='dotted')
ax2.set_xticks(bar_positions)
ax2.set_xticklabels(categories, fontsize=20)
ax2.set_xlim(-0.5, 1.5)
#ax2.set_xlabel('Percentage of Observations', fontsize=20)
#ax3.set_yticks([-0.5, 0, 0.5],size=15)
ax3.tick_params(axis='y', labelsize=15)
ax2.set_yticks([])
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['left'].set_visible(False)
plt.show()


df_sel_s = df_sel.sort_values(['Scale'])
df_sel_s=df_sel_s[df_sel_s['Confidence']>0.6] 
df_sel_s=df_sel_s[df_sel_s['Scale']<10000]
df_keep_1 = df_sel_s.index
ind_keep_mse=ind_keep_mse[df_keep_1]
ttest_1samp(df_sel.iloc[:,0],0)[1]
    
df_try=pd.concat([df_sel_s.iloc[:,0],pd.Series([0]*(111-len(df_sel_s)))])
ttest_1samp(df_try,0)

# bins = [-99,-5,-4,-3,-2, -1.5,-1,-0.75,-0.5,-0.25, 0,0.25,0.5, 0.75,1, 1.5,2,3,4,5,99]
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
# ax1.hist(df_sel['log MSE'], bins=bins, edgecolor='black')
# ax1.set_title('All points')
# ax1.set_ylabel('Frequency')
# ax2.hist(df_sel_s['log MSE'], bins=bins, edgecolor='black')
# ax2.set_title('confidence >0.6')
# ax2.set_xlabel('log MSE')
# ax2.set_ylabel('Frequency')
# plt.tight_layout()
# plt.xlim(-6,6)
# plt.show()

# fig,ax = plt.subplots(figsize=(12,8))
# plt.scatter(n_df_sel['Confidence'],n_df_sel['log MSE'])
# plt.scatter(p_df_sel['Confidence'],p_df_sel['log MSE'],marker='x',color='black')
# plt.scatter(sub_p_df_sel['Confidence'],sub_p_df_sel['log MSE'],marker='o',color='grey')
# plt.scatter(sub_n_df_sel['Confidence'],sub_n_df_sel['log MSE'],marker='o',color='purple')
# plt.hlines(0,0.2,2)
# plt.vlines(0.602,-5,5,linestyles='--',color='purple')
# plt.ylim(-5,5)
# plt.ylabel('MSE Log ratio',size=20)
# plt.xlabel('Confidence, p*log(N)',size=20)
# plt.show()

df_sel = pd.concat([df_tot_res.iloc[:,0].reset_index(drop=True),pr_scale,pr_main,len_mat],axis=1)
df_sel = df_sel.dropna()
df_sel.columns=['log MSE','Scale','Main_Pr','N_Matches']
df_sel['Confidence']=df_sel.iloc[:,2]*np.log10(df_sel.iloc[:,3])
#n_df_sel= df_sel[df_sel['log MSE'] <= -0.2]
#p_df_sel= df_sel[df_sel['log MSE'] >= 0.2]
n_df_sel= df_sel[df_sel['log MSE'] <= 0]
p_df_sel= df_sel[df_sel['log MSE'] > 0]

plt.figure(figsize=(14, 10))
plt.scatter(n_df_sel.iloc[:,1], n_df_sel.iloc[:,2] , color='darkgrey', s=70)
plt.scatter(p_df_sel.iloc[:,1], p_df_sel.iloc[:,2], color='black', s=70)
plt.xscale('log')
plt.xlabel('Severity, Number of fatalities',size=20)
plt.ylabel('Confidence, p*log(N)',size=20)
plt.hlines(0.6,0.6,1000000, color='black', linestyle='--', linewidth=2)
plt.vlines(10000,0,2, color='black', linestyle='--', linewidth=2)
plt.fill_betweenx(y=[0, 0.6], x1=0, x2=10000, color='grey', alpha=0.1, hatch='/',label='Confidence too low')
plt.fill_betweenx(y=[0.6, 2], x1=10000, x2=100000000, color='grey', alpha=0.1, hatch='\\',label='Severity too high')
plt.fill_betweenx(y=[0, 0.6], x1=10000, x2=100000000, color='grey', alpha=0.1, hatch='x')
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0.6,200000)
plt.ylim(0.1,1.8)
ax = plt.gca()
xticks = ax.get_xticks()
yticks = ax.get_yticks()
ax.set_xticklabels([f'{tick:.0f}' if tick != 10000 else f'$\mathbf{{{int(tick):,}}}$' for tick in xticks],fontsize=20)
ax.set_yticklabels([f'{tick:.1f}' if tick != 0.6000000000000001 else f'$\mathbf{{{tick:.1f}}}$' for tick in yticks],fontsize=20)
plt.savefig("out/compound_select.jpeg",dpi=400,bbox_inches="tight")
plt.show()

##################
### Evaluation ###
##################

# Function to get difference explained
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
plt.text(1600000, 0.265, "DTP", size=20, color='black')
plt.text(1570000, 0.231, "ViEWS", size=20, color='black')
plt.text(1500000, 0.005, "Null", size=20, color='black')
plt.text(3050000, 0.29, "t-1", size=20, color='black')
ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25,0.3,0.35])
ax.set_xticks([0, 1500000,3000000,4500000,6000000,7500000])
plt.savefig("out/scatter1.jpeg",dpi=400,bbox_inches="tight")

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
plt.text(0.285, 0.004, "t-1", size=20, color='dimgray')
plt.text(0.035,-0.158, "Null", size=20, color='dimgray')
plt.text(0.026, -0.03, "ViEWS", size=20, color='dimgray')
plt.text(0.09, -0.003, 'DTP', size=20,color="dimgray")
plt.text(0.033, 0.008, 'Compound', size=20,color="black")
ax.set_yticks([-0.2,-0.15,-0.1,-0.05,0,0.05])
ax.set_xticks([0, 0.1,0.2,0.3,0.4,0.5])
plt.savefig("out/scatter2.jpeg",dpi=400,bbox_inches="tight")
plt.show()

### Find best performing cases ###

for i in range(len(df_input.columns)): 
    if (df_input.iloc[-24:-24+horizon,i]==0).all()==False:
        if mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i])<mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]):
            print(f'{df_input.columns[i]} - 2022')
            print((mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]))-(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i])))
    if (df_input.iloc[-12:,i]==0).all()==False:
        if mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i])<mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]):
            print(f'{df_input.columns[i]} - 2023')
            print((mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]))-(mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i])))

for i in range(len(df_input.columns)): 
    if (df_input.iloc[-24:-24+horizon,i]==0).all()==False:
        if d_nn[i]<d_b[i]:
            print(f'{df_input.columns[i]} - 2022')
            print(d_b[i]-d_nn[i])
    if (df_input.iloc[-12:,i]==0).all()==False:
        if d_nn2[i]<d_b2[i]:
            print(f'{df_input.columns[i]} - 2023')
            print(d_b2[i]-d_nn2[i])

for i in range(len(df_input.columns)): 
    if (df_input.iloc[-24:-24+horizon,i]==0).all()==False:
        if mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i])<mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]):
            if d_nn[i]<d_b[i]:
                print(f'{df_input.columns[i]} - 2022')
                print((mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]))-(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i])))
                print(d_b[i]-d_nn[i])
    if (df_input.iloc[-12:,i]==0).all()==False:
        if mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i])<mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]):
            if d_nn2[i]<d_b2[i]:
                print(f'{df_input.columns[i]} - 2023')
                print((mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]))-(mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i])))
                print(d_b2[i]-d_nn2[i])

### Illustrate metrics #####       
for i in range(len(df_input.columns)):
    print(i)
    print(df_input.columns[i]) 

# Create a figure and a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(12,8))
# Plot data in each subplot
axs[0, 0].plot(df_sf_2.iloc[:,47], label='DTP model', linestyle="dashed",color='black',linewidth=2)
axs[0, 0].plot(df_preds_test_2.iloc[:12,47].reset_index(drop=True), label='ViEWS ensemble',linestyle="dotted",color='black',linewidth=2)
axs[0, 0].plot(df_obs_2.iloc[:,47].reset_index(drop=True),label='Actuals',linestyle="solid",color="black",linewidth=2)
axs[0, 0].set_xticks([])
axs[0, 0].set_yticks([])
axs[0, 0].spines['top'].set_visible(False)
axs[0, 0].spines['left'].set_visible(False)
axs[0, 0].spines['bottom'].set_visible(False)
axs[0, 0].spines['right'].set_visible(False)
axs[0, 0].set_title("Worse MSE",size=20)
axs[0, 0].set_ylabel("Better DE",size=20)
axs[0, 0].set_xlabel("Mali---2023",size=15)

axs[0, 1].plot(df_sf_1.iloc[:,117], linestyle="dashed",color='black',linewidth=2)
axs[0, 1].plot(df_preds_test_1.iloc[:12,117].reset_index(drop=True),linestyle="dotted",color='black',linewidth=2)
axs[0, 1].plot(df_obs_1.iloc[:,117].reset_index(drop=True),linestyle="solid",color="black",linewidth=2)
axs[0, 1].set_xticks([])
axs[0, 1].set_yticks([])
axs[0, 1].set_title('Better MSE',size=20)
axs[0, 1].set_xlabel("Saudi Arabia---2022",size=15)
axs[0, 1].spines['top'].set_visible(False)
axs[0, 1].spines['left'].set_visible(False)
axs[0, 1].spines['bottom'].set_visible(False)
axs[0, 1].spines['right'].set_visible(False)

axs[1, 0].plot(df_sf_1.iloc[:,88], linestyle="dashed",color='black',linewidth=2)
axs[1, 0].plot(df_preds_test_1.iloc[:12,88].reset_index(drop=True),linestyle="dotted",color='black',linewidth=2)
axs[1, 0].plot(df_obs_1.iloc[:,88].reset_index(drop=True),linestyle="solid",color="black",linewidth=2)
axs[1, 0].set_xticks([])
axs[1, 0].set_yticks([])
axs[1, 0].spines['top'].set_visible(False)
axs[1, 0].spines['left'].set_visible(False)
axs[1, 0].spines['bottom'].set_visible(False)
axs[1, 0].spines['right'].set_visible(False)
axs[1, 0].set_ylabel("Worse DE",size=20)
axs[1, 0].set_xlabel("Poland---2022",size=15)

axs[1, 1].plot(df_sf_1.iloc[:,152], linestyle="dashed",color='black',linewidth=2)
axs[1, 1].plot(df_preds_test_1.iloc[:12,152].reset_index(drop=True),linestyle="dotted",color='black',linewidth=2)
axs[1, 1].plot(df_obs_1.iloc[:,152].reset_index(drop=True),linestyle="solid",color="black",linewidth=2)
axs[1, 1].set_xticks([])
axs[1, 1].set_yticks([])
axs[1, 1].spines['top'].set_visible(False)
axs[1, 1].spines['left'].set_visible(False)
axs[1, 1].spines['bottom'].set_visible(False)
axs[1, 1].spines['right'].set_visible(False)
axs[1, 1].set_xlabel("Congo, DRC---2022",size=15)

#fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=5,fontsize=12)
plt.tight_layout()
plt.savefig("out/cross_tab.jpeg",dpi=400,bbox_inches="tight")
plt.show()
             
# =============================================================================
# Horizon MSE
# =============================================================================

def err(y_true,y_pred):
    return (np.log(y_true+1)-np.log(y_pred+1))**2

# Calculate MSE
dict_hor = {'DTP':[],'ViEWS':[],'Zeros':[],'t-1':[],'Compound':[]}
dict_hor = {'DTP':[],'ViEWS':[],'Zeros':[],'t-1':[]}
for h in range(12):
    err_sf_pr=[]
    err_views=[]
    err_zero=[]
    err_t1=[]
    err_mix=[]
    for i in range(len(df_input.columns)):   
        if (df_input.iloc[-34:-24,i]==0).all() == True:
            err_mix.append(err(df_input.iloc[-24+h,i], pd.Series(np.zeros((1,))).iloc[0]))
        elif i in ind_keep_mse:
            err_mix.append(err(df_input.iloc[-24+h,i], df_sf_1.iloc[h,i]))
        elif i not in ind_keep_mse.tolist():
            err_mix.append(err(df_input.iloc[-24+h,i], df_preds_test_1.iloc[h,i]))
        err_sf_pr.append(err(df_input.iloc[-24+h,i], df_sf_1.iloc[h,i]))
        err_views.append(err(df_input.iloc[-24+h,i], df_preds_test_1.iloc[h,i]))
        err_zero.append(err(df_input.iloc[-24+h,i], pd.Series(np.zeros((1,))).iloc[0]))
        err_t1.append(err(df_input.iloc[-24+h,i],df_input.iloc[-36+h,i])) 
        
        if (df_input.iloc[-22:-12,i]==0).all() == True:
            err_mix.append(err(df_input.iloc[-12+h,i], pd.Series(np.zeros((1,))).iloc[0]))
        elif i in ind_keep_mse-191:
            err_mix.append(err(df_input.iloc[-12+h,i], df_sf_2.iloc[h,i]))
        elif i not in ind_keep_mse-191:
            err_mix.append(err(df_input.iloc[-12+h,i], df_preds_test_2.iloc[h,i]))
            
        err_sf_pr.append(err(df_input.iloc[-12+h,i], df_sf_2.iloc[h,i]))
        err_views.append(err(df_input.iloc[-12+h,i], df_preds_test_2.iloc[h,i]))
        err_zero.append(err(df_input.iloc[-12+h,i], pd.Series(np.zeros((1,))).iloc[0]))
        err_t1.append(err(df_input.iloc[-12+h,i],df_input.iloc[-24+h,i]))
    err_sf_pr = np.log((pd.Series(err_sf_pr)+1)/(pd.Series(err_mix)+1))
    err_views = np.log((pd.Series(err_views)+1)/(pd.Series(err_mix)+1))
    err_zero = np.log((pd.Series(err_zero)+1)/(pd.Series(err_mix)+1))
    err_t1 = np.log((pd.Series(err_t1)+1)/(pd.Series(err_mix)+1))
    #err_mix = np.log((pd.Series(err_mix)+1)/(pd.Series(err_sf_pr)+1))
    
    dict_hor['DTP'].append([err_sf_pr.mean(),err_sf_pr.std()])
    dict_hor['ViEWS'].append([err_views.mean(),err_views.std()])
    dict_hor['Zeros'].append([err_zero.mean(),err_zero.std()])
    dict_hor['t-1'].append([err_t1.mean(),err_t1.std()])
    #dict_hor['Compound'].append([err_mix.mean(),err_mix.std()])

test = pd.DataFrame(dict_hor)
horizons = np.arange(len(test))+1
mean_mse=[]
sd_mse=[]
plt.figure(figsize=(14, 8))
for column,line in zip(test.columns[[0,1,2,3]],["dashed","dotted","dashed","dotted"]):
    means = test[column].apply(lambda x: x[0])
    stds = test[column].apply(lambda x: x[1])
    mean_mse.append(means)
    sd_mse.append(stds)
    plt.errorbar(horizons, means, yerr=1.96*stds/np.sqrt(382), label=column, capsize=5,color="black",linewidth=2,linestyle=line)
plt.xlabel('Prediction horizon',size=25)
plt.ylabel('Mean-squared error of log-transformed values',size=25)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=5,fontsize=20)
plt.axhline(0)
#plt.savefig("out/horizon_mse.jpeg",dpi=400,bbox_inches="tight")
plt.show()

mean_mse=pd.DataFrame(mean_mse).T
sd_mse=pd.DataFrame(sd_mse).T

# =============================================================================
# Horizon DE
# =============================================================================

def diff_explained_h(df_input,pred,k=5,horizon=12):
    real = df_input.copy()
    real=real.reset_index(drop=True)
    sf = pred.copy()
    sf=sf.reset_index(drop=True)
    max_s=[]
    if (real==0).all()==False:
        for value in real[1:].index:
            if (real[value]==real[value-1]):
                max_exp=0
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
            max_s.append(max_exp)
    else:
        max_s.append([0]*(horizon-1)) 
    return(max_s)

horizon=12
err_sf_pr=[]
err_views=[]
err_zero=[]
err_t1=[]
err_mix=[]
for i in range(len(df_input.columns)):   
    if (df_input.iloc[-24:-24+horizon,i]==0).all()==False:
        err_sf_pr.append(diff_explained_h(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i]))
        err_views.append(diff_explained_h(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]))
        err_zero.append(diff_explained_h(df_input.iloc[-24:-24+horizon,i], pd.Series(np.zeros((horizon,)))))
        err_t1.append(diff_explained_h(df_input.iloc[-24:-24+horizon,i],df_input.iloc[-24-horizon:-24,i]))
        if i in ind_keep_mse:
            err_mix.append(diff_explained_h(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i]))
        else:
            err_mix.append(diff_explained_h(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i]))
for i in range(len(df_input.columns)):   
    if (df_input.iloc[-12:,i]==0).all()==False:
        err_sf_pr.append(diff_explained_h(df_input.iloc[-12:,i], df_sf_2.iloc[:,i]))
        err_views.append(diff_explained_h(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]))
        err_zero.append(diff_explained_h(df_input.iloc[-12:,i], pd.Series(np.zeros((horizon,)))))
        err_t1.append(diff_explained_h(df_input.iloc[-12:,i],df_input.iloc[-24:-24+horizon,i]))
        if i in ind_keep_mse-191:
            err_mix.append(diff_explained_h(df_input.iloc[-12:,i], df_sf_2.iloc[:,i]))
        else:
            err_mix.append(diff_explained_h(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i]))

mean_de=[]
std_de=[]
sum_tot_m=[]
sd_tot_m=[]
mod2 = pd.DataFrame(err_mix)
for mod in [err_sf_pr,err_views,err_zero,err_t1]:
    mod = pd.DataFrame(mod)
    mean_de.append(mod.mean())
    std_de.append(mod.std())
    sum_de=[]
    sd_m=[]
    for i in range(11):
        sum_de.append(np.log((mod.iloc[:,i]+1)/(mod2.iloc[:,i]+1)).mean())
        sd_m.append(np.log((mod.iloc[:,i]+1)/(mod2.iloc[:,i]+1)).std())
    sum_tot_m.append(sum_de)
    sd_tot_m.append(sd_m)

mean_de=pd.DataFrame(sum_tot_m).T
std_de =pd.DataFrame(sd_tot_m).T

mean_de.columns=['DTP','ViEWS','Zeros','t-1']
std_de.columns = ['DTP','ViEWS','Zeros','t-1']

#mean_de.drop('Zeros', axis=1, inplace=True)
#std_de.drop('Zeros', axis=1, inplace=True)

horizons = np.arange(11)+1
plt.figure(figsize=(14, 8))
for column,line in zip(mean_de.columns,["dashed","dotted","solid","dashdot"]):
    means = mean_de[column]
    stds = std_de[column]
    plt.errorbar(horizons, means, yerr=1.96*stds/np.sqrt(136), label=column, capsize=5,color="black",linewidth=2,linestyle=line)
plt.xlabel('Prediction horizon',size=25)
plt.ylabel('Difference explained (DE)',size=25)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=5,fontsize=20)
plt.axhline(0)
#plt.savefig("out/horizon_de.jpeg",dpi=400,bbox_inches="tight")
plt.show()




# Define colors and labels
colors = ['black', 'gray', 'royalblue',"lightskyblue"]
letters=['A','B','C','D']
mean_mse.columns=['DTP','ViEWS','Null','t-1']
mean_de.columns=['DTP','ViEWS','Null','t-1']

name_m = mean_mse.iloc[:, [0, 1, 2,3]].columns.tolist()
fig,ax = plt.subplots(figsize=(12,8))
for i, col in enumerate(mean_mse.iloc[:, [0, 1, 2,3]].columns):
    x_values = []
    y_values = []
    line_x=[]
    line_y=[]
    for k in range(4):
        if k == 4:
            1
        else:
            x = mean_mse[col].iloc[(k * 4):((k + 1) * 4)].mean()
            y = mean_de.loc[k * 4:(k + 1) * 4, col].mean()
            x_values.append(x)
            y_values.append(y)
            line_x.append(x)
            line_y.append(y)

    plt.plot(x_values, y_values, color=colors[i], linestyle='-', linewidth=0,marker='o',markersize=15,zorder=5)
    plt.plot(line_x, line_y, color=colors[i], linestyle='-', linewidth=2)
    for j in range(len(line_x) - 1):
        dx = (line_x[j + 1] - line_x[j])/2
        dy = (line_y[j + 1] - line_y[j])/2
        plt.annotate('', xy=(line_x[j]+dx, line_y[j]+dy), xytext=(line_x[j], line_y[j]),
                     arrowprops=dict(arrowstyle='->', color=colors[i]),size=20)
for i, col in enumerate(mean_mse.iloc[:, [0, 1, 2,3]].columns):
    for k in range(5):
        if k == 4:
            plt.scatter(-1000, 0, 
                        color=colors[i], 
                        marker='o',
                        label=name_m[i], s=50)
        else:
            x = mean_mse[col].iloc[(k * 4):((k + 1) * 4)].mean()
            y = mean_de.loc[k * 4:(k + 1) * 4, col].mean()
            plt.scatter(x, y, 
                        color='white', 
                        marker='$'+letters[k]+'$', s=100,zorder=10)
plt.plot(0, 0, marker='o',linewidth=0, label='Compound',color='lightgrey',markersize=8,zorder=10)
plt.xlabel('Accuracy ratio (MSE reversed)')
plt.ylabel('Difference explained ratio (DE)')
#plt.title('Accuracy and MSE for Horizon sections',fontsize=20)
plt.axhline(0, linestyle='--', alpha=0.4)
plt.axvline(0, linestyle='--', alpha=0.4)
plt.xlim(0.23, -0.008)
plt.ylim(-0.07, 0.007)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=5,fontsize=15)
plt.savefig("out/horizon_test.jpeg",dpi=400,bbox_inches="tight")
plt.show()



# Define colors and labels
colors = ['black', 'gray', 'royalblue',"lightskyblue"]
letters=['A','B','C','D']
mean_mse.columns=['DTP','ViEWS','Null','t-1']
mean_de.columns=['DTP','ViEWS','Null','t-1']

name_m = mean_mse.iloc[:, [0, 1,3]].columns.tolist()
fig,ax = plt.subplots(figsize=(12,8))
for i, col in enumerate(mean_mse.iloc[:, [0, 1,3]].columns):
    x_values = []
    y_values = []
    line_x=[]
    line_y=[]
    for k in range(4):
        if k == 4:
            1
        else:
            x = mean_mse[col].iloc[(k * 4):((k + 1) * 4)].mean()
            y = mean_de.loc[k * 4:(k + 1) * 4, col].mean()
            x_values.append(x)
            y_values.append(y)
            line_x.append(x)
            line_y.append(y)

    plt.plot(x_values, y_values, color=colors[i], linestyle='-', linewidth=0,marker='o',markersize=15,zorder=5)
    plt.plot(line_x, line_y, color=colors[i], linestyle='-', linewidth=2)
    for j in range(len(line_x) - 1):
        dx = (line_x[j + 1] - line_x[j])/2
        dy = (line_y[j + 1] - line_y[j])/2
        plt.annotate('', xy=(line_x[j]+dx, line_y[j]+dy), xytext=(line_x[j], line_y[j]),
                     arrowprops=dict(arrowstyle='->', color=colors[i]),size=20)
for i, col in enumerate(mean_mse.iloc[:, [0, 1,3]].columns):
    for k in range(5):
        if k == 4:
            plt.scatter(-1000, 0, 
                        color=colors[i], 
                        marker='o',
                        label=name_m[i], s=50)
        else:
            x = mean_mse[col].iloc[(k * 4):((k + 1) * 4)].mean()
            y = mean_de.loc[k * 4:(k + 1) * 4, col].mean()
            plt.scatter(x, y, 
                        color='white', 
                        marker='$'+letters[k]+'$', s=100,zorder=10)
plt.plot(0, 0, marker='o',linewidth=0, label='Compound',color='lightgrey',markersize=8,zorder=10)
plt.xlabel('Accuracy ratio (MSE reversed)')
plt.ylabel('Difference explained ratio (DE)')
#plt.title('Accuracy and MSE for Horizon sections',fontsize=20)
plt.axhline(0, linestyle='--', alpha=0.4)
plt.axvline(0, linestyle='--', alpha=0.4)
plt.xlim(0.052, -0.008)
plt.ylim(-0.017, 0.007)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=5,fontsize=15)
plt.savefig("out/horizon_test_short.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# =============================================================================
# DE exp factor 
# =============================================================================

de_res = []
de_res_std=[]
for k_fac in range(1,11):    
    # Calculate difference explaines
    d_nn = diff_explained(df_input.iloc[-24:-24+horizon],df_sf_1,k=k_fac)
    d_nn2 = diff_explained(df_input.iloc[-12:],df_sf_2,k=k_fac)
    d_nn = np.concatenate([d_nn,d_nn2])
    
    d_b = diff_explained(df_input.iloc[-24:-24+horizon],df_preds_test_1.iloc[:12],k=k_fac)
    d_b2 = diff_explained(df_input.iloc[-12:],df_preds_test_2.iloc[:12],k=k_fac)
    d_b = np.concatenate([d_b,d_b2])
    
    d_null = diff_explained(df_input.iloc[-24:-24+horizon],pd.DataFrame(np.zeros((horizon,len(df_input.columns)))),k=k_fac)
    d_null2 = diff_explained(df_input.iloc[-12:],pd.DataFrame(np.zeros((horizon,len(df_input.columns)))),k=k_fac)
    d_null = np.concatenate([d_null,d_null2])
    
    d_t1 = diff_explained(df_input.iloc[-24:-24+horizon],df_input.iloc[-24-horizon:-24],k=k_fac)
    d_t12 = diff_explained(df_input.iloc[-12:],df_input.iloc[-24:-24+horizon],k=k_fac)
    d_t1= np.concatenate([d_t1,d_t12])
    
    d_mix = d_b.copy()
    d_mix[ind_keep[df_keep_1]] = d_nn[ind_keep[df_keep_1]]
    d_nn = d_nn[~np.isnan(d_nn)]
    d_b = d_b[~np.isnan(d_b)]
    d_null = d_null[~np.isnan(d_null)]
    d_t1= d_t1[~np.isnan(d_t1)]
    d_mix = d_mix[~np.isnan(d_mix)]
    
    de_res.append([d_nn.mean(),d_b.mean(),d_null.mean(),d_t1.mean(),d_mix.mean()])
    de_res_std.append([d_nn.std(),d_b.std(),d_null.std(),d_t1.std(),d_mix.std()])

de_res=pd.DataFrame(de_res)
de_res_std = pd.DataFrame(de_res_std)

#de_res.columns=test.columns
#de_res_std.columns=test.columns
horizons = np.arange(len(de_res))+1

plt.figure(figsize=(14, 8))
for column,line in zip(de_res.columns,["dashed","dotted","solid","dashdot",(5,(10,0))]):
    means = de_res[column]
    stds = de_res_std[column]
    plt.errorbar(horizons, means, yerr=1.96*stds/np.sqrt(382), label=column, capsize=5, color="black",linestyle=line)
plt.xlabel('Exponetial factor',size=25)
plt.ylabel('Differecne explained',size=25)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=5,fontsize=20)
plt.savefig("out/de_exp.jpeg",dpi=400,bbox_inches="tight")
plt.show()


df_t = pd.DataFrame([0,0.2,0,0.5,0,0.96,0.2,0,0.15,0])
df_1 = pd.DataFrame([0.05,0.05,0.6,0.25,0,0.7,0.05,0,0,0])
df_2 = pd.DataFrame([1,0,1,0,1,0,1,0,1,0])
# df_3 = pd.DataFrame([0.1,0.2,0.05,0.25,0.1,0.4,0.1,0,0.05,0])

# df_3 = pd.DataFrame([0.2,0.3,0.2,0.25,0.2,0.35,0.15,0,0,0.1])
df_3 = pd.DataFrame([0.5,0.6,0.5,0.55,0.5,0.65,0.55,0.12,0,0.2])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})

ax1.plot(df_1, label='Model 1',color='dimgray',linewidth=3,linestyle="solid")
ax1.plot(df_2, label='Model 2',color='darkgray',linewidth=2,linestyle="dashed")
ax1.plot(df_3, label='Model 3',color='darkgray',linewidth=2,linestyle="dotted")

ax1.plot(df_t, label='Actuals',color='black',linewidth=3)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5,fontsize=12)

k_values = [1, 5, 10]
models = ['Model 1', 'Model 2', 'Model 3']
differences = {
    'Model 1': [diff_explained(df_t, df_1, k, horizon=10)[0] for k in k_values],
    'Model 2': [diff_explained(df_t, df_2, k, horizon=10)[0] for k in k_values],
    'Model 3': [diff_explained(df_t, df_3, k, horizon=10)[0] for k in k_values],
}
podiums = {k: sorted(models, key=lambda m: differences[m][i]) for i, k in enumerate(k_values)}
bar_width = 0.2
col=["gray","darkgray","lightgray","black"]
x = np.arange(len(k_values))
for i, model in enumerate(models):
    ax2.bar(x + i * bar_width, [differences[model][j] for j in range(len(k_values))], bar_width, label=model,color=col[i])
ax2.set_xticks(x + bar_width)
ax2.set_xticklabels(['k=1', 'k=5', 'k=10'],fontsize=15)
ax2.set_title('Difference explained (DE)',fontsize=20)
ax1.set_title('Actuals and predicted outcome',fontsize=20)
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_yticks([])
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=5,fontsize=12)
plt.tight_layout()
plt.savefig("out/de_exp_fictional.jpeg",dpi=400,bbox_inches="tight")
plt.show()



# =============================================================================
# Variable ? 
# =============================================================================

def approximate_entropy(U, m, r):
    def _phi(m):
        N = len(U)
        X = [U[i:i + m] for i in range(N - m + 1)]
        C = []
        for x in X:
            C.append(sum([np.linalg.norm(np.array(x) - np.array(x2), ord=np.inf) <= r for x2 in X]) / (N - m + 1.0))
        return np.sum(np.log(C)) / (N - m + 1.0)
    return _phi(m) - _phi(m + 1)



mean_d=[]
std_d=[]
per_d=[]
mean_m=[]
std_m=[]
per_m=[]
sca=[]
ap_ent=[]
countries=[]
for i in range(len(df_input.columns)):   
    if (df_input.iloc[-34:-24,i]==0).all()==False:
        countries.append(f"{df_input.columns[i]}--2022")
        ser = (df_input.iloc[-34:-24,i] - df_input.iloc[-34:-24,i].min())/(df_input.iloc[-34:-24,i].max()-df_input.iloc[-34:-24,i].min())
        diff = ser.diff()
        mean_d.append(abs(diff).mean())
        std_d.append(abs(diff).std())
        per_d.append((diff>0).mean())
        mean_m.append(ser.mean())
        std_m.append(ser.std())
        per_m.append((ser>ser.mean()).mean())
        sca.append(df_input.iloc[-34:-24,i].sum())
        ap_ent.append(approximate_entropy(ser.tolist(),3,0.15))
for i in range(len(df_input.columns)):   
    if (df_input.iloc[-22:-12,i]==0).all()==False:
        countries.append(f"{df_input.columns[i]}--2023")
        ser = (df_input.iloc[-22:-12,i] - df_input.iloc[-22:-12,i].min())/(df_input.iloc[-22:-12,i].max()-df_input.iloc[-22:-12,i].min())
        diff = ser.diff()
        mean_d.append(abs(diff).mean())
        std_d.append(abs(diff).std())
        per_d.append((diff>0).mean()) 
        mean_m.append(ser.mean())
        std_m.append(ser.std())
        per_m.append((ser>ser.mean()).mean())
        sca.append(df_input.iloc[-22:-12,i].sum())
        ap_ent.append(approximate_entropy(ser.tolist(),3,0.15))
        
df_var = pd.DataFrame([mean_d,std_d,per_d,sca,df_sel['log MSE'],mean_m,std_m,per_m,ap_ent]).T



X = df_var.iloc[:, [1,5]]
y = df_var.iloc[:, 4]
model = DecisionTreeRegressor(random_state=42, max_depth=2,min_samples_split=10,min_samples_leaf=10)
model.fit(X, y)
#viz_model = dtreeviz.model(model,X,y,feature_names=['Diff SD', 'Mean'],target_name='Log Ratio MSE')
#v = viz_model.view()  
#v.show()  


fig,ax=plt.subplots(figsize=(9,6))
plot_tree(model, feature_names=[ 'Diff SD','Mean'], rounded=True,impurity=False) 
for text in ax.texts:
    # Custom logic to change node labels
    if 'Mean <= 0.395\nsamples = 111\nvalue = -0.076' in text.get_text():
        text.set_text('Mean $<=$ 0.395\nSamples = 111\nValue = -0.076')  # Replace with desired label
for text in ax.texts:
    # Custom logic to change node labels
    if 'Mean <= 0.286\nsamples = 90\nvalue = -0.196' in text.get_text():
        text.set_text('Mean $<=$ 0.286\nSamples = 90\nValue = -0.196')  # Replace with desired label
for text in ax.texts:
    # Custom logic to change node labels
    if 'samples = 71\nvalue = -0.063' in text.get_text():
        text.set_text('Samples = 71\nValue = -0.063')  # Replace with desired label
for text in ax.texts:
    # Custom logic to change node labels
    if 'samples = 19\nvalue = -0.691' in text.get_text():
        text.set_text('Samples = 19\nValue = -0.691')  # Replace with desired label                    
for text in ax.texts:
    # Custom logic to change node labels
    if 'Diff SD <= 0.252\nsamples = 21\nvalue = 0.437' in text.get_text():
        text.set_text('Diff SD $<=$ 0.252\nSamples = 21\nValue = 0.437')  # Replace with desired label                    
for text in ax.texts:
    # Custom logic to change node labels
    if 'samples = 10\nvalue = 0.944' in text.get_text():
        text.set_text('Samples = 10\nValue = 0.944') 
        text.set_color('darkblue')
for text in ax.texts:
    # Custom logic to change node labels
    if 'samples = 11\nvalue = -0.023' in text.get_text():
        text.set_text('Samples = 11\nValue = -0.023')  #
plt.savefig("out/decision_tree_best.jpeg",dpi=400,bbox_inches="tight")
plt.show()


df_subr=df_var[(df_var.iloc[:,1]<0.2516) & (df_var.iloc[:,5]>0.3951)]
norm = mcolors.Normalize(vmin=-1, vmax=1)
cmap = plt.get_cmap('RdBu')
color = cmap(norm(df_subr))
fig,ax = plt.subplots(figsize=(12,8))
plt.fill_betweenx(y=[0.3951, 0.7], x1=0.2516, color="gray", alpha=0.2,linestyle='--')
plt.axhline(0.3951,xmax=0.3316,linestyle='--',color='black',alpha=0.2)
plt.axvline(0.2516,ymin=0.5551,linestyle='--',color='black',alpha=0.2)
plt.scatter(df_var.iloc[:,1],df_var.iloc[:,5],c=df_var.iloc[:,4],cmap='RdBu',label='STD Diff',vmin=-1,vmax=1,s=np.log(df_var.iloc[:,3])*20,alpha=0.2)
plot=plt.scatter(df_subr.iloc[:,1],df_subr.iloc[:,5],c=df_subr.iloc[:,4],cmap='RdBu',label='STD Diff',vmin=-1,vmax=1,s=np.log(df_subr.iloc[:,3])*20)
plt.axhline(0.3951,linestyle='--',color='black',alpha=0.2)
plt.axvline(0.2516,linestyle='--',color='black',alpha=0.2)
plt.xlim(0.1,0.55)
plt.ylim(0.04,0.67)
plt.xlabel('Standard deviation of the differentiated input series (Diff SD)',fontsize=15)
plt.ylabel('Mean of all data points in input series (Mean)',fontsize=15)
cax_cb = fig.add_axes([0.95, 0.23, 0.015, 0.4])  # [left, bottom, width, height]
cbar = plt.colorbar(plot, ax=ax,cax=cax_cb,orientation='vertical')
cbar.ax.set_title('Log-ratio', fontsize=15, pad=15)
size_ranges = [20, 80, 160, 240]
legend_labels = ['0', '4', '8', '12']
legend_handles = [mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=np.sqrt(size), linestyle='', markeredgewidth=1) for size in size_ranges]
ax.legend(legend_handles, legend_labels,loc='center left', bbox_to_anchor=(1, 0.9), title='Fatalities (log)', title_fontsize='15', fontsize='15',frameon=False)
for i in range(len(df_var)):
    if df_var.iloc[i,1] < 0.2516 and df_var.iloc[i,5] > 0.3951 and df_var.iloc[i,4]>0.3:
        ax.annotate(countries[i], (df_var.iloc[i,1], df_var.iloc[i,5]-0.005),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontsize=9)
plt.savefig("out/scatter_best.jpeg",dpi=400,bbox_inches="tight")
plt.show()







X = df_var.iloc[:, [6,5]]
y = df_var.iloc[:, 4]
w = pd.Series([1]*len(df_var),index = df_var.index)
w.loc[63] = 3
w.loc[22] = 2
model_2 = DecisionTreeRegressor(random_state=42, max_depth=2,max_features=1)
model_2.fit(X,y,w)
#viz_model = dtreeviz.model(model_2,X,y,feature_names=[ 'SD','Mean'])
#v = viz_model.view(colors={'highlight':'darkred','scatter_marker':'darkred'})  
#v.show() 


fig,ax=plt.subplots(figsize=(9,6))
plot_tree(model_2, feature_names=[ 'SD','Mean'], rounded=True,impurity=False)  
for text in ax.texts:
    # Custom logic to change node labels
    if 'SD <= 0.375\nsamples = 111\nvalue = -0.102' in text.get_text():
        text.set_text('SD $<=$ 0.375\nSamples = 111\nValue = -0.102')
for text in ax.texts:
    # Custom logic to change node labels
    if 'SD <= 0.29\nsamples = 96\nvalue = 0.035' in text.get_text():
        text.set_text('SD $<=$ 0.29\nSamples = 96\nValue = 0.035')
for text in ax.texts:
    # Custom logic to change node labels
    if 'samples = 5\nvalue = 1.09' in text.get_text():
        text.set_text('Samples = 5\nValue = 1.09')
for text in ax.texts:
    # Custom logic to change node labels
    if 'samples = 91\nvalue = -0.023' in text.get_text():
        text.set_text('Samples = 91\nValue = -0.023')
for text in ax.texts:
    # Custom logic to change node labels
    if 'Mean <= 0.436\nsamples = 15\nvalue = -0.835' in text.get_text():
        text.set_text('Mean $<=$ 0.436\nSamples = 15\nValue = -0.835')        
for text in ax.texts:
    # Custom logic to change node labels
    if 'samples = 14\nvalue = -1.139' in text.get_text():
        text.set_text('Samples = 14\nValue = -1.139')    
        text.set_color('darkred')
for text in ax.texts:
    # Custom logic to change node labels
    if 'samples = 1\nvalue = 0.682' in text.get_text():
        text.set_text('Samples = 1\nValue = 0.682')                         
plt.savefig("out/decision_tree_worst.jpeg",dpi=400,bbox_inches="tight")
plt.show()

df_subr=df_var[(df_var.iloc[:,6]>0.37) & (df_var.iloc[:,5]<0.43)]
norm = mcolors.Normalize(vmin=-1, vmax=1)
cmap = plt.get_cmap('RdBu')
color = cmap(norm(df_subr))
fig,ax = plt.subplots(figsize=(12,8))
plt.fill_betweenx(y=[0.43, 0.05], x1=0.37,x2=0.45, color="gray", alpha=0.2,linestyle='--')
plt.axhline(0.43,xmin=0.6, linestyle='--', color='gray', alpha=1)
plt.axvline(0.37,ymax=0.62, linestyle='--', color='gray', alpha=1)
plt.scatter(df_var.iloc[:,6],df_var.iloc[:,5],c=df_var.iloc[:,4],cmap='RdBu',label='STD Diff',vmin=-1,vmax=1,s=np.log(df_var.iloc[:,3])*20,alpha=0.2)
plt.scatter(df_subr.iloc[:,6],df_subr.iloc[:,5],c=df_subr.iloc[:,4],cmap='RdBu',label='STD Diff',vmin=-1,vmax=1,s=np.log(df_subr.iloc[:,3])*20)
plt.axhline(0.43, linestyle='--', color='black', alpha=0.2)
plt.axvline(0.37, linestyle='--', color='black', alpha=0.2)
plt.xlim(0.25,0.45)
plt.ylim(0.05,0.67)
plt.xlabel('Standard deviation of the input series (SD)')
plt.ylabel('Mean of all data points in input series (Mean)')
cax_cb = fig.add_axes([0.95, 0.23, 0.015, 0.4])  # [left, bottom, width, height]
cbar = plt.colorbar(plot, ax=ax,cax=cax_cb,orientation='vertical')
cbar.ax.set_title('Log-ratio', fontsize=15, pad=15)
size_ranges = [20, 80, 160, 240]
legend_labels = ['0', '4', '8', '12']
legend_handles = [mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=np.sqrt(size), linestyle='', markeredgewidth=1) for size in size_ranges]
ax.legend(legend_handles, legend_labels,loc='center left', bbox_to_anchor=(1, 0.9), title='Fatalities (log)', title_fontsize='15', fontsize='15',frameon=False)
for i in range(len(df_var)):
    if df_var.iloc[i,6] > 0.35 and df_var.iloc[i,5] < 0.43 and df_var.iloc[i,4]<-0.8:
        ax.annotate(countries[i], (df_var.iloc[i,6], df_var.iloc[i,5]-0.005),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontsize=9)
plt.savefig("out/scatter_worst.jpeg",dpi=400,bbox_inches="tight")
plt.show()

# =============================================================================
# Worse cases
# =============================================================================

for i in range(len(df_input.columns)):
    print(i)
    print(df_input.columns[i])
    
def remove_box(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

fig, axs = plt.subplots(4, 2, figsize=(13, 12))
axs[0, 0].plot(df_input.iloc[-34:-24, 119].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[0, 0])
axs[0, 0].set_xticks([])
axs[0, 0].set_yticks([])
axs[0, 1].plot(df_input.iloc[-24:-12, 119].reset_index(drop=True), linewidth=2, color='black')
axs[0, 1].plot(df_sf_1.iloc[:, 119].reset_index(drop=True),linestyle="dashed",color='black',linewidth=2)
axs[0, 1].plot(df_preds_test_1.iloc[:12, 119].reset_index(drop=True), linestyle="dotted",color='black',linewidth=2)
remove_box(axs[0, 1])
axs[0, 1].set_xticks([])
axs[0, 1].set_yticks([])
axs[0, 1].set_title(f'{df_input.columns[119]}---2022')
axs[0, 0].set_title(f'{df_input.columns[119]}---2021')

axs[1, 0].plot(df_input.iloc[-22:-12, 123].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[1, 0])
axs[1, 0].set_xticks([])
axs[1, 0].set_yticks([])
axs[1, 1].plot(df_input.iloc[-12:, 123].reset_index(drop=True), linewidth=2, color='black')
axs[1, 1].plot(df_sf_2.iloc[:, 123].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
axs[1, 1].plot(df_preds_test_2.iloc[:12, 123].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
remove_box(axs[1, 1])
axs[1, 1].set_xticks([])
axs[1, 1].set_yticks([])
axs[1, 0].set_title(f'{df_input.columns[123]}---2022')
axs[1, 1].set_title(f'{df_input.columns[123]}---2023')

axs[2, 0].plot(df_input.iloc[-34:-24, 190].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[2, 0])
axs[2, 0].set_xticks([])
axs[2, 0].set_yticks([])
axs[2, 1].plot(df_input.iloc[-24:-12, 190].reset_index(drop=True), linewidth=2, color='black')
axs[2, 1].plot(df_sf_1.iloc[:, 190].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
axs[2, 1].plot(df_preds_test_1.iloc[:12, 190].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
remove_box(axs[2, 1])
axs[2, 1].set_xticks([])
axs[2, 1].set_yticks([])
axs[2, 0].set_title(f'{df_input.columns[190]}--2021')
axs[2, 1].set_title(f'{df_input.columns[190]}--2022')

axs[3, 0].plot(df_input.iloc[-22:-12, 109].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[3, 0])
axs[3, 0].set_xticks([])
axs[3, 0].set_yticks([])
axs[3, 1].plot(df_input.iloc[-12:, 109].reset_index(drop=True), linewidth=2, color='black')
axs[3, 1].plot(df_sf_2.iloc[:, 109].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
axs[3, 1].plot(df_preds_test_2.iloc[:12, 109].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
remove_box(axs[3, 1])
axs[3, 1].set_xticks([])
axs[3, 1].set_yticks([])
axs[3, 0].set_title(f'{df_input.columns[109]}---2022')
axs[3, 1].set_title(f'{df_input.columns[109]}---2023')
plt.tight_layout()

plt.savefig("out/cases_worst.jpeg",dpi=400,bbox_inches="tight")

plt.show()

        
fig, axs = plt.subplots(4, 1, figsize=(10, 13))
axs[0].plot([0,1,2,3,4,5,6,7,8,9],df_input.iloc[-34:-24, 119].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[0])
axs[0].set_xticks([])
axs[0].set_yticks([])
ax2 = axs[0].twinx()
ax2.set_yticks([])
remove_box(ax2)
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_input.iloc[-24:-12, 119].reset_index(drop=True), linewidth=2, color='black')
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_sf_1.iloc[:, 119].reset_index(drop=True),linestyle="dashed",color='black',linewidth=2)
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_preds_test_1.iloc[:12, 119].reset_index(drop=True), linestyle="dotted",color='black',linewidth=2)
axs[0].set_title(f'{df_input.columns[119]} 2021---2022')
ax2.plot([9,10],[6,27], linestyle="solid",color='black',linewidth=2)
axs[0].axvline(x=10, color='black', linestyle='--', linewidth=1)

axs[1].plot([0,1,2,3,4,5,6,7,8,9],df_input.iloc[-22:-12, 123].reset_index(drop=True), linewidth=2, color='black')   
remove_box(axs[1])
axs[1].set_xticks([])
axs[1].set_yticks([])
ax2 = axs[1].twinx()
ax2.set_yticks([])
remove_box(ax2)
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_input.iloc[-12:, 123].reset_index(drop=True), linewidth=2, color='black')
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_sf_2.iloc[:, 123].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_preds_test_2.iloc[:12, 123].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
axs[1].set_title(f'{df_input.columns[123]} 2022---2023')
axs[1].axvline(x=10, color='black', linestyle='--', linewidth=1)
ax2.plot([9,10],[0.009,0], linestyle="solid",color='black',linewidth=2)

axs[2].plot([0,1,2,3,4,5,6,7,8,9],df_input.iloc[-34:-24, 190].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[2])
axs[2].set_xticks([])
axs[2].set_yticks([])
ax2 = axs[2].twinx()
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_input.iloc[-24:-12, 190].reset_index(drop=True), linewidth=2, color='black')
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_sf_1.iloc[:, 190].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_preds_test_1.iloc[:12, 190].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
ax2.set_yticks([])
remove_box(ax2)
axs[2].set_title(f'{df_input.columns[190]} 2021---2022')
axs[2].axvline(x=10, color='black', linestyle='--', linewidth=1)
ax2.plot([9,10],[38,0], linestyle="solid",color='black',linewidth=2)

axs[3].plot([0,1,2,3,4,5,6,7,8,9],df_input.iloc[-22:-12, 109].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[3])
axs[3].set_xticks([])
axs[3].set_yticks([])
ax2 = axs[3].twinx()
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_input.iloc[-12:, 109].reset_index(drop=True), linewidth=2, color='black')
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_sf_2.iloc[:, 109].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_preds_test_2.iloc[:12, 109].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
ax2.set_yticks([])
remove_box(ax2)
axs[3].set_title(f'{df_input.columns[109]} 2022---2023')
axs[3].axvline(x=10, color='black', linestyle='--', linewidth=1)
ax2.plot([9,10],[0,0], linestyle="solid",color='black',linewidth=2)

plt.tight_layout()
plt.savefig("out/cases_worst2.jpeg",dpi=400,bbox_inches="tight")
plt.show()

    
#def remove_box(ax):
#    ax.spines['top'].set_visible(False)
#    ax.spines['right'].set_visible(False)
#    ax.spines['left'].set_visible(False)
#    ax.spines['bottom'].set_visible(False)
      
#fig, axs = plt.subplots(3, 2, figsize=(15, 10))
#axs[0, 0].plot(df_input.iloc[-34:-24, 119].reset_index(drop=True), linewidth=2, color='black')
#remove_box(axs[0, 0])
#axs[0, 0].set_xticks([])
#axs[0, 0].set_yticks([])
#axs[0, 1].plot(df_input.iloc[-24:-12, 119].reset_index(drop=True), linewidth=2, color='black')
#axs[0, 1].plot(df_sf_1.iloc[:, 119].reset_index(drop=True), linewidth=5, color='purple')
#axs[0, 1].plot(df_preds_test_1.iloc[:12, 119].reset_index(drop=True), linewidth=2, color='grey')
#remove_box(axs[0, 1])
#axs[0, 1].set_xticks([])
#axs[0, 1].set_yticks([])
#axs[0, 1].set_title(f'{df_input.columns[119]} - 2023')


#axs[1, 0].plot(df_input.iloc[-22:-12, 123].reset_index(drop=True), linewidth=2, color='black')
#remove_box(axs[1, 0])
#axs[1, 0].set_xticks([])
#axs[1, 0].set_yticks([])
#axs[1, 1].plot(df_input.iloc[-12:, 123].reset_index(drop=True), linewidth=2, color='black')
#axs[1, 1].plot(df_sf_2.iloc[:, 123].reset_index(drop=True), linewidth=5, color='purple')
#axs[1, 1].plot(df_preds_test_2.iloc[:12, 123].reset_index(drop=True), linewidth=2, color='grey')
#remove_box(axs[1, 1])
#axs[1, 1].set_xticks([])
#axs[1, 1].set_yticks([])
#axs[1, 1].set_title(f'{df_input.columns[123]} - 2023')


#axs[2, 0].plot(df_input.iloc[-34:-24, 190].reset_index(drop=True), linewidth=2, color='black')
#remove_box(axs[2, 0])
#axs[2, 0].set_xticks([])
#axs[2, 0].set_yticks([])
#axs[2, 1].plot(df_input.iloc[-24:-12, 190].reset_index(drop=True), linewidth=2, color='black')
#axs[2, 1].plot(df_sf_1.iloc[:, 190].reset_index(drop=True), linewidth=5, color='purple')
#axs[2, 1].plot(df_preds_test_1.iloc[:12, 190].reset_index(drop=True), linewidth=2, color='grey')
#remove_box(axs[2, 1])
#axs[2, 1].set_xticks([])
#axs[2, 1].set_yticks([])
#axs[2, 1].set_title(f'{df_input.columns[190]} - 2022')
#plt.tight_layout()
#plt.show()


# =============================================================================
# Best cases  
# =============================================================================

fig, axs = plt.subplots(4, 2, figsize=(13, 12))
axs[0, 0].plot(df_input.iloc[-34:-24, 177].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[0, 0])
axs[0, 0].set_xticks([])
axs[0, 0].set_yticks([])
axs[0, 1].plot(df_input.iloc[-24:-12, 177].reset_index(drop=True), linewidth=2, color='black')
axs[0, 1].plot(df_sf_1.iloc[:, 177].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
axs[0, 1].plot(df_preds_test_1.iloc[:12, 177].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
remove_box(axs[0, 1])
axs[0, 1].set_xticks([])
axs[0, 1].set_yticks([])
axs[0, 0].set_title(f'{df_input.columns[177]}--2021')
axs[0, 1].set_title(f'{df_input.columns[177]}--2022')

axs[1, 0].plot(df_input.iloc[-34:-24, 179].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[1, 0])
axs[1, 0].set_xticks([])
axs[1, 0].set_yticks([])
axs[1, 1].plot(df_input.iloc[-24:-12, 179].reset_index(drop=True), linewidth=2, color='black')
axs[1, 1].plot(df_sf_1.iloc[:, 179].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
axs[1, 1].plot(df_preds_test_1.iloc[:12, 179].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
remove_box(axs[1, 1])
axs[1, 1].set_xticks([])
axs[1, 1].set_yticks([])
axs[1, 0].set_title(f'{df_input.columns[179]}--2021')
axs[1, 1].set_title(f'{df_input.columns[179]}--2022')

axs[2, 0].plot(df_input.iloc[-22:-12, 63].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[2, 0])
axs[2, 0].set_xticks([])
axs[2, 0].set_yticks([])
axs[2, 1].plot(df_input.iloc[-12:, 63].reset_index(drop=True), linewidth=2, color='black')
axs[2, 1].plot(df_sf_2.iloc[:, 63].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
axs[2, 1].plot(df_preds_test_2.iloc[:12, 63].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
remove_box(axs[2, 1])
axs[2, 1].set_xticks([])
axs[2, 1].set_yticks([])
axs[2, 0].set_title(f'{df_input.columns[63]}---2022')
axs[2, 1].set_title(f'{df_input.columns[63]}---2023')

axs[3, 0].plot(df_input.iloc[-22:-12, 26].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[3, 0])
axs[3, 0].set_xticks([])
axs[3, 0].set_yticks([])
axs[3, 1].plot(df_input.iloc[-12:, 26].reset_index(drop=True), linewidth=2, color='black')
axs[3, 1].plot(df_sf_2.iloc[:, 26].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
axs[3, 1].plot(df_preds_test_2.iloc[:12, 26].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
remove_box(axs[3, 1])
axs[3, 1].set_xticks([])
axs[3, 1].set_yticks([])
axs[3, 0].set_title(f'{df_input.columns[26]}---2022')
axs[3, 1].set_title(f'{df_input.columns[26]}---2023')

plt.tight_layout()
plt.savefig("out/cases_best.jpeg",dpi=400,bbox_inches="tight")
plt.show()





fig, axs = plt.subplots(4, 1, figsize=(10, 13))
axs[0].plot([0,1,2,3,4,5,6,7,8,9],df_input.iloc[-34:-24, 177].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[0])
axs[0].set_xticks([])
axs[0].set_yticks([])
ax2 = axs[0].twinx()
ax2.set_yticks([])
remove_box(ax2)
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_input.iloc[-24:-12, 177].reset_index(drop=True), linewidth=2, color='black')
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_sf_1.iloc[:, 177].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_preds_test_1.iloc[:12, 177].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
axs[0].set_title(f'{df_input.columns[177]} 2021---2022')
ax2.plot([9,10],[6,85], linestyle="solid",color='black',linewidth=2)
axs[0].axvline(x=10, color='black', linestyle='--', linewidth=1)

axs[1].plot([0,1,2,3,4,5,6,7,8,9],df_input.iloc[-34:-24, 179].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[1])
axs[1].set_xticks([])
axs[1].set_yticks([])
ax2 = axs[1].twinx()
ax2.set_yticks([])
remove_box(ax2)
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_input.iloc[-24:-12, 179].reset_index(drop=True), linewidth=2, color='black')
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_sf_1.iloc[:, 179].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_preds_test_1.iloc[:12, 179].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
axs[1].set_title(f'{df_input.columns[179]} 2021---2022')
axs[1].axvline(x=10, color='black', linestyle='--', linewidth=1)
ax2.plot([9,10],[68.3,35], linestyle="solid",color='black',linewidth=2)

axs[2].plot([0,1,2,3,4,5,6,7,8,9],df_input.iloc[-22:-12, 63].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[2])
axs[2].set_xticks([])
axs[2].set_yticks([])
ax2 = axs[2].twinx()
ax2.set_yticks([])
remove_box(ax2)
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_input.iloc[-12:, 63].reset_index(drop=True), linewidth=2, color='black')
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_sf_2.iloc[:, 63].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_preds_test_2.iloc[:12, 63].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
axs[2].set_title(f'{df_input.columns[63]} 2022---2023')
axs[2].axvline(x=10, color='black', linestyle='--', linewidth=1)
ax2.plot([9,10],[57,19], linestyle="solid",color='black',linewidth=2)

axs[3].plot([0,1,2,3,4,5,6,7,8,9],df_input.iloc[-22:-12, 26].reset_index(drop=True), linewidth=2, color='black')
remove_box(axs[3])
axs[3].set_xticks([])
axs[3].set_yticks([])
ax2 = axs[3].twinx()
ax2.set_yticks([])
remove_box(ax2)
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_input.iloc[-12:, 26].reset_index(drop=True), linewidth=2, color='black')
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_sf_2.iloc[:, 26].reset_index(drop=True), linewidth=2, color='black',linestyle="dashed")
ax2.plot([10,11,12,13,14,15,16,17,18,19,20,21],df_preds_test_2.iloc[:12, 26].reset_index(drop=True), linewidth=2, color='black',linestyle="dotted")
axs[3].set_title(f'{df_input.columns[26]} 2022---2023')
axs[3].axvline(x=10, color='black', linestyle='--', linewidth=1)
ax2.plot([9,10],[9.2,10], linestyle="solid",color='black',linewidth=2)


plt.tight_layout()
plt.savefig("out/cases_best2.jpeg",dpi=400,bbox_inches="tight")
plt.show()


#fig, axs = plt.subplots(3, 2, figsize=(15, 10))
#axs[0, 0].plot(df_input.iloc[-34:-24, 177].reset_index(drop=True), linewidth=2, color='black')
#remove_box(axs[0, 0])
#axs[0, 0].set_xticks([])
#axs[0, 0].set_yticks([])
#axs[0, 1].plot(df_input.iloc[-24:-12, 177].reset_index(drop=True), linewidth=2, color='black')
#axs[0, 1].plot(df_sf_1.iloc[:, 177].reset_index(drop=True), linewidth=5, color='purple')
#axs[0, 1].plot(df_preds_test_1.iloc[:12, 177].reset_index(drop=True), linewidth=2, color='grey')
#remove_box(axs[0, 1])
#axs[0, 1].set_xticks([])
#axs[0, 1].set_yticks([])
#axs[0, 1].set_title(f'{df_input.columns[177]} - 2022')


#axs[1, 0].plot(df_input.iloc[-34:-24, 179].reset_index(drop=True), linewidth=2, color='black')
#remove_box(axs[1, 0])
#axs[1, 0].set_xticks([])
#axs[1, 0].set_yticks([])
#axs[1, 1].plot(df_input.iloc[-24:-12, 179].reset_index(drop=True), linewidth=2, color='black')
#axs[1, 1].plot(df_sf_1.iloc[:, 179].reset_index(drop=True), linewidth=5, color='purple')
#axs[1, 1].plot(df_preds_test_1.iloc[:12, 179].reset_index(drop=True), linewidth=2, color='grey')
#remove_box(axs[1, 1])
#axs[1, 1].set_xticks([])
#axs[1, 1].set_yticks([])
#axs[1, 1].set_title(f'{df_input.columns[179]} - 2022')



#axs[2, 0].plot(df_input.iloc[-22:-12, 110].reset_index(drop=True), linewidth=2, color='black')
#remove_box(axs[2, 0])
#axs[2, 0].set_xticks([])
#axs[2, 0].set_yticks([])
#axs[2, 1].plot(df_input.iloc[-12:, 110].reset_index(drop=True), linewidth=2, color='black')
#axs[2, 1].plot(df_sf_2.iloc[:, 110].reset_index(drop=True), linewidth=5, color='purple')
#axs[2, 1].plot(df_preds_test_2.iloc[:12, 110].reset_index(drop=True), linewidth=2, color='grey')
#remove_box(axs[2, 1])
#axs[2, 1].set_xticks([])
#axs[2, 1].set_yticks([])
#axs[2, 1].set_title(f'{df_input.columns[110]} - 2023')
#plt.tight_layout()
#plt.show()

# =============================================================================
# Create the MSE of Better/Worse
# =============================================================================

df_subr=df_var[(df_var.iloc[:,1]<0.27) & (df_var.iloc[:,5]>0.44)][4].mean()

### Exemples 
# Good shape
df_input = df_input.fillna(0)
h_train=10
h=12

fig, axs = plt.subplots(6, 6, figsize=(18,12))
row=0
col=0
nume=30
flag=True
while flag==True:
    nume=nume+12
    for coun in range(len(df_input.columns)):
        if not (df_input.iloc[nume:nume+h_train,coun]==0).all():
            ser=df_input.iloc[nume:nume+h_train,coun]
            ser=(ser-ser.min())/(ser.max()-ser.min())
            diff = ser.diff()
            if (abs(diff).std()<0.25) & (ser.mean()>0.44):
                axs[row, col].plot(ser.reset_index(drop=True), linewidth=2, color='black')
                remove_box(axs[row, col])
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                axs[row, col].set_title(f"{df_input.columns[coun]}; {str(df_input.iloc[nume:nume+h_train,coun].index[0]+timedelta(days=365))[:7]}---{str(df_input.iloc[nume:nume+h_train,coun].index[-1]+timedelta(days=365))[:7]}",size=12)
                row=row+1
                if row==6:
                    col=col+1
                    row=0
                if col==6:
                    flag=False
plt.tight_layout()
plt.savefig("out/cases_best_grid.jpeg",dpi=400,bbox_inches="tight")
plt.show()


# Bad shape
fig, axs = plt.subplots(5, 5, figsize=(18,12))
row=0
col=0
nume=1
flag=True
while flag==True:
    nume=nume+12
    for coun in range(len(df_input.columns)):
        if not (df_input.iloc[nume:nume+h_train,coun]==0).all():
            ser=df_input.iloc[nume:nume+h_train,coun]
            ser=(ser-ser.min())/(ser.max()-ser.min())
            diff = ser.diff()
            if (ser.std()>0.37) & (ser.mean()<0.4):
                axs[row, col].plot(ser.reset_index(drop=True), linewidth=2, color='black')
                remove_box(axs[row, col])
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                axs[row, col].set_title(df_input.columns[coun])
                axs[row, col].set_title(f"{df_input.columns[coun]}; {str(df_input.iloc[nume:nume+h_train,coun].index[0]+timedelta(days=365))[:7]}---{str(df_input.iloc[nume:nume+h_train,coun].index[-1]+timedelta(days=365))[:7]}",size=12)

                row=row+1
                if row==5:
                    col=col+1
                    row=0
                if col==5:
                    flag=False
plt.tight_layout()
plt.savefig("out/cases_worst_grid.jpeg",dpi=400,bbox_inches="tight")
plt.show()


########### Best MSE 

# mse_be_w=[]
# mse_be_n=[]
# df_match=pd.DataFrame()
# for nume in range(1,380,5):
#     for coun in range(len(df_input.columns)):
#         if not (df_input.iloc[nume:nume+h_train,coun]==0).all():
#             ser=df_input.iloc[nume:nume+h_train,coun]
#             ser=(ser-ser.min())/(ser.max()-ser.min())
#             diff = ser.diff()
#             if (abs(diff).std()<0.25) & (ser.mean()>0.44):
#                 df_copy=df_input.copy()
#                 df_copy.iloc[nume:nume+h_train,coun]=np.nan
#                 shape = Shape()
#                 shape.set_shape(df_input.iloc[nume:nume+h_train,coun]) 
#                 find = finder(df_copy.iloc[:-h,:],shape)
#                 find.find_patterns_lim(min_d=0.1,select=True,metric='dtw',dtw_sel=2,min_mat=3,d_increase=0.05)
#                 mat = [i[0].sum() for i in find.sequences]
#                 mat = pd.cut( pd.Series(mat), bins=[0, 10, 100, 1000,np.inf], labels=['<10', '10-100', '100-1000','>1000'], right=False)
#                 find.create_sce_predict(horizon=12)
#                 df_match=pd.concat([df_match,mat.value_counts()],axis=1)
#                 pred = find.val_sce
#                 pred = pred[pred.index == pred.index.max()].mean()
#                 true=df_input.iloc[nume:nume+12,coun]
#                 true = (true-df_input.iloc[nume:nume+h_train,coun].min())/(df_input.iloc[nume:nume+h_train,coun].max()-df_input.iloc[nume:nume+h_train,coun].min())
#                 mse_be_w.append(df_input.iloc[nume:nume+12,coun].sum())
#                 mse_be_n.append(mean_squared_error(true,pred))

err_sf_pr_n=[]
we_n=[]
for i in range(len(df_input.columns)):  
    if (df_input.iloc[-34:-24,i]==0).all()==False:
        true = df_input.iloc[-24:-24+horizon,i]
        true = (true-df_input.iloc[-34:-24,i].min())/(df_input.iloc[-34:-24,i].max()-df_input.iloc[-34:-24,i].min())
        pred = df_sf_1.iloc[:,i]
        pred = (pred-df_input.iloc[-34:-24,i].min())/(df_input.iloc[-34:-24,i].max()-df_input.iloc[-34:-24,i].min())
        err_sf_pr_n.append(mean_squared_error(true, pred))
        we_n.append(df_input.iloc[-24:-24+horizon,i].sum())

for i in range(len(df_input.columns)):   
    if (df_input.iloc[-22:-12,i]==0).all()==False:
        true = df_input.iloc[-12:,i]
        true = (true-df_input.iloc[-22:-12,i].min())/(df_input.iloc[-22:-12,i].max()-df_input.iloc[-22:-12,i].min())
        pred = df_sf_2.iloc[:,i]
        pred = (pred-df_input.iloc[-22:-12,i].min())/(df_input.iloc[-22:-12,i].max()-df_input.iloc[-22:-12,i].min())
        err_sf_pr_n.append(mean_squared_error(true, pred))
        we_n.append(df_input.iloc[-12:,i].sum())


err_sf_pr_n = [i+0.001 for i in err_sf_pr_n]
# mse_be_n = [i+0.001 for i in mse_be_n]
we_n = [i+1 for i in we_n]
# mse_be_w = [i+1 for i in mse_be_w]

df_norm = pd.DataFrame([err_sf_pr_n,we_n]).T
df_norm.to_csv('df_norm.csv')

# df_be = pd.DataFrame([mse_be_n,mse_be_w]).T
# df_be.to_csv('df_be.csv')

df_be=pd.read_csv('df_be.csv',index_col=0)
mse_be_w = df_be.iloc[:,1]
mse_be_n = df_be.iloc[:,0]

df_norm=pd.read_csv('df_norm.csv',index_col=0)
we_n = df_norm.iloc[:,1]
err_sf_pr_n = df_norm.iloc[:,0]

plt.figure(figsize=(10,6))
plt.scatter(np.log(we_n),np.log(err_sf_pr_n),color='blue',alpha=0.6)
plt.scatter(np.log(mse_be_w),np.log(mse_be_n),color='black',marker='x',alpha=1)
# coeffs1 = np.polyfit(np.log(we_n), np.log(err_sf_pr_n), 2)
# poly1 = np.polyval(coeffs1, np.log(we_n))
# plt.plot(np.log(we_n), poly1, color='blue', linestyle='--', label='Poly fit 1')
slope1, intercept1, r_value1, p_value1, std_err1 = linregress(np.log(we_n), np.log(err_sf_pr_n))
plt.plot(np.log(we_n), intercept1 + slope1 * np.log(we_n), color='blue', linestyle='--')
plt.xlabel('Sum of Fatalies in forecasted window (log)')
plt.ylabel('Normalized MSE (log)')
#plt.savefig("out/best_reg.jpeg",dpi=400,bbox_inches="tight")
plt.show()

df_match_w=pd.DataFrame()
rea=[]
err_spe=[]
inde=[]
with open('test1.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
for i in range(len(df_input.columns)):   
    if (df_input.iloc[-34:-24,i]==0).all()==False:
        ser = (df_input.iloc[-34:-24,i] - df_input.iloc[-34:-24,i].min())/(df_input.iloc[-34:-24,i].max()-df_input.iloc[-34:-24,i].min())
        if (ser.std()>0.37) & (ser.mean()<0.4):
            seq = dict_m[df_input.columns[i]]
            mat = [i[0].sum() for i in seq]
            mat = pd.cut( pd.Series(mat), bins=[0, 10, 100, 1000,np.inf], labels=['<10', '10-100', '100-1000','>1000'], right=False)
            df_match_w=pd.concat([df_match_w,mat.value_counts(normalize=True)],axis=1)
            rea.append(df_input.iloc[-34:-24,i].sum())
            err_s=mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_sf_1.iloc[:,i])
            err_s_v=mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:12,i])
            err_spe.append(np.log((err_s_v+1)/(err_s+1)))
            inde.append(df_input.columns[i]+' 2022')

with open('test2.pkl', 'rb') as f:
    dict_m = pickle.load(f) 
for i in range(len(df_input.columns)):   
    if (df_input.iloc[-22:-12,i]==0).all()==False:
        ser = (df_input.iloc[-22:-12,i] - df_input.iloc[-22:-12,i].min())/(df_input.iloc[-22:-12,i].max()-df_input.iloc[-22:-12,i].min())
        if (ser.std()>0.37) & (ser.mean()<0.4):
            seq = dict_m[df_input.columns[i]]
            mat = [i[0].sum() for i in seq]
            mat = pd.cut( pd.Series(mat), bins=[0, 10, 100, 1000,np.inf], labels=['<10', '10-100', '100-1000','>1000'], right=False)
            df_match_w=pd.concat([df_match_w,mat.value_counts(normalize=True)],axis=1)
            rea.append(df_input.iloc[-22:-12,i].sum())
            err_s=mean_squared_error(df_input.iloc[-12:,i], df_sf_2.iloc[:,i])
            err_s_v=mean_squared_error(df_input.iloc[-12:,i], df_preds_test_2.iloc[:12,i])
            err_spe.append(np.log((err_s_v+1)/(err_s+1)))
            inde.append(df_input.columns[i]+' 2023')

df_match_w = df_match_w.T        
df_match_w = pd.concat([df_match_w.reset_index(drop=True),pd.cut(pd.Series(rea), bins=[0, 10, 100, 1000,np.inf], labels=['<10', '10-100', '100-1000','>1000'], right=False)],axis=1)
df_match_w.index=inde
df_match_w['MSE']=err_spe
latex_table = df_match_w.sort_values('MSE').iloc[:5, :].to_latex(index=True, caption='Top 5 Entries by Log Ratio MSE', label='tab:top5_mse', float_format="%.2f")
print(latex_table)



########### Best DE 

df_input=df_input.fillna(0)

mse_be_w=[]
mse_be_n=[]
df_match=pd.DataFrame()
for nume in range(1,380,5):
    for coun in range(len(df_input.columns)):
        if not (df_input.iloc[nume:nume+h_train,coun]==0).all():
            ser=df_input.iloc[nume:nume+h_train,coun]
            ser=(ser-ser.min())/(ser.max()-ser.min())
            diff = ser.diff()
            if (abs(diff).std()<0.25) & (ser.mean()>0.44):
                df_copy=df_input.copy()
                df_copy.iloc[nume:nume+h_train,coun]=np.nan
                shape = Shape()
                shape.set_shape(df_input.iloc[nume:nume+h_train,coun]) 
                find = finder(df_copy.iloc[:-h,:],shape)
                find.find_patterns_lim(min_d=0.1,select=True,metric='dtw',dtw_sel=2,min_mat=3,d_increase=0.05)
                mat = [i[0].sum() for i in find.sequences]
                mat = pd.cut( pd.Series(mat), bins=[0, 10, 100, 1000,np.inf], labels=['<10', '10-100', '100-1000','>1000'], right=False)
                find.create_sce_predict(horizon=12)
                df_match=pd.concat([df_match,mat.value_counts()],axis=1)
                pred = find.val_sce
                pred = pred[pred.index == pred.index.max()].mean()
                true=df_input.iloc[nume:nume+12,coun]
                true = (true-df_input.iloc[nume:nume+h_train,coun].min())/(df_input.iloc[nume:nume+h_train,coun].max()-df_input.iloc[nume:nume+h_train,coun].min())
                mse_be_w.append(df_input.iloc[nume:nume+12,coun].sum())
                mse_be_n.append(diff_explained_h(true,pred).sum())

err_sf_pr_n=[]
we_n=[]
for i in range(len(df_input.columns)):  
    if (df_input.iloc[-34:-24,i]==0).all()==False:
        true = df_input.iloc[-24:-24+horizon,i]
        true = (true-df_input.iloc[-34:-24,i].min())/(df_input.iloc[-34:-24,i].max()-df_input.iloc[-34:-24,i].min())
        pred = df_sf_1.iloc[:,i]
        pred = (pred-df_input.iloc[-34:-24,i].min())/(df_input.iloc[-34:-24,i].max()-df_input.iloc[-34:-24,i].min())
        err_sf_pr_n.append(diff_explained_h(true, pred).sum())
        we_n.append(df_input.iloc[-24:-24+horizon,i].sum())

for i in range(len(df_input.columns)):   
    if (df_input.iloc[-22:-12,i]==0).all()==False:
        true = df_input.iloc[-12:,i]
        true = (true-df_input.iloc[-22:-12,i].min())/(df_input.iloc[-22:-12,i].max()-df_input.iloc[-22:-12,i].min())
        pred = df_sf_2.iloc[:,i]
        pred = (pred-df_input.iloc[-22:-12,i].min())/(df_input.iloc[-22:-12,i].max()-df_input.iloc[-22:-12,i].min())
        err_sf_pr_n.append(diff_explained_h(true, pred).sum())
        we_n.append(df_input.iloc[-12:,i].sum())



err_sf_pr_n = [np.mean(i)+0.001 for i in err_sf_pr_n]
mse_be_n = [np.mean(i)+0.001 for i in mse_be_n]
we_n = [i+1 for i in we_n]
mse_be_w = [i+1 for i in mse_be_w]


plt.figure(figsize=(10,6))
plt.scatter(np.log(we_n),err_sf_pr_n,color='blue',alpha=0.6)
plt.scatter(np.log(mse_be_w),mse_be_n,color='black',marker='x',alpha=1)
slope1, intercept1, r_value1, p_value1, std_err1 = linregress(np.log(we_n), err_sf_pr_n)
plt.plot(np.log(we_n), intercept1 + slope1 * np.log(we_n), color='blue', linestyle='--')
slope2, intercept2, r_value2, p_value2, std_err2 = linregress(np.log(mse_be_w), mse_be_n)
plt.plot(np.log(mse_be_w), intercept2 + slope2 * np.log(mse_be_w), color='black', linestyle='--')
plt.xlabel('Sum of Fatalies in forecasted window (log)')
plt.ylabel('Diff Explained (log)')
plt.show()




########### Worse MSE 

df_subr=df_var[(df_var.iloc[:,6]>0.37) & (df_var.iloc[:,5]<0.4)][4].mean()
mse_be_w=[]
mse_be_n=[]
df_match=pd.DataFrame()
for nume in range(1,380,5):
    for coun in range(len(df_input.columns)):
        if not (df_input.iloc[nume:nume+h_train,coun]==0).all():
            ser=df_input.iloc[nume:nume+h_train,coun]
            ser=(ser-ser.min())/(ser.max()-ser.min())
            diff = ser.diff()
            if (ser.std()>0.37) & (ser.mean()<0.4):
                df_copy=df_input.copy()
                df_copy.iloc[nume:nume+h_train,coun]=np.nan
                shape = Shape()
                shape.set_shape(df_input.iloc[nume:nume+h_train,coun]) 
                find = finder(df_copy.iloc[:-h,:],shape)
                find.find_patterns_lim(min_d=0.1,select=True,metric='dtw',dtw_sel=2,min_mat=3,d_increase=0.05)
                mat = [i[0].sum() for i in find.sequences]
                mat = pd.cut( pd.Series(mat), bins=[0, 10, 100, 1000,np.inf], labels=['<10', '10-100', '100-1000','>1000'], right=False)
                find.create_sce_predict(horizon=12)
                df_match=pd.concat([df_match,mat.value_counts()],axis=1)
                pred = find.val_sce
                pred = pred[pred.index == pred.index.max()].mean()
                true=df_input.iloc[nume:nume+12,coun]
                true = (true-df_input.iloc[nume:nume+h_train,coun].min())/(df_input.iloc[nume:nume+h_train,coun].max()-df_input.iloc[nume:nume+h_train,coun].min())
                mse_be_w.append(df_input.iloc[nume:nume+12,coun].sum())
                mse_be_n.append(mean_squared_error(true,pred))

err_sf_pr_n=[]
we_n=[]
for i in range(len(df_input.columns)):  
    if (df_input.iloc[-34:-24,i]==0).all()==False:
        true = df_input.iloc[-24:-24+horizon,i]
        true = (true-df_input.iloc[-34:-24,i].min())/(df_input.iloc[-34:-24,i].max()-df_input.iloc[-34:-24,i].min())
        pred = df_sf_1.iloc[:,i]
        pred = (pred-df_input.iloc[-34:-24,i].min())/(df_input.iloc[-34:-24,i].max()-df_input.iloc[-34:-24,i].min())
        err_sf_pr_n.append(mean_squared_error(true, pred))
        we_n.append(df_input.iloc[-24:-24+horizon,i].sum())

for i in range(len(df_input.columns)):   
    if (df_input.iloc[-22:-12,i]==0).all()==False:
        true = df_input.iloc[-12:,i]
        true = (true-df_input.iloc[-22:-12,i].min())/(df_input.iloc[-22:-12,i].max()-df_input.iloc[-22:-12,i].min())
        pred = df_sf_2.iloc[:,i]
        pred = (pred-df_input.iloc[-22:-12,i].min())/(df_input.iloc[-22:-12,i].max()-df_input.iloc[-22:-12,i].min())
        err_sf_pr_n.append(mean_squared_error(true, pred))
        we_n.append(df_input.iloc[-12:,i].sum())


err_sf_pr_n = [i+0.001 for i in err_sf_pr_n]
mse_be_n = [i+0.001 for i in mse_be_n]
we_n = [i+1 for i in we_n]
mse_be_w = [i+1 for i in mse_be_w]

df_norm = pd.DataFrame([err_sf_pr_n,we_n]).T
df_norm.to_csv('df_norm_wo.csv')

df_be = pd.DataFrame([mse_be_n,mse_be_w]).T
df_be.to_csv('df_wo.csv')

df_be=pd.read_csv('df_wo.csv',index_col=0)
mse_be_w = df_be.iloc[:,1]
mse_be_n = df_be.iloc[:,0]


df_norm=pd.read_csv('df_norm_wo.csv',index_col=0)
we_n = df_norm.iloc[:,1]
err_sf_pr_n = df_norm.iloc[:,0]

plt.figure(figsize=(10,6))
plt.scatter(np.log(we_n),np.log(err_sf_pr_n),color='blue',alpha=0.6)
plt.scatter(np.log(mse_be_w),np.log(mse_be_n),color='black',marker='x',alpha=1)
slope1, intercept1, r_value1, p_value1, std_err1 = linregress(np.log(we_n), np.log(err_sf_pr_n))
plt.plot(np.log(we_n), intercept1 + slope1 * np.log(we_n), color='gray', linestyle='--')
slope2, intercept2, r_value2, p_value2, std_err2 = linregress(np.log(mse_be_w), np.log(mse_be_n))
plt.xlabel('Sum of fatalies in forecasting window (log)')
plt.ylabel('Normalized MSE (log)')
#plt.savefig("out/best_reg.jpeg",dpi=400,bbox_inches="tight")
plt.show()
