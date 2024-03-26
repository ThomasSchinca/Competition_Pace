# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 00:34:39 2024

@author: thoma
"""

import CRPS.CRPS as pscore
import pandas as pd 
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
np.random.seed(1)
from shape import Shape,finder
import pickle

df_input = pd.read_csv('df_input.csv',index_col=(0),parse_dates=True)
df_tot_m = pd.read_csv('df_tot_m.csv',index_col=(0),parse_dates=True)
h_train=10
h=12
dict_pois={i :[] for i in df_input.columns}
dict_exp={i :[] for i in df_input.columns}
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
        pred = find.predict_distrib(df_input_sub.iloc[-h_train:,coun].max(),df_input_sub.iloc[-h_train:,coun].min(),horizon=12,quantile=[i/100 for i in range(1, 100)])
        dict_pois[df_input.columns[coun]]=pred
        pred_exp = find.predict_distrib(df_input_sub.iloc[-h_train:,coun].max(),df_input_sub.iloc[-h_train:,coun].min(),horizon=12,mode='expo',quantile=[i/100 for i in range(1, 100)])
        dict_exp[df_input.columns[coun]]=pred_exp
    else :
        pass
with open('test1_prob_pois.pkl', 'wb') as f:
    pickle.dump(dict_pois, f)
with open('test1_prob_exp.pkl', 'wb') as f:
    pickle.dump(dict_exp, f)

h_train=10
h=12
dict_pois={i :[] for i in df_input.columns}
dict_exp={i :[] for i in df_input.columns}
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
        pred = find.predict_distrib(df_input_sub.iloc[-h_train:,coun].max(),df_input_sub.iloc[-h_train:,coun].min(),horizon=12,quantile=[i/100 for i in range(1, 100)])
        dict_pois[df_input.columns[coun]]=pred
        pred_exp = find.predict_distrib(df_input_sub.iloc[-h_train:,coun].max(),df_input_sub.iloc[-h_train:,coun].min(),horizon=12,mode='expo',quantile=[i/100 for i in range(1, 100)])
        dict_exp[df_input.columns[coun]]=pred_exp
    else :
        pass
with open('test2_prob_pois.pkl', 'wb') as f:
    pickle.dump(dict_pois, f)
with open('test2_prob_exp.pkl', 'wb') as f:
    pickle.dump(dict_exp, f)    
    
