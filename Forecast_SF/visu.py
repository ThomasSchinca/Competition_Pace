# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:48:27 2023

@author: thoma
"""

import pandas as pd 
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
np.random.seed(1)
from shapefinder import Shape,finder
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error

df_tot_tot=pd.read_csv('data.csv',index_col=0)
bench1_tot = pd.read_csv('bench1.csv',index_col=0)
bench2_tot = pd.read_csv('bench2.csv',index_col=0)

tot_df_nn=pd.read_csv('10_h15.csv',index_col=0)
tot_df_ar=pd.read_csv('ar.csv',index_col=0)
tot_df_ar2=pd.read_csv('ar2.csv',index_col=0)
tot_df_obs=pd.read_csv('obs.csv',index_col=0)


scaler=MinMaxScaler((0,1))
df = scaler.fit_transform(df_tot_tot) 
df=pd.DataFrame(df)
tot_df_nn=tot_df_nn.fillna(0)
tot_df_nn=scaler.transform(tot_df_nn)
tot_df_nn=pd.DataFrame(tot_df_nn)
tot_df_ar=scaler.transform(tot_df_ar)
tot_df_ar=pd.DataFrame(tot_df_ar)
tot_df_ar2=scaler.transform(tot_df_ar2)
tot_df_ar2=pd.DataFrame(tot_df_ar2)
tot_df_obs=scaler.transform(tot_df_obs)
tot_df_obs=pd.DataFrame(tot_df_obs)
err_nn=[]
err_b=[]
err_b2=[]
for i in range(len(tot_df_nn.columns)):
    for y in range(4):
        err_nn.append(mean_squared_error(tot_df_obs.iloc[y*12:(y+1)*12,i], tot_df_nn.iloc[y*12:(y+1)*12,i]))
        err_b.append(mean_squared_error(tot_df_obs.iloc[y*12:(y+1)*12,i], tot_df_ar.iloc[y*12:(y+1)*12,i]))
        err_b2.append(mean_squared_error(tot_df_obs.iloc[y*12:(y+1)*12,i], tot_df_ar2.iloc[y*12:(y+1)*12,i]))
    err_nn.append(mean_squared_error(tot_df_obs.iloc[:,i], tot_df_nn.iloc[:,i]))
    err_b.append(mean_squared_error(tot_df_obs.iloc[:,i], tot_df_ar.iloc[:,i]))
    err_b2.append(mean_squared_error(tot_df_obs.iloc[:,i], tot_df_ar2.iloc[:,i]))

err_nn = np.array(err_nn).reshape((5,191),order='F')
err_b = np.array(err_b).reshape((5,191),order='F')
err_b2 = np.array(err_b2).reshape((5,191),order='F')

d_mse = err_nn/err_b2

x_labels = ['Bench1','Bench2']
# mea_2=(err_b2[:4, :].flatten()-err_nn[:4, :].flatten())
# mea_1=(err_b[:4, :].flatten()-err_nn[:4, :].flatten())

mea_2=np.log(err_b2[:4, :].flatten()+2**(-16)/err_nn[:4, :].flatten())
mea_1=np.log(err_b[:4, :].flatten()/err_nn[:4, :].flatten())

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
plt.yscale('log')
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


d_nn=[]
d_b=[]
d_b2=[]
win=2
for i in range(len(tot_df_nn.columns)):
    for y in range(4):
        real = tot_df_obs.iloc[y*12:(y+1)*12,i]
        sf=tot_df_nn.iloc[y*12:(y+1)*12,i]
        b1=tot_df_ar.iloc[y*12:(y+1)*12,i]
        b2=tot_df_ar2.iloc[y*12:(y+1)*12,i]
        max_s=0
        max_b1=0
        max_b2=0
        for wi in range(12-win):
            real_win = real.iloc[wi:wi+win+1]
            sf_win= sf.iloc[wi:wi+win+1]
            b1_win=b1.iloc[wi:wi+win+1]
            b2_win=b2.iloc[wi:wi+win+1]
            t_v=[]
            for value in real_win[1:].index:
                if ((real_win[value]<1.05*real_win[value-1]) and (real_win[value]>0.95*real_win[value-1])) or (real_win[value-1]==0 and real_win[value]==0) :
                    t_v.append(1)
                elif (real_win[value]>1.05*real_win[value-1]):
                    t_v.append(2)
                elif (real_win[value]<0.95*real_win[value-1]):
                    t_v.append(0)
            if (pd.Series(t_v)==1).all()==False:
                sf_v=[]
                for value in sf_win[1:].index:
                    if ((sf_win[value]<1.05*sf_win[value-1]) and (sf_win[value]>0.95*sf_win[value-1])) or (sf_win[value-1]==0 and sf_win[value]==0) :
                        sf_v.append(1)
                    elif (sf_win[value]>1.05*sf_win[value-1]):
                        sf_v.append(2)
                    elif (sf_win[value]<0.95*sf_win[value-1]):
                        sf_v.append(0)
                mat=0        
                for index in range(len(sf_v)):
                    if t_v[index] ==sf_v[index]:
                        mat += 1  
                max_s=max_s+mat/len(sf_v)
                
                b1_v=[]
                for value in b1_win[1:].index:
                    if ((b1_win[value]<1.05*b1_win[value-1]) and (b1_win[value]>0.95*b1_win[value-1])) or (b1_win[value-1]==0 and b1_win[value]==0) :
                        b1_v.append(1)
                    elif (b1_win[value]>1.05*b1_win[value-1]):
                        b1_v.append(2)
                    elif (b1_win[value]<0.95*b1_win[value-1]):
                        b1_v.append(0)
                mat=0        
                for index in range(len(b1_v)):
                    if t_v[index] ==b1_v[index]:
                        mat += 1  
                max_b1=max_b1+mat/len(b1_v)
                
                b2_v=[]
                for value in b2_win[1:].index:
                    if ((b2_win[value]<1.05*b2_win[value-1]) and (b2_win[value]>0.95*b2_win[value-1])) or (b2_win[value-1]==0 and b2_win[value]==0) :
                        b2_v.append(1)
                    elif (b2_win[value]>1.05*b2_win[value-1]):
                        b2_v.append(2)
                    elif (b2_win[value]<0.95*b2_win[value-1]):
                        b2_v.append(0)
                mat=0        
                for index in range(len(b2_v)):
                    if t_v[index] ==b2_v[index]:
                        mat += 1  
                max_b2=max_b2+mat/len(b2_v)
        d_nn.append(max_s)
        d_b.append(max_b1)
        d_b2.append(max_b2)

d_nn = np.array(d_nn).reshape((4,191),order='F')
d_b = np.array(d_b).reshape((4,191),order='F')
d_b2 = np.array(d_b2).reshape((4,191),order='F')

x_labels = ['Bench1','Bench2']
mea_2=(d_nn[:4, :].flatten()-d_b2[:4, :].flatten())
mea_1=(d_nn[:4, :].flatten()-d_b[:4, :].flatten())
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
plt.ylabel('Classif difference',color=blue_color)
plt.tick_params(axis='y', colors=blue_color)
plt.scatter(mean2_data.index, mean2_data['mean'], color=blue_color, marker='o', s=marker_size)
plt.errorbar(mean2_data.index, mean2_data['mean'], yerr=mean2_data['std'], fmt='none', color=blue_color, linewidth=linewidth)
plt.grid(False)
plt.hlines(0,-0.5,1.5,linestyles='--',color='r')
plt.xticks([0,1],['Benchmark 1','Benchmark 2'])
plt.show()

d_comp=d_nn-d_b2

# year=['2021','2020','2019','2018']
# for i in pd.DataFrame(d_comp).mean().sort_values(ascending=False).index[:10]:
#     for y in range(4):
#         real = tot_df_obs.iloc[y*12:(y+1)*12,i]
#         sf=tot_df_nn.iloc[y*12:(y+1)*12,i]
#         b1=tot_df_ar.iloc[y*12:(y+1)*12,i]
#         b2=tot_df_ar2.iloc[y*12:(y+1)*12,i]
#         plt.figure(figsize=(15, 10))
#         plt.plot(sf, label='ShapeFinder', marker='o',color='r')
#         #plt.plot(b1, label='Bench1', marker='o',color='g')
#         plt.plot(b2, label='Bench2', marker='o',color='b')
#         plt.plot(real,label='Obs',marker='o',linewidth=5)
#         plt.legend()
#         plt.grid(True)
#         plt.title(df_tot_tot.columns[i]+year[y])
#         plt.show()
        
#         print(df_tot_tot.columns[i]+year[y])
#         print(mean_squared_error(real,sf))
#         print(mean_squared_error(real,b2))
#         print('      ')
        


d_nn=[]
d_b=[]
d_b2=[]
win=4
for i in range(len(tot_df_nn.columns)):
    for y in range(4):
        real = tot_df_obs.iloc[y*12:(y+1)*12,i]
        sf=tot_df_nn.iloc[y*12:(y+1)*12,i]
        b1=tot_df_ar.iloc[y*12:(y+1)*12,i]
        b2=tot_df_ar2.iloc[y*12:(y+1)*12,i]
        max_s=[]
        max_b1=[]
        max_b2=[]
        for wi in range(3):
            real_win = real.iloc[(win*wi):(win*(wi+1))]
            sf_win= sf.iloc[(win*wi):(win*(wi+1))]
            b1_win=b1.iloc[(win*wi):(win*(wi+1))]
            b2_win=b2.iloc[(win*wi):(win*(wi+1))]
            
            max_s.append(abs(real_win.std()-sf_win.std()))
            max_b1.append(abs(real_win.std()-b1_win.std()))
            max_b2.append(abs(real_win.std()-b2_win.std()))
            
        d_nn.append(np.mean(max_s))
        d_b.append(np.mean(max_b1))
        d_b2.append(np.mean(max_b2))

d_nn = np.array(d_nn).reshape((4,191),order='F')
d_b = np.array(d_b).reshape((4,191),order='F')
d_b2 = np.array(d_b2).reshape((4,191),order='F')

x_labels = ['Bench1','Bench2']
mea_2=(d_b2[:4, :].flatten()-d_nn[:4, :].flatten())
mea_1=(d_b[:4, :].flatten()-d_nn[:4, :].flatten())
plt.plot(mea_2)
means = [mea_1.mean(),mea_2.mean()]
std = [2*mea_1.std()/np.sqrt(len(mea_1)),2*mea_2.std()/np.sqrt(len(mea_2))]
mean2_data = pd.DataFrame({
    'mean': means,
    'std': std
})

blue_color = '#404040'  # Dark grey shade
orange_color = '#A0A0A0'  # Light grey shade

plt.figure(figsize=(12,8))
marker_size = 150
linewidth = 3
fonts=25
plt.rc('font', size=24)
plt.ylabel('STD difference',color=blue_color)
plt.tick_params(axis='y', colors=blue_color)
plt.scatter(mean2_data.index, mean2_data['mean'], color=blue_color, marker='o', s=marker_size)
plt.errorbar(mean2_data.index, mean2_data['mean'], yerr=mean2_data['std'], fmt='none', color=blue_color, linewidth=linewidth)
plt.grid(False)
plt.hlines(0,-0.5,1.5,linestyles='--',color='r')
plt.xticks([0,1],['Benchmark 1','Benchmark 2'])
plt.show()








d_nn=[]
d_b=[]
d_b2=[]
inte=0.1
for i in range(len(tot_df_nn.columns)):
    for y in range(4):
        real = tot_df_obs.iloc[y*12:(y+1)*12,i]
        real=real.reset_index(drop=True)
        sf=tot_df_nn.iloc[y*12:(y+1)*12,i]
        sf=sf.reset_index(drop=True)
        b1=tot_df_ar.iloc[y*12:(y+1)*12,i]
        b1=b1.reset_index(drop=True)
        b2=tot_df_ar2.iloc[y*12:(y+1)*12,i]
        b2=b2.reset_index(drop=True)
        max_s=0
        max_b1=0
        max_b2=0
        for value in real[1:].index:
            if ((real[value]<1.05*real[value-1]) and (real[value]>0.95*real[value-1])) or (real[value-1]==0 and real[value]==0) :
                1
            else:
                if (real[value]>real[value-1]):
                    inter= [(real[value]-real[value-1])*(1-inte),(real[value]-real[value-1])*(1+inte)]
                    if ((sf[value]-sf[value-1])>inter[0] and (sf[value]-sf[value-1])<inter[1]):
                        max_s=max_s+1
                    else:
                        if value==11:
                            if ((sf[value-1]-sf[value-2])>inter[0] and (sf[value-1]-sf[value-2])<inter[1]):
                                max_s=max_s+0.5
                        elif value==1:
                            if ((sf[value+1]-sf[value])>inter[0] and (sf[value+1]-sf[value])<inter[1]):
                                max_s=max_s+0.5
                        else : 
                            if ((sf[value+1]-sf[value])>inter[0] and (sf[value+1]-sf[value])<inter[1]) or ((sf[value-1]-sf[value-2])>inter[0] and (sf[value-1]-sf[value-2])<inter[1]):
                                max_s=max_s+0.5
                else:
                    inter= [(real[value]-real[value-1])*(1-inte),(real[value]-real[value-1])*(1+inte)]
                    if ((sf[value]-sf[value-1])<inter[0] and (sf[value]-sf[value-1])>inter[1]):
                        max_s=max_s+1
                    else:
                        if value==11:
                            if ((sf[value-1]-sf[value-2])<inter[0] and (sf[value-1]-sf[value-2])>inter[1]):
                                max_s=max_s+0.5
                        elif value==1:
                            if ((sf[value+1]-sf[value])<inter[0] and (sf[value+1]-sf[value])>inter[1]):
                                max_s=max_s+0.5
                        else : 
                            if ((sf[value+1]-sf[value])<inter[0] and (sf[value+1]-sf[value])>inter[1]) or ((sf[value-1]-sf[value-2])<inter[0] and (sf[value-1]-sf[value-2])>inter[1]):
                                max_s=max_s+0.5
                     
                                
                     
                        
                if (real[value]>real[value-1]):
                    inter= [(real[value]-real[value-1])*(1-inte),(real[value]-real[value-1])*(1+inte)]
                    if ((b1[value]-b1[value-1])>inter[0] and (b1[value]-b1[value-1])<inter[1]):
                        max_b1=max_b1+1
                    else:
                        if value==11:
                            if ((b1[value-1]-b1[value-2])>inter[0] and (b1[value-1]-b1[value-2])<inter[1]):
                                max_b1=max_b1+0.5
                        elif value==1:
                            if ((b1[value+1]-b1[value])>inter[0] and (b1[value+1]-b1[value])<inter[1]):
                                max_b1=max_b1+0.5
                        else : 
                            if ((b1[value+1]-b1[value])>inter[0] and (b1[value+1]-b1[value])<inter[1]) or ((b1[value-1]-b1[value-2])>inter[0] and (b1[value-1]-b1[value-2])<inter[1]):
                                max_b1=max_b1+0.5
                else:
                    inter= [(real[value]-real[value-1])*(1-inte),(real[value]-real[value-1])*(1+inte)]
                    if ((b1[value]-b1[value-1])<inter[0] and (b1[value]-b1[value-1])>inter[1]):
                        max_s=max_s+1
                    else:
                        if value==11:
                            if ((b1[value-1]-b1[value-2])<inter[0] and (b1[value-1]-b1[value-2])>inter[1]):
                                max_b1=max_b1+0.5
                        elif value==1:
                            if ((b1[value+1]-b1[value])<inter[0] and (b1[value+1]-b1[value])>inter[1]):
                                max_b1=max_b1+0.5
                        else : 
                            if ((b1[value+1]-b1[value])<inter[0] and (b1[value+1]-b1[value])>inter[1]) or ((b1[value-1]-b1[value-2])<inter[0] and (b1[value-1]-b1[value-2])>inter[1]):
                                max_b1=max_b1+0.5
                
                
                
                if (real[value]>real[value-1]):
                    inter= [(real[value]-real[value-1])*(1-inte),(real[value]-real[value-1])*(1+inte)]
                    if ((b2[value]-b2[value-1])>inter[0] and (b2[value]-b2[value-1])<inter[1]):
                        max_b2=max_b2+1
                    else:
                        if value==11:
                            if ((b2[value-1]-b2[value-2])>inter[0] and (b2[value-1]-b2[value-2])<inter[1]):
                                max_b2=max_b2+0.5
                        elif value==1:
                            if ((b2[value+1]-b2[value])>inter[0] and (b2[value+1]-b2[value])<inter[1]):
                                max_b2=max_b2+0.5
                        else : 
                            if ((b2[value+1]-b2[value])>inter[0] and (b2[value+1]-b2[value])<inter[1]) or ((b2[value-1]-b2[value-2])>inter[0] and (b2[value-1]-b2[value-2])<inter[1]):
                                max_b2=max_b2+0.5
                else:
                    inter= [(real[value]-real[value-1])*(1-inte),(real[value]-real[value-1])*(1+inte)]
                    if ((b2[value]-b2[value-1])<inter[0] and (b2[value]-b2[value-1])>inter[1]):
                        max_s=max_s+1
                    else:
                        if value==11:
                            if ((b2[value-1]-b2[value-2])<inter[0] and (b2[value-1]-b2[value-2])>inter[1]):
                                max_b2=max_b2+0.5
                        elif value==1:
                            if ((b2[value+1]-b2[value])<inter[0] and (b2[value+1]-b2[value])>inter[1]):
                                max_b2=max_b2+0.5
                        else : 
                            if ((b2[value+1]-b2[value])<inter[0] and (b2[value+1]-b2[value])>inter[1]) or ((b2[value-1]-b2[value-2])<inter[0] and (b2[value-1]-b2[value-2])>inter[1]):
                                max_b2=max_b2+0.5                
                
        d_nn.append(max_s)
        d_b.append(max_b1)
        d_b2.append(max_b2)        

d_nn = np.array(d_nn).reshape((4,191),order='F')
d_b = np.array(d_b).reshape((4,191),order='F')
d_b2 = np.array(d_b2).reshape((4,191),order='F')

x_labels = ['Bench1','Bench2']
mea_2=(d_nn[:4, :].flatten()-d_b2[:4, :].flatten())
mea_1=(d_nn[:4, :].flatten()-d_b[:4, :].flatten())
means = [mea_1.mean(),mea_2.mean()]
std = [2*mea_1.std()/np.sqrt(len(mea_1)),2*mea_2.std()/np.sqrt(len(mea_2))]
mean2_data = pd.DataFrame({
    'mean': means,
    'std': std
})

blue_color = '#404040'  # Dark grey shade
orange_color = '#A0A0A0'  # Light grey shade

plt.figure(figsize=(12,8))
marker_size = 150
linewidth = 3
fonts=25
plt.rc('font', size=24)
plt.ylabel('Good diff model spotted difference',color=blue_color)
plt.tick_params(axis='y', colors=blue_color)
plt.scatter(mean2_data.index, mean2_data['mean'], color=blue_color, marker='o', s=marker_size)
plt.errorbar(mean2_data.index, mean2_data['mean'], yerr=mean2_data['std'], fmt='none', color=blue_color, linewidth=linewidth)
plt.grid(False)
plt.hlines(0,-0.5,1.5,linestyles='--',color='r')
plt.xticks([0,1],['Benchmark 1','Benchmark 2'])
plt.show()






d_nn=[]
d_b=[]
d_b2=[]
k=5
for i in range(len(tot_df_nn.columns)):
    for y in range(4):
        real = tot_df_obs.iloc[y*12:(y+1)*12,i]
        real=real.reset_index(drop=True)
        sf=tot_df_nn.iloc[y*12:(y+1)*12,i]
        sf=sf.reset_index(drop=True)
        b1=tot_df_ar.iloc[y*12:(y+1)*12,i]
        b1=b1.reset_index(drop=True)
        b2=tot_df_ar2.iloc[y*12:(y+1)*12,i]
        b2=b2.reset_index(drop=True)
        max_s=0
        max_b1=0
        max_b2=0
        if (real==0).all()==False:
            for value in real[1:].index:
                # if ((real[value]<1.05*real[value-1]) and (real[value]>0.95*real[value-1])) or (real[value-1]==0 and real[value]==0):
                #     1
                if (real[value]==real[value-1]):
                    1
                else:
                    max_exp=0
                    if (real[value]-real[value-1])/(sf[value]-sf[value-1])>0 and sf[value]-sf[value-1] != 0:
                        t=abs(((real[value]-real[value-1])-(sf[value]-sf[value-1]))/(real[value]-real[value-1]))
                        max_exp = np.exp(-(k*t))
                    else:
                        if value==11:
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
                    if (real[value]-real[value-1])/(b1[value]-b1[value-1])>0 and b1[value]-b1[value-1] != 0:
                        t=abs(((real[value]-real[value-1])-(b1[value]-b1[value-1]))/(real[value]-real[value-1]))
                        max_exp = np.exp(-(k*t))
                    else:
                        if value==11:
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
                    if (real[value]-real[value-1])/(b2[value]-b2[value-1])>0 and b2[value]-b2[value-1] != 0:
                        t=abs(((real[value]-real[value-1])-(b2[value]-b2[value-1]))/(real[value]-real[value-1]))
                        max_exp = np.exp(-(k*t))
                    else:
                        if value==11:
                            if (real[value]-real[value-1])/(b2[value-1]-b2[value-2])>0 and b2[value-1]-b2[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(b2[value-1]-b2[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                        elif value==1:
                            if (real[value]-real[value-1])/(b2[value+1]-b2[value])>0 and b2[value+1]-b2[value] != 0:
                                t=abs(((real[value]-real[value-1])-(b2[value+1]-b2[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                        else : 
                            if (real[value]-real[value-1])/(b2[value-1]-b2[value-2])>0 and b2[value-1]-b2[value-2] != 0:
                                t=abs(((real[value]-real[value-1])-(b2[value-1]-b2[value-2]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-t):
                                    max_exp = np.exp(-(k*t))
                            if (real[value]-real[value-1])/(b2[value+1]-b2[value])>0 and b2[value+1]-b2[value] != 0:
                                t=abs(((real[value]-real[value-1])-(b2[value+1]-b2[value]))/(real[value]-real[value-1]))/2
                                if max_exp<np.exp(-(k*t)):
                                    max_exp = np.exp(-(k*t))
                    max_b2=max_b2+max_exp         
                
        d_nn.append(max_s)
        d_b.append(max_b1)
        d_b2.append(max_b2)        

d_nn = np.array(d_nn).reshape((4,191),order='F')
d_b = np.array(d_b).reshape((4,191),order='F')
d_b2 = np.array(d_b2).reshape((4,191),order='F')

x_labels = ['Bench1','Bench2']
mea_2=(d_nn[:4, :].flatten()-d_b2[:4, :].flatten())
mea_1=(d_nn[:4, :].flatten()-d_b[:4, :].flatten())
means = [mea_1.mean(),mea_2.mean()]
std = [2*mea_1.std()/np.sqrt(len(mea_1)),2*mea_2.std()/np.sqrt(len(mea_2))]
mean2_data = pd.DataFrame({
    'mean': means,
    'std': std
})

blue_color = '#404040'  # Dark grey shade
orange_color = '#A0A0A0'  # Light grey shade

plt.figure(figsize=(12,8))
marker_size = 150
linewidth = 3
fonts=25
plt.rc('font', size=24)
plt.ylabel('Diff exp',color=blue_color)
plt.tick_params(axis='y', colors=blue_color)
plt.scatter(mean2_data.index, mean2_data['mean'], color=blue_color, marker='o', s=marker_size)
plt.errorbar(mean2_data.index, mean2_data['mean'], yerr=mean2_data['std'], fmt='none', color=blue_color, linewidth=linewidth)
plt.grid(False)
plt.hlines(0,-0.5,1.5,linestyles='--',color='r')
plt.xticks([0,1],['Benchmark 1','Benchmark 2'])
plt.show()

d_diff = d_nn-d_b2
d_mse = d_mse[:4,:]

df_plot = pd.DataFrame([d_diff.flatten(),d_mse.flatten()])
df_plot = df_plot.T
df_plot=df_plot.sort_values([0],ascending=False)


year=['2021','2020','2019','2018']
ind=[4,1,9]
yea=[3,0,2]
for k in range(3):
    y=yea[k]
    i=pd.DataFrame(d_diff).mean().sort_values(ascending=False).index[ind[k]]
    real = tot_df_obs.iloc[y*12:(y+1)*12,i]
    sf=tot_df_nn.iloc[y*12:(y+1)*12,i]
    b1=tot_df_ar.iloc[y*12:(y+1)*12,i]
    b2=tot_df_ar2.iloc[y*12:(y+1)*12,i]
    plt.figure(figsize=(15, 10))
    plt.plot(sf, label='ShapeFinder', marker='o',color='r')
    #plt.plot(b1, label='Bench1', marker='o',color='g')
    plt.plot(b2, label='Bench2', marker='o',color='b')
    plt.plot(real,label='Obs',marker='o',color='black',linewidth=5)
    #plt.legend()
    plt.grid(True)
    plt.title(df_tot_tot.columns[i]+year[y])
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.show()
    
    print(df_tot_tot.columns[i]+year[y])
    print(d_nn[y,i]/d_b2[y,i])
    print('      ')    
    
    
plt.figure(figsize=(15, 10))
sf=[0.6,0.8,0.05,0.4,0.3]
real=[0,1,0.5,0.8,0]
b2=[0.45,0.45,0.45,0.45,0.45]
plt.plot(sf, label='ShapeFinder', marker='o',color='r')
plt.plot(b2, label='Bench2', marker='o',color='b')
plt.plot(real,label='Obs',marker='o',color='black',linewidth=5)
print(mean_squared_error(real,sf)/mean_squared_error(real,b2))
plt.xticks([0,1,2,3,4],[1,2,3,4,5])
plt.grid()
plt.show()


tot_df_nn=pd.read_csv('10_h15.csv',index_col=0)
tot_df_nn=tot_df_nn.fillna(0)
tot_df_nn = pd.concat([tot_df_nn.iloc[36:],tot_df_nn.iloc[24:36],tot_df_nn.iloc[12:24],tot_df_nn.iloc[:12]])
bench1_tot = pd.read_csv('bench1.csv',index_col=0)


tot_df_nn.columns = bench1_tot.columns
tot_df_nn.index = bench1_tot.index

tot_df_nn.reset_index(inplace=True)
new_df = pd.melt(tot_df_nn, id_vars=['month_id'], var_name='country_id', value_name='outcome')
new_df['draw'] = 1
new_df.columns = ['month_id', 'country_id', 'outcome', 'draw']
new_df = new_df[['month_id', 'country_id', 'draw', 'outcome']]

new_df = new_df.apply(pd.to_numeric)
new_df=new_df.sort_values(['month_id','country_id'])

df_2018 = new_df.iloc[:2292,:]
df_2019 = new_df.iloc[2292:2292*2,:]
df_2020 = new_df.iloc[2292*2:2292*3,:]
df_2021 = new_df.iloc[2292*3:,:]

# df_2018.to_parquet('pred_2018.parquet')
# df_2019.to_parquet('pred_2019.parquet')
# df_2020.to_parquet('pred_2020.parquet')
# df_2021.to_parquet('pred_2021.parquet')

