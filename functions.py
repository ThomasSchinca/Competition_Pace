# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 20:10:07 2023

@author: thoma
"""

import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from pmdarima.arima import auto_arima,ARIMA
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from random import sample
import math

# =============================================================================
# Input/Output and Clusters 
# =============================================================================

# =============================================================================

def get_dynamic_input_output(y: pd.Series,
                             ar: int=1,
                             n_clu: int=5,
                             number_s: int=5,
                             model=TimeSeriesKMeans(n_clusters=5, 
                                                    metric="dtw",
                                                    max_iter_barycenter=100,
                                                    verbose=0,
                                                    random_state=0),
                             output: bool=True,
                             train_test_split: float=0.7):
    
    """This function takes a time series as input, and returns
    time sequences of length number_s, and a dummy set of the 
    obtained clusters. If no_output is set to False, the function 
    also returns the output, for an autoregressive setup. 
    
    Parameters:
        y (pd.Series): Data input
        ar (int): Number of observations to exclude from beginning of the time series
        n_clu (int): Number of clusters
        number_s (int): Length of time window
        model: Model for clustering
        output (bool): Return output for autoregressive setup
        train_test_split (float): Cut-off for train test data split
        
    Return: 
        (dict): Containing input (time series and clusters) and output
        if output == True for autoregressive setup
    """
 
    # Training data
    ex=y.iloc[:int(train_test_split*len(y))]
    ts_seq=[]
    
    # Make list of lists, 
    # each sub-list contains number_s observations
    for i in range(number_s,len(ex)):
        ts_seq.append(y.iloc[i-number_s:i])
        
    # Convert into array,
    # each row is a time series of number_s observations 
    ts_seq=np.array(ts_seq)
    
    # Sacling
    ts_seq_l= pd.DataFrame(ts_seq).T
    ts_seq_l=(ts_seq_l-ts_seq_l.min())/(ts_seq_l.max()-ts_seq_l.min())
    ts_seq_l = ts_seq_l.fillna(0) # if seq uniform 
    ts_seq_l=np.array(ts_seq_l.T)
    
    # Reshape array,
    # each sub array contains times series of number_s observations
    ts_seq_l=ts_seq_l.reshape(len(ts_seq_l),number_s,1)
    
    # Clustering and convert into dummy set
    model.n_clusters=n_clu
    m_dba = model.fit(ts_seq_l)
    cl= m_dba.labels_
    cl=pd.Series(cl)
    cl=pd.get_dummies(cl)
        
    # Test data    
    ts_seq_2=[]
    
    # Make list of lists, 
    # each sub-list contains number_s observations
    for i in range(len(ex),len(y)):
        ts_seq_2.append(y.iloc[i-number_s:i])
        
    # Convert into array,
    # each row is a time series of number_s observations   
    ts_seq_2=np.array(ts_seq_2)
    
    # Sacling
    ts_seq_l= pd.DataFrame(ts_seq_2).T
    ts_seq_l=(ts_seq_l-ts_seq_l.min())/(ts_seq_l.max()-ts_seq_l.min())
    ts_seq_l = ts_seq_l.fillna(0)
    ts_seq_l=np.array(ts_seq_l.T)
    
    # Reshape array,
    # each sub array contains times series of number_s observations
    ts_seq_l=ts_seq_l.reshape(len(ts_seq_l),number_s,1)
    
    # Use trained knn model to predict clusters in test data
    # and convert into dummy set
    y_test = m_dba.predict(ts_seq_l)
    y_test=pd.Series(y_test)
    y_test=pd.get_dummies(y_test)
    
    # Make sure that length of dummy set is equal to n_clu
    # If not, add empty column 
    y_t = pd.DataFrame(columns=range(n_clu))
    y_test= pd.concat([y_t,y_test],axis=0)   
    y_test = y_test.fillna(0)

    # Merge test and training data
    ts_seq=np.concatenate([ts_seq,ts_seq_2],axis=0)
    cl=pd.concat([cl,y_test])
    
    # Return output and input     
    if output == True:
        return({'input':np.concatenate([np.array(ts_seq[:,-ar:]),np.array(cl)],axis=1)[:-1,:],
                'output':ts_seq[1:,-1]})
    # Return only input
    elif output == False:
        return({'input':np.concatenate([np.array(ts_seq[:,-ar:]),np.array(cl)],axis=1)})
 
# =============================================================================       

def get_dynamic_clusters(y: pd.Series,
                         n_clu: int=5,
                         number_s: int=5,
                         model=TimeSeriesKMeans(n_clusters=5, 
                                                metric="dtw",
                                                max_iter_barycenter=100,
                                                verbose=0,
                                                random_state=0),
                         plot: str=None, 
                         plot_params: dict=None,
                         path: str=None):
    
    """This function takes time series as input and performs a cluster analysis. 
    The centroids, cluster assignments and time series are returned. 
    
    Parameters:
        y (pd.Series): Data input
        n_clu (int): Number of clusters
        number_s (int): Length of time window
        model: Model for clustering
        plot (str): Whether to return no plot ("none"), one plot ("one")
                    or multiple ("multiple")
        
    Return: 
        (dict): Containing centroids of cluster, cluster assignments and time series
        if plot = 'one' or 'multiple', figure showing centroids is returned
    """
    
    ts_seq=[]
    
    # Make list of lists, 
    # each sub-list contains number_s observations
    for i in range(number_s,len(y)):
        ts_seq.append(y.iloc[i-number_s:i])
        
    # Convert into array,
    # each row is a time series of number_s observations 
    ts_seq=np.array(ts_seq)
    
    # Scaling 
    ts_seq_l= pd.DataFrame(ts_seq).T
    ts_seq_l=(ts_seq_l-ts_seq_l.min())/(ts_seq_l.max()-ts_seq_l.min())
    ts_seq_l = ts_seq_l.fillna(0) # if seq uniform 
    ts_seq_l=np.array(ts_seq_l.T)
    
    
    # Reshape array,
    # each sub array contains times series of number_s observations
    ts_seq_l=ts_seq_l.reshape(len(ts_seq_l),number_s,1)
    
    # Clustering
    model.n_clusters=n_clu
    m_dba = model.fit(ts_seq_l)
    cl= m_dba.labels_
    
    # Plot none
    if plot==None:
       return({'cluster_shape':m_dba.cluster_centers_,
               'seqences_clusters':cl,
               'sequences':ts_seq})
   
    # Plot none
    if plot_params!=None: 
            plt.rcParams.update(plot_params)
    
    # Plot one
    if plot=='one':
        fig = plt.figure()
        col = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i in np.unique(m_dba.labels_):
            plt.plot(m_dba.cluster_centers_[i],
                         color=col[i],
                         label='Cluster '+str(i+1))
        plt.legend(loc=(1.04, 0))
        # Save
        if path!=None:
            plt.savefig(path,
                        dpi=100,
                        bbox_inches='tight')
            
        return({'cluster_shape':m_dba.cluster_centers_,
               'seqences_clusters':cl,
               'sequences':ts_seq},
               fig)
    
    # Plot multiple
    if plot=='multiple':
        fig, axes = plt.subplots(nrows=math.ceil(n_clu/3),ncols=3)
        plt.setp(axes, xticks=[], yticks=[])
        for i, ax in zip(np.unique(m_dba.labels_), axes.ravel()):
            ax.plot(m_dba.cluster_centers_[i])
        # Remove empty subplots 
        diff = math.ceil(n_clu/3) *3 - n_clu
        for d in range(1, diff+1): 
            axes.flat[(-d)].set_visible(False)
        # Save
        if path!=None:
            plt.savefig(path,
                        dpi=100,
                        bbox_inches='tight')
        return({'cluster_shape':m_dba.cluster_centers_,
               'seqences_clusters':cl,
               'sequences':ts_seq},
               fig)

# =============================================================================

def extract_b_clu(y: pd.Series,
                  test_clu: list,
                  test_win: list,
                  model=TimeSeriesKMeans(n_clusters=5, 
                                         metric="dtw",
                                         max_iter_barycenter=100,
                                         verbose=0,
                                         random_state=0),
                  train_test_split: float=0.7,
                  select: str='top',
                  top: int=5,
                  thres: float=0.15):
    
    """ """
    
    df_std=pd.DataFrame()
    X=pd.DataFrame()
    
    # Training data
    y_eff = y.iloc[:int(train_test_split*len(y))]
    
    # For varying cluster number
    for n_clu in test_clu:
        # For varyinf window length
        for number_s in test_win:
            # Update number of clusters in model
            model.n_clusters=n_clu
            
            # Get clusters and time sequences
            clus=get_dynamic_clusters(y_eff,
                                      number_s=number_s,
                                      n_clu=n_clu,
                                      model=model)
            # Get input
            clu_input=get_dynamic_input_output(y,
                                               n_clu=n_clu,
                                               number_s=number_s,
                                               model=model,
                                               train_test_split=train_test_split,
                                               output=False)
                            # Last row of time series, without first observation
            data=pd.concat([pd.Series(clus['sequences'][1:,-1]),
                            # Last row of cluster assignments, with first observation
                            pd.Series(clus['seqences_clusters'][:-1])],
                            axis=1)
            # Std of time series for each cluster
            df_std=pd.concat([df_std,data.groupby(1).std()],
                             axis=0)
            # Get input, exlduing first column
            X=pd.concat([X,
                         pd.DataFrame(clu_input['input'][:,1:],
                         index=range(number_s,len(y)))],
                         axis=1)
            
    # Rename columns
    X.columns=range(len(X.columns))
    df_std.index=range(len(df_std))
    
    # Select top 5, with lowest std
    if select=='top':
        out = X.iloc[:,df_std.sort_values(0).index[:top]]
    # Select if std below threshold
    elif select=='threshold':
        out = X.iloc[:,df_std[df_std[0]<thres].index]
    out=out.fillna(0)    
    out= out.loc[:,out.apply(pd.Series.nunique)!=1]
    return(out)

# =============================================================================
# ARIMA
# =============================================================================

def DARIMA(y, 
           X=None,
           number_s=5,
           n_clu=5):
    y_eff = y[number_s:]
    ts_seq=[]
    for i in range(number_s,len(y)):
        ts_seq.append(y.iloc[i-number_s:i])
    ts_seq=np.array(ts_seq)
    ts_seq_l= pd.DataFrame(ts_seq).T
    ts_seq_l=(ts_seq_l-ts_seq_l.min())/(ts_seq_l.max()-ts_seq_l.min())
    ts_seq_l=np.array(ts_seq_l.T)
    ts_seq_l=ts_seq_l.reshape(len(ts_seq_l),number_s,1)
    km_dba = TimeSeriesKMeans(n_clusters=n_clu, metric="dtw",max_iter_barycenter=100,verbose=0,random_state=0).fit(ts_seq_l)
    cl= km_dba.labels_
    cl=pd.Series(cl)
    cl=pd.get_dummies(cl)
    if X==None:
        model_f = auto_arima(y_eff,np.array(cl))
    else:    
        model_f = auto_arima(y_eff,np.concatenate([X,np.array(cl)],axis=1))
    return model_f

def Compare_fit(y,
                X=None,
                arima=None,
                darima=None,
                number_s=5,
                n_clu=5,
                metric='aic',
                plot_res=False):
    y_eff = y[number_s:]
    if arima != None:
        if metric=='aic':
            ar_met=arima.aic()
        elif metric =='resid':
            ar_met=abs(arima.resid()).mean() 
    else:
        if X is None:
            arima = auto_arima(y_eff)
        else:
            arima = auto_arima(y_eff,X)
        if metric=='aic':
            ar_met=arima.aic()
        elif metric =='resid':
            ar_met=abs(arima.resid()).mean()   
    if darima != None:
        if metric=='aic':
            arx_met=darima.aic()
        elif metric =='resid':
            arx_met=abs(darima.resid()).mean()   
    else:
        if X is None:
            darima = DARIMA(y,number_s=number_s,n_clu=n_clu)
        else:
            darima = DARIMA(y,X,number_s=number_s,n_clu=n_clu)
        if metric=='aic':
            arx_met=darima.aic()
        elif metric =='resid':
            arx_met=abs(darima.resid()).mean()       
    if plot_res==True:
        plt.plot(y_eff,label='Observed')
        plt.plot(y_eff-arima.resid(),label='ARIMA fit')
        plt.plot(y_eff-darima.resid(),label='DARIMA fit')
        plt.legend()
        plt.show()
        
    return(pd.DataFrame([ar_met,arx_met,((ar_met-arx_met)/abs(ar_met))*100],index=['ARIMA_AIC','DARIMA_AIC','%_Improv']))

def Compare_pred(y,X=None,number_s=5,n_clu=5,plot_res=False,train_test_split=0.7):
    ex=y.iloc[:int(train_test_split*len(y))]
    ex_test=y.iloc[int(train_test_split*len(y)):]
    ex_train = ex[number_s:]
    ts_seq=[]
    for i in range(number_s,len(ex)):
        ts_seq.append(y.iloc[i-number_s:i])
    ts_seq=np.array(ts_seq)
    ts_seq_l= pd.DataFrame(ts_seq).T
    ts_seq_l=(ts_seq_l-ts_seq_l.min())/(ts_seq_l.max()-ts_seq_l.min())
    ts_seq_l=np.array(ts_seq_l.T)
    ts_seq_l=ts_seq_l.reshape(len(ts_seq_l),number_s,1)
    km_dba = TimeSeriesKMeans(n_clusters=n_clu, metric="dtw",max_iter_barycenter=100,verbose=0,random_state=0).fit(ts_seq_l)
    cl= km_dba.labels_
    cl=pd.Series(cl)
    cl=pd.get_dummies(cl)
    while len(cl.columns)<n_clu:
        cl[len(cl.columns)]=[0]*len(cl)
    ts_seq=[]
    for i in range(len(ex),len(y)):
        ts_seq.append(y.iloc[i-number_s:i])
    ts_seq=np.array(ts_seq)
    ts_seq_l= pd.DataFrame(ts_seq).T
    ts_seq_l=(ts_seq_l-ts_seq_l.min())/(ts_seq_l.max()-ts_seq_l.min())
    ts_seq_l=np.array(ts_seq_l.T)
    ts_seq_l=ts_seq_l.reshape(len(ts_seq_l),number_s,1)
    y_test = km_dba.predict(ts_seq_l)
    y_test=pd.Series(y_test)
    y_test=pd.get_dummies(y_test)
    while len(y_test.columns)<n_clu:
        y_test[len(y_test.columns)]=[0]*len(y_test)    
    
    if X is None:
        arima = auto_arima(ex_train)
        order = arima.order
        seas = arima.seasonal_order 
        pred_ar=[]    
        for i in range(len(ex_test)):
            model_test=ARIMA(order,seas).fit(y.iloc[number_s:int(train_test_split*len(y))+i])
            pred_ar.append(model_test.predict(n_periods=1)[0])
            
        model_f = auto_arima(ex_train,np.array(cl))
        order = model_f.order
        seas = model_f.seasonal_order   
        pred=[]    
        for i in range(len(ex_test)):
            model_f_t=ARIMA(order,seas).fit(y.iloc[number_s:int(train_test_split*len(y))+i],np.array(cl))
            exog = y_test.iloc[i:i+1,:]
            pred.append(model_f_t.predict(n_periods=1,X=np.array(exog).reshape(1,n_clu))[0])
            cl=np.concatenate([cl,exog])       
    else:
        x_train=X[number_s:int(train_test_split*len(y)),:]
        x_test=X[int(train_test_split*len(y)):,:]
        arima = auto_arima(ex_train,x_train)
        pred_ar=[]    
        for i in range(len(ex_test)):
            model_test=ARIMA(order,seas).fit(y.iloc[number_s:int(train_test_split*len(y))+i],X[number_s:int(train_test_split*len(y))+i,:])
            pred_ar.append(model_test.predict(n_periods=1,X=x_test[i:i+1,:])[0])
        
        model_f = auto_arima(ex_train,np.concatenate([x_train,np.array(cl)],axis=1))
        order = model_f.order
        seas = model_f.seasonal_order   
        exo_in=np.concatenate([x_train,np.array(cl)],axis=1)
        pred=[]    
        for i in range(len(ex_test)):
            model_f_t=ARIMA(order,seas).fit(y.iloc[number_s:int(train_test_split*len(y))+i],exo_in)
            exog = np.concatenate([x_test[i:i+1,:],y_test.iloc[i:i+1,:]],axis=1)
            pred.append(model_f_t.predict(n_periods=1,X=exog)[0])
            exo_in=np.concatenate([exo_in,exog])   

    mse_ar = mean_squared_error(ex_test,pred_ar)
    mse_arx = mean_squared_error(ex_test,pred)    
    #print('MSE of the ARIMA = '+str(round(mse_ar,3)))
    #print('MSE of the DARIMA = '+str(round(mse_arx,3)))
    #print('Percentage of improvement = '+str(round(((mse_ar-mse_arx)/mse_ar)*100))+'%')
    if plot_res==True:
        plt.plot(ex_test.reset_index(drop=True),label='Observed')
        plt.plot(pred_ar,label='ARIMA prediction')
        plt.plot(pred,label='DARIMA prediction')
        plt.legend()
        plt.show()
    return({'results_table':pd.DataFrame([mse_ar,mse_arx,((mse_ar-mse_arx)/mse_ar)*100],index=['ARIMA_MSE','DARIMA_MSE','%_Improv']),'arima_pred':pred_ar,'Darima_pred':pred})
   
def Compare_pred_exo(y,X,X1,plot_res=False,train_test_split=0.7):
    ex_train=y.iloc[:int(train_test_split*len(y))]
    ex_test=y.iloc[int(train_test_split*len(y)):]
    
    x_train=X[:int(train_test_split*len(y)),:]
    x_test=X[int(train_test_split*len(y)):,:]
    
    arima = auto_arima(ex_train,x_train)
    order=arima.order
    seas = arima.seasonal_order
    exo_in=x_train
    pred_ar=[]    
    for i in range(len(ex_test)):
        model_test=ARIMA(order,seas).fit(y.iloc[:int(train_test_split*len(y))+i],exo_in)
        exog = x_test[i:i+1,:]
        pred_ar.append(model_test.predict(n_periods=1,X=exog).iloc[0])
        exo_in=np.concatenate([exo_in,exog])   
    
    X = np.concatenate([X1,X],axis=1)
    x_train=X[:int(train_test_split*len(y)),:]
    x_test=X[int(train_test_split*len(y)):,:]
    
    model_f = auto_arima(ex_train,x_train)
    order = model_f.order
    seas = model_f.seasonal_order   
    exo_in=x_train
    pred=[]    
    for i in range(len(ex_test)):
        model_f_t=ARIMA(order,seas).fit(y.iloc[:int(train_test_split*len(y))+i],exo_in)
        exog = x_test[i:i+1,:]
        pred.append(model_f_t.predict(n_periods=1,X=exog).iloc[0])
        exo_in=np.concatenate([exo_in,exog])   

    mse_ar = mean_squared_error(ex_test,pred_ar)
    mse_arx = mean_squared_error(ex_test,pred)    
    if plot_res==True:
        plt.plot(ex_test.reset_index(drop=True),label='Observed')
        plt.plot(pred_ar,label='ARIMA prediction')
        plt.plot(pred,label='DARIMA prediction')
        plt.legend()
        plt.show()
    return({'results_table':pd.DataFrame([mse_ar,mse_arx,((mse_ar-mse_arx)/mse_ar)*100],index=['ARIMA_MSE','DARIMA_MSE','%_Improv']),'arima_pred':pred_ar,'Darima_pred':pred,'Obs':ex_test})
       
def Compare_pred_exo_h(y,X,X1,h,plot_res=False,train_test_split=0.7):
    ex_train=y.iloc[:int(train_test_split*len(y))]
    ex_test=y.iloc[int(train_test_split*len(y))+h:]
    
    x_train=X[:int(train_test_split*len(y)),:]
    x_test=X[int(train_test_split*len(y)):-h,:]
    
    arima = auto_arima(ex_train,x_train)
    order=arima.order
    seas = arima.seasonal_order
    exo_in=x_train
    pred_ar=[]    
    for i in range(len(ex_test)):
        model_test=ARIMA(order,seas).fit(y.iloc[:int(train_test_split*len(y))+i],exo_in)
        exog=x_test[i:i+1,:]
        for rep in range(h):
            exog =  np.concatenate([exog,x_test[i:i+1,:]])
        pred_ar.append(model_test.predict(n_periods=h+1,X=exog).iloc[h])
        exo_in=np.concatenate([exo_in,x_test[i:i+1,:]]) 
    
    X = np.concatenate([X1,X],axis=1)
    x_train=X[:int(train_test_split*len(y)),:]
    x_test=X[int(train_test_split*len(y)):-h,:]
    
    model_f = auto_arima(ex_train,x_train)
    order = model_f.order
    seas = model_f.seasonal_order   
    exo_in=x_train
    pred=[]    
    for i in range(len(ex_test)):
        model_f_t=ARIMA(order,seas).fit(y.iloc[:int(train_test_split*len(y))+i],exo_in)
        exog=x_test[i:i+1,:]
        for rep in range(h):
            exog =  np.concatenate([exog,x_test[i:i+1,:]])
        pred.append(model_f_t.predict(n_periods=h+1,X=exog).iloc[h])
        exo_in=np.concatenate([exo_in,x_test[i:i+1,:]])   

    mse_ar = mean_squared_error(ex_test,pred_ar)
    mse_arx = mean_squared_error(ex_test,pred)    
    if plot_res==True:
        plt.plot(ex_test.reset_index(drop=True),label='Observed')
        plt.plot(pred_ar,label='ARIMA prediction')
        plt.plot(pred,label='DARIMA prediction')
        plt.legend()
        plt.show()
    return({'results_table':pd.DataFrame([mse_ar,mse_arx,((mse_ar-mse_arx)/mse_ar)*100],index=['ARIMA_MSE','DARIMA_MSE','%_Improv']),'arima_pred':pred_ar,'Darima_pred':pred,'Obs':ex_test})
       
def Compare_pred_exo_only_h(y,X,X1,h,plot_res=False,train_test_split=0.7):
    ex_train=y.iloc[:int(train_test_split*len(y))]
    ex_test=y.iloc[int(train_test_split*len(y))+h:]
    
    X = np.concatenate([X1,X],axis=1)
    x_train=X[:int(train_test_split*len(y)),:]
    x_test=X[int(train_test_split*len(y)):-h,:]
    model_f = auto_arima(ex_train,x_train)
    order = model_f.order
    seas = model_f.seasonal_order   
    exo_in=x_train
    pred=[]    
    for i in range(len(ex_test)):
        model_f_t=ARIMA(order,seas).fit(y.iloc[:int(train_test_split*len(y))+i],exo_in)
        exog=x_test[i:i+1,:]
        for rep in range(h):
            exog =  np.concatenate([exog,x_test[i:i+1,:]])
        pred.append(model_f_t.predict(n_periods=h+1,X=exog).iloc[h])
        exo_in=np.concatenate([exo_in,x_test[i:i+1,:]])    
    return({'Darima_pred':pred})
       

def Compare_pred_exo_only(y,X,X1,plot_res=False,train_test_split=0.7):
    ex_train=y.iloc[:int(train_test_split*len(y))]
    ex_test=y.iloc[int(train_test_split*len(y)):]
    
    X = np.concatenate([X1,X],axis=1)
    x_train=X[:int(train_test_split*len(y)),:]
    x_test=X[int(train_test_split*len(y)):,:]
    model_f = auto_arima(ex_train,x_train)
    order = model_f.order
    seas = model_f.seasonal_order   
    exo_in=x_train
    pred=[]    
    for i in range(len(ex_test)):
        model_f_t=ARIMA(order,seas).fit(y.iloc[:int(train_test_split*len(y))+i],exo_in)
        exog = x_test[i:i+1,:]
        pred.append(model_f_t.predict(n_periods=1,X=exog).iloc[0])
        exo_in=np.concatenate([exo_in,exog])   
    return({'Darima_pred':pred})
       




def opti_clu_fit(y,test_clu,test_win,opti='brut',metric='aic',iterac=None):
    if opti=='brut':
        if metric=='aic':
            min_aic=np.inf
            for n_clu in test_clu:
                for number_s in test_win:
                    darima = DARIMA(y,number_s=number_s,n_clu=n_clu)
                    score=darima.aic()
                    if score<min_aic:
                        min_aic=score
                        para=[n_clu,number_s]
        elif metric=='resid':
            min_resid=np.inf
            for n_clu in test_clu:
                for number_s in test_win:
                    darima = DARIMA(y,number_s=number_s,n_clu=n_clu)
                    score=abs(darima.resid()).mean()
                    if score<min_resid:
                        min_resid=score
                        para=[n_clu,number_s]
    elif opti=='random':
        if iterac is None:
            iterac=10    
        p_param=[]    
        for n_clu in test_clu:
            for number_s in test_win:
                p_param.append([n_clu,number_s])
        if iterac>len(p_param):
            iterac=len(p_param)
        p_param=sample(p_param,k=iterac)    
        if metric=='aic':
            min_aic=np.inf
            for comb in p_param:
                darima = DARIMA(y,number_s=number_s,n_clu=n_clu)
                score=darima.aic()
                if score<min_aic:
                    min_aic=score
                    para=[n_clu,number_s]
        elif metric=='resid':
            min_resid=np.inf
            for comb in p_param:
                darima = DARIMA(y,number_s=number_s,n_clu=n_clu)
                score=abs(darima.resid()).mean()
                if score<min_resid:
                    min_resid=score
                    para=[n_clu,number_s]
    return(para)        




    

def clu_select(y,
               X=None,
               n_clu=5,
               number_s=5,
               model=TimeSeriesKMeans(n_clusters=5, 
                                                metric="dtw",
                                                max_iter_barycenter=100,
                                                verbose=0,
                                                random_state=0),
               select='p_val',
               metric='aic',
               val_set=0.2,
               test_set=0.3):
    y_eff = y[:int(len(y)*(1-test_set))]

    if metric=='aic':   
        if X is None:
            arima = auto_arima(y_eff[number_s:])
        else:
            arima = auto_arima(y_eff[number_s:],X[:int(len(y)*(1-test_set))])
        met_ar=arima.aic()
        if X is None:
            darima = DARIMA(y_eff,number_s=number_s,n_clu=n_clu)
        else:
            darima = DARIMA(y_eff,X[:int(len(y)*(1-test_set))],number_s=number_s,n_clu=n_clu)
        met_arx=darima.aic()    
    elif metric=='resid':   
        if X is None:
            arima = auto_arima(y_eff[number_s:])
        else:
            arima = auto_arima(y_eff[number_s:],X[:int(len(y)*(1-test_set))])
        met_ar=abs(arima.resid()).mean() 
        if X is None:
            darima = DARIMA(y_eff,number_s=number_s,n_clu=n_clu)
        else:
            darima = DARIMA(y_eff,X[:int(len(y)*(1-test_set))],number_s=number_s,n_clu=n_clu)
        met_arx=abs(darima.resid()).mean()  
       
    if met_arx<met_ar:
        out = {'pred':Compare_pred(y,X=X,number_s=number_s,n_clu=n_clu,plot_res=False,train_test_split=1-test_set),'fit':Compare_fit(y,X=X,number_s=number_s,n_clu=n_clu,metric=metric)}
    
    else:    
        p_v = pd.DataFrame([darima.pvalues()[:n_clu].tolist(),range(n_clu)]).T
        p_v=p_v.sort_values(0)
        cl=pd.DataFrame(get_dynamic_input_output(y_eff,
                                                 n_clu=n_clu,
                                                 number_s=number_s,
                                                 model=model,
                                                 output=False)['input'][:,1:])
        cl_test=pd.DataFrame(get_dynamic_input_output(y,
                                                      n_clu=n_clu,
                                                      number_s=number_s,
                                                      model=model,
                                                      output=False)['input'][:,1:])
        flag_end=False
        clu_f=1
        while ((flag_end !=True)&(clu_f < n_clu)):
            cl = pd.DataFrame(cl).drop(int(p_v.iloc[-clu_f,1]),axis=1)
            cl_test=pd.DataFrame(cl_test).drop(int(p_v.iloc[-clu_f,1]),axis=1)
            clu_f=clu_f+1  
            darima=auto_arima(y_eff[number_s:],X=np.array(cl))
            if metric=='aic':
                met_arx=darima.aic()
            elif metric=='resid':     
                met_arx=abs(darima.resid()).mean()  
            if met_arx<met_ar:
                arima = auto_arima(y[number_s:])
                darima= auto_arima(y[number_s:],X=np.array(cl_test))
                if metric=='aic':
                    l_pred = Compare_pred_exo(y[number_s:],X=np.array(cl_test),plot_res=False,train_test_split=1-test_set)
                    l_fit = pd.DataFrame([arima.aic(),darima.aic(),((arima.aic()-darima.aic())/abs(arima.aic()))*100],index=['ARIMA_AIC','DARIMA_AIC','%_Improv'])
                elif metric=='resid':     
                    l_pred = Compare_pred_exo(y[number_s:],X=np.array(cl_test),plot_res=False,train_test_split=1-test_set)
                    l_fit = pd.DataFrame([abs(arima.resid()).mean(),abs(darima.resid()).mean(),((abs(arima.resid()).mean()-abs(darima.resid()).mean())/abs(arima.resid()).mean())*100],index=['ARIMA_AIC','DARIMA_AIC','%_Improv']) 
                out = {'pred':l_pred,'fit':l_fit}
                flag_end=True
        if flag_end==False:
            arima = auto_arima(y[number_s:])
            predic = Compare_pred_exo(y[number_s:],X=np.array(cl_test),plot_res=False,train_test_split=1-test_set)
            l_pred = {'results_table':pd.DataFrame([predic['results_table'].iloc[0][0],predic['results_table'].iloc[0][0],0],index=['ARIMA_MSE','DARIMA_MSE','%_Improv']),'arima_pred':predic['arima_pred'],'Darima_pred':predic['arima_pred']}
            if metric=='aic':
                l_fit = pd.DataFrame([arima.aic(),arima.aic(),0],index=['ARIMA_AIC','DARIMA_AIC','%_Improv'])
            elif metric=='resid':  
                l_fit = pd.DataFrame([abs(arima.resid()).mean(),abs(arima.resid()).mean(),0],index=['ARIMA_AIC','DARIMA_AIC','%_Improv']) 
            out = {'pred' : l_pred, 'fit' : l_fit}
    return(out)


# =============================================================================
# General model application
# =============================================================================

def model_pred(model,
               y,
               X=None,
               ar=1,
               n_clu=5,
               number_s=5,
               model_cl=TimeSeriesKMeans(n_clusters=5, 
                                                metric="dtw",
                                                max_iter_barycenter=100,
                                                verbose=0,
                                                random_state=0),
               metric='mse',
               train_test_split=0.7,
               plot=False,
               opti_grid=None,
               compare=False):
    in_out=get_dynamic_input_output(y,ar,n_clu,number_s,model=model_cl)
    if X is not None:
        x = np.concatenate([in_out['input'],X],axis=1)
    else:
        x = in_out['input']
    x_train = x[:int(train_test_split*len(x)),:]
    x_test = x[int(train_test_split*len(x)):,:]
    y_train = in_out['output'][:int(train_test_split*len(x))]
    y_test = in_out['output'][int(train_test_split*len(x)):]
    
    if opti_grid is not None:
        train_indices=range(int(len(x_train)*0.8))
        test_indices=range(int(len(x_train)*0.8),len(x_train))
        custom_cv = [(list(train_indices), list(test_indices))]
        opti_model=RandomizedSearchCV(estimator=model,param_distributions=opti_grid,n_iter=50,cv =custom_cv, verbose=0,random_state=0,n_jobs=-1)
        opti_model.fit(x_train,y_train)
        pred=opti_model.predict(x_test)
    else:    
        model.fit(x_train,y_train)
        pred=model.predict(x_test)
    
    if metric=='mse':
        eva=mean_squared_error(y_test, pred)
    if compare==True:
        x_train = x[:int(train_test_split*len(x)),:ar]
        x_test = x[int(train_test_split*len(x)):,:ar]
        if opti_grid is not None:
            train_indices=range(int(len(x_train)*0.8))
            test_indices=range(int(len(x_train)*0.8),len(x_train))
            custom_cv = [(list(train_indices), list(test_indices))]
            opti_model=RandomizedSearchCV(estimator=model,param_distributions=opti_grid,n_iter=50,cv =custom_cv, verbose=0,random_state=0,n_jobs=-1)
            opti_model.fit(x_train,y_train)
            pred_rf=opti_model.predict(x_test)
        else:    
            model.fit(x_train,y_train)
            pred_rf=model.predict(x_test)
        eva_rf=mean_squared_error(y_test, pred_rf)
        out= {metric:[eva,eva_rf],'Prediction':[pred,pred_rf],'Observed value':y_test}
    else:
        out= {metric:eva,'Prediction':pred,'Observed value':y_test}
    if plot==True:
        if compare ==False:
            plt.plot(y_test,label='Observed values')
            plt.plot(pred,label='Predicted values')
            plt.legend()
            plt.show()        
        elif compare ==True:
            plt.plot(y_test,label='Observed values')
            plt.plot(pred,label='Predicted values - Dyn')
            plt.plot(pred_rf,label='Predicted values')
            plt.legend()
            plt.show()   
    return(out)
        
        
def Compare_RF_exo(y,
                   X,X1,
                   ar=1,
                   n_clu=1,
                   number_s=5,
                   model=TimeSeriesKMeans(n_clusters=5, 
                                                metric="dtw",
                                                max_iter_barycenter=100,
                                                verbose=0,
                                                random_state=0),
                   plot_res=False,
                   train_test_split=0.7,
                   opti_grid=None):
    
    in_out=get_dynamic_input_output(y,ar,n_clu,number_s,model=model)
    x = np.concatenate([in_out['input'][:,:ar],X,X1],axis=1)
    x_train = x[:int(train_test_split*len(x)),:]
    x_test = x[int(train_test_split*len(x)):,:]
    y_train = in_out['output'][:int(train_test_split*len(x))]
    y_test = in_out['output'][int(train_test_split*len(x)):]
    
    model = RandomForestRegressor(random_state=0)
    if opti_grid is not None:
        train_indices=range(int(len(x_train)*0.8))
        test_indices=range(int(len(x_train)*0.8),len(x_train))
        custom_cv = [(list(train_indices), list(test_indices))]
        opti_model=RandomizedSearchCV(estimator=model,param_distributions=opti_grid,n_iter=50,cv =custom_cv, verbose=0,random_state=0,n_jobs=-1)
        opti_model.fit(x_train,y_train)
        pred=opti_model.predict(x_test)
    else:    
        model.fit(x_train,y_train)
        pred=model.predict(x_test)
    
    x = np.concatenate([in_out['input'][:,:ar],X],axis=1)
    x_train = x[:int(train_test_split*len(x)),:]
    x_test = x[int(train_test_split*len(x)):,:]
    model = RandomForestRegressor(random_state=0)
    if opti_grid is not None:
        train_indices=range(int(len(x_train)*0.8))
        test_indices=range(int(len(x_train)*0.8),len(x_train))
        custom_cv = [(list(train_indices), list(test_indices))]
        opti_model=RandomizedSearchCV(estimator=model,param_distributions=opti_grid,n_iter=50,cv =custom_cv, verbose=0,random_state=0,n_jobs=-1)
        opti_model.fit(x_train,y_train)
        pred_ar=opti_model.predict(x_test)
    else:    
        model.fit(x_train,y_train)
        pred_ar=model.predict(x_test)
    
    mse_rf = mean_squared_error(y_test,pred_ar)
    mse_rfx = mean_squared_error(y_test,pred)    
    if plot_res==True:
        plt.figure(figsize=(20,5))
        plt.plot(y_test,label='Observed')
        plt.plot(pred_ar,label='RF prediction')
        plt.plot(pred,label='RFX prediction')
        plt.legend()
        plt.show()
    return({'results_table':pd.DataFrame([mse_rf,mse_rfx,((mse_rf-mse_rfx)/mse_rf)*100],index=['RF_MSE','RFX_MSE','%_Improv']),'rf_pred':pred_ar,'rfx_pred':pred,'obs':y_test})
  
def Compare_RF_exo_h(y,X,X1,h,ar=1,n_clu=1,number_s=5,
                   model=TimeSeriesKMeans(n_clusters=5, 
                                                metric="dtw",
                                                max_iter_barycenter=100,
                                                verbose=0,
                                                random_state=0),
                   plot_res=False,
                   train_test_split=0.7,
                   opti_grid=None):
    
    in_out=get_dynamic_input_output(y,ar,n_clu,number_s,model=model)
    x = np.concatenate([in_out['input'][:,:ar],X,X1],axis=1)
    x_train = x[:int(train_test_split*len(x))-h,:]
    x_test = x[int(train_test_split*len(x)):-h,:]
    y_train = in_out['output'][h:int(train_test_split*len(x))]
    y_test = in_out['output'][h+int(train_test_split*len(x)):]
    
    model = RandomForestRegressor(random_state=0)
    if opti_grid is not None:
        train_indices=range(int(len(x_train)*0.8))
        test_indices=range(int(len(x_train)*0.8),len(x_train))
        custom_cv = [(list(train_indices), list(test_indices))]
        opti_model=RandomizedSearchCV(estimator=model,param_distributions=opti_grid,n_iter=50,cv =custom_cv, verbose=0,random_state=0,n_jobs=-1)
        opti_model.fit(x_train,y_train)
        pred=opti_model.predict(x_test)
    else:    
        model.fit(x_train,y_train)
        pred=model.predict(x_test)
    
    x = np.concatenate([in_out['input'][:,:ar],X],axis=1)
    x_train = x[:int(train_test_split*len(x))-h,:]
    x_test = x[int(train_test_split*len(x)):-h,:]
    model = RandomForestRegressor(random_state=0)
    if opti_grid is not None:
        train_indices=range(int(len(x_train)*0.8))
        test_indices=range(int(len(x_train)*0.8),len(x_train))
        custom_cv = [(list(train_indices), list(test_indices))]
        opti_model=RandomizedSearchCV(estimator=model,param_distributions=opti_grid,n_iter=50,cv =custom_cv, verbose=0,random_state=0,n_jobs=-1)
        opti_model.fit(x_train,y_train)
        pred_ar=opti_model.predict(x_test)
    else:    
        model.fit(x_train,y_train)
        pred_ar=model.predict(x_test)
    
    mse_rf = mean_squared_error(y_test,pred_ar)
    mse_rfx = mean_squared_error(y_test,pred)    
    if plot_res==True:
        plt.figure(figsize=(20,5))
        plt.plot(y_test,label='Observed')
        plt.plot(pred_ar,label='RF prediction')
        plt.plot(pred,label='RFX prediction')
        plt.legend()
        plt.show()
    return({'results_table':pd.DataFrame([mse_rf,mse_rfx,((mse_rf-mse_rfx)/mse_rf)*100],index=['RF_MSE','RFX_MSE','%_Improv']),'rf_pred':pred_ar,'rfx_pred':pred,'obs':y_test})

def Compare_RF_exo_only_h(y,
                   X,X1,h,
                   ar=1,
                   n_clu=1,
                   number_s=5,
                   model=TimeSeriesKMeans(n_clusters=5, 
                                                metric="dtw",
                                                max_iter_barycenter=100,
                                                verbose=0,
                                                random_state=0),
                   plot_res=False,
                   train_test_split=0.7,
                   opti_grid=None):
    in_out=get_dynamic_input_output(y,ar,n_clu,number_s,model=model)
    x = np.concatenate([in_out['input'][:,:ar],X,X1],axis=1)
    x_train = x[:int(train_test_split*len(x))-h,:]
    x_test = x[int(train_test_split*len(x)):-h,:]
    y_train = in_out['output'][h:int(train_test_split*len(x))]
    
    model = RandomForestRegressor(random_state=0)
    if opti_grid is not None:
        train_indices=range(int(len(x_train)*0.8))
        test_indices=range(int(len(x_train)*0.8),len(x_train))
        custom_cv = [(list(train_indices), list(test_indices))]
        opti_model=RandomizedSearchCV(estimator=model,param_distributions=opti_grid,n_iter=50,cv =custom_cv, verbose=0,random_state=0,n_jobs=-1)
        opti_model.fit(x_train,y_train)
        pred=opti_model.predict(x_test)
    else:    
        model.fit(x_train,y_train)
        pred=model.predict(x_test)
        
    return({'rfx_pred':pred})
            
def Compare_RF_exo_only(y,
                   X,X1,
                   ar=1,
                   n_clu=1,
                   number_s=5,
                   model=TimeSeriesKMeans(n_clusters=5, 
                                                metric="dtw",
                                                max_iter_barycenter=100,
                                                verbose=0,
                                                random_state=0),
                   plot_res=False,
                   train_test_split=0.7,
                   opti_grid=None):
    in_out=get_dynamic_input_output(y,ar,n_clu,number_s,model=model)
    
    x = np.concatenate([in_out['input'][:,:ar],X,X1],axis=1)
    x_train = x[:int(train_test_split*len(x)),:]
    x_test = x[int(train_test_split*len(x)):,:]
    y_train = in_out['output'][:int(train_test_split*len(x))]
    model = RandomForestRegressor(random_state=0)
    if opti_grid is not None:
        train_indices=range(int(len(x_train)*0.8))
        test_indices=range(int(len(x_train)*0.8),len(x_train))
        custom_cv = [(list(train_indices), list(test_indices))]
        opti_model=RandomizedSearchCV(estimator=model,param_distributions=opti_grid,n_iter=50,cv =custom_cv, verbose=0,random_state=0,n_jobs=-1)
        opti_model.fit(x_train,y_train)
        pred=opti_model.predict(x_test)
    else:    
        model.fit(x_train,y_train)
        pred=model.predict(x_test)
        
    return({'rfx_pred':pred})
              
        


# =============================================================================
# METRICS
# =============================================================================

def chicken_mse(y_obs,y_pred):
    """
    The name was found by Hannah Frank. 
    """
    weight=abs(np.array(y_pred)-np.array(y_pred).mean())+10**(-22)
    return(mean_squared_error(y_obs, y_pred,sample_weight=weight))

def diff_mse(y_obs,y_pred):
    """
    The name was NOT found by Hannah Frank beacause it's NOT funny. 
    """
    y_obs=pd.Series(y_obs).diff()
    y_pred=pd.Series(y_pred).diff()
    return(mean_squared_error(y_obs[1:], y_pred[1:]))
