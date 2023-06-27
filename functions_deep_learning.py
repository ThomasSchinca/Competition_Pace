#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 12:37:18 2023

@author: hannahfrank
"""

import pandas as pd
import numpy as np
from functions import Compare_pred,Compare_fit,DARIMA,get_dynamic_clusters,get_dynamic_input_output,model_pred
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,LSTM,Bidirectional,Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from numpy.random import seed
seed(1)
tf.random.set_seed(1)

# =============================================================================
# General model application
# =============================================================================


def nn_model_pred(y,X=None,mod='lstm',ar=1,n_clu=5,number_s=5,train_test_split=0.7,opti_grid=None,metric='mse',plot=False,compare=False):
    seed(1)
    tf.random.set_seed(1)
    tf.random.set_seed(0)
    in_out=get_dynamic_input_output(y,ar,n_clu,number_s)
    if X is not None:
        x = np.concatenate([in_out['input'],X],axis=1)
    else:
        x = in_out['input']
    x_train = x[:int(train_test_split*len(x)),:]
    x_test = x[int(train_test_split*len(x)):,:]
    y_train = in_out['output'][:int(train_test_split*len(x))]
    y_test = in_out['output'][int(train_test_split*len(x)):]
    x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1]))
    x_test= x_test.reshape((x_test.shape[0],1,x_test.shape[1]))
    
    min_eva=np.inf
    if mod=='lstm':
        if opti_grid is not None:
            for acti in opti_grid['activation']:
                for n_layer in opti_grid['n_layer']:
                    for drop in opti_grid['drop']:
                        for l_r in opti_grid['l_r']:
                            nb_hidden = int(len(x[:,1])/(10*len(x[1,:])))
                            if nb_hidden<32:
                                nb_hidden=32
                            model = keras.Sequential()
                            if n_layer==1:
                                model.add(LSTM(nb_hidden,activation=acti,input_shape=(x_train.shape[1],x_train.shape[2])))
                                model.add(Dropout(drop))
                            if n_layer==2:
                                model.add(LSTM(nb_hidden,activation=acti,return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
                                model.add(Dropout(drop))
                                model.add(LSTM(nb_hidden,activation=acti))
                                model.add(Dropout(drop))
                            model.add(Dense(1))
                            model.compile(loss='mse',optimizer= Adam(learning_rate=l_r))
                            n_epochs=5000
                            es=EarlyStopping(monitor='loss', mode ='min',verbose=0,patience=50)
                            model.fit(x_train,y_train,epochs=n_epochs,batch_size=16,verbose=0,shuffle=True,callbacks=[es])
                            pred = model.predict(x_test,verbose=0)
                            if metric=='mse':
                                eva=mean_squared_error(y_test, pred)
                                if eva < min_eva :
                                    m_pred = pred
                                    min_eva = eva
            if plot==True:
                plt.plot(y_test,label='Observed values')
                plt.plot(m_pred,label='Predicted values')
                plt.legend()
                plt.show()                        
                                    
        else:
            model = keras.Sequential()
            nb_hidden = int(len(x[:,1])/(10*len(x[1,:])))
            if nb_hidden<32:
                nb_hidden=32
            model.add(LSTM(nb_hidden,activation='relu',input_shape=(x_train.shape[1],x_train.shape[2])))
            model.add(Dropout(0.1))
            model.add(Dense(1))
            model.compile(loss='mse',optimizer= Adam(learning_rate=0.001))
            n_epochs=5000
            es=EarlyStopping(monitor='loss', mode ='min',verbose=0,patience=50)
            model.fit(x_train,y_train,epochs=n_epochs,batch_size=16,verbose=0,shuffle=True,callbacks=[es])
            m_pred = model.predict(x_test,verbose=0)
            if plot==True:
                plt.plot(y_test[:100],label='Observed values')
                plt.plot(pred[:100],label='Predicted values')
                plt.legend()
                plt.show()
            eva=mean_squared_error(y_test, m_pred)   
        if compare==True:    
            min_eva_ar=np.inf
            x_train = x[:int(train_test_split*len(x)),:ar]
            x_test = x[int(train_test_split*len(x)):,:ar]
            y_train = in_out['output'][:int(train_test_split*len(x))]
            y_test = in_out['output'][int(train_test_split*len(x)):]
            x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1]))
            x_test= x_test.reshape((x_test.shape[0],1,x_test.shape[1]))
            if opti_grid is not None:
                for acti in opti_grid['activation']:
                    for n_layer in opti_grid['n_layer']:
                        for drop in opti_grid['drop']:
                            for l_r in opti_grid['l_r']:
                                nb_hidden = int(len(x[:,1])/(10*len(x[1,:])))
                                if nb_hidden<32:
                                    nb_hidden=32
                                model = keras.Sequential()
                                if n_layer==1:
                                    model.add(LSTM(nb_hidden,activation=acti,input_shape=(x_train.shape[1],x_train.shape[2])))
                                    model.add(Dropout(drop))
                                if n_layer==2:
                                    model.add(LSTM(nb_hidden,activation=acti,return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
                                    model.add(Dropout(drop))
                                    model.add(LSTM(nb_hidden,activation=acti))
                                    model.add(Dropout(drop))
                                model.add(Dense(1))
                                model.compile(loss='mse',optimizer= Adam(learning_rate=l_r))
                                n_epochs=5000
                                es=EarlyStopping(monitor='loss', mode ='min',verbose=0,patience=50)
                                model.fit(x_train,y_train,epochs=n_epochs,batch_size=16,verbose=0,shuffle=True,callbacks=[es])
                                pred_ar = model.predict(x_test,verbose=0)
                                if metric=='mse':
                                    eva_ar=mean_squared_error(y_test, pred_ar)
                                    if eva_ar < min_eva_ar :
                                        m_pred_ar = pred_ar
                                        min_eva_ar = eva_ar                     
                                       
            else:
                model = keras.Sequential()
                nb_hidden = int(len(x[:,1])/(10*len(x[1,:])))
                if nb_hidden<32:
                    nb_hidden=32
                model.add(LSTM(nb_hidden,activation='relu',input_shape=(x_train.shape[1],x_train.shape[2])))
                model.add(Dropout(0.1))
                model.add(Dense(1))
                model.compile(loss='mse',optimizer= Adam(learning_rate=0.001))
                n_epochs=5000
                es=EarlyStopping(monitor='loss', mode ='min',verbose=0,patience=50)
                model.fit(x_train,y_train,epochs=n_epochs,batch_size=16,verbose=0,shuffle=True,callbacks=[es])
                m_pred_ar = model.predict(x_test,verbose=0)
                eva_ar=mean_squared_error(y_test, m_pred_ar)   
            out={metric:[eva,eva_ar],'Prediction':[m_pred,m_pred_ar],'Observed value':y_test}
        else:
            out={metric:eva,'Prediction':m_pred,'Observed value':y_test}
    return(out)
        
        
def Compare_nn_exo(y,X,X1,ar=1,n_clu=1,number_s=5,plot_res=False,train_test_split=0.7,opti_grid=None):
    seed(1)
    tf.random.set_seed(1)
    in_out=get_dynamic_input_output(y,ar,n_clu,number_s)
    x = np.concatenate([in_out['input'][:,:ar],X,X1],axis=1)
    x_train = x[:int(train_test_split*len(x)),:]
    x_test = x[int(train_test_split*len(x)):,:]
    y_train = in_out['output'][:int(train_test_split*len(x))]
    y_test = in_out['output'][int(train_test_split*len(x)):]
    
    x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1]))
    x_test= x_test.reshape((x_test.shape[0],1,x_test.shape[1]))
    
    min_eva=np.inf
    if opti_grid is not None:
        for acti in opti_grid['activation']:
            for n_layer in opti_grid['n_layer']:
                for drop in opti_grid['drop']:
                    for l_r in opti_grid['l_r']:
                        nb_hidden = int(len(x[:,1])/(10*len(x[1,:])))
                        if nb_hidden<32:
                            nb_hidden=32
                        model = keras.Sequential()
                        if n_layer==1:
                            model.add(LSTM(nb_hidden,activation=acti,input_shape=(x_train.shape[1],x_train.shape[2])))
                            model.add(Dropout(drop))
                        if n_layer==2:
                            model.add(LSTM(nb_hidden,activation=acti,return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
                            model.add(Dropout(drop))
                            model.add(LSTM(nb_hidden,activation=acti))
                            model.add(Dropout(drop))
                        model.add(Dense(1))
                        model.compile(loss='mse',optimizer= Adam(learning_rate=l_r))
                        n_epochs=5000
                        es=EarlyStopping(monitor='loss', mode ='min',verbose=0,patience=50)
                        model.fit(x_train,y_train,epochs=n_epochs,batch_size=16,verbose=0,shuffle=True,callbacks=[es])
                        pred = model.predict(x_test,verbose=0)

                        eva=mean_squared_error(y_test, pred)
                        if eva < min_eva :
                            m_pred = pred
                            min_eva = eva
            if plot_res==True:
                plt.plot(y_test,label='Observed values')
                plt.plot(m_pred,label='Predicted values')
                plt.legend()
                plt.show()                        
    
    else:
        model = keras.Sequential()
        nb_hidden = int(len(x[:,1])/(10*len(x[1,:])))
        if nb_hidden<32:
            nb_hidden=32
        model.add(LSTM(nb_hidden,activation='relu',input_shape=(x_train.shape[1],x_train.shape[2])))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        model.compile(loss='mse',optimizer= Adam(learning_rate=0.001))
        n_epochs=5000
        es=EarlyStopping(monitor='loss', mode ='min',verbose=0,patience=50)
        model.fit(x_train,y_train,epochs=n_epochs,batch_size=16,verbose=0,shuffle=True,callbacks=[es])
        m_pred = model.predict(x_test,verbose=0)
        eva=mean_squared_error(y_test, m_pred)   
   
    min_eva_ar=np.inf
    x = np.concatenate([in_out['input'][:,:ar],X],axis=1)
    x_train = x[:int(train_test_split*len(x)),:]
    x_test = x[int(train_test_split*len(x)):,:]
    y_train = in_out['output'][:int(train_test_split*len(x))]
    y_test = in_out['output'][int(train_test_split*len(x)):]
    x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1]))
    x_test= x_test.reshape((x_test.shape[0],1,x_test.shape[1]))
    if opti_grid is not None:
        for acti in opti_grid['activation']:
            for n_layer in opti_grid['n_layer']:
                for drop in opti_grid['drop']:
                    for l_r in opti_grid['l_r']:
                        nb_hidden = int(len(x[:,1])/(10*len(x[1,:])))
                        if nb_hidden<32:
                            nb_hidden=32
                        model = keras.Sequential()
                        if n_layer==1:
                            model.add(LSTM(nb_hidden,activation=acti,input_shape=(x_train.shape[1],x_train.shape[2])))
                            model.add(Dropout(drop))
                        if n_layer==2:
                            model.add(LSTM(nb_hidden,activation=acti,return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
                            model.add(Dropout(drop))
                            model.add(LSTM(nb_hidden,activation=acti))
                            model.add(Dropout(drop))
                        model.add(Dense(1))
                        model.compile(loss='mse',optimizer= Adam(learning_rate=l_r))
                        n_epochs=5000
                        es=EarlyStopping(monitor='loss', mode ='min',verbose=0,patience=50)
                        model.fit(x_train,y_train,epochs=n_epochs,batch_size=16,verbose=0,shuffle=True,callbacks=[es])
                        pred_ar = model.predict(x_test,verbose=0)
                        eva_ar=mean_squared_error(y_test, pred_ar)
                        if eva_ar < min_eva_ar :
                            m_pred_ar = pred_ar
                            min_eva_ar = eva_ar                     
    else:
        model = keras.Sequential()
        nb_hidden = int(len(x[:,1])/(10*len(x[1,:])))
        if nb_hidden<32:
            nb_hidden=32
        model.add(LSTM(nb_hidden,activation='relu',input_shape=(x_train.shape[1],x_train.shape[2])))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        model.compile(loss='mse',optimizer= Adam(learning_rate=0.001))
        n_epochs=5000
        es=EarlyStopping(monitor='loss', mode ='min',verbose=0,patience=50)
        model.fit(x_train,y_train,epochs=n_epochs,batch_size=16,verbose=0,shuffle=True,callbacks=[es])
        m_pred_ar = model.predict(x_test,verbose=0)
        eva_ar=mean_squared_error(y_test, m_pred_ar)   
    if plot_res==True:
        plt.figure(figsize=(20,5))
        plt.plot(y_test,label='Observed')
        plt.plot(m_pred_ar,label='RF prediction')
        plt.plot(m_pred,label='RFX prediction')
        plt.legend()
        plt.show()            
    return({'results_table':pd.DataFrame([eva_ar,eva,((eva_ar-eva)/eva_ar)*100],index=['LSTM_MSE','LSTMX_MSE','%_Improv']),'rf_pred':m_pred_ar,'rfx_pred':m_pred})
          
def Compare_nn_exo_h(y,X,X1,h,ar=1,n_clu=1,number_s=5,plot_res=False,train_test_split=0.7,opti_grid=None):
    tf.random.set_seed(0)
    in_out=get_dynamic_input_output(y,ar,n_clu,number_s)
    x = np.concatenate([in_out['input'][:,:ar],X,X1],axis=1)
    x_train = x[:int(train_test_split*len(x))-h,:]
    x_test = x[int(train_test_split*len(x)):-h,:]
    y_train = in_out['output'][h:int(train_test_split*len(x))]
    y_test = in_out['output'][h+int(train_test_split*len(x)):]
    
    x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1]))
    x_test= x_test.reshape((x_test.shape[0],1,x_test.shape[1]))
    
    min_eva=np.inf
    if opti_grid is not None:
        for acti in opti_grid['activation']:
            for n_layer in opti_grid['n_layer']:
                for drop in opti_grid['drop']:
                    for l_r in opti_grid['l_r']:
                        nb_hidden = int(len(x[:,1])/(10*len(x[1,:])))
                        if nb_hidden<32:
                            nb_hidden=32
                        model = keras.Sequential()
                        if n_layer==1:
                            model.add(LSTM(nb_hidden,activation=acti,input_shape=(x_train.shape[1],x_train.shape[2])))
                            model.add(Dropout(drop))
                        if n_layer==2:
                            model.add(LSTM(nb_hidden,activation=acti,return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
                            model.add(Dropout(drop))
                            model.add(LSTM(nb_hidden,activation=acti))
                            model.add(Dropout(drop))
                        model.add(Dense(1))
                        model.compile(loss='mse',optimizer= Adam(learning_rate=l_r))
                        n_epochs=5000
                        es=EarlyStopping(monitor='loss', mode ='min',verbose=0,patience=50)
                        model.fit(x_train,y_train,epochs=n_epochs,batch_size=16,verbose=0,shuffle=True,callbacks=[es])
                        pred = model.predict(x_test,verbose=0)

                        eva=mean_squared_error(y_test, pred)
                        if eva < min_eva :
                            m_pred = pred
                            min_eva = eva
            if plot_res==True:
                plt.plot(y_test,label='Observed values')
                plt.plot(m_pred,label='Predicted values')
                plt.legend()
                plt.show()                        
    
    else:
        model = keras.Sequential()
        nb_hidden = int(len(x[:,1])/(10*len(x[1,:])))
        if nb_hidden<32:
            nb_hidden=32
        model.add(LSTM(nb_hidden,activation='relu',input_shape=(x_train.shape[1],x_train.shape[2])))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        model.compile(loss='mse',optimizer= Adam(learning_rate=0.001))
        n_epochs=5000
        es=EarlyStopping(monitor='loss', mode ='min',verbose=0,patience=50)
        model.fit(x_train,y_train,epochs=n_epochs,batch_size=16,verbose=0,shuffle=True,callbacks=[es])
        m_pred = model.predict(x_test,verbose=0)
        eva=mean_squared_error(y_test, m_pred)   
   
    min_eva_ar=np.inf
    
    x = np.concatenate([in_out['input'][:,:ar],X],axis=1)
    x_train = x[:int(train_test_split*len(x))-h,:]
    x_test = x[int(train_test_split*len(x)):-h,:]
    y_train = in_out['output'][h:int(train_test_split*len(x))]
    y_test = in_out['output'][h+int(train_test_split*len(x)):]
    
    x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1]))
    x_test= x_test.reshape((x_test.shape[0],1,x_test.shape[1]))
    if opti_grid is not None:
        for acti in opti_grid['activation']:
            for n_layer in opti_grid['n_layer']:
                for drop in opti_grid['drop']:
                    for l_r in opti_grid['l_r']:
                        nb_hidden = int(len(x[:,1])/(10*len(x[1,:])))
                        if nb_hidden<32:
                            nb_hidden=32
                        model = keras.Sequential()
                        if n_layer==1:
                            model.add(LSTM(nb_hidden,activation=acti,input_shape=(x_train.shape[1],x_train.shape[2])))
                            model.add(Dropout(drop))
                        if n_layer==2:
                            model.add(LSTM(nb_hidden,activation=acti,return_sequences=True,input_shape=(x_train.shape[1],x_train.shape[2])))
                            model.add(Dropout(drop))
                            model.add(LSTM(nb_hidden,activation=acti))
                            model.add(Dropout(drop))
                        model.add(Dense(1))
                        model.compile(loss='mse',optimizer= Adam(learning_rate=l_r))
                        n_epochs=5000
                        es=EarlyStopping(monitor='loss', mode ='min',verbose=0,patience=50)
                        model.fit(x_train,y_train,epochs=n_epochs,batch_size=16,verbose=0,shuffle=True,callbacks=[es])
                        pred_ar = model.predict(x_test,verbose=0)
                        eva_ar=mean_squared_error(y_test, pred_ar)
                        if eva_ar < min_eva_ar :
                            m_pred_ar = pred_ar
                            min_eva_ar = eva_ar                     
    else:
        model = keras.Sequential()
        nb_hidden = int(len(x[:,1])/(10*len(x[1,:])))
        if nb_hidden<32:
            nb_hidden=32
        model.add(LSTM(nb_hidden,activation='relu',input_shape=(x_train.shape[1],x_train.shape[2])))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        model.compile(loss='mse',optimizer= Adam(learning_rate=0.001))
        n_epochs=5000
        es=EarlyStopping(monitor='loss', mode ='min',verbose=0,patience=50)
        model.fit(x_train,y_train,epochs=n_epochs,batch_size=16,verbose=0,shuffle=True,callbacks=[es])
        m_pred_ar = model.predict(x_test,verbose=0)
        eva_ar=mean_squared_error(y_test, m_pred_ar)   
    if plot_res==True:
        plt.figure(figsize=(20,5))
        plt.plot(y_test,label='Observed')
        plt.plot(m_pred_ar,label='RF prediction')
        plt.plot(m_pred,label='RFX prediction')
        plt.legend()
        plt.show()            
    return({'results_table':pd.DataFrame([eva_ar,eva,((eva_ar-eva)/eva_ar)*100],index=['LSTM_MSE','LSTMX_MSE','%_Improv']),'rf_pred':m_pred_ar,'rfx_pred':m_pred})
          

def reset_tf_memory():
    """
    Resets the memory of TensorFlow.
    """
    tf.keras.backend.clear_session()
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        