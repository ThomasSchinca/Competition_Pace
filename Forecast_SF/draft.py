# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 21:54:16 2024

@author: thoma
"""

#### Weights

# include_w=[0.5,0.5,8]
# w_l=[0.75,0.1,3.5]
# pred_tot_pr=[]
# horizon=9
# for coun in range(len(df_input_sub.columns)):
#     if not (df_input_sub.iloc[-h_train:,coun]==0).all():
#         inp=[df_conf[df_input_sub.iloc[-h_train:,coun].name],df_input_sub.iloc[-h_train:,coun].index.year[int(horizon/2)],np.log10(df_input_sub.iloc[-h_train:,coun].sum())]
#         l_find=dict_m[df_input.columns[coun]]
#         tot_seq = [[series.name, series.index[-1],series.min(),series.max(),series.sum()] for series, weight in l_find]
#         pred_seq=[]
#         co=[]
#         deca=[]
#         scale=[]
#         for col,last_date,mi,ma,somme in tot_seq:
#             date=df_tot_m.iloc[:-24].index.get_loc(last_date)
#             if date+horizon<len(df_tot_m.iloc[:-24]):
#                 seq=df_tot_m.iloc[:-24].iloc[date+1:date+1+horizon,df_tot_m.iloc[:-24].columns.get_loc(col)].reset_index(drop=True)
#                 seq = (seq - mi) / (ma - mi)
#                 pred_seq.append(seq.tolist())
#                 co.append(df_conf[col])
#                 deca.append(last_date.year)
#                 scale.append(somme)
#         tot_seq=pd.DataFrame(pred_seq)
#         linkage_matrix = linkage(tot_seq, method='ward')
#         clusters = fcluster(linkage_matrix, horizon/2, criterion='distance')
#         df_sce=pd.DataFrame([clusters,co,deca,np.log10(scale)]).T
#         df_sce.columns=["Sce","Region","Decade","Scale"]
        
#         tot_wei=[]
#         for i in range(len(df_sce)):
#             if df_sce.iloc[i,1]==inp[0]:
#                 weight=include_w[0]
#             else:
#                 weight=w_l[0]*include_w[0]
#             weight = weight + include_w[1]*(1-w_l[1]*abs(inp[1]-df_sce.iloc[i,2]))
#             if abs(inp[2]-df_sce.iloc[i,3])>w_l[2]:
#                 pass
#             else:
#                 weight = weight + include_w[2]*(1-(1/(w_l[2]**2))*(abs(inp[2]-df_sce.iloc[i,3])**2))
#             if (pd.Series(include_w)==0).all():
#                 weight=1
#             tot_wei.append(weight)
#         pred_ori=pd.Series([(tot_seq.iloc[:,i]*tot_wei).sum()/pd.Series(tot_wei).sum() for i in range(len(tot_seq.columns))])
#         pred_tot_pr.append(pred_ori*(df_input_sub.iloc[-h_train:,coun].max()-df_input_sub.iloc[-h_train:,coun].min())+df_input_sub.iloc[-h_train:,coun].min())
#     else:
#         pred_tot_pr.append(pd.Series(np.zeros((horizon,))))
# plot_res()


def plot_res():    
    err_sf=[]
    err_sf_pr=[]
    err_views=[]
    err_zero=[]
    err_t1=[]
    for i in range(len(df_input.columns)):
        err_sf.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], pred_tot_min[i]))
        err_sf_pr.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], pred_tot_pr[i]))
        err_views.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:horizon,i]))
        err_zero.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], pd.Series(np.zeros((horizon,)))))
        err_t1.append(mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_input.iloc[-24-horizon:-24,i]))
        if (mean_squared_error(df_input.iloc[-24:-24+horizon,i], pred_tot_pr[i])>2*mean_squared_error(df_input.iloc[-24:-24+horizon,i], df_preds_test_1.iloc[:horizon,i])) & (df_input.iloc[-24:-24+horizon,i].sum()>10):
            plt.plot(df_input.iloc[-24:-24+horizon,i].reset_index(drop=True),label='True')
            plt.plot(pred_tot_min[i],label='Min',marker='s')
            plt.plot(df_preds_test_1.iloc[:horizon,i].reset_index(drop=True),label='Views')
            plt.plot(pred_tot_pr[i],label='Pr')
            plt.legend()
            plt.show()
        
        
    err_sf = np.log(np.array(err_sf)+1)
    err_sf_pr = np.log(np.array(err_sf_pr)+1)
    err_views = np.log(np.array(err_views)+1)
    err_zero = np.log(np.array(err_zero)+1)
    err_t1 = np.log(np.array(err_t1)+1)
    
    means = [err_views.mean(),err_sf.mean(),err_sf_pr.mean(),err_zero.mean(),err_t1.mean()]
    std_error = [2*err_views.std()/np.sqrt(len(err_views)),2*err_sf.std()/np.sqrt(len(err_sf)),2*err_sf_pr.std()/np.sqrt(len(err_sf_pr)),2*err_zero.std()/np.sqrt(len(err_zero)),2*err_t1.std()/np.sqrt(len(err_t1))]
    mean_mse = pd.DataFrame({
        'mean': means,
        'std': std_error
    })
    
                
    # plt.figure(figsize=(12,8))
    # ax = plt.axes()
    # marker_size = 150
    # linewidth = 3
    # fonts=25
    # plt.rc('font', size=24)
    # plt.ylabel('Mean Squared Error')
    # plt.tick_params(axis='y')
    # plt.scatter(mean_mse.index, mean_mse['mean'], color="black", marker='o', s=marker_size)
    # plt.errorbar(mean_mse.index, mean_mse['mean'], yerr=mean_mse['std'], fmt='none', color="black", linewidth=linewidth)
    # plt.grid(False)
    # plt.xticks([0,1,2,3],['Views','Shape Finder',"Null",'T-1'])
    # plt.show()
    
    
    
    # model_no_z=pd.DataFrame([np.log((err_views+1)/(err_sf+1)),np.log((err_zero+1)/(err_sf+1))])
    # model_no_z=model_no_z.T
    # model_no_z.columns=["ratio_1", "ratio_2"]
    # med_1 = model_no_z['ratio_1'].median()
    # med_2 = model_no_z['ratio_2'].median()
    
    # melted_df = pd.melt(model_no_z, value_vars=["ratio_1", "ratio_2"], var_name="Model", value_name="Log ratio")
    # sns.set(style="ticks",rc={"figure.figsize": (7, 8)})
    # b = sns.boxplot(data = melted_df,           
    #                     x = "Model",       # x axis column from data
    #                     y = "Log ratio",       # y axis column from data
    #                     width = 0.4,        # The width of the boxes
    #                     color = "white",  # Box colour
    #                     linewidth = 2,      # Thickness of the box lines
    #                     showfliers = False)  # Sop showing the fliers
    # b = sns.stripplot(data = melted_df,           
    #                     x = "Model",       # x axis column from data
    #                     y = "Log ratio",     # y axis column from data
    #                       color = "darkgrey", # Colours the dots
    #                       linewidth = 1,     # Dot outline width
    #                       alpha = 0.4)       # Makes them transparent
    # b.set_ylabel("Log Ratio", fontsize = 20)
    # b.set_xlabel("Model", fontsize = 20)
    # b.set_xticklabels(['VIEWS', 'Zeros'])
    # b.tick_params(axis='both', which='both', labelsize=20)
    # b.axhline(y=0, linestyle='--', color='black', linewidth=1)
    # sns.despine(offset = 5, trim = True)
    # b.set_ylim(-2,2)
    # plt.show()
    
    
        
        
        
    # Difference explained
    d_nn=[]
    d_nn1=[]
    d_b=[]
    d_null=[]
    d_t1=[]
    
    k=5
    for i in range(len(df_input.columns)):
        real = df_input.iloc[-24:-24+horizon,i]
        real=real.reset_index(drop=True)
        sf=pred_tot_min[i]
        sf=sf.reset_index(drop=True)
        sf1=pred_tot_pr[i]
        sf1=sf1.reset_index(drop=True)
        b1=df_preds_test_1.iloc[:horizon,i]
        b1=b1.reset_index(drop=True)
        null=pd.Series(np.zeros((horizon,)))
        null=null.reset_index(drop=True)
        t1=df_input.iloc[-24-horizon:-24,i]
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
    d_nn = np.array(d_nn)
    d_nn1 = np.array(d_nn1)
    d_b = np.array(d_b)
    d_null = np.array(d_null)
    d_t1 = np.array(d_t1)
    
    means = [d_b.mean(),d_nn.mean(),d_nn1.mean(),d_null.mean(),d_t1.mean()]
    std = [2*d_b.std()/np.sqrt(len(d_b)),2*d_nn.std()/np.sqrt(len(d_nn)),2*d_nn1.std()/np.sqrt(len(d_nn1)),2*d_null.std()/np.sqrt(len(d_null)),2*d_t1.std()/np.sqrt(len(d_t1))]
    mean_de = pd.DataFrame({
        'mean': means,
        'std': std
    })
    
    # plt.figure(figsize=(12,8))
    # marker_size = 150
    # linewidth = 3
    # fonts=25
    # plt.rc('font', size=24)
    # plt.ylabel('Difference Explained (DE)')
    # plt.tick_params(axis='y')
    # plt.scatter(mean_de.index, mean_de['mean'], color="black", marker='o', s=marker_size)
    # plt.errorbar(mean_de.index, mean_de['mean'], yerr=mean_de['std'], fmt='none', color="black", linewidth=linewidth)
    # plt.grid(False)
    # plt.xticks([0,1,2,3],['ViEWS','Shape Finder',"Null","t-1"])
    # plt.show()
    
    fig,ax = plt.subplots(figsize=(12,8))
    plt.scatter(mean_mse["mean"][0],mean_de["mean"][0],color="black",s=150)
    plt.scatter(mean_mse["mean"][1],mean_de["mean"][1],color="black",s=150)
    plt.scatter(mean_mse["mean"][2],mean_de["mean"][2],color="black",s=150)
    plt.scatter(mean_mse["mean"][3],mean_de["mean"][3],color="black",s=150)
    plt.scatter(mean_mse["mean"][4],mean_de["mean"][4],color="black",s=150)
    plt.plot([mean_mse["mean"][0],mean_mse["mean"][0]],[mean_de["mean"][0]-mean_de["std"][0],mean_de["mean"][0]+mean_de["std"][0]],linewidth=3,color="black")
    plt.plot([mean_mse["mean"][0]-mean_mse["std"][0],mean_mse["mean"][0]+mean_mse["std"][0]],[mean_de["mean"][0],mean_de["mean"][0]],linewidth=3,color="black")
    plt.text(mean_mse["mean"][0], mean_de["mean"][0], "ViEWS", size=20, color='black')
    plt.plot([mean_mse["mean"][1],mean_mse["mean"][1]],[mean_de["mean"][1]-mean_de["std"][1],mean_de["mean"][1]+mean_de["std"][1]],linewidth=3,color="black")
    plt.plot([mean_mse["mean"][1]-mean_mse["std"][1],mean_mse["mean"][1]+mean_mse["std"][1]],[mean_de["mean"][1],mean_de["mean"][1]],linewidth=3,color="black")
    plt.text(mean_mse["mean"][1], mean_de["mean"][1], "SF Min", size=20, color='black')
    plt.plot([mean_mse["mean"][2],mean_mse["mean"][2]],[mean_de["mean"][2]-mean_de["std"][2],mean_de["mean"][2]+mean_de["std"][2]],linewidth=3,color="black")
    plt.plot([mean_mse["mean"][2]-mean_mse["std"][2],mean_mse["mean"][2]+mean_mse["std"][2]],[mean_de["mean"][2],mean_de["mean"][2]],linewidth=3,color="black")
    plt.text(mean_mse["mean"][2], mean_de["mean"][2], "SF Proba", size=20, color='black')
    plt.plot([mean_mse["mean"][3],mean_mse["mean"][3]],[mean_de["mean"][3]-mean_de["std"][3],mean_de["mean"][3]+mean_de["std"][3]],linewidth=3,color="black")
    plt.plot([mean_mse["mean"][3]-mean_mse["std"][3],mean_mse["mean"][3]+mean_mse["std"][3]],[mean_de["mean"][3],mean_de["mean"][3]],linewidth=3,color="black")
    plt.text(mean_mse["mean"][3], mean_de["mean"][3], "Null", size=20, color='black')
    plt.plot([mean_mse["mean"][4],mean_mse["mean"][4]],[mean_de["mean"][4]-mean_de["std"][4],mean_de["mean"][4]+mean_de["std"][4]],linewidth=3,color="black")
    plt.plot([mean_mse["mean"][4]-mean_mse["std"][4],mean_mse["mean"][4]+mean_mse["std"][4]],[mean_de["mean"][4],mean_de["mean"][4]],linewidth=3,color="black")
    plt.text(mean_mse["mean"][4], mean_de["mean"][4], "t-1", size=20, color='black')
    plt.xlabel("Accuracy (MSE reversed)")
    plt.ylabel("Difference explained (DE)")
    ax.invert_xaxis()
    plt.show()