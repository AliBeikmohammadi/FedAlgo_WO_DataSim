#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt

def filenamegenerator(Learning_rate, dataset, type, Method, batch_size=64, global_iter=400):
    fname= dataset+'-'+type+'-batch'+str(batch_size)
    if Method['Algo']=='FedAVG':
        Mname=Method['Algo']+'-LocIter'+str(Method['loc_iter'])                
    elif Method['Algo']=='EC_FedAVG':
        Mname=Method['Algo']+'-LocIter'+str(Method['loc_iter'])+'-topk'+str(Method['topk'])           
    elif Method['Algo']=='FedProx2':
        Mname=Method['Algo']+'-LocIter'+str(Method['loc_iter'])+'-alpha'+str(Method['alpha']) 
    elif Method['Algo']=='EC_FedProx2':
        Mname=Method['Algo']+'-LocIter'+str(Method['loc_iter'])+'-alpha'+str(Method['alpha'])+'-topk'+str(Method['topk'])
    Lname=Learning_rate['lr']+'-c'+str(Learning_rate['c'])
    if Learning_rate['lr']=='Diminishing':
        Lname=Learning_rate['lr']+'-c'+str(Learning_rate['c'])+'-v'+str(Learning_rate['v'])
    elif Learning_rate['lr']=='Step-decay':
        Lname=Learning_rate['lr']+'-c'+str(Learning_rate['c'])+'-st'+str(Learning_rate['step'])+'-gam'+str(Learning_rate['gamma'])
    filename=fname+'-'+Mname+'-'+Lname+'-GlobIter'+str(global_iter)
    return filename
    
    
def filelistgenerator(p=False):
    Dataset=['MNIST', 'FMNIST']
    Datatype=['iid', 'non_iid2', 'non_iid1']
    
    Method1={'Algo':'FedAVG', 'loc_iter': 30}
    Method2={'Algo':'EC_FedAVG', 'loc_iter': 30, 'topk': 1}
    Method3={'Algo':'FedProx2', 'loc_iter': 30, 'alpha':0.1}
    Method4={'Algo':'EC_FedProx2', 'loc_iter': 30, 'alpha':0.1, 'topk': 1} 
    
    Learning_rate1={'lr':'Fix', 'c': 2}
    Learning_rate2={'lr':'Diminishing', 'c': 0.8,  'v': 0.51}
    Learning_rate3={'lr':'Step-decay', 'c': 0.8,  'step': 50, 'gamma': 0.5}
    
    Methods=[Method1, Method2, Method3, Method4]  
    Learning_rates=[Learning_rate1, Learning_rate2, Learning_rate3]    
    
    filenamelist=[]
    for Learning_rate in Learning_rates:
        if p:
            print('*', Learning_rate)
        for dataset in Dataset:
            if p:
                print('**', dataset)
            for type in Datatype:
                if p:
                    print('***', type)
                for Method in Methods:
                    filename = filenamegenerator(Learning_rate, dataset, type, Method)
                    filenamelist.append(filename)
                    if p:
                        print(filename)
    return filenamelist, len(Methods), len(Datatype)


def myplot(Metrics_Y, L_Metrics_Y, Load_path, save_fig_path, Coef, conf):
    print(conf)
    filenamelist, len_m, len_d= filelistgenerator()
    filelim={'MNIST-Fix': 0, 'FMNIST-Fix': 1, 'MNIST-Diminishing': 2, 
             'FMNIST-Diminishing': 3, 'MNIST-Step-decay': 4, 'FMNIST-Step-decay': 5}
    n= len_d
    flist = filenamelist[filelim[conf]*len_d*len_m:(filelim[conf]+1)*len_d*len_m]
    isExist = os.path.exists(save_fig_path)
    if not isExist:
        os.makedirs(save_fig_path)
    fig, axes = plt.subplots(nrows=n, ncols=len(Metrics_Y), figsize=(10, 10))

    for d in range(n):
        Flist=flist[d*len_m : (d+1)*len_m]
        for i in range(len(Metrics_Y)):
            for f, ex in enumerate(Flist):
                if "FMNIST" in ex:
                    load_path = Load_path + 'FMNIST/'
                elif "MNIST" in ex:
                    load_path = Load_path + 'MNIST/'
                if "EC_FedProx2" in ex:
                    Label= "EF-FedProx"
                elif "FedProx2" in ex:
                    Label= "FedProx"
                elif "EC_FedAVG" in ex:
                    Label= "EF-FedAVG"
                elif "FedAVG" in ex:
                    Label= "FedAVG"
                data = pd.read_csv(load_path+ex+'.csv')
                if d==0 and i ==0:
                    data.plot(x='Unnamed: 0', y=Metrics_Y[i]+'_mean',ax=axes[d, i],label=Label, linewidth=1) 
                else:
                    data.plot(x='Unnamed: 0', y=Metrics_Y[i]+'_mean',ax=axes[d, i],legend=False, linewidth=1) 
                axes[d, i].fill_between(data['Unnamed: 0'], data[Metrics_Y[i]+'_mean'] - Coef*data[Metrics_Y[i]+'_std'], data[Metrics_Y[i]+'_mean'] + Coef*data[Metrics_Y[i]+'_std'], alpha=0.3)
                axes[d, i].set_ylabel(L_Metrics_Y[i], fontsize=18)
                axes[d, i].set_xlabel(None, fontsize=18)
            axes[n-1, i].set_xlabel('Communication Round', fontsize=18)
      
        axes[d, 1].set_ylim([0, 100])
        #axes[d, 0].semilogy()
    legend = axes[0, 0].legend(fontsize=14)
    fig.tight_layout(pad=0.25)
    plt.figtext(0.5,1.03, "IID", ha="center", va="top", fontsize=20, color="black")
    plt.figtext(0.5,0.69, "Non-IID2", ha="center", va="top", fontsize=20, color="black")
    plt.figtext(0.5,0.36, "Non-IID1", ha="center", va="top", fontsize=20, color="black")
    plt.subplots_adjust(hspace=0.25)
    plt.rcParams['font.size'] = 14
    #plt.show()
    plt.savefig(save_fig_path+conf+'.png', dpi=256, format=None, metadata=None, bbox_inches='tight', pad_inches=0,
                facecolor='auto', edgecolor='auto', backend=None, transparent=False)
    print("The figures saved as: "+save_fig_path+conf+'.png')
    return

# set the path where the CSV files are stored
Load_path = './CSV_Aggregate/'
save_fig_path = './Plots_Paper/'

Metrics_Y=['server_train_loss','server_test_acc'] 
L_Metrics_Y=['Training Loss','Test Accuracy'] 
Coef= 1

#'MNIST-Fix', 'FMNIST-Fix', 'MNIST-Diminishing', 'FMNIST-Diminishing', 'MNIST-Step-decay', 'FMNIST-Step-decay'
myplot(Metrics_Y, L_Metrics_Y, Load_path, save_fig_path, Coef, conf='MNIST-Fix')

