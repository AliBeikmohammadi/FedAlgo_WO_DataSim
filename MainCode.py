#!/usr/bin/env python
# coding: utf-8

import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import FashionMNIST, MNIST
from torchvision import datasets, transforms
from torch.utils.data import Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import pandas as pd


def seed_set(SEED):
    # Set a seed on the three different libraries
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(SEED)

def non_iid_split(dataset, non_iid_type, nb_nodes, batch_size, shuffle, seed, g, shuffle_digits=False):
    if nb_nodes !=10:
        raise NotImplementedError('')

    n_classes = 10
    if non_iid_type == 'non_iid1':
        class_per_client = 1
    elif non_iid_type == 'non_iid2':
        class_per_client = 2
    else:
        raise NotImplementedError('')
        
        
    # Split the dataset into 20 parts, each containing half of one digit
    split_datasets = [[] for _ in range(n_classes)]
    for digit in range(n_classes):
        indices = (dataset.targets == digit).nonzero().flatten()
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, class_per_client)
        for split in split_indices:
            split_datasets[digit].append(Subset(dataset, split))

    # Randomly concatenate two datasets to create new datasets with half of two classes
    data_splitted = []
    if non_iid_type == 'non_iid1':
        for i in range(0, nb_nodes, 1):
            combined_partition = ConcatDataset(split_datasets[i])
            data_splitted.append(torch.utils.data.DataLoader(combined_partition,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     worker_init_fn=seed,
                                                     generator=g))
         
    elif non_iid_type == 'non_iid2':
        for i in range(0, nb_nodes, 1):
            digit1, digit2 = i, (i + 1) % 10
            combined_partition = ConcatDataset([split_datasets[digit1][0], split_datasets[digit2][1]])
            data_splitted.append(torch.utils.data.DataLoader(combined_partition,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     worker_init_fn=seed,
                                                     generator=g))
  
    return data_splitted

def iid_split(dataset, nb_nodes, batch_size, shuffle, seed,g):
    
    # load and shuffle n_samples_per_node from the dataset
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=int(len(dataset)/nb_nodes),
                                        shuffle=shuffle,
                                        worker_init_fn=seed,
                                                generator=g)
    dataiter = iter(loader)
    
    data_splitted=list()
    for _ in range(nb_nodes):
        data_splitted.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(next(dataiter))), 
                                                         batch_size=batch_size, 
                                                         shuffle=shuffle,
                                                        worker_init_fn=seed,
                                                    generator=g))

    return data_splitted

def  get_DATA(dataset, type, n_clients, batch_size, shuffle, seed):
    if dataset=='MNIST':
        dataset_loaded_train = MNIST(
                root="./data",
                train=True,
                download=True,
                transform=transforms.ToTensor()
        ) 
        dataset_loaded_test = MNIST(
                root="./data",
                train=False,
                download=True,
                transform=transforms.ToTensor()
        ) 
        
    if dataset=='FashionMNIST' or dataset=='FMNIST':
        dataset_loaded_train = FashionMNIST(
                root="./data",
                train=True,
                download=True,
                transform = transforms.Compose([ transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor()])
        ) 
        dataset_loaded_test = FashionMNIST(
                root="./data",
                train=False,
                download=True,
                transform = transforms.Compose([ transforms.RandomHorizontalFlip(),  
                                    transforms.ToTensor()])
        )  
    
    seed_set(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    if type=="iid":
        train=iid_split(dataset_loaded_train, n_clients, batch_size, shuffle, seed,g)
        test=iid_split(dataset_loaded_test, n_clients, batch_size, shuffle, seed,g)
    elif type=="non_iid1" or type=="non_iid2":
        non_iid_type = type
        train=non_iid_split(dataset_loaded_train, non_iid_type, n_clients, batch_size, shuffle, seed,g)
        test=non_iid_split(dataset_loaded_test , non_iid_type, n_clients, batch_size, shuffle, seed,g)
    else:
        train=[]
        test=[]

    return train, test

  
def plot_samples(data, channel:int, title=None, plot_name="", n_examples =20):
    n_rows = int(n_examples / 5)
    plt.figure(figsize=(1* n_rows, 1*n_rows))
    if title: plt.suptitle(title)
    X, y= data
    for idx in range(n_examples):
        ax = plt.subplot(n_rows, 5, idx + 1)
        image = 255 - X[idx, channel].view((28,28))
        ax.imshow(image, cmap='gist_gray')
        ax.axis("off")

    if plot_name!="":plt.savefig(f"plots/"+plot_name+".png")
    plt.tight_layout()
    
def plot_acc_loss(title:str, loss_hist:list, acc_hist:list, All=False):
    plt.figure()
    plt.suptitle(title)
    plt.subplot(1,2,1)
    if All:
        lines=plt.plot(loss_hist)
        plt.title("Loss")
        legend= [f'C{i+1}' for i in range(len(loss_hist[0])-1)]
        legend.append('Avg')
        plt.legend(lines,legend)
        plt.subplot(1,2,2)
        lines=plt.plot(acc_hist )
        plt.title("Accuracy")
        plt.legend(lines, legend)
    else:
        loss_avg=[loss_hist[i][-1] for i in range(len(loss_hist))]
        acc_avg=[acc_hist[i][-1] for i in range(len(acc_hist))]
        lines=plt.plot(loss_avg)
        plt.title("Loss")
        plt.legend(lines,['Avg'])
        plt.subplot(1,2,2)
        lines=plt.plot(acc_avg )
        plt.title("Accuracy")
        plt.legend(lines, ['Avg'])
    
    
class CNN(nn.Module):

    """ConvNet -> Max_Pool -> RELU -> ConvNet ->
    Max_Pool -> RELU -> FC -> RELU -> FC -> SOFTMAX"""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def loss_classifier(predictions,labels):
    loss = nn.CrossEntropyLoss(reduction="mean") #It is equivalent to the combination of LogSoftmax and NLLLoss.
    return loss(predictions,labels.view(-1))

def loss_dataset(model, dataset, loss_f, epoch_or_iter):
    """Compute the loss of `model` on `dataset`"""
    loss=0
    if epoch_or_iter == 'epoch':
        for idx,(features,labels) in enumerate(dataset):
            predictions= model(features)
            loss+=loss_f(predictions,labels)
        loss/=idx+1
        return loss
    if epoch_or_iter == 'iteration':
        # Select a random batch index
        random_batch_index = torch.randint(high=len(dataset), size=(1,)).item()
        for idx,(features,labels) in enumerate(dataset):
            if idx == random_batch_index:
                predictions= model(features)
                loss = loss_f(predictions,labels)
        return loss


def accuracy_dataset(model, dataset):
    """Compute the accuracy of `model` on `dataset`"""
    correct=0
    for features,labels in iter(dataset):
        predictions= model(features)
        _,predicted=predictions.max(1,keepdim=True)
        correct+=torch.sum(predicted.view(-1,1)==labels.view(-1, 1)).item()
    accuracy = 100*correct/len(dataset.dataset)
    return accuracy

def train_step(model, model_0, MU:int, optimizer, train_data, loss_f, epoch_or_iter, Prox, Prox2, epsilon):
    """Train `model` on one epoch of `train_data`"""
    total_loss=0
    if epoch_or_iter == 'epoch' and not Prox and not Prox2:
        for idx, (features,labels) in enumerate(train_data):
            optimizer.zero_grad()
            predictions= model(features)
            loss=loss_f(predictions,labels)
            total_loss+=loss
            loss.backward()
            optimizer.step()
        return total_loss/(idx+1)
    elif epoch_or_iter == 'iteration' and not Prox and not Prox:
        # Select a random batch index
        random_batch_index = torch.randint(high=len(train_data), size=(1,)).item()
        for idx, (features,labels) in enumerate(train_data):
            if idx == random_batch_index:
                optimizer.zero_grad()
                predictions= model(features)
                loss=loss_f(predictions,labels)
                loss.backward()
                optimizer.step()
        return loss
    
    elif epoch_or_iter == 'epoch' and (Prox or Prox2):
        raise NotImplementedError('epoch version of Prox has not been implemented yet.')

    elif epoch_or_iter == 'iteration' and Prox and not Prox2:
        # Select a random batch index
        random_batch_index = torch.randint(high=len(train_data), size=(1,)).item()
        counter=0
        grad=10*epsilon
        while grad>epsilon:
            counter+=1
            for idx, (features,labels) in enumerate(train_data):
                if idx == random_batch_index:
                    optimizer.zero_grad()
                    predictions= model(features)
                    loss=loss_f(predictions,labels)
                    loss+=MU/2*difference_models_norm_2(model,model_0)
                    total_loss+=loss
                    loss.backward()
                    total_norm = 0
                    for param in model.parameters():
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item()**2
                    grad = total_norm **(0.5)
                    print('counter:', counter, ' grad', grad)
                    if grad>epsilon:
                        optimizer.step()
        print('counter', counter)
        return total_loss/counter
    
    elif epoch_or_iter == 'iteration' and Prox2 and not Prox:
        # Select a random batch index
        random_batch_index = torch.randint(high=len(train_data), size=(1,)).item()
        for e in range(loc_iter):
            for idx, (features,labels) in enumerate(train_data):
                if idx == random_batch_index:
                    optimizer.zero_grad()
                    predictions= model(features)
                    loss=loss_f(predictions,labels)
                    loss+=MU/2*difference_models_norm_2(model,model_0)
                    total_loss+=loss
                    loss.backward()
                    optimizer.step()
        return total_loss/loc_iter
       

def local_learning(model, MU:float, epsilon:float, optimizer, train_data, loc_iter:int, loss_f, epoch_or_itertaion, Prox, Prox2):
    model_0=deepcopy(model)
    if Prox:
        local_loss=train_step(model,model_0,MU,optimizer,train_data,loss_f, epoch_or_itertaion, Prox, Prox2, epsilon)
    elif Prox2:
        local_loss=train_step(model,model_0,MU,optimizer,train_data,loss_f, epoch_or_itertaion, Prox, Prox2, epsilon)
        
    else:
        for e in range(loc_iter):
            local_loss=train_step(model,model_0,MU,optimizer,train_data,loss_f, epoch_or_itertaion, Prox, Prox2, epsilon)
             
    return float(local_loss.detach().numpy())


def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters
    """
    tensor_1=list(model_1.parameters())
    tensor_2=list(model_2.parameters())
    norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) for i in range(len(tensor_1))])
    return norm


def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""
    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)

def average_models(model, clients_models_hist:list , weights:list):
    """Creates the new model of a given iteration with the models of the other
    clients
    -model is the current global model
    - clients_models_hist is the list of each client's updated local model
    - weights is the list of each client's weight"""
    new_model=deepcopy(model)
    set_to_zero_model_weights(new_model)

    for k,client_hist in enumerate(clients_models_hist):
        for idx, layer_weights in enumerate(new_model.parameters()):
            contribution=client_hist[idx].data*weights[k]
            layer_weights.data.add_(contribution)
    return new_model

def error_and_compress(model, clients_models_hist:list, error_clients: list, topK: int, weights:list):
    new_error_clients=deepcopy(error_clients)
    new_model=deepcopy(model)
    new_model_hist=[tens_param.detach() for tens_param in list(new_model.parameters())] 
                                                                                        

    #calculate  x_i^k,T - x^k + e_i^k for all clients 
    diff_term = []     
    for k,client_hist in enumerate(clients_models_hist):
        temp = []
        for i, client_hist_s in enumerate(client_hist): 
            temp.append(torch.add(torch.sub(client_hist_s, new_model_hist[i], alpha=1), new_error_clients[k][i], alpha=1))
        diff_term.append(temp)
        
    #apply Q over diff_term Q(x_i^k,T - x^k + e_i^k) for all clients
    compress_term = deepcopy(diff_term)
    for i, client_i in enumerate(compress_term): 
        flatten_client_i = []
        for client_i_x in client_i:
            flatten_client_i.append(torch.flatten(client_i_x))
        flatten_client_i = torch.cat(flatten_client_i)
        tk = min(topK, len(flatten_client_i))
        topk_i, _ = torch.topk(torch.abs(flatten_client_i), tk)
        threshold_i = topk_i[-1]
        for z, compress_term_i_x in enumerate(compress_term[i]):
            mask_i_x = torch.abs(compress_term_i_x) >= threshold_i
            compress_term[i][z]= compress_term[i][z] * mask_i_x.type_as(compress_term[i][z])
    
    #compute new_error_clients =  diff_term - compress_term 
    for i,diff_term_i in enumerate(diff_term):
        for x, diff_term_i_x in enumerate(diff_term_i): 
            new_error_clients[i][x] = torch.sub(diff_term_i_x, compress_term[i][x], alpha=1)
     
    #compute global model using Q x_new=x_old+ weight*Q
    for k,compress_term_k in enumerate(compress_term):
        for idx, layer_weights in enumerate(new_model.parameters()):
            contribution=compress_term_k[idx].data*weights[k]
            layer_weights.data.add_(contribution)

    return new_error_clients, new_model
        

def error_0(model, K):
    new_model=deepcopy(model)
    set_to_zero_model_weights(new_model)
    list_params=[tens_param.detach() for tens_param in list(new_model.parameters())] #e_i^0,T error of client k in iter 0 = 0
    error_init_clients=[]
    for k in range(K):
        error_init_clients.append(list_params) #e_i^0,T  for all K
    return error_init_clients
        

def FedMain(training_sets, testing_sets, Method, global_iter, Learning_rate, epoch_or_iter, fname, seed): 
    seed_set(seed)
    torch.manual_seed(seed)
    model = CNN()
    loss_f=loss_classifier
    
    Trainable_parameters=sum(param.numel() for param in model.parameters() if param.requires_grad)
    #Variables initialization
    K=len(training_sets) #number of clients
    n_samples=sum([len(db.dataset) for db in training_sets])
    weights=([len(db.dataset)/n_samples for db in training_sets])
    
    if Method['Algo']=='FedAVG':
        loc_iter=Method['loc_iter']
        EC = False
        Prox = False
        Prox2 = False
        MU = 0 
        epsilon = 0 
        #print('FedAVG with '+ str(loc_iter)+ ' local iteration')
        Mname=Method['Algo']+'-LocIter'+str(Method['loc_iter'])
        
    elif Method['Algo']=='FedProx':
        epsilon=Method['epsilon']
        alpha=Method['alpha']
        EC = False
        Prox = True
        Prox2 = False
        loc_iter = 0 
        #print('FedProx with epsilon: '+ str(epsilon)+ ' and alpha: '+str(alpha))
        Mname=Method['Algo']+'-eps'+str(Method['epsilon'])+'-alpha'+str(Method['alpha'])
        
    elif Method['Algo']=='EC_FedAVG':
        loc_iter=Method['loc_iter']
        EC = True
        Prox = False
        Prox2 = False
        topK= Method['topk']*Trainable_parameters//100
        MU = 0 
        epsilon = 0 
        #print('EC-FedAVG with '+ str(loc_iter)+' local iteration and TopK: '+str(topK))
        Mname=Method['Algo']+'-LocIter'+str(Method['loc_iter'])+'-topk'+str(Method['topk'])
        
    elif Method['Algo']=='EC_FedProx':
        epsilon=Method['epsilon']
        alpha=Method['alpha']
        EC = True
        Prox = True
        Prox2 = False
        topK= Method['topk']*Trainable_parameters//100
        loc_iter=0 
        #print('EC-FedProx with epsilon: '+ str(epsilon)+ ' and alpha: '+str(alpha)+ ' and TopK: '+str(topK))
        Mname=Method['Algo']+'-eps'+str(Method['epsilon'])+'-alpha'+str(Method['alpha'])+'-topk'+str(Method['topk'])
        
    elif Method['Algo']=='FedProx2':
        alpha=Method['alpha']
        EC = False
        Prox = False
        Prox2 = True
        loc_iter = Method['loc_iter']
        epsilon=0
        #print('FedProx2 with '+ str(loc_iter)+' local iteration and alpha: '+str(alpha))
        Mname=Method['Algo']+'-LocIter'+str(Method['loc_iter'])+'-alpha'+str(Method['alpha'])
        
    elif Method['Algo']=='EC_FedProx2':
        alpha=Method['alpha']
        EC = True
        Prox = False
        Prox2 = True
        topK= Method['topk']*Trainable_parameters//100
        loc_iter= Method['loc_iter']
        epsilon=0
        #print('EC-FedProx2 with '+ str(loc_iter)+' local iteration and alpha: '+str(alpha)+ ' and TopK: '+str(topK))
        Mname=Method['Algo']+'-LocIter'+str(Method['loc_iter'])+'-alpha'+str(Method['alpha'])+'-topk'+str(Method['topk'])
    
    lr=Learning_rate['lr']  
    c=Learning_rate['c']
    LR = c/(global_iter**0.5)
    Lname=Learning_rate['lr']+'-c'+str(Learning_rate['c'])
    if Learning_rate['lr']=='Diminishing':
        v=Learning_rate['v']
        Lname=Learning_rate['lr']+'-c'+str(Learning_rate['c'])+'-v'+str(Learning_rate['v'])
        LR = c
    elif Learning_rate['lr']=='Step-decay':
        step=Learning_rate['step']
        gamma=Learning_rate['gamma']
        LR=c
        Lname=Learning_rate['lr']+'-c'+str(Learning_rate['c'])+'-st'+str(Learning_rate['step'])+'-gam'+str(Learning_rate['gamma'])
        
    filename=fname+'-'+Mname+'-'+Lname+'-GlobIter'+str(global_iter)+'-seed'+str(seed)
    if ('FashionMNIST' in filename) or ('FMNIST' in filename):
        datasetname='FMNIST'
    else:
        datasetname='MNIST'
        
    writer = SummaryWriter('./tensorboard/'+datasetname+'/'+filename)
    
    print('Config: ', filename)
    print('Trainable parameters:', Trainable_parameters)
    print("Clients' weights:",weights)
    lr_hist=[LR]
    loss_hist=[[float(loss_dataset(model, dl, loss_f, epoch_or_iter='epoch').detach()) for dl in training_sets]] 
    acc_hist=[[accuracy_dataset(model, dl) for dl in testing_sets]] 
    server_hist=[[tens_param.detach().numpy()
        for tens_param in list(model.parameters())]] 
    models_hist = []

    server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
    server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])
    
    loss_hist[-1].append(server_loss) 
    acc_hist[-1].append(server_acc)  
    
    print(f'====> i: 0 Server Loss: {server_loss} Server Test Accuracy: {server_acc}')
            
    for cl in range(len(weights)):
        writer.add_scalar("Loss/train_"+str(cl+1), loss_hist[-1][cl], 0)
        writer.add_scalar("Acc/test_"+str(cl+1), acc_hist[-1][cl], 0)
    writer.add_scalar("Server/Loss", loss_hist[-1][-1], 0)
    writer.add_scalar("Server/Test", acc_hist[-1][-1], 0)

    if EC: #error_corection
        error_clients = error_0(deepcopy(model), K)

    for i in range(global_iter):
        clients_models=[] 
        clients_params=[] 
        clients_losses=[] 
        #DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        if lr=='Fix': 
            LR = c/(global_iter**0.5)
        elif lr== 'Step-decay': 
            LR=c*(gamma**np.floor(i/step))
        elif lr=='Diminishing':
            LR = c/((i+1)**v)
   
        if Prox or Prox2: 
            MU = 1/LR
    
        for k in range(K):
            local_model=deepcopy(model)
            if Prox or Prox2: 
                local_optimizer=optim.SGD(local_model.parameters(),lr=alpha)
            else:
                local_optimizer=optim.SGD(local_model.parameters(),lr=LR)
                
            local_loss=local_learning(local_model,MU, epsilon, local_optimizer, training_sets[k],
                                      loc_iter,loss_f, epoch_or_iter, Prox, Prox2)
            if Prox:
                print('Client ', k+1, 'converged with epsilon >= norm of gradient')
            clients_losses.append(local_loss)

            #GET THE PARAMETER TENSORS OF THE MODEL
            list_params=list(local_model.parameters())
            list_params=[tens_param.detach() for tens_param in list_params] 
            clients_params.append(list_params) #x_i^k,T  for all K
            clients_models.append(deepcopy(local_model))
            
        #CREATE THE NEW GLOBAL MODEL
        if EC:
            error_clients, model = error_and_compress(deepcopy(model), clients_params, deepcopy(error_clients), topK=topK, weights=weights)
        else:
            model = average_models(deepcopy(model), clients_params, weights=weights)
        models_hist.append(clients_models)
        lr_hist.append(LR)
        #COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        loss_hist+=[[float(loss_dataset(model, dl, loss_f, epoch_or_iter='epoch').detach())
            for dl in training_sets]]
        acc_hist+=[[accuracy_dataset(model, dl) for dl in testing_sets]]

        server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
        server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])
        
        loss_hist[-1].append(server_loss) 
        acc_hist[-1].append(server_acc)  

        print(f'====> i: {i+1} LR: {LR} Server Loss: {server_loss} Server Test Accuracy: {server_acc}')
         
        for cl in range(len(weights)):
            writer.add_scalar("Loss/train_"+str(cl+1), loss_hist[-1][cl], i+1)
            writer.add_scalar("Acc/test_"+str(cl+1), acc_hist[-1][cl], i+1)
        writer.add_scalar("Server/Loss", loss_hist[-1][-1], i+1)
        writer.add_scalar("Server/Test", acc_hist[-1][-1], i+1)
        writer.add_scalar("Server/LR", LR, i+1)

        server_hist.append([tens_param.detach().cpu().numpy() for tens_param in list(model.parameters())])
        
        writer.flush()
    writer.close()
    # Save the results to a CSV file
    results_dict = {
        'server_train_loss': [sublist[-1] for sublist in loss_hist],
        'server_test_acc': [sublist[-1] for sublist in acc_hist],
        'server_lr': lr_hist,
    }
    for cl in range(len(weights)):
        results_dict['train_loss_'+str(cl+1)] = [sublist[cl] for sublist in loss_hist]
        results_dict['test_acc_'+str(cl+1)] = [sublist[cl] for sublist in acc_hist]
    results_df = pd.DataFrame(results_dict)
    # Check whether the specified path exists or not
    isExist = os.path.exists('./CSV/'+datasetname+'/')
    if not isExist:
   # Create a new directory because it does not exist
       os.makedirs('./CSV/'+datasetname+'/')
    results_df.to_csv('./CSV/'+datasetname+'/'+filename+'.csv', index=True)

    return model, loss_hist, acc_hist

def Main(dataset='MNIST', type='iid', shuffle=True, 
         n_clients=10, batch_size=64, global_iter=200, 
         Method={'Algo':'FedAVG', 'loc_iter': 3}, 
         Learning_rate={'lr':'Fix', 'c': 1},
         epoch_or_iter='iteration', seed=0):
    
    """ 
        - `dataset`: 'MNIST' 'FashionMNIST';
        - `type`: 'iid' 'non_iid1' 'non_iid2';
        NA- `n_samples_train`: 'All' or int; number of training samples in each client, 
                            if set 'All' then each client will have #total_train/n_clients
        NA- `n_samples_test`: 'All' or int; number of testing samples in each client, 
                            if set 'All' then each client will have #total_test/n_clients
        - `shuffle`: True False; shuffle dataset before starting the process of data allocation to each client                                                        
        - `n_clients`: int; number of clients
        - `batch_size`: int;
        - `global_iter`: int; Number of round of communication
        - `Method`: dic; 
                    Method={'Algo':'FedAVG', 'loc_iter': int, }
                    Method={'Algo':'FedProx', 'epsilon':float, 'alpha':float}
                    Method={'Algo':'EC_FedAVG', 'loc_iter': int, 'topk': int}
                    Method={'Algo':'EC_FedProx', 'epsilon':float, 'alpha':float, 'topk': int} 
                    Method={'Algo':'FedProx2', 'loc_iter': int, 'alpha':float}
                    Method={'Algo':'EC_FedProx2', 'loc_iter': int, 'alpha':float, 'topk': int} 
                    
                    - `Algo`: 'FedAVG' 'FedProx' 'EC_FedAVG' 'EC_FedProx' 'FedProx2' 'EC_FedProx2';
                    - `loc_iter`: int; Number of local update
                    - `EC`: True False; error feedback based algorithm
                    - `topK`: int; it set k in top_K compression phase of error feedback.
                    - `epsilon`: float; threshold to stop prox solver function  
                    - `alpha`: float; step-size of prox solver function
         `Method`: dic; 
                    Learning_rate={'lr':'Fix', 'c': 1}
                    Learning_rate={'lr':'Diminishing', 'c': float,  'v': float}
                    Learning_rate={'lr':'Step-decay', 'c': float,  'step': int, 'gamma': float}

                    - `lr`: 'Fix' 'Diminishing' 'Step-decay';
                    - `c`: float; \gamma_iter = \gamma = c/\sqrt{global_iter} , \gamma_iter = c/(iter+1)^{v}
                    - `v`: float; (1/2,1)
                    - `step`: int; step-decay  LR=c*(gamma**floor(iter/step))  ###\gamma_iter = c/{\sqrt{S}{(iter+1)}^\v}
                    - `gamma`: float;
    
        - `epoch_or_iter`: 'iteration' 'epoch'; if `epoch_or_iter`: 'iteration', we use just one batch with batch_size sample on each client.
                                                if `epoch_or_iter`: 'epoch', we use all batchs with batch_size sample on each client, which
                                                means n_samples_train/batch_size times update happen in each local update step
        - `seed`: int; SEED number
         
        """
    fname= dataset+'-'+type+'-batch'+str(batch_size)
    training_sets, testing_sets = get_DATA(dataset=dataset, type=type,  
                                           n_clients=n_clients, batch_size=batch_size, shuffle=shuffle, seed=seed)
    
    model, loss_hist, acc_hist = FedMain(training_sets= training_sets, testing_sets= testing_sets, Method= Method,  
                                         global_iter=global_iter, Learning_rate=Learning_rate, epoch_or_iter=epoch_or_iter, 
                                         fname= fname, seed=seed)
    return model, loss_hist, acc_hist

def maincode(args):
    Method1={'Algo':'FedAVG', 'loc_iter': 30}
    Method2={'Algo':'EC_FedAVG', 'loc_iter': 30, 'topk': 1}
    Method3={'Algo':'FedProx2', 'loc_iter': 30, 'alpha':0.1}
    Method4={'Algo':'EC_FedProx2', 'loc_iter': 30, 'alpha':0.1, 'topk': 1} 
    
    Learning_rate1={'lr':'Fix', 'c': 2}
    Learning_rate2={'lr':'Diminishing', 'c': 0.8,  'v': 0.51}
    Learning_rate3={'lr':'Step-decay', 'c': 0.8,  'step': 50, 'gamma': 0.5}
    
    Methods=[Method1, Method2, Method3, Method4]  
    Learning_rates=[Learning_rate1, Learning_rate2, Learning_rate3]    

    model, loss_hist, acc_hist = Main(dataset= args.dataset, type = args.type ,
                                                  batch_size = args.batch_size, global_iter = args.global_iter ,
                                                  Method = Methods[args.Method], 
                                                  Learning_rate = Learning_rates[args.Learning_rate],
                                                  seed = args.seed)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('--dataset', type=str, default='MNIST',
                        help='MNIST, FMNIST')
    parser.add_argument('--type', type=str, default='iid',
                        help='iid, non_iid1 non_iid2')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='')
    parser.add_argument('--global_iter', type=int, default=400,
                        help='')
    parser.add_argument('--Method', type=int, default= 0, 
                        help='')
    parser.add_argument('--Learning_rate', type=int, default= 0, 
                        help='')
    parser.add_argument('--seed', type=int, default=0,
                        help='')

args = parser.parse_args()
maincode(args)   