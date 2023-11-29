# FedAlgo_WO_DataSim
Implementation of **FedAlgo_WO_DataSim**, as presented in:
* On the Convergence of Federated Learning Algorithms without Data Similarity. Submitted to the IEEE Transactions on Big Data.


# Numerical Evaluations on Deep Neural Networks
We included (additional) experiments from running FedAvg, FedProx, error-feedback FedAvg, and error-feedback FedAvg with fixed, diminishing, and step-decay step sizes over the MNIST dataset and FashionMNIST datasets. 
We reported the average and standard deviation of training loss and test accuracy from running the algorithms with diminishing step sizes in Figures 1 and 2, with diminishing step sizes in Figures 3 and 4, and with the step-decay step sizes in Figures 5 and 6.
The shaded regions correspond to the standard deviation of the average evaluation over five trials.


1. (Figure 1) Performance of FedAvg, error-feedback FedAvg, FedProx, and error-feedback FedProx with the fixed step size in (left plots -) training loss and (right plots -) test accuracy on MNIST dataset considering three different partitioned data among the workers.

![Figure 1](https://github.com/AliBeikmohammadi/FedAlgo_WO_DataSim/blob/main/Plots_Paper/MNIST-Fix.png)
   
2. (Figure 2) Performance of FedAvg, error-feedback FedAvg, FedProx, and error-feedback FedProx with the fixed step size in (left plots -) training loss and (right plots -) test accuracy on FashionMNIST dataset considering three different partitioned data among the workers.

![Figure 2](https://github.com/AliBeikmohammadi/FedAlgo_WO_DataSim/blob/main/Plots_Paper/FMNIST-Fix.png)
  
3. (Figure 3) Performance of FedAvg, error-feedback FedAvg, FedProx, and error-feedback FedProx with the diminishing step size in (left plots -) training loss and (right plots -) test accuracy on MNIST dataset considering three different partitioned data among the workers.

![Figure 3](https://github.com/AliBeikmohammadi/FedAlgo_WO_DataSim/blob/main/Plots_Paper/MNIST-Diminishing.png)
   
4. (Figure 4) Performance of FedAvg, error-feedback FedAvg, FedProx, and error-feedback FedProx with the diminishing step size in (left plots -) training loss and (right plots -) test accuracy on FashionMNIST dataset considering three different partitioned data among the workers.

![Figure 4](https://github.com/AliBeikmohammadi/FedAlgo_WO_DataSim/blob/main/Plots_Paper/FMNIST-Diminishing.png)
   
5. (Figure 5) Performance of FedAvg, error-feedback FedAvg, FedProx, and error-feedback FedProx with the step-decay step size in (left plots -) training loss and (right plots -) test accuracy on MNIST dataset considering three different partitioned data among the workers.

![Figure 5](https://github.com/AliBeikmohammadi/FedAlgo_WO_DataSim/blob/main/Plots_Paper/MNIST-Step-decay.png)
    
6. (Figure 6) Performance of FedAvg, error-feedback FedAvg, FedProx, and error-feedback FedProx with the step-decay step size in (left plots -) training loss and (right plots -) test accuracy on FashionMNIST dataset considering three different partitioned data among the workers.

![Figure 6](https://github.com/AliBeikmohammadi/FedAlgo_WO_DataSim/blob/main/Plots_Paper/FMNIST-Step-decay.png)   
  



# Importing

> To run a new test .
```
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import transforms
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import os
```
> To aggregate CSV files.
```
import glob
import pandas as pd
import os
```
> To draw output figures.
```
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
```


# Usage
## How to Run Experiments
### Training a FCNN on [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
> The script below starts a new training process on the MNIST dataset with customized settings.
```
python MNIST_SGDM_v1.py -h

usage: MNIST_SGDM_v1.py [-h] [--seed_number SEED_NUMBER]
                        [--learning_rate LEARNING_RATE] [--beta BETA]
                        [--L2 L2] [--num_epochs NUM_EPOCHS]
                        [--num_nodes NUM_NODES] [--method METHOD]
                        [--max_norm MAX_NORM] [--top_k TOP_K]
                        [--clip_value CLIP_VALUE]

Train a neural network on MNIST

optional arguments:
  -h, --help            show this help message and exit
  --seed_number SEED_NUMBER
                        seed number
  --learning_rate LEARNING_RATE
                        learning rate for the optimizer
  --beta BETA           beta for the optimizer
  --L2 L2               weight_decay (L2 penalty)
  --num_epochs NUM_EPOCHS
                        number of epochs to train
  --num_nodes NUM_NODES
                        number of nodes
  --method METHOD       method: none, norm, clip_grad_norm, clip_grad_value,
                        Top-K
  --max_norm MAX_NORM   gradient norm clipping max value (necessary if method:
                        clip_grad_norm)
  --top_k TOP_K         number of top elements to keep in compressed gradient
                        (necessary if method: Top-K)
  --clip_value CLIP_VALUE
                        gradient clipping value (necessary if method:
                        clip_grad_value)
```
> Set --beta 1 in case you need to train DistributedSGD instead of DistributedSGDM.
> 
> As a result of running this code; data folder, MNIST_CSV folder , and runs folder will be created.

* dataset will be downloaded in data folder.

* You can simultaneously monitor training progress through tensorboard by the files saved in runs folder.

* Training process will be loged also in a CSV file in MNIST_CSV folder.


### Training a ResNet-18 on [FashionMNIST dataset](https://github.com/zalandoresearch/fashion-mnist)
> The script below starts a new training process on the FashionMNIST dataset with customized settings.
```
python FashionMNIST_ResNet18_SGDM_v1.py -h

usage: FashionMNIST_ResNet18_SGDM_v1.py [-h] [--seed_number SEED_NUMBER]
                                        [--learning_rate LEARNING_RATE]
                                        [--beta BETA] [--L2 L2]
                                        [--num_epochs NUM_EPOCHS]
                                        [--num_nodes NUM_NODES]
                                        [--method METHOD]
                                        [--max_norm MAX_NORM] [--top_k TOP_K]
                                        [--clip_value CLIP_VALUE]

Train a neural network on FashionMNIST ResNet18

optional arguments:
  -h, --help            show this help message and exit
  --seed_number SEED_NUMBER
                        seed number
  --learning_rate LEARNING_RATE
                        learning rate for the optimizer
  --beta BETA           beta for the optimizer
  --L2 L2               weight_decay (L2 penalty)
  --num_epochs NUM_EPOCHS
                        number of epochs to train
  --num_nodes NUM_NODES
                        number of nodes
  --method METHOD       method: none, norm, clip_grad_norm, clip_grad_value,
                        Top-K
  --max_norm MAX_NORM   gradient norm clipping max value (necessary if method:
                        clip_grad_norm)
  --top_k TOP_K         number of top elements to keep in compressed gradient
                        (necessary if method: Top-K)
  --clip_value CLIP_VALUE
                        gradient clipping value (necessary if method:
                        clip_grad_value)
```
> Set --beta 1 in case you need to train DistributedSGD instead of DistributedSGDM.
> 
> As a result of running this code; data folder, FashionMNIST_ResNet18_CSV folder , and runs folder will be created.
* dataset will be downloaded in data folder.
* You can simultaneously monitor training progress through tensorboard by the files saved in runs folder.
* Training process will be loged also in a CSV file in FashionMNIST_ResNet18_CSV folder.

## How to Aggregate CSV Files and generate Mean and STD over different trials
> Use `aggregateCSVs.ipynb` to generate a single CSV file containing the mean and standard deviation of 5 runs on each experiment's setup.

## How to Plot the Results
> To draw output figures with the desired features use `PlotResults.ipynb`.


# Citation
* Submitted to the IEEE Transactions on Signal Processing.

Please cite the accompanied paper, if you find this useful:
```
To be completed
```
