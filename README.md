# FedAlgo_WO_DataSim
Implementation of **FedAlgo_WO_DataSim**, as presented in:
* On the Convergence of Federated Learning Algorithms without Data Similarity. Published in the IEEE Transactions on Big Data. [Link](https://ieeexplore.ieee.org/document/10587070)


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
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import FashionMNIST, MNIST
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import argparse
import json
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
### Training on [MNIST dataset](http://yann.lecun.com/exdb/mnist/) or [FashionMNIST dataset](https://github.com/zalandoresearch/fashion-mnist)
> The script below starts a new training process with customized settings.
```
python MainCode.py -h

usage: MainCode.py [-h] [--dataset DATASET] [--type TYPE] [--batch_size BATCH_SIZE] [--global_iter GLOBAL_ITER]
                   [--Method METHOD] [--Learning_rate LEARNING_RATE] [--seed SEED]

Federated Learning

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     MNIST, FMNIST
  --type TYPE           iid, non_iid1 non_iid2
  --batch_size BATCH_SIZE
  --global_iter GLOBAL_ITER
  --Method METHOD
               0, 1, 2, 3
               Method0={'Algo':'FedAVG', 'loc_iter': 30}
               Method1={'Algo':'EC_FedAVG', 'loc_iter': 30, 'topk': 1}
               Method2={'Algo':'FedProx2', 'loc_iter': 30, 'alpha':0.1}
               Method3={'Algo':'EC_FedProx2', 'loc_iter': 30, 'alpha':0.1, 'topk': 1} 
  --Learning_rate LEARNING_RATE
               0, 1, 2
               Learning_rate0={'lr':'Fix', 'c': 2}
               Learning_rate1={'lr':'Diminishing', 'c': 0.8,  'v': 0.51}
               Learning_rate2={'lr':'Step-decay', 'c': 0.8,  'step': 50, 'gamma': 0.5}

  --seed SEED
```

> As a result of running this code; data folder, CSV/[dataset name] folder, and tenorboard folder will be created.

* dataset will be downloaded in data folder.

* You can simultaneously monitor training progress through tensorboard by the files saved in tensorboard folder.

* Training process will be loged also in a CSV file in CSV/[dataset name] folder.


## How to Aggregate CSV Files and generate Mean and STD over different trials
> Use `AggregateCSVs.ipynb` to generate a single CSV file containing the mean and standard deviation of 5 runs on each experiment's setup.

## How to Plot the Results
> To draw output figures with the desired features use `PlotResults.ipynb`.

# Citation
* Published in the IEEE Transactions on Big Data.

Please cite the accompanied paper, if you find this useful:
```
@ARTICLE{Beikmohammadi_FedAlgo_WO_DataSim,
  author={Beikmohammadi, Ali and Khirirat, Sarit and Magnússon, Sindri},
  journal={IEEE Transactions on Big Data}, 
  title={On the Convergence of Federated Learning Algorithms Without Data Similarity}, 
  year={2024},
  volume={},
  number={},
  pages={1-10},
  keywords={Convergence;Federated learning;Stochastic processes;Optimization;Data models;Schedules;Computational modeling;Compression algorithms;federated learning;gradient methods;machine learning},
  doi={10.1109/TBDATA.2024.3423693}}
```
