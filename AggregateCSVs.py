#!/usr/bin/env python
# coding: utf-8

import glob
import pandas as pd
import os

# set the path where the CSV files are stored
#load_path = './CSV/MNIST/'
#save_path = './CSV_Aggregate/MNIST/'

load_path = './CSV/FMNIST/'
save_path = './CSV_Aggregate/FMNIST/'

# create a list of all CSV files in the folder
csv_files = glob.glob(load_path + '*.csv')

# loop over the CSV files and group them by hyperparameters
file_name_list=[]
for csv_file in csv_files:
    # extract hyperparameters from file name
    file_name = os.path.basename(csv_file)
    file_name_wo_seed = file_name.split('-seed')[0]
    file_name_list.append(file_name_wo_seed) if file_name_wo_seed not in file_name_list else file_name_list
    
# loop over the CSV files and group them by hyperparameters
for i in file_name_list:
    sameseed_files = glob.glob(load_path+i+ '*.csv')
    # Load each CSV file into a dataframe and store it in a list
    dfs = [pd.read_csv(filename) for filename in sameseed_files]
    # Concatenate the dataframes into a single dataframe
    df = pd.concat(dfs)
    # Use groupby to calculate the average and standard deviation for each metric
    aggregated_df = df.groupby('Unnamed: 0').agg({'server_train_loss': ['mean', 'std'],
                                             'server_test_acc': ['mean', 'std'],
                                             'server_lr': ['mean', 'std']})
    # Flatten the multi-level column index
    aggregated_df.columns = ['_'.join(col).strip() for col in aggregated_df.columns.values]
    # Reset the index to make the epoch column a regular column
    aggregated_df = aggregated_df.reset_index()
    # Save the aggregated dataframe to a new CSV file
    isExist = os.path.exists(save_path)
    if not isExist:
   # Create a new directory because it does not exist
       os.makedirs(save_path)
    aggregated_df.to_csv(save_path+i+'.csv', index=False)

