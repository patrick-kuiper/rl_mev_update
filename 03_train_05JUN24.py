import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import math, time
import itertools
from datetime import datetime
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import pickle
import numpy as np
import random
import pandas as pd
import os
import sys
from pylab import mpl, plt
import time
from collections import defaultdict
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import confusion_matrix
from sparsemax import Sparsemax
import functools
import math as ma

from scipy.spatial.distance import hamming
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from rl_evt_base_models import MyDataset, lstm, split_data, mlp, tensorize_list_of_tensors, get_factors, mlp_relu_noBatch, mlp_relu_layer
from tqdm import tqdm
import yaml
import argparse
from transformers import get_scheduler
from datasets import load_dataset
from transformers import EncoderDecoderModel, AutoTokenizer, TrainingArguments, Trainer, modeling_outputs
import argparse

from scipy.stats import multivariate_normal


parser = argparse.ArgumentParser(
                prog = 'EVT RL',
                description = 'Representative Learning Model for EVT.')
parser.add_argument('filename')
args = parser.parse_args()
with open(args.filename, 'r') as f:
    params = yaml.safe_load(f)
print("##################################################################################")
print("#################################MODLE PARAMS#####################################")
for key in list(params.keys()):
    print( key, ": ", params[key])
print("##################################################################################")

# Define parameters
num_channels = params["num_channels"]
num_points = params["num_points"]
num_samples = params["num_samples"] #num samples
start_samples = params["start_samples"]
end_samples = params["end_samples"] # Duration 

sigma = params["sigma"]  # noise standard deviation
mean = params["mean"]
output_data = params["output_data"]
plot_base_data = params["plot_base_data"]

hidden_dim_encode = params["hidden_dim_encode"]
output_dim_encode = params["output_dim_encode"]

input_dim_decode = params["input_dim_decode"]
hidden_dim_decode = params["hidden_dim_decode"]
output_dim_decode = params["output_dim_decode"]

num_epochs = params["num_epochs"]
learning_rate = params["learning_rate"]
batchSize = params["batchSize"]
shuffle = params["shuffle"]

rate = params["rate"]
valid_size = params["valid_size"]

plot_folder = "plots/"
data_folder = "data/"
model_folder = "./model_folder"

if not os.path.isdir(plot_folder): 
    os.mkdir(plot_folder)
if not os.path.isdir(model_folder): 
    os.mkdir(model_folder)

num_samples_train = math.floor((1 - valid_size) * num_samples)

file = open(data_folder + "{}_pts{}_mn{}_sg{}.p".format(output_data, num_points, mean, sigma), 'rb')
all_data_df = pickle.load(file)
file.close()

noisy_trig_data = all_data_df['noisy_trig_data'].T
no_noisy_trig_data = all_data_df['no_noisy_trig_data'].T
t = all_data_df['domain'].T
max_data = all_data_df['interval_data']

print("noisy_trig_data.shape: ", noisy_trig_data.shape)

#this provides the max seperated by the periods measured in radians
max_intervals = [noisy_trig_data[:, (t > max_data[i-1]) & 
                                 (t < max_data[i]), :].max(0) for i 
                 in range(1, len(max_data))]


#Define models
traindata = MyDataset(noisy_trig_data[:,:num_samples_train], noisy_trig_data[:,:num_samples_train])
trainloader = torch.utils.data.DataLoader(traindata, batch_size = batchSize, shuffle = True)

model_encode_min = mlp_relu_layer(input_dim = num_samples_train, 
                    hidden_dim = hidden_dim_encode, 
                    output_dim = output_dim_encode).to(device)

model_decode_min = mlp_relu_layer(input_dim = input_dim_decode, 
                    hidden_dim = hidden_dim_decode, 
                    output_dim = num_samples_train).to(device)

num_training_steps = num_epochs * len(trainloader)
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(list(model_encode_min.parameters()) + list(model_decode_min.parameters()), lr = learning_rate)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

#Create random block max sizes
batchSize_factors = get_factors(batchSize) #this function gets common denominators for block sizes
num_batchtes = len(trainloader)
random_blocksize_list = random.choices(batchSize_factors, k = num_batchtes)

loss_hist = []
encode_data = []
display_debug = False
for t in tqdm(range(num_epochs)):
    for i_batch, (x_batch, y_batch) in enumerate(trainloader):
        
        x_batch = x_batch.reshape(batchSize, -1)
        y_batch = y_batch.reshape(batchSize, -1)
        blockSize = random_blocksize_list[i_batch] 
        blockFact = int(batchSize/blockSize) #blockFact is the number of blocks in the batch
        y_batch_reshape = y_batch.float().to(device).reshape(blockFact, blockSize, -1)
        x_batch_reshape = x_batch.float().to(device).reshape(blockFact, blockSize, -1)
        
        random_radial_list = torch.tensor(np.random.exponential(scale = 1 / rate, size = (blockFact)))
        A_n = torch.cumsum(random_radial_list, dim=0).view(blockFact, 1).float().to(device) # (blockFact x 1) - each block gets unique A_n

         
        x_encode = model_encode_min(x_batch_reshape)
        
        block_min_ambient = y_batch_reshape.max(1)[0] #(blockFact x num_samples) get min over each
        x_encode_min = torch.max(x_encode, dim = 1)[0]
        
        x_encode_min_scaled = x_encode_min - torch.log(A_n)
        x_decode_min_scaled = model_decode_min(x_encode_min_scaled)
        loss_extreme = criterion(x_decode_min_scaled, block_min_ambient)

        if display_debug: 
            for i in range(batchSize):
                plt.plot(x_batch[i, :], color = "b", alpha = 0.25)
                plt.plot(y_batch[i, :], color = "g", alpha = 0.25)
            for i in range(blockFact):
                plt.plot(block_min_ambient[i, :], color = "r", alpha = 0.5)
            plt.plot(no_noisy_trig_data[0, :, 0], label = "no_noisy_trig_data")
            plt.legend()
            plt.savefig("x_y_batch_data_debug_plot.pdf")
            
            print("blockFact, blockSize: ", blockFact, blockSize) 
            print("x_batch, y_batch, A_n:", x_batch.shape, y_batch.shape, A_n.shape)
            print("x_batch_reshape, y_batch_reshape, A_n:", x_batch_reshape.shape, y_batch_reshape.shape, A_n.shape)
            print("block_min_ambient.shape: ",  block_min_ambient.shape)
            print("x_encode: ", x_encode.shape)
            print("x_encode_max: ", x_encode_min.shape)
            print("x_encode_scaled: ", x_encode_min_scaled.shape)
            print("loss_extreme: ", loss_extreme)
            assert False
        optimizer.zero_grad()
        loss_extreme.backward()
        optimizer.step()
        lr_scheduler.step()
    torch.save(model_encode_min.state_dict(), model_folder + "/model_encode_min_bs{}_rt{}_sig{}_ep{}_hd{}.pt".format(batchSize, rate, sigma, num_epochs, output_dim_encode))
    torch.save(model_decode_min.state_dict(), model_folder + "/model_decode_min_bs{}_rt{}_sig{}_ep{}_hd{}.pt".format(batchSize, rate, sigma, num_epochs, output_dim_encode))
        
    loss_hist.append(loss_extreme)
    
with open(data_folder + "Loss_data_pts{}_mn{}_sg{}.p".format(num_points, mean, sigma), 'wb') as f: 
    pickle.dump(tensorize_list_of_tensors(loss_hist), f)
    
torch.save(model_encode_min.state_dict(), model_folder + "/model_encode_min_bs{}_rt{}_sig{}_ep{}_hd{}.pt".format(batchSize, rate, sigma, num_epochs, output_dim_encode))
torch.save(model_decode_min.state_dict(), model_folder + "/model_decode_min_bs{}_rt{}_sig{}_ep{}_hd{}.pt".format(batchSize, rate, sigma, num_epochs, output_dim_encode))
        
        
plt.plot(torch.log(tensorize_list_of_tensors(loss_hist)))
plt.savefig(plot_folder + "loss_bs{}_rt{}_sig{}_ep{}_hd{}.pdf".format(batchSize, rate, sigma, num_epochs, output_dim_encode))
plt.close()

all_input_data_encode = model_encode_min(torch.tensor(noisy_trig_data[:,:num_samples_train].reshape(-1,num_samples_train)).float().to(device))
random_radial_list = torch.tensor(np.random.exponential(scale = 1 / rate, 
                                                        size = (all_input_data_encode.shape[0])))
A_n = torch.cumsum(random_radial_list, dim=0).view(-1, 1).float().to(device)
all_input_data_encode = all_input_data_encode.to(device) - torch.log(A_n)

#Create random block max sizes
batchSize_factors = get_factors(batchSize) #this function gets common denominators for block sizes
num_batchtes = ma.floor(all_input_data_encode.shape[0] / batchSize)
random_blocksize_list = random.choices(batchSize_factors, k = num_batchtes)

data_block_out = {}
for i, blockSize in enumerate(batchSize_factors[1:]):
    
    all_input_data_encode_reshape = all_input_data_encode.reshape(-1, blockSize, input_dim_decode)

    min_values = all_input_data_encode_reshape.max(1)[0]
    out_values_min = model_decode_min(min_values)
    print(out_values_min.shape)
    data_block_out[blockSize] = out_values_min

print(batchSize_factors)
for i in batchSize_factors[1:]:
    plt.plot(data_block_out[i].mean(0).cpu().detach().numpy(), label = "{}".format(i))
plt.legend()
plt.savefig(plot_folder + "compare1_bs{}_rt{}_sig{}_ep{}_hd{}.pdf".format(batchSize, rate, sigma, num_epochs, output_dim_encode))
plt.close()

for i in batchSize_factors[1:]:
    plt.plot(data_block_out[i].mean(0).cpu().detach().numpy(), label = "{}".format(i))
plt.legend()
plt.plot(noisy_trig_data[:,num_samples_train:].reshape(-1,num_samples - num_samples_train).max(0), color = "r")
plt.savefig(plot_folder + "compare2_bs{}_rt{}_sig{}_ep{}_hd{}.pdf".format(batchSize, rate, sigma, num_epochs, output_dim_encode))
plt.close()