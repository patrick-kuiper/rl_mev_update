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
import pandas as pd
import os
from pylab import mpl, plt
import time
from collections import defaultdict
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import confusion_matrix
from sparsemax import Sparsemax
import functools
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

from scipy.spatial.distance import hamming
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def datetime_range(start=None, end=None):
    span = end - start
    for i in range(span.days + 1):
        yield start + timedelta(days=i)

def make_labels_and_features_new(data_raw, lookback):
    data = []
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    out_data = tensorize_list_of_tensors(data)
    out_data_label = out_data[:,-1,-1] #get the last entry, last column for label
#     out_data_label_mean = out_data_label.mean(1)
    out_data_features = out_data[:,:,:-1]
    return out_data_features, out_data_label

def make_labels_and_features(data_raw, lookback):
    data = []
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback])
    out_data = tensorize_list_of_tensors(data)
    out_data_label = out_data[:,:,0]
    out_data_label_mean = out_data_label.mean(1)
    out_data_features = out_data[:,:,1:]
    return out_data_features, out_data_label_mean

def get_factors(x):
    factors = []
    for i in range(1, x + 1):
        if x % i == 0:
            factors.append(i)
    return factors

def tensorize_list_of_tensors(list_of_tensors):
    if type(list_of_tensors[0]) != np.ndarray:
        tensorize_data = np.array([tensor.cpu().detach().numpy() for tensor in list_of_tensors])
        tensorized_data = torch.from_numpy(tensorize_data).type(torch.Tensor)
    elif type(list_of_tensors[0]) == np.ndarray:
        tensorize_data = np.array(list_of_tensors)
        tensorized_data = torch.from_numpy(tensorize_data).type(torch.Tensor)
    else:
        tensorize_data = np.array([tensor.cpu().detach().numpy() for tensor in list_of_tensors])
        tensorized_data = torch.from_numpy(tensorize_data).type(torch.Tensor)
    return tensorized_data

class MyDataset(Dataset):
    
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0] # assuming shape[0] = dataset size
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    
class mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fco = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fco(x)
        return x
    
class mlp_relu_batch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp_relu_batch, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
#         self.relu = nn.ReLU(self.hidden_dim)
        self.relu = nn.LeakyReLU(self.hidden_dim)
        self.batchnorm = nn.BatchNorm1d(self.hidden_dim)
        self.fco = nn.Linear(self.hidden_dim, self.output_dim)
        
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.batchnorm(x)
        x = self.fco(x)
        return x
    
class mlp_relu_layer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp_relu_layer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
#         self.relu = nn.ReLU(self.hidden_dim)
        self.relu = nn.LeakyReLU(self.hidden_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.fco = nn.Linear(self.hidden_dim, self.output_dim)
        
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.layer_norm(x)
        x = self.fco(x)
        return x
    
    
    
class mlp_relu_noBatch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp_relu_noBatch, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
#         self.relu = nn.ReLU(self.hidden_dim)
        self.relu = nn.LeakyReLU(self.hidden_dim)
        self.fco = nn.Linear(self.hidden_dim, self.output_dim)
        
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fco(x)
        return x
    
class mlp_batch(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mlp_batch, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.batchnorm = nn.BatchNorm1d(self.hidden_dim)
        self.fco = nn.Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.batchnorm(x)
        x = self.fco(x)
        return x
    
class lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(lstm, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        h0 = torch.zeros(self.num_layers, self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out_final = self.fc(out[:, -1, :]) 
        return out_final
    
    
def split_data(data_raw, lookback, factor):
    data = []
    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback): 
        data.append(data_raw[index: index + lookback + 1])  
    
    data = np.array(data);
    test_set_size = int(np.round(factor*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]
    
