import matplotlib.pyplot as plt
import numpy as np
import math as ma
import pickle
import argparse
from tqdm import tqdm
import yaml
import os

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
end_samples = params["end_samples"] * np.pi # Duration 
freq = params["freq"] #frequency
sigma = params["sigma"]  # noise standard deviation
mean = params["mean"]
output_data = params["output_data"]
plot_base_data = params["plot_base_data"]
batchSize = params["batchSize"]
rate = params["rate"]

plot_folder = "plots/"

data_folder = "data/"

if not os.path.isdir(data_folder): 
    os.mkdir(data_folder)

if not os.path.isdir(plot_folder): 
    os.mkdir(plot_folder)

dim = (num_channels, num_samples, num_points)

# Generate time vector
t = np.linspace(start_samples, end_samples, num_samples)
trig_data = np.log(t).reshape(1, -1) * np.sin(t).reshape(1, -1)
repeat_trig_data = np.repeat(trig_data[...,np.newaxis], num_points, axis=2)
# noise = np.random.lognormal(mean = mean, sigma = sigma, size = dim)
noise = np.random.lognormal(mean = mean, sigma = sigma, size = (num_channels, 1, num_points))
noisy_trig_data = noise * repeat_trig_data
no_noisy_trig_data = repeat_trig_data

interval_data = []
for j in range(ma.ceil(end_samples / ma.pi)):
    interval_data.append(j * np.pi)




# # Plot the results
if plot_base_data:
    
    plt.figure(figsize = (15, 5))
    for j in interval_data:
        plt.axvline(x = j, color = 'r')
    plt.plot(t, noisy_trig_data.max(2)[0])
    plt.savefig(plot_folder + "Sine_pts{}_mn{}_sg{}_plot_max.pdf".format(num_points, mean, sigma))
    
    plt.figure(figsize = (15, 5))
    plot_points = 10 #plot first 10 points
    for i in range(plot_points):
        plt.plot(t, no_noisy_trig_data[0,:,i], label='Noisy Sine Normal Data for point: {}'.format(i))
    for j in interval_data:
        plt.axvline(x = j, color = 'r')
    plt.axvline(x = end_samples, color = 'r')
    interval_data.append(end_samples)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Normal Data Noisy Sine Wave Generation')
    plt.grid(True)
    plt.savefig(plot_folder + "Sine_pts{}_mn{}_sg{}_plot_no_noise.png".format(num_points, mean, sigma))
    plt.close()
    
    plt.figure(figsize = (15, 5))
    plot_points = 10 #plot first 10 points
    for i in range(plot_points):
        plt.plot(t, noisy_trig_data[0,:,i], label='Noisy Sine Normal Data for point: {}'.format(i))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Normal Data Noisy Sine Wave Generation')
    plt.grid(True)
    plt.savefig(plot_folder + "Sine_pts{}_mn{}_sg{}_plot_noise.png".format(num_points, mean, sigma))
    plt.close()
    
with open(data_folder + "{}_pts{}_mn{}_sg{}.p".format(output_data, num_points, mean, sigma), 'wb') as f: 
    pickle.dump({"noisy_trig_data": noisy_trig_data, "no_noisy_trig_data": no_noisy_trig_data, "domain": t, "interval_data": interval_data}, f)
    
