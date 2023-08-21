import os 
os.system("unset LD_LIBRARY_PATH")
import numpy as np
from itertools import product

import torch
from torch.utils.data import Dataset

class DatasetConfig():
    def __init__(self):
        self.horizon = 48
        self.start_time = 0
        self.end_time = 2*np.pi 

        self.minimum_amplitude = 0.1
        self.maximum_amplitude = 1.0 
        self.amplitude_interval = 0.05
        self.amplitudes = list(np.arange(self.minimum_amplitude, self.maximum_amplitude + self.amplitude_interval, self.amplitude_interval))
    
        self.minimum_frequency = 0.1
        self.maximum_frequency = 1 
        self.frequency_interval = 0.05
        self.frequencies = list(np.arange(self.minimum_frequency, self.maximum_frequency + self.frequency_interval, self.frequency_interval))

        self.minimum_phase = 0.0 
        self.maximum_phase = 2*np.pi
        self.phase_interval = 5*np.pi/180
        self.phases = list(np.arange(self.minimum_phase, self.maximum_phase, self.phase_interval))

        self.sine_params = list(product(self.amplitudes, self.frequencies, self.phases))

        self.train_test_ratio = 0.95

class Sinusoidal():
    def __init__(self, config):
        self.config = config

        time_stamps = np.linspace(self.config.start_time, self.config.end_time, self.config.horizon)
        data = [params[0]*np.sin(2*np.pi*params[1]*time_stamps + params[2]) for params in self.config.sine_params]
        
        self.data = np.array(data)
        self.labels = np.array([np.array(params) for params in self.config.sine_params])

        self.data = torch.tensor(self.data).type(torch.FloatTensor)
        self.data = self.data.unsqueeze(1)
        self.labels = torch.tensor(self.labels).type(torch.FloatTensor)

        indices = torch.randperm(self.data.size(0))
        num_training_indices = int(self.data.size(0)*self.config.train_test_ratio)
        train_indices = indices[:num_training_indices]
        test_indices = indices[num_training_indices:]
        self.train_data = self.data[train_indices]
        self.train_labels = self.labels[train_indices]
        self.test_data = self.data[test_indices]
        self.test_labels = self.labels[test_indices]

def generate_dataset_obj(dataset_config):
    sine_dataset_obj = Sinusoidal(config=dataset_config)
    return sine_dataset_obj

class GetDataset(Dataset):
    def __init__(self, dataset_obj, dataset_config, train=True):
        self.horizon = dataset_config.horizon
        if train:
            self.dataset = dataset_obj.train_data
            self.labels = dataset_obj.train_labels
        else:
            self.dataset = dataset_obj.test_data 
            self.labels = dataset_obj.test_labels
    
    def __getitem__(self, org_index):
        return {"observed_data": self.dataset[org_index], "labels": self.labels[org_index], 'timepoints': torch.arange(self.horizon)}

    def __len__(self):
        return self.dataset.shape[0]