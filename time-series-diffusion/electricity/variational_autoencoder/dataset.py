import os 
os.system("unset LD_LIBRARY_PATH")
import numpy as np

import torch
from torch.utils.data import Dataset

class ElectricityLoadPattern():
    def __init__(self, config):
        self.config = config 
        
        dataset_dir = self.config.dataset_dir 
        per_user_mean_dict_loc = os.path.join(dataset_dir, 'per_user_mean_dict.npy')
        per_user_std_dict_loc = os.path.join(dataset_dir, 'per_user_std_dict.npy')
        per_user_dataset_loc = os.path.join(dataset_dir, 'per_user_dataset.npy')
        per_user_labels_loc = os.path.join(dataset_dir, 'per_user_dataset_labels.npy')

        self.per_user_mean_dict = np.load(per_user_mean_dict_loc, allow_pickle=True)
        self.per_user_std_dict = np.load(per_user_std_dict_loc, allow_pickle=True)
        self.per_user_dataset = np.load(per_user_dataset_loc, allow_pickle=True)
        self.per_user_dataset = torch.tensor(self.per_user_dataset).type(torch.FloatTensor)
        self.per_user_labels = np.load(per_user_labels_loc, allow_pickle=True)

        self.horizon = self.config.horizon
        assert self.horizon == self.per_user_dataset.shape[1]

        indices = torch.randperm(self.per_user_dataset.shape[0])
        num_training_indices = int(len(indices)*self.config.train_test_ratio)

        train_indices = indices[:num_training_indices]
        test_indices = indices[num_training_indices:]
        
        self.train_data = self.per_user_dataset[train_indices]
        self.train_labels = self.per_user_labels[train_indices]
        
        self.test_data = self.per_user_dataset[test_indices]
        self.test_labels = self.per_user_labels[test_indices]

class GetDataset(Dataset):
    def __init__(self, dataset_config, train=True):
        dataset_obj = ElectricityLoadPattern(config=dataset_config)
        self.horizon = dataset_config.horizon

        if train:
            self.dataset = dataset_obj.train_data
            self.labels = dataset_obj.train_labels
        else:
            self.dataset = dataset_obj.test_data 
            self.labels = dataset_obj.test_labels
    
    def __getitem__(self, org_index):
        return self.dataset[org_index].unsqueeze(0)

    def __len__(self):
        return self.dataset.shape[0]