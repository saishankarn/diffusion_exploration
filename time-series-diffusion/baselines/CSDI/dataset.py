import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import os


class Electricity_Dataset(Dataset):
    def __init__(self, dataset_path):
        # self.dataset_path = dataset_path
        # self.dataset = np.load(self.dataset_path, allow_pickle=True)
        # self.dataset = torch.from_numpy(self.dataset).type(torch.FloatTensor)
        # self.dataset = self.dataset.unsqueeze(-1).permute(0,2,1)

        # self.dataset = np.random.normal(size=(10000,2))
        # self.dataset = torch.from_numpy(self.dataset).type(torch.FloatTensor)
        # self.dataset = self.dataset.unsqueeze(-1).permute(0,2,1)

        freqs = list(np.linspace(1, 10, 10))
        time = np.linspace(0, 2*np.pi, 1000)
        data = [np.sin(2 * np.pi * freq * time) for freq in freqs]
        data = np.array(data)
        size = 96
        intervals = [(0+i,size+i) for i in range(0,1000-size)]
        dataset = np.zeros((len(intervals)*10, size))
        for i in range(10):
            orig_dataset = data[i]
            for j in range(len(intervals)):
                min_ = intervals[j][0]
                max_ = intervals[j][1]
                idx = i*len(intervals)+j
                dataset[idx] = orig_dataset[min_:max_]

        self.dataset = dataset
        self.dataset = torch.from_numpy(self.dataset).type(torch.FloatTensor)
        self.dataset = self.dataset.unsqueeze(-1).permute(0,2,1)

    def __getitem__(self, org_index):
        return self.dataset[org_index]#.unsqueeze(1).permute(1,0)

    def __len__(self):
        return self.dataset.shape[0] 


def get_dataloader_electricity(dataset_path, batch_size):
    dataset = Electricity_Dataset(dataset_path=dataset_path)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=True
    )

    return train_loader