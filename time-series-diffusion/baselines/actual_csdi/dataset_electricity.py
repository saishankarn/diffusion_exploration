import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch


class Electricity_Dataset(Dataset):
    def __init__(self, eval_length=36, target_dim=1, mode="train", validindex=0):
        self.eval_length = eval_length
        self.target_dim = target_dim

        num_freqs = 20
        time_steps = 1000
        freqs = list(np.linspace(0.05, 1, num_freqs))
        time = np.linspace(0, 2*np.pi, time_steps)
        data = [np.sin(2 * np.pi * freq * time) for freq in freqs]
        data = np.array(data)
        intervals = [(0+i,eval_length+i) for i in range(0,time_steps-eval_length-1)]
        dataset = np.zeros((len(intervals)*num_freqs, eval_length))

        for i in range(num_freqs):
            orig_dataset = data[i]
            for j in range(len(intervals)):
                min_ = intervals[j][0]
                max_ = intervals[j][1]

                idx = i*len(intervals)+j
                dataset[idx] = orig_dataset[min_:max_]
        self.dataset = torch.tensor(dataset)
        self.dataset = self.dataset.unsqueeze(1)

    def __getitem__(self, org_index):
        ts = self.dataset[org_index]
        return ts

    def __len__(self):
        return self.dataset.shape[0]


def get_dataloader(batch_size, device, validindex=0):
    dataset = Electricity_Dataset(mode="train", validindex=validindex)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=True
    )

    return train_loader