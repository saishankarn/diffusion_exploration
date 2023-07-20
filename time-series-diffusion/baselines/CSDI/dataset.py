import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import os


class Electricity_Dataset(Dataset):
    def __init__(self, dataset_path, ts_dim=96):
        self.ts_dim = ts_dim
        self.dataset_path = dataset_path
        self.dataset = np.load(self.dataset_path, allow_pickle=True)
        self.dataset = torch.from_numpy(self.dataset)

    def __getitem__(self, org_index):
        return self.dataset[org_index]

    def __len__(self):
        return self.dataset.shape[0]


def get_dataloader_electricity(dataset_path, ts_dim, batch_size, device):
    dataset = Electricity_Dataset(dataset_path=dataset_path, ts_dim=ts_dim)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=1, shuffle=True
    )

    return train_loader