from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torch.nn as nn
import copy

from time_series_encoder import RecurrentAutoencoder

class Sinusoidal():
    def __init__(self, eval_length=36, n_features=1, train_test_ratio=0.9):
        self.eval_length = eval_length
        self.n_features = n_features 

        num_freqs = 100
        time_steps = 1000
        freqs = list(np.linspace(1, 20, num_freqs))
        time = np.linspace(0, 2*np.pi, time_steps)

        data = [np.sin(2 * np.pi * freq * time) for freq in freqs]
        data = np.array(data)
        intervals = [(0+i,eval_length+i) for i in range(0,time_steps-eval_length)]
        dataset = np.zeros((len(intervals)*num_freqs, eval_length))

        for i in range(num_freqs):
            orig_dataset = data[i] 
            for j in range(len(intervals)):
                min_ = intervals[j][0]
                max_ = intervals[j][1]
 
                idx = i*len(intervals)+j
                dataset[idx] = orig_dataset[min_:max_]
        self.dataset = torch.tensor(dataset)
        self.dataset = self.dataset.unsqueeze(-1)

        indices = torch.randperm(self.dataset.size(0))
        num_training_indices = int(self.dataset.size(0)*train_test_ratio)
        train_indices = indices[:num_training_indices]
        test_indices = indices[num_training_indices:]
        self.train_dataset = self.dataset[train_indices]
        self.test_dataset = self.dataset[test_indices]

horizon = 48
input_features = 1
sinusoidal_data = Sinusoidal(eval_length=horizon, n_features=input_features)

class GetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset 
    
    def __getitem__(self, org_index):
        return self.dataset[org_index]

    def __len__(self):
        return self.dataset.shape[0]
    
train_dataset = GetDataset(sinusoidal_data.train_dataset)
test_dataset = GetDataset(sinusoidal_data.test_dataset)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

model = RecurrentAutoencoder(horizon, input_features, 8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def train_model(model, train_dataloader, val_dataloader, n_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0
    
    for epoch in range(1, n_epochs + 1):
        model = model.train()

        train_losses = []
        for seq_true in train_dataloader:
            optimizer.zero_grad()

            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataloader:

                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)

                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    model.load_state_dict(best_model_wts)
    return model.eval(), history

model, history = train_model(model, train_dataloader, test_dataloader, 200)