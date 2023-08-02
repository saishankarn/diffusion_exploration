import copy
import json
import yaml
import os
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split

import torch 
import torch.nn as nn 
import torch.functional as F 
from torch.utils.data import Dataset, DataLoader

class Sinusoidal():
    def __init__(self, eval_length=36, n_features=1, train_test_ratio=0.95):
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
        self.dataset = torch.tensor(dataset).type(torch.FloatTensor)
        self.dataset = self.dataset.unsqueeze(-1)

        indices = torch.randperm(self.dataset.size(0))
        num_training_indices = int(self.dataset.size(0)*train_test_ratio)
        train_indices = indices[:num_training_indices]
        test_indices = indices[num_training_indices:]
        self.train_dataset = self.dataset[train_indices]
        self.test_dataset = self.dataset[test_indices]

class GetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset 
    
    def __getitem__(self, org_index):
        return self.dataset[org_index]

    def __len__(self):
        return self.dataset.shape[0]
    
class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=16):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features, # 1
            hidden_size=self.hidden_dim, # 32
            num_layers=1,
            batch_first=True
        )
        
        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim, 
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        x, (_, _) = self.rnn1(x) # the shape of x is 1x140x32
        x, (hidden_n, _) = self.rnn2(x)
        flat_dim = self.rnn2.num_layers*2*self.embedding_dim if self.rnn2.bidirectional else self.rnn2.num_layers*self.embedding_dim
        return hidden_n.permute(1,0,2).view(-1,flat_dim).unsqueeze(1)
  
class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = x.repeat(1, self.seq_len, 1)
        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((-1, self.hidden_dim))
        x = self.output_layer(x)
        return x.reshape(-1, self.seq_len, self.n_features)
  
class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
    
def plot_results(model, val_dataloader, num_plots, log_dir, epoch, device):
    fig, axs = plt.subplots(
        nrows=1,
        ncols=num_plots,
        sharey=True,
        sharex=True,
        figsize=(32, 8)
    )

    model = model.eval()
    for data in val_dataloader:
        data = data.to(device)
        predictions = model(data)
        for i in range(num_plots):
            pred = predictions[i].squeeze(-1).cpu().detach().numpy()
            true = data[i].squeeze(-1).cpu().numpy()
            axs[i].plot(true, label='true')
            axs[i].plot(pred, label='reconstructed')
        break
    fig.tight_layout()

    save_dir = os.path.join(log_dir, 'qualitative')
    os.makedirs(save_dir, exist_ok=True)
    save_loc = os.path.join(save_dir, str(epoch)+'.png')
    plt.savefig(save_loc)
    plt.close('all')

  
def train_model(model, train_dataloader, val_dataloader, n_epochs, device, log_dir, num_plots=6, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000000000.0
  
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

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
        plot_results(model, val_dataloader, num_plots, log_dir, epoch, device)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            save_dir = os.path.join(log_dir, 'weights')
            os.makedirs(save_dir, exist_ok=True)
            save_loc = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_loc)        

    model.load_state_dict(best_model_wts)
    return model.eval(), history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TS-Autoencoder") 
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument("--modelfolder", type=str, default="")

    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path = "config/" + args.config 
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
    foldername = ("./save/sine" + "_" + current_time + "/")

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    horizon = config["model"]["horizon"]
    input_features = config["model"]["input_dim"]
    sinusoidal_data = Sinusoidal(eval_length=horizon, n_features=input_features)

    train_dataset = GetDataset(sinusoidal_data.train_dataset)
    test_dataset = GetDataset(sinusoidal_data.test_dataset)

    batch_size = config["train"]["batch_size"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

    latent_dim = config["model"]["latent_dim"]
    model = RecurrentAutoencoder(seq_len=horizon, n_features=input_features, embedding_dim=latent_dim)
    model = model.to(device)
    print(model)

    num_epochs = config["train"]["epochs"]
    log_dir = os.path.join(foldername, 'results')
    os.makedirs(log_dir, exist_ok=True)

    num_plots = config["train"]["num_plots"]
    lr = config["train"]["lr"]

    model, history = train_model(
        model, 
        train_dataloader, 
        val_dataloader, 
        n_epochs=num_epochs,
        device=device,
        log_dir=log_dir,
        num_plots=num_plots,
        lr=lr
    )


