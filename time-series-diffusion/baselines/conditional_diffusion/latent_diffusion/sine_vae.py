import copy
import json
import yaml
import os
import datetime
import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn 
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
    def __init__(self, seq_len, n_features, embedding_dim=16, num_layers=4, bidirectional=True):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.flat_dim = self.num_layers*2*self.embedding_dim if self.bidirectional else self.num_layers*self.embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features, # 1
            hidden_size=32, # 32
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )
        
        self.mean = nn.LSTM(
            input_size=32*2, 
            hidden_size=embedding_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )

        self.var = nn.LSTM(
            input_size=32*2, 
            hidden_size=embedding_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional
        )

    def forward(self, x):
        x, (_, _) = self.rnn1(x) # the shape of x is 1x140x32
        _, (mu, _) = self.mean(x)
        _, (sigma, _) = self.var(x)
        mu = mu.permute(1,0,2).reshape(-1, self.flat_dim)
        sigma = sigma.permute(1,0,2).reshape(-1, self.flat_dim)
        return mu, sigma
  
class Decoder(nn.Module):
    def __init__(self, seq_len, embedding_dim=64, n_features=1, num_layers=4, bidirectional=True):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.bidirectional = bidirectional 
        self.input_size = self.num_layers*2*embedding_dim if self.bidirectional else self.num_layers*embedding_dim
        self.seq_len = seq_len
        self.output_features = embedding_dim*2 if self.bidirectional else embedding_dim
        self.n_features = n_features

        self.rnn1 = nn.LSTM(
            input_size=self.input_size,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.rnn2 = nn.LSTM(
            input_size=32*2,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=self.bidirectional
        )

        self.rnn3 = nn.LSTM(
            input_size=embedding_dim*2,
            hidden_size=1,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.unsqueeze(1)
        x = x.repeat(1, self.seq_len, 1)
        x, (_, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)
        x, (_, _) = self.rnn3(x)
        return x
  
class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()
        print(embedding_dim)
        self.num_layers = 4
        self.bidirectional = True 
        self.encoder = Encoder(seq_len, n_features, embedding_dim, self.num_layers, self.bidirectional)
        self.decoder = Decoder(seq_len, embedding_dim, n_features, self.num_layers, self.bidirectional)

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)        
        z = mean + var*epsilon
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterization(mu, torch.exp(0.5 * log_var))
        x = self.decoder(z)
        return x, mu, log_var
    
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
        predictions, _, _ = model(data)
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

def loss_function(x, x_hat, mean, log_var, criterion):
    reproduction_loss = criterion(x_hat, x)
    KLD = - 0.5 * torch.mean(1+ log_var - mean.pow(2) - log_var.exp())
    print(reproduction_loss, KLD)
    return reproduction_loss, KLD

def train_model(model, train_dataloader, val_dataloader, n_epochs, device, log_dir, num_plots=6, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000000000.0
  
    for epoch in range(10, n_epochs + 1):
        model = model.train()

        train_losses = []
        for seq_true in train_dataloader:
            optimizer.zero_grad()

            seq_true = seq_true.to(device)
            seq_pred, mu, log_var = model(seq_true)
            reconstruction_loss, kldivergence_loss = loss_function(seq_true, seq_pred, mu, log_var, criterion)
            if (epoch // 10) % 2:
                loss = reconstruction_loss + 10 * kldivergence_loss
            else:
                loss = reconstruction_loss + 10 * kldivergence_loss
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for seq_true in val_dataloader:
                seq_true = seq_true.to(device)
                seq_pred, mu, log_var = model(seq_true)

                reconstruction_loss, kld_loss = loss_function(seq_true, seq_pred, mu, log_var, criterion)
                val_loss = reconstruction_loss + kld_loss
                val_losses.append(val_loss.item())

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

    pretrained_weights_loc = config["train"]["pretrained_loc"]
    if pretrained_weights_loc != '':
        model.load_state_dict(torch.load(pretrained_weights_loc))

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


