import os 
from tqdm import tqdm
os.system("unset LD_LIBRARY_PATH")
import numpy as np
from itertools import product

import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader   

from main_model import CSDI_base

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

    
class GetDataset(Dataset):
    def __init__(self, data, labels):
        self.dataset = data
        self.labels = labels
        self.horizon = data.shape[-1]
    
    def __getitem__(self, org_index):
        return {"observed_data": self.dataset[org_index], "labels": self.labels[org_index], 'timepoints': torch.arange(self.horizon)}

    def __len__(self):
        return self.dataset.shape[0]
    
class CLTSPConfig():
    def __init__(self):
        # train config parameters
        self.batch_size = 64
        self.n_epochs = 2000
        self.learning_rate = 1e-3
        self.n_plots = 6

        # encoder parameters
        self.n_features = 1
        self.dim_val = 128
        self.dropout_pos_enc = 0.2 
        self.max_seq_len = 48
        self.n_heads = 8
        self.n_encoder_layers = 1
        self.n_decoder_layers = 1
        self.batch_first = True
        self.channels = 64

        # parameter encoder parameters 
        self.n_parameters = 3 
        self.clap_latent_dim = self.max_seq_len*2

        # model initialization 
        self.pretrained_loc = ''

def plot_results(model, val_dataloader, num_plots, log_dir, epoch, device):
    fig, axs = plt.subplots(nrows=1, ncols=num_plots, sharey=True, sharex=True, figsize=(32, 8))
    model = model.eval()
    for val_batch in val_dataloader:
        seq_true = val_batch["observed_data"]
        predictions = model(val_batch)
        for i in range(num_plots):
            pred = predictions[i].squeeze(0).cpu().detach().numpy()
            true = seq_true[i].squeeze(0).cpu().numpy()
            axs[i].plot(true, label='true')
            axs[i].plot(pred, label='reconstructed')
        break
    fig.tight_layout()

    save_dir = os.path.join(log_dir, 'qualitative')
    os.makedirs(save_dir, exist_ok=True)
    save_loc = os.path.join(save_dir, str(epoch)+'.png')
    plt.savefig(save_loc)
    plt.close('all')

def train(model, train_loader, val_loader, model_config, log_dir, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate, weight_decay=1e-6)
    p1 = int(0.75 * model_config.n_epochs)
    p2 = int(0.9 * model_config.n_epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1) 
    criterion = nn.L1Loss(reduction='sum').to(device)

    best_training_loss = 1e10
    for epoch_no in range(model_config.n_epochs):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                seq_true = train_batch["observed_data"].to(device)
                seq_pred = model(train_batch)
                # loss = F.mse_loss(seq_pred, seq_true)
                loss = criterion(seq_pred, seq_true)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()
        
        if avg_loss < best_training_loss:
            best_training_loss = avg_loss
            print(
                "\n best loss is updated to ",
                avg_loss / batch_no,
                "at",
                epoch_no,
            )
            best_model_path = os.path.join(log_dir, "model_best.pth")
            torch.save(model.state_dict(), best_model_path)

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                seq_true = val_batch["observed_data"].to(device)
                seq_pred = model(val_batch)
                val_loss = criterion(seq_pred, seq_true)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        print(f'Epoch {epoch_no}: val loss {val_loss}')
        plot_results(model, val_loader, model_config.n_plots, log_dir, epoch_no, device)


############################################################################################################

dataset_config = DatasetConfig() 
cltsp_config = CLTSPConfig()

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
foldername = ("./save/sine" + "_" + current_time + "/")

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)

num_epochs = cltsp_config.n_epochs
log_dir = os.path.join(foldername, 'results')
os.makedirs(log_dir, exist_ok=True)


sine_dataset_obj = Sinusoidal(config=dataset_config)
train_dataset = GetDataset(data=sine_dataset_obj.train_data, labels=sine_dataset_obj.train_labels)
test_dataset = GetDataset(data=sine_dataset_obj.test_data, labels=sine_dataset_obj.test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=cltsp_config.batch_size, num_workers=0, shuffle=True)
val_dataloader = DataLoader(test_dataset, batch_size=cltsp_config.batch_size, num_workers=0, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = CSDI_base(model_config=cltsp_config, device=device)
model = model.to(device)
if cltsp_config.pretrained_loc != '':
    model.load_state_dict(torch.load(cltsp_config.pretrained_loc))

model, history = train(model, train_dataloader, val_dataloader, cltsp_config, log_dir, device)

# for batch_no, train_batch in enumerate(train_dataloader):
#     seq_true = train_batch["observed_data"].to(device).float()
#     seq_pred = model(train_batch)
#     print(seq_pred.shape)