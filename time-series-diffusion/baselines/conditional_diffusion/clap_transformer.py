import os 
from tqdm import tqdm
os.system("unset LD_LIBRARY_PATH")
import numpy as np
from itertools import product

import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader   

from timeseries_encoder import TimeSeriesEncoder

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
        self.train_batch_size = 64
        self.val_batch_size = 1
        self.n_epochs = 2000
        self.learning_rate = 1e-4
        self.n_plots = 6
        self.val_epoch_interval = 5

        # encoder parameters
        self.n_features = 1
        self.dim_val = 128
        self.dropout_pos_enc = 0.2 
        self.max_seq_len = 48
        self.n_heads = 8
        self.n_encoder_layers = 2
        self.n_decoder_layers = 1
        self.batch_first = True
        self.channels = 64

        # parameter encoder parameters 
        self.n_parameters = 3 
        self.clap_latent_dim = self.max_seq_len*2

        # model initialization 
        self.pretrained_loc = ''

class ParamEncoder(nn.Module):
    def __init__(self, model_config, device):
        super(ParamEncoder, self).__init__()
        self.device = device 

        self.fc1 = nn.Linear(model_config.n_parameters, model_config.clap_latent_dim // 2)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(model_config.clap_latent_dim // 2, model_config.clap_latent_dim)
        self.act2 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

  
    def forward(self, x):
        x = x.to(self.device)
        x = self.dropout(self.act1(self.fc1(x)))
        x = self.dropout(self.act2(self.fc2(x)))
        return x
    
class CLTSPModel(nn.Module):
    def __init__(self, model_config, device):
        super(CLTSPModel, self).__init__()

        self.timeseries_autoencoder = TimeSeriesEncoder(model_config=model_config, device=device)
        self.parameters_encoder = ParamEncoder(model_config=model_config, device=device)

    def forward(self, batch):
        timeseries_latent = self.timeseries_autoencoder(batch)
        parameters_latent = self.parameters_encoder(batch["labels"])
        return timeseries_latent, parameters_latent

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def cltsp_loss(timeseries_latent, parameters_latent):
    logits = (parameters_latent @ timeseries_latent.T)
    parameters_similarity = parameters_latent @ parameters_latent.T
    timeseries_similarity = timeseries_latent @ timeseries_latent.T
    targets = F.softmax((parameters_similarity + timeseries_similarity) / 2, dim=-1)
    parameters_loss = cross_entropy(logits, targets, reduction='none')
    timeseries_loss = cross_entropy(logits.T, targets.T, reduction='none')
    similarity_loss =  (parameters_loss + timeseries_loss) / 2.0
    similarity_loss = similarity_loss.mean()
    return similarity_loss

def train(model, train_loader, val_loader, val_labels, model_config, log_dir, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate, weight_decay=1e-6)
    p1 = int(0.75 * model_config.n_epochs)
    p2 = int(0.9 * model_config.n_epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1) 

    best_training_loss = 1e10
    for epoch_no in range(model_config.n_epochs):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:  
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                timeseries_latent, parameters_latent = model(train_batch)
                loss = 100*cltsp_loss(timeseries_latent, parameters_latent)                
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(ordered_dict={"avg_loss": avg_loss / batch_no, "epoch": epoch_no}, refresh=False)
            lr_scheduler.step()
        
        if avg_loss < best_training_loss:
            best_training_loss = avg_loss
            print("\n best loss is updated to ", avg_loss / batch_no, "at", epoch_no)
            best_model_path = os.path.join(log_dir, "model_best.pth")
            torch.save(model.state_dict(), best_model_path)

        if epoch_no % model_config.val_epoch_interval == 0:
            # model.eval()
            avg_val_loss = 0
            with torch.no_grad():
                val_parameters_latent = model.parameters_encoder(val_labels)
                num_labels = val_parameters_latent.shape[0]
                top1_acc = 0
                top3_acc = 0
                top5_acc = 0
            
                for val_batch_no, val_batch in tqdm(enumerate(val_loader)):
                    timeseries_latent, parameters_latent = model(val_batch)
                    val_loss = cltsp_loss(timeseries_latent, parameters_latent)
                    avg_val_loss += val_loss

                    scores = (val_parameters_latent @ timeseries_latent.T).squeeze(-1)
                    query_label = val_batch["labels"][0].to(device)

                    # top 1
                    top1_indices = torch.topk(scores, k=1).indices
                    top1_corresponding_labels = val_labels[top1_indices]
                    query_label_exists_in_top1 = torch.any(torch.all(top1_corresponding_labels == query_label, dim=1))
                    top1_acc += query_label_exists_in_top1

                    # top 3
                    top3_indices = torch.topk(scores, k=3).indices
                    top3_corresponding_labels = val_labels[top3_indices]
                    query_label_exists_in_top3 = torch.any(torch.all(top3_corresponding_labels == query_label, dim=1))
                    top3_acc += query_label_exists_in_top3
                
                    # top 5
                    top5_indices = torch.topk(scores, k=5).indices
                    top5_corresponding_labels = val_labels[top5_indices]
                    query_label_exists_in_top5 = torch.any(torch.all(top5_corresponding_labels == query_label, dim=1))
                    top5_acc += query_label_exists_in_top5

                top1_acc = top1_acc / num_labels
                top3_acc = top3_acc / num_labels
                top5_acc = top5_acc / num_labels
                print("validation loss is %f"%(avg_val_loss/val_batch_no))
                print("top1 accuracy = %f, top3 accuracy = %f,  top5 accuracy = %f"%(top1_acc, top3_acc, top5_acc))
        
    last_model_path = os.path.join(log_dir, "model_last.pth")
    torch.save(model.state_dict(), last_model_path)

############################################################################################################

if __name__ == "__main__":
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

    train_dataloader = DataLoader(train_dataset, batch_size=cltsp_config.train_batch_size, num_workers=0, shuffle=True)
    val_dataloader = DataLoader(test_dataset, batch_size=cltsp_config.val_batch_size, num_workers=0, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CLTSPModel(model_config=cltsp_config, device=device)
    model = model.to(device)
    if cltsp_config.pretrained_loc != '':
        model.load_state_dict(torch.load(cltsp_config.pretrained_loc))

    val_labels = sine_dataset_obj.test_labels
    val_labels = val_labels.to(device)

    train(model, train_dataloader, val_dataloader, val_labels, cltsp_config, log_dir, device)



