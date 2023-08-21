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
        self.train_batch_size = 64
        self.val_batch_size = 1
        self.n_epochs = 2000
        self.learning_rate = 1e-5
        self.n_plots = 6
        self.val_epoch_interval = 5

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
        self.pretrained_loc = 'save/sine_20230814_030340/results/model_best.pth'

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

        self.timeseries_autoencoder = CSDI_base(model_config=model_config, device=device)
        self.parameters_encoder = ParamEncoder(model_config=model_config, device=device)

    def forward(self, batch):
        timeseries_latent, seq_pred = self.timeseries_autoencoder(batch)
        parameters_latent = self.parameters_encoder(batch["labels"])
        return seq_pred, timeseries_latent, parameters_latent

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

def plot_results(model, val_dataloader, num_plots, log_dir, epoch, device):
    fig, axs = plt.subplots(nrows=1, ncols=num_plots, sharey=True, sharex=True, figsize=(32, 8))
    model = model.eval()
    for i, val_batch in enumerate(val_dataloader):
        if i >= num_plots:
            break
        seq_true = val_batch["observed_data"]
        predictions, _, _ = model(val_batch)
        pred = predictions[0].squeeze(0).cpu().detach().numpy()
        true = seq_true[0].squeeze(0).cpu().numpy()
        axs[i].plot(true, label='true')
        axs[i].plot(pred, label='reconstructed')
        
    fig.tight_layout()

    save_dir = os.path.join(log_dir, 'qualitative')
    os.makedirs(save_dir, exist_ok=True)
    save_loc = os.path.join(save_dir, str(epoch)+'.png')
    plt.savefig(save_loc)
    plt.close('all')

def train(model, train_loader, val_loader, val_labels, model_config, log_dir, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate, weight_decay=1e-6)
    p1 = int(0.75 * model_config.n_epochs)
    p2 = int(0.9 * model_config.n_epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1) 
    criterion = nn.L1Loss(reduction='sum').to(device)

    best_training_loss = 1e10
    for epoch_no in range(model_config.n_epochs):
        avg_reconstruction_loss = 0
        avg_similarity_loss = 0
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:  
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                seq_true = train_batch["observed_data"].to(device)
                seq_pred, timeseries_latent, parameters_latent = model(train_batch)
                
                reconstruction_loss = criterion(seq_pred, seq_true)
                similarity_loss = cltsp_loss(timeseries_latent, parameters_latent)
                loss = 0*reconstruction_loss + similarity_loss
                
                loss.backward()
                
                avg_reconstruction_loss += reconstruction_loss.item()
                avg_similarity_loss += similarity_loss.item()
                avg_loss += loss.item()
                
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_recon_loss": avg_reconstruction_loss / batch_no,
                        "avg_sim_loss": avg_similarity_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()
        
        if avg_loss < best_training_loss:
            best_training_loss = avg_loss
            print("\n best loss is updated to ", avg_loss / batch_no, "at", epoch_no)
            best_model_path = os.path.join(log_dir, "model_best.pth")
            torch.save(model.state_dict(), best_model_path)

        if epoch_no % model_config.val_epoch_interval == 0:
            model.eval()
            avg_reconstruction_loss = 0
            avg_similarity_loss = 0
            
            with torch.no_grad():
                val_parameters_latent = model.parameters_encoder(val_labels)
                num_labels = val_parameters_latent.shape[0]
                top1_acc = 0
                top3_acc = 0
                top5_acc = 0
            
                for batch_no, val_batch in tqdm(enumerate(val_loader)):
                    seq_true = val_batch["observed_data"].to(device)
                    seq_pred, timeseries_latent, parameters_latent = model(val_batch)
                    val_reconstruction_loss = criterion(seq_pred, seq_true)
                    val_similarity_loss = cltsp_loss(timeseries_latent, parameters_latent)
                    avg_reconstruction_loss += val_reconstruction_loss
                    avg_similarity_loss += val_similarity_loss

                    scores = torch.matmul(val_parameters_latent, timeseries_latent.T).squeeze(-1)
                    query_label = val_batch["labels"][0].to(device)

                    # top 1
                    # print("top 1")
                    top_indices = torch.topk(scores, k=1).indices
                    # print(top_indices)
                    corresponding_labels = val_labels[top_indices]
                    # print(corresponding_labels.shape)
                    
                    label_exists = torch.any(torch.all(query_label == corresponding_labels, dim=1))
                    top1_acc += label_exists

                    # top 3
                    top_indices = torch.topk(scores, k=3).indices
                    # print(top_indices)
                    corresponding_labels = val_labels[top_indices]
                    # print(corresponding_labels.shape)
                    label_exists = torch.any(torch.all(query_label == corresponding_labels, dim=1))
                    top3_acc += label_exists
                
                    # top 5
                    top_indices = torch.topk(scores, k=5).indices
                    # print(top_indices)
                    corresponding_labels = val_labels[top_indices]
                    # print(corresponding_labels.shape)
                    label_exists = torch.any(torch.all(corresponding_labels == query_label, dim=1))
                    top5_acc += label_exists

                print(top1_acc, top3_acc, top5_acc)
                top1_acc = top1_acc / num_labels
                top3_acc = top3_acc / num_labels
                top5_acc = top5_acc / num_labels
                print("validation reconstruction loss is %f, validation similarity loss is %f"%(avg_reconstruction_loss/batch_no, avg_similarity_loss/batch_no))
                print("top1 accuracy = %f, top3 accuracy = %f,  top5 accuracy = %f"%(top1_acc, top3_acc, top5_acc))
        
        plot_results(model, val_loader, model_config.n_plots, log_dir, epoch_no, device)

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
    cltsp_model = CLTSPModel(model_config=cltsp_config, device=device)
    cltsp_model.load_state_dict(torch.load(cltsp_config.pretrained_loc))
    for param in cltsp_model.parameters():
        param.requires_grad = False

    cltsp_decoder = CLTSPDecoder(model_config=cltsp_config, cltsp_model=cltsp_model, device=device)
    for param in cltsp_decoder.parameters():
        param.requires_grad = True


    val_labels = sine_dataset_obj.test_labels
    val_labels = val_labels.to(device)

    

    # Autoencoder, history = train(model, train_dataloader, val_dataloader, val_labels, cltsp_config, log_dir, device)

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
    
class CLTSPDecoder(nn.Module):
    def __init__(self, model_config, cltsp_model, device):
        super(CLTSPModel, self).__init__()
        self.device = device 
        self.model_config = model_config 
        self.decompress_encoded = cltsp_model.timeseries_autoencoder.diffmodel.decompress_encoded
        self.decoder_layers = cltsp_model.timeseries_autoencoder.diffmodel.decoder_layers

    def get_side_info(self, encoded):
        B = encoded.shape
        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1) # (B, L, K, emb)
        side_info = time_embed.permute(0, 3, 2, 1)  # (B,*,K,L)
        return side_info

    def forward(self, encoded, batch):
        encoded = encoded.to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        decoded = self.decompress_encoded(encoded)
        decoded = decoded.reshape(decoded.shape[0], self.channels, self.model_config.n_features, self.model_config.max_seq_len)
        for layer in self.decoder_layers:
            decoded = layer(decoded, cond_info)
        decoded = decoded.reshape(B, self.channels, K * L)
        decoded = F.relu(decoded)
        decoded = self.output_projection(decoded)  # (B,1,K*L)
        x_pred = decoded.reshape(B, K, L)

        return encoded.view(B, -1), x_pred