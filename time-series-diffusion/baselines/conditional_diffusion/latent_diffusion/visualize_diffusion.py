import json
import yaml
from tqdm import tqdm
import os
import datetime
import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch  
from torch.optim import Adam
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader

from config import DatasetConfig, VAEConfig
from autoencoder import TimeSeriesVariationalAutoEncoder
from main_model import LDM

class GetDiffusionDataset(Dataset):
    def __init__(self, mean_dataset, var_dataset, labels, ts):
        self.mean_dataset = mean_dataset
        self.var_dataset = var_dataset
        self.labels = labels
        self.ts = ts
        assert self.mean_dataset.shape[0] == self.var_dataset.shape[0] == self.labels.shape[0] == self.ts.shape[0]
    
    def __getitem__(self, org_index):
        return {"mean": self.mean_dataset[org_index],
                "var": self.var_dataset[org_index],
                "label": self.labels[org_index], 
                "ts": self.ts[org_index]}

    def __len__(self):
        return self.mean_dataset.shape[0]
    
def plot_results(plot_dict, plot_path):
    gt_ts_plot = plot_dict["gt_ts_plot"] 
    gt_label_plot = plot_dict["gt_label_plot"]
    pred_ts_dict = plot_dict["pred_ts_plot"]

    timestamps_to_plot = list(pred_ts_dict.keys())
    num_timestamps_to_plot = len(timestamps_to_plot)
    num_plots = gt_ts_plot.shape[0]
    fig, axs = plt.subplots(nrows=num_plots, ncols=num_timestamps_to_plot, sharey=True, sharex=True, figsize=(48, 32))
    
    for plot_idx in range(num_plots):
        gt_label_elem = gt_label_plot[plot_idx]
        gt_ts_elem = gt_ts_plot[plot_idx]
    
        for timestamp_idx, timestamp in enumerate(timestamps_to_plot):
            title = 'a=' + str(gt_label_elem[0]) + ', ' + 'f=' + str(gt_label_elem[1]) + ', ' + 'p=' + str(gt_label_elem[2]) + ', ' + 't=' + str(timestamp)
    
            pred_ts_elem = pred_ts_dict[timestamp][plot_idx]          
            axs[plot_idx][timestamp_idx].plot(np.arange(gt_ts_elem.shape[0]), gt_ts_elem, label='gt')
            axs[plot_idx][timestamp_idx].plot(np.arange(pred_ts_elem.shape[0]), pred_ts_elem, label='pred')

            axs[plot_idx][timestamp_idx].set_title(title, fontsize=20)
            axs[plot_idx][timestamp_idx].legend(fontsize=20)
            axs[plot_idx][timestamp_idx].set_ylim(-1.2,1.2)

    fig.tight_layout()
    plt.savefig(plot_path)
    plt.close('all') 

def visualize_diffusion(diffmodel, config, val_loader, autoencoder, foldername=""):
    # for plotting
    autoencoder = autoencoder.eval()
    num_plots = config["train"]["num_plots"]
    plot_dir = os.path.join(foldername, 'diff_viz')
    os.makedirs(plot_dir, exist_ok=True)
    timestamps_to_plot = [0, 20, 40, 43, 46, 49] # this corresponds to 49, 29, 9, 6, 3, 0
    plot_dict = {}

    diffmodel.eval()
    with torch.no_grad():
        for val_idx, val_batch in enumerate(val_loader):
            gt_label = val_batch["label"]
            gt_ts = val_batch["ts"]
            gt_ts_plot = gt_ts[:num_plots].cpu().numpy()
            gt_ts_plot = gt_ts_plot.squeeze(1)
            gt_label_plot = gt_label[:num_plots].cpu().numpy()
            plot_dict["gt_ts_plot"] = gt_ts_plot
            plot_dict["gt_label_plot"] = gt_label_plot


            outputs = diffmodel.synthesize_for_visualization(gt_label)
            pred_ts_dict = {}
            for tidx, output in enumerate(outputs):
                encoded = output.reshape(gt_label.shape[0], config["model"]["latent_dim"])
                pred_ts = autoencoder.decode(encoded)
                pred_ts = pred_ts.detach().cpu()
                pred_ts_plot = pred_ts[:num_plots].cpu().numpy()
                pred_ts_plot = pred_ts_plot.squeeze(1)
                if tidx in timestamps_to_plot:
                    pred_ts_dict[tidx] = pred_ts_plot

            plot_dict["pred_ts_plot"] = pred_ts_dict
            plot_path = os.path.join(plot_dir, str(val_idx) + '.png')
            plot_results(plot_dict, plot_path)
            break 
    
    # print(plot_dict)
    
    #             plot_results(gt_ts_plot, pred_ts_plot, gt_label_plot, plot_path)

    # last_model_path = os.path.join(foldername, "model_last.pth")
    # torch.save(diffmodel.state_dict(), last_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TS-Autoencoder") 
    parser.add_argument("--config", type=str, default="diffusion.yaml")

    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # loading all the configs
    path = "config/" + args.config 
    with open(path, "r") as f:
        diffusion_config = yaml.safe_load(f)
    dataset_config = DatasetConfig() 
    vae_config = VAEConfig()
    print(json.dumps(diffusion_config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
    foldername = ("./save/diffusion" + "_" + current_time + "/")

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(diffusion_config, f, indent=4)

    # get the train dataloader
    mean_train_dataset = torch.load(os.path.join(vae_config.pretrained_dir, "results/mean_train.pth")) 
    var_train_dataset = torch.load(os.path.join(vae_config.pretrained_dir, "results/log_var_train.pth"))
    labels_train_dataset = torch.load(os.path.join(vae_config.pretrained_dir, "results/labels_train.pth"))
    ts_train_dataset = torch.load(os.path.join(vae_config.pretrained_dir, "results/ts_train.pth"))
    train_dataset = GetDiffusionDataset(mean_dataset=mean_train_dataset, var_dataset=var_train_dataset, labels=labels_train_dataset, ts=ts_train_dataset)
    batch_size = diffusion_config["train"]["batch_size"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

    # get the val dataloader
    mean_val_dataset = torch.load(os.path.join(vae_config.pretrained_dir, "results/mean_val.pth")) 
    var_val_dataset = torch.load(os.path.join(vae_config.pretrained_dir, "results/log_var_val.pth"))
    labels_val_dataset = torch.load(os.path.join(vae_config.pretrained_dir, "results/labels_val.pth"))
    ts_val_dataset = torch.load(os.path.join(vae_config.pretrained_dir, "results/ts_val.pth"))
    val_dataset = GetDiffusionDataset(mean_dataset=mean_val_dataset, var_dataset=var_val_dataset, labels=labels_val_dataset, ts=ts_val_dataset)
    batch_size = diffusion_config["train"]["batch_size"]
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

    row_matches = torch.eq(labels_val_dataset[:, None, :], labels_train_dataset).all(dim=2)
    any_row_match = torch.any(row_matches, dim=1)
    assert torch.sum(any_row_match).item() == 0

    autoencoder = TimeSeriesVariationalAutoEncoder(model_config=vae_config, device=device)
    autoencoder = autoencoder.to(device)
    if vae_config.pretrained_dir != '':
        pretrained_loc = os.path.join(vae_config.pretrained_dir, 'results/model_best.pth')
        autoencoder.load_state_dict(torch.load(pretrained_loc))
    print("loaded pretrained weights")

    denoising_model = LDM(diffusion_config, device)
    denoising_model = denoising_model.to(device)
    denoising_pretrained_loc = diffusion_config["train"]["pretrained_loc"]
    if os.path.exists(denoising_pretrained_loc):
        denoising_model.load_state_dict(torch.load(denoising_pretrained_loc))

    visualize_diffusion(denoising_model, diffusion_config, val_dataloader, autoencoder, foldername)

