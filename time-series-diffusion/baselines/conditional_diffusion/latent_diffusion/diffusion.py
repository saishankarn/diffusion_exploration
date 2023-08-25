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
    
def plot_results(gt_ts_plot, pred_ts_plot, gt_label_plot, plot_path):
    fig, axs = plt.subplots(nrows=1, ncols=gt_ts_plot.shape[0], sharey=True, sharex=True, figsize=(32, 8))
    for i in range(gt_ts_plot.shape[0]):
        axs[i].plot(np.arange(gt_ts_plot[i][0].shape[0]), gt_ts_plot[i][0], label='gt')
        axs[i].plot(np.arange(pred_ts_plot[i][0].shape[0]), pred_ts_plot[i][0], label='pred')
        title = 'a=' + str(gt_label_plot[i][0]) + ', ' + 'f=' + str(gt_label_plot[i][1]) + ', ' + 'p=' + str(gt_label_plot[i][2])
        axs[i].set_title(title, fontsize=20)
        axs[i].legend(fontsize=20)
    fig.tight_layout()
    plt.savefig(plot_path)
    plt.close('all') 

def train(diffmodel, config, train_loader, val_loader, autoencoder, foldername=""):
    # for plotting
    autoencoder = autoencoder.eval()
    num_plots = config["train"]["num_plots"]
    plot_dir = os.path.join(foldername, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # initializing the optimizer
    optimizer = Adam(diffmodel.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
    p1 = int(0.75 * config["train"]["epochs"])
    p2 = int(0.9 * config["train"]["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1) 

    best_training_loss = 1e10
    for epoch_no in range(config["train"]["epochs"]):
        avg_denoising_loss = 0
        diffmodel.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                mu_batch = train_batch["mean"]
                sigma_batch = train_batch["var"]
                x = [mu_batch + sigma_batch * torch.randn_like(sigma_batch) for idx in range(config["train"]["repeat_latent_num"])]
                x = torch.cat(x,0)
                
                label_batch = train_batch["label"]
                y = [label_batch for idx in range(config["train"]["repeat_latent_num"])]
                y = torch.cat(y,0)

                denoising_loss = diffmodel(data=x, label=y)
                denoising_loss.backward()
                avg_denoising_loss += denoising_loss.item()
                optimizer.step()
                it.set_postfix(ordered_dict={"avg_epoch_loss": avg_denoising_loss / batch_no, "epoch": epoch_no}, refresh=False)
            lr_scheduler.step()

        if avg_denoising_loss < best_training_loss:
            best_training_loss = avg_denoising_loss
            print("\n best loss is updated to ", avg_denoising_loss / batch_no, "at", epoch_no)
            best_model_path = os.path.join(foldername, "model_best.pth")
            torch.save(diffmodel.state_dict(), best_model_path)

        if epoch_no%config["train"]["run_eval_after_every"] == 0:
            diffmodel.eval()
            with torch.no_grad():
                avg_val_loss = 0
                for val_idx, val_batch in enumerate(val_loader):
                    gt_label = val_batch["label"]
                    gt_ts = val_batch["ts"]

                    output = diffmodel.synthesize(gt_label)
                    encoded = output.reshape(gt_label.shape[0], config["model"]["latent_dim"])
                    pred_ts = autoencoder.decode(encoded)
                    pred_ts = pred_ts.detach().cpu()
                    val_batch_loss = F.l1_loss(pred_ts, gt_ts)
                    avg_val_loss += val_batch_loss.item()
                    
                    gt_ts_plot = gt_ts[:num_plots].cpu().numpy()
                    gt_label_plot = gt_label[:num_plots].cpu().numpy()
                    pred_ts_plot = pred_ts[:num_plots].cpu().numpy()

                    plot_path = os.path.join(plot_dir, str(epoch_no) + '_' + str(val_idx) + '.png')
                    plot_results(gt_ts_plot, pred_ts_plot, gt_label_plot, plot_path)

                print("l1 loss for the validation set : ", avg_val_loss/val_idx)

    last_model_path = os.path.join(foldername, "model_last.pth")
    torch.save(diffmodel.state_dict(), last_model_path)

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
    print(torch.sum(any_row_match))

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

    train(denoising_model, diffusion_config, train_dataloader, val_dataloader, autoencoder, foldername)
