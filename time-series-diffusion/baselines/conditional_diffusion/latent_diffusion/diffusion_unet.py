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
from torch.utils.data import Dataset, DataLoader

from diffusion_dataset import RecurrentAutoencoder
from model import LDM, LDMUNet

class GetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset 
        self.dataset = self.dataset.view(-1, config["model"]["latent_channels"], config["model"]["latent_dim"])
    
    def __getitem__(self, org_index):
        return self.dataset[org_index]

    def __len__(self):
        return self.dataset.shape[0]
    
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


def plot_timeseries(ts, plot_path):
    x = np.arange(ts.shape[0])
    print(ts)
    plt.plot(x, ts)
    plt.savefig(plot_path)
    plt.close()

def train(model, config, train_loader, autoencoder, foldername=""):
    optimizer = Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
    p1 = int(0.75 * config["train"]["epochs"])
    p2 = int(0.9 * config["train"]["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1) 

    plot_dir = os.path.join(foldername, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    best_training_loss = 1e10
    for epoch_no in range(config["train"]["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(ordered_dict={"avg_epoch_loss": avg_loss / batch_no, "epoch": epoch_no}, refresh=False)
            lr_scheduler.step()
        
        if avg_loss < best_training_loss:
            best_training_loss = avg_loss
            print(
                "\n best loss is updated to ",
                avg_loss / batch_no,
                "at",
                epoch_no,
            )
            best_model_path = os.path.join(foldername, "model_best.pth")
            torch.save(model.state_dict(), best_model_path)


        model.eval()
        with torch.no_grad():
            latent_vector = model.synthesize()
            predicted = autoencoder.decoder(latent_vector)
            predicted = predicted.squeeze(-1).cpu().detach().numpy()

            fig, axs = plt.subplots(nrows=1, ncols=config["train"]["num_plots"], sharey=True, sharex=True, figsize=(32, 8))
            for i in range(config["train"]["num_plots"]):
                pred = predicted[i]
                axs[i].plot(pred, label='reconstructed')

            fig.tight_layout()
            save_loc = os.path.join(plot_dir, str(epoch_no)+'.png')
            plt.savefig(save_loc)
            plt.close('all')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TS-Autoencoder") 
    parser.add_argument("--config", type=str, default="diffusion.yaml")

    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path = "config/" + args.config 
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
    foldername = ("./save/diffusion" + "_" + current_time + "/")

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    dataset_tensor = torch.load(config["train"]["dataset_loc"])
    print(dataset_tensor.shape)
    m = torch.mean(dataset_tensor, axis = 0)
    s = torch.std(dataset_tensor, axis = 0)
    train_dataset = GetDataset(dataset_tensor)
    batch_size = config["train"]["batch_size"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    
    # t = torch.randint(0, config["diffusion"]["num_steps"], [6]).to(device)
    # input = torch.randn((6, 1, 16)).to(device)
    # output = diffusion_model(input, t)["sample"]
    # print(output.shape)

    denoising_model = LDMUNet(config, device)

    horizon = config["model"]["horizon"]
    input_features = config["model"]["input_dim"]
    latent_dim = config["model"]["latent_dim"]
    autoencoder = RecurrentAutoencoder(seq_len=horizon, n_features=input_features, embedding_dim=latent_dim)
    autoencoder = autoencoder.to(device)

    train(denoising_model, config, train_dataloader, autoencoder, foldername)
