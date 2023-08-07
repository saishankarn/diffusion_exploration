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
import torch.functional as F 
from torch.utils.data import Dataset, DataLoader

from diffusion_dataset import RecurrentAutoencoder
from main_model import LDM

class GetDataset(Dataset):
    def __init__(self, config, dataset):
        self.dataset = dataset 
        self.latent_dim = config["model"]["latent_dim"]
    
    def __getitem__(self, org_index):
        return {"mean": self.dataset[org_index][:self.latent_dim],
                "var": self.dataset[org_index][self.latent_dim:]}

    def __len__(self):
        return self.dataset.shape[0]
    
def plot_results(latent, autoencoder, plot_path):
    fig, axs = plt.subplots(nrows=1, ncols=latent.shape[0], sharey=True, sharex=True, figsize=(32, 8))
    autoencoder = autoencoder.eval()
    for i in range(latent.shape[0]):
        to_decoder = latent[i].detach()
        ts = autoencoder.decoder(to_decoder)
        ts = ts[0][:,-1].detach().cpu().numpy()
        axs[i].plot(np.arange(ts.shape[0]), ts)
    fig.tight_layout()
    plt.savefig(plot_path)
    plt.close('all')

def train(diffmodel, config, train_loader, autoencoder, foldername=""):
    plot_dir = os.path.join(foldername, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    optimizer = Adam(diffmodel.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
    p1 = int(0.75 * config["train"]["epochs"])
    p2 = int(0.9 * config["train"]["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1) 

    best_training_loss = 1e10
    for epoch_no in range(config["train"]["epochs"]):
        avg_loss = 0
        diffmodel.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                mu_batch = train_batch["mean"]
                sigma_batch = train_batch["var"]
                z = [mu_batch + sigma_batch * torch.randn_like(sigma_batch) for idx in range(10)]
                z = torch.cat(z,0)
                
                loss = diffmodel(z)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(ordered_dict={"avg_epoch_loss": avg_loss / batch_no, "epoch": epoch_no}, refresh=False)
            lr_scheduler.step()

        if avg_loss < best_training_loss:
            best_training_loss = avg_loss
            print("\n best loss is updated to ", avg_loss / batch_no, "at", epoch_no)
            best_model_path = os.path.join(foldername, "model_best.pth")
            torch.save(diffmodel.state_dict(), best_model_path)

        diffmodel.eval()
        with torch.no_grad():
            output = diffmodel.synthesize()
            plot_path = os.path.join(plot_dir, str(epoch_no) + '.png')
            plot_results(output, autoencoder, plot_path)

    
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

    mean_dataset_loc = torch.load(os.path.join(config["train"]["dataset_dir"], "mean.pth")) 
    var_dataset_loc = torch.load(os.path.join(config["train"]["dataset_dir"], "var.pth"))
    dataset_tensor = torch.cat((mean_dataset_loc, var_dataset_loc), 1)
    
    dataset = GetDataset(config=config, dataset=dataset_tensor)
    batch_size = config["train"]["batch_size"]
    train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)

    denoising_model = LDM(config, device)
    denoising_model = denoising_model.to(device)

    autoencoder = RecurrentAutoencoder(seq_len=config["model"]["horizon"], n_features=1, embedding_dim=16)
    autoencoder = autoencoder.to(device)
    pretrained_loc = os.path.join(config["train"]["dataset_dir"], 'results/weights/best_model.pth')
    if os.path.exists(pretrained_loc):
        autoencoder.load_state_dict(torch.load(pretrained_loc))
    print("loaded pretrained weights")

    train(denoising_model, config, train_dataloader, autoencoder, foldername)
