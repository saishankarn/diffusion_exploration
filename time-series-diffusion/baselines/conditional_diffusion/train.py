import os 
os.system("unset LD_LIBRARY_PATH")
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from config import DatasetConfig, VAEConfig
from dataset import GetDataset
from model import TimeSeriesVariationalAutoEncoder

def plot_results(model, val_dataloader, num_plots, log_dir, epoch, device):
    fig, axs = plt.subplots(nrows=1, ncols=num_plots, sharey=True, sharex=True, figsize=(32, 8))
    model = model.eval()
    for val_batch in val_dataloader:
        seq_true = val_batch["observed_data"].to(device)
        _, _, predictions= model(val_batch)
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

    best_training_loss = 1e10
    for epoch_no in range(model_config.n_epochs):
        avg_reconstruction_loss = 0
        avg_kl_divergence_loss = 0
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:  
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                true_data = train_batch["observed_data"].to(device)
                mean, log_var, pred_data = model(train_batch)

                reconstruction_loss = torch.nn.functional.l1_loss(pred_data, true_data, reduction='sum')
                kl_divergence_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim = 1), dim = 0)
                # kl_divergence_loss = -0.5 * torch.mean(1 + log_var - mean ** 2 - log_var.exp())
                loss = reconstruction_loss + kl_divergence_loss              
                loss.backward()
                avg_reconstruction_loss += reconstruction_loss.item()
                avg_kl_divergence_loss += kl_divergence_loss.item()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(ordered_dict={"avg_recon_loss": avg_reconstruction_loss / batch_no, "avg_kld_loss": avg_kl_divergence_loss / batch_no, "epoch": epoch_no}, refresh=False)
  
            lr_scheduler.step()
        
        if avg_loss < best_training_loss:
            best_training_loss = avg_loss
            print("\n best loss is updated to ", avg_loss / batch_no, "at", epoch_no)
            best_model_path = os.path.join(log_dir, "model_best.pth")
            torch.save(model.state_dict(), best_model_path)

        val_losses = []
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                seq_true = val_batch["observed_data"]
                true_data = seq_true.to(device)
                mean, log_var, pred_data = model(val_batch)
                reconstruction_loss = torch.nn.functional.l1_loss(pred_data, true_data, reduction='sum')
                kl_divergence_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean ** 2 - log_var.exp(), dim = 1), dim = 0)
                loss = reconstruction_loss + kl_divergence_loss 
                val_losses.append(loss.item())

        print("the validation loss is : ", sum(val_losses) / len(val_losses))
        plot_results(model, val_dataloader, model_config.n_plots, log_dir, epoch_no, device)
    
        
    last_model_path = os.path.join(log_dir, "model_last.pth")
    torch.save(model.state_dict(), last_model_path)

if __name__ == "__main__":
    dataset_config = DatasetConfig() 
    model_config = VAEConfig()

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
    foldername = ("./save/sine" + "_" + current_time + "/")

    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)

    num_epochs = model_config.n_epochs
    log_dir = os.path.join(foldername, 'results')
    os.makedirs(log_dir, exist_ok=True)

    train_dataset = GetDataset(dataset_config=dataset_config, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=model_config.train_batch_size, num_workers=0, shuffle=True)

    test_dataset = GetDataset(dataset_config=dataset_config, train=False)    
    val_dataloader = DataLoader(test_dataset, batch_size=model_config.val_batch_size, num_workers=0, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TimeSeriesVariationalAutoEncoder(model_config=model_config, device=device)
    model = model.to(device)
    if model_config.pretrained_loc != '':
        model.load_state_dict(torch.load(model_config.pretrained_loc))

    train(model, train_dataloader, val_dataloader, model_config, log_dir, device)



