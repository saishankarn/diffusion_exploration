import os
import numpy as np
import torch 
import yaml
import argparse

from sine_vae import RecurrentAutoencoder

class Sinusoidal():
    def __init__(self, config, device):
        self.eval_length = config["model"]["horizon"]
        self.embedding_dim = config["model"]["latent_dim"]

        num_freqs = 10
        time_steps = 1000
        freqs = list(np.linspace(1, 5, num_freqs))
        time = np.linspace(0, 2*np.pi, time_steps)

        data = [np.sin(2 * np.pi * freq * time) for freq in freqs]
        data = np.array(data)
        intervals = [(0+i,self.eval_length+i) for i in range(0,time_steps-self.eval_length)]
        dataset = np.zeros((len(intervals)*num_freqs, self.eval_length))

        for i in range(num_freqs):
            orig_dataset = data[i] 
            for j in range(len(intervals)):
                min_ = intervals[j][0]
                max_ = intervals[j][1]
 
                idx = i*len(intervals)+j
                dataset[idx] = orig_dataset[min_:max_]
        self.dataset = torch.tensor(dataset).type(torch.FloatTensor)
        self.dataset = self.dataset.unsqueeze(-1)

        latent_extractor = RecurrentAutoencoder(seq_len=self.eval_length, n_features=1, embedding_dim=16)
        print(latent_extractor)
        latent_extractor = latent_extractor.to(device)

        pretrained_loc = os.path.join(config["train"]["dataset_dir"], 'results/weights/best_model.pth')
        if os.path.exists(pretrained_loc):
            latent_extractor.load_state_dict(torch.load(pretrained_loc))
        print("loaded pretrained weights")

        self.latent_mean_dataset = torch.zeros((self.dataset.shape[0], self.embedding_dim))
        self.latent_var_dataset = torch.zeros((self.dataset.shape[0], self.embedding_dim))
        for i in range(self.dataset.shape[0]):
            print(i)
            input = self.dataset[i].unsqueeze(0).to(device)
            with torch.no_grad():
                latent_mean, latent_var = latent_extractor.encoder(input)
                self.latent_mean_dataset[i] = latent_mean 
                self.latent_var_dataset[i] = latent_var 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TS-Autoencoder") 
    parser.add_argument("--config", type=str, default="diffusion.yaml")

    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path = "config/" + args.config 
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    dataset_obj = Sinusoidal(config, device)
    mean_dataset = dataset_obj.latent_mean_dataset
    var_dataset = dataset_obj.latent_var_dataset

    mean_dataset_save_path = os.path.join(config["train"]["dataset_dir"], 'mean.pth')
    torch.save(mean_dataset, mean_dataset_save_path)
    var_dataset_save_path = os.path.join(config["train"]["dataset_dir"], 'var.pth')
    torch.save(var_dataset, var_dataset_save_path)

