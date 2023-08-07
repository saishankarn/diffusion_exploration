import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import diff_LDM

from diffusers import UNet1DModel


class LDMUNet(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config

        self.diffmodel = UNet1DModel(sample_size=config["model"]["latent_dim"], 
                                in_channels=config["model"]["latent_channels"]+2*config["model"]["time_embedding_dim"], 
                                out_channels=config["model"]["latent_channels"],
                                timestep_embedding_size=config["model"]["time_embedding_dim"]).to(device)    
        
        self.num_steps = config["diffusion"]["num_steps"]
        self.beta = np.linspace(config["diffusion"]["beta_start"], config["diffusion"]["beta_end"], config["diffusion"]["num_steps"])
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def calc_loss(self, observed_data):
        B, _, _ = observed_data.shape # B is the batch size
        t = torch.randint(0, self.num_steps, [B]).to(self.device) # randomly sampling noise levels for each element in the batch 
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        predicted_noise = self.diffmodel(noisy_data, t)["sample"]  # (B,K,L)
        loss = F.mse_loss(predicted_noise, noise)
        return loss

    def synthesize(self):
        B = self.config["train"]["num_plots"]
        L = self.config["model"]["latent_dim"]
        K = self.config["model"]["latent_channels"] 
        synthesized_output = torch.randn((B,K,L)).to(self.device) # step 1

        for tidx in reversed(range(self.num_steps)):            
            t = torch.tensor([tidx]).to(self.device)
            predicted_noise = self.diffmodel(synthesized_output, t)["sample"]

            coeff1 = 1 / self.alpha_hat[tidx] ** 0.5
            coeff2 = (1 - self.alpha_hat[tidx]) / (1 - self.alpha[tidx]) ** 0.5

            synthesized_output = coeff1 * (synthesized_output - coeff2 * predicted_noise)

            if tidx > 0:
                noise = torch.randn_like(synthesized_output)
                sigma = ((1.0 - self.alpha[tidx - 1]) / (1.0 - self.alpha[tidx]) * self.beta[tidx]) ** 0.5
                synthesized_output += sigma * noise

        return synthesized_output

    def forward(self, batch):
        observed_data = batch.to(self.device).float()
        loss_func = self.calc_loss
        return loss_func(observed_data)

class LDM(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.latent_dim = config["model"]["latent_dim"]
        self.latent_channels = config["model"]["latent_channels"] 
        self.emb_time_dim = config["model"]["time_embedding_dim"] # 128 

        self.diffmodel = diff_LDM(config)
        self.diffmodel.to(self.device)

        # parameters for diffusion models
        self.num_steps = config["diffusion"]["num_steps"]
        if config["diffusion"]["schedule"] == "quad":
            self.beta = np.linspace(
                config["diffusion"]["beta_start"] ** 0.5, config["diffusion"]["beta_end"] ** 0.5, config["diffusion"]["num_steps"]
            ) ** 2
        elif config["diffusion"]["schedule"] == "linear":
            self.beta = np.linspace(
                config["diffusion"]["beta_start"], config["diffusion"]["beta_end"], config["diffusion"]["num_steps"]
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)


    def calc_loss(self, observed_data):
        B, K, L = observed_data.shape # B is the batch size, K is the number of features for each time step, L is the total no of time steps
        t = torch.randint(0, self.num_steps, [B]).to(self.device) # randomly sampling noise levels for each element in the batch 
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = noisy_data.unsqueeze(1) # process the input before forward pass to the neural network
        predicted = self.diffmodel(total_input, t)  # (B,K,L)
        loss = F.mse_loss(predicted, noise)
        return loss

    def synthesize(self):
        B = self.config["train"]["num_plots"]
        L = self.config["model"]["latent_dim"]
        K = self.config["model"]["latent_channels"] 
        synthesized_output = torch.randn((B,K,L)).to(self.device) # step 1

        for tidx in reversed(range(self.num_steps)):            
            input = synthesized_output.unsqueeze(1)
            t = torch.tensor([tidx]).to(self.device)
            predicted_noise = self.diffmodel(input, t)

            coeff1 = 1 / self.alpha_hat[tidx] ** 0.5
            coeff2 = (1 - self.alpha_hat[tidx]) / (1 - self.alpha[tidx]) ** 0.5

            synthesized_output = coeff1 * (synthesized_output - coeff2 * predicted_noise)

            if tidx > 0:
                noise = torch.randn_like(synthesized_output)
                sigma = ((1.0 - self.alpha[tidx - 1]) / (1.0 - self.alpha[tidx]) * self.beta[tidx]) ** 0.5
                synthesized_output += sigma * noise

        return synthesized_output

    def forward(self, batch, is_train=1):
        observed_data = batch.to(self.device).float()

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data)