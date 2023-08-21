'''
This follows the DDPM implementation
https://arxiv.org/pdf/2006.11239.pdf
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_models import Denoiser


class LDM(nn.Module):
    def __init__(self, config, device): 
        super().__init__()
        self.device = device
        self.config = config
 
        self.pos_embedding_dim = self.config["model"]["pos_embedding_dim"] # 128 dim vector representing each position of the latent vector
        self.latent_dim = self.config["model"]["latent_dim"] # 16 dim latent vector
        self.num_latent_fetures = self.config["model"]["num_latent_features"] # always 1
        self.horizon = self.latent_dim # same a latent dim

        self.config["model"]["side_dim"] = self.pos_embedding_dim 
        self.diffmodel = Denoiser(self.config) 

        # parameters for diffusion models
        self.num_steps = self.config["diffusion"]["num_steps"] # 50 steps
        self.beta = np.linspace(self.config["diffusion"]["beta_start"], self.config["diffusion"]["beta_end"], self.config["diffusion"]["num_steps"]) # same beta as in DDPM paper
        self.alpha_hat = 1 - self.beta # alpha_hat here is alpha in the paper
        self.alpha = np.cumprod(self.alpha_hat) # alpha here is alpha bar in the paper
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1) # alpha torch is also equivalent to alpha bar in the paper

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_side_info(self, time_points):
        # contains the embedding representing each position of the latent vector
        time_embed = self.time_embedding(time_points, self.pos_embedding_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2) # (B, L, 1, emb)
        side_info = time_embed.permute(0,3,2,1)

        return side_info.to(self.device)

    def calc_loss(self, data, label, side_info):
        B = data.shape[0] # B is the batch size, K is the number of features for each time step, L is the total no of time steps
        t = torch.randint(0, self.num_steps, [B]).to(self.device) # randomly sampling noise levels for each element in the batch 
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(data).to(self.device).float()
        noisy_data = (current_alpha ** 0.5) * data + (1.0 - current_alpha) ** 0.5 * noise
        predicted = self.diffmodel(noisy_input=noisy_data, cond_input=label, side_info=side_info, diffusion_step=t)  # (B,K,L)
        loss = F.l1_loss(predicted, noise)
        return loss

    def synthesize(self, label):
        B = label.shape[0]
        timepoints = torch.arange(self.latent_dim).unsqueeze(0).repeat(B, 1).to(self.device).float()
        side_info = self.get_side_info(timepoints)
        synthesized_output = torch.randn((B,self.num_latent_fetures,self.horizon)).to(self.device) # step 1
        label = label.to(self.device)
        for tidx in reversed(range(self.num_steps)):    
            noisy_data = synthesized_output        
            t = torch.ones(B).int()*tidx
            t = t.to(self.device)
            predicted_noise = self.diffmodel(noisy_input=noisy_data, cond_input=label, side_info=side_info, diffusion_step=t)
            coeff1 = 1 / (self.alpha_hat[tidx] ** 0.5)
            coeff2 = (1 - self.alpha_hat[tidx]) / ((1 - self.alpha[tidx]) ** 0.5)
            synthesized_output = coeff1 * (synthesized_output - coeff2 * predicted_noise)
            if tidx > 0:
                noise = torch.randn_like(synthesized_output).to(self.device)
                sigma = ((1.0 - self.alpha[tidx - 1]) / (1.0 - self.alpha[tidx]) * self.beta[tidx]) ** 0.5
                synthesized_output += sigma * noise
        return synthesized_output
    
    def synthesize_for_visualization(self, label):
        B = label.shape[0]
        timepoints = torch.arange(self.latent_dim).unsqueeze(0).repeat(B, 1).to(self.device).float()
        side_info = self.get_side_info(timepoints)
        synthesized_output = torch.randn((B,self.num_latent_fetures,self.horizon)).to(self.device) # step 1
        label = label.to(self.device)

        intermediate_outputs = []
        for tidx in reversed(range(self.num_steps)):    
            noisy_data = synthesized_output        
            t = torch.ones(B).int()*tidx
            t = t.to(self.device)
            predicted_noise = self.diffmodel(noisy_input=noisy_data, cond_input=label, side_info=side_info, diffusion_step=t)
            coeff1 = 1 / (self.alpha_hat[tidx] ** 0.5)
            coeff2 = (1 - self.alpha_hat[tidx]) / ((1 - self.alpha[tidx]) ** 0.5)
            synthesized_output = coeff1 * (synthesized_output - coeff2 * predicted_noise)
            if tidx > 0:
                noise = torch.randn_like(synthesized_output).to(self.device)
                sigma = ((1.0 - self.alpha[tidx - 1]) / (1.0 - self.alpha[tidx]) * self.beta[tidx]) ** 0.5
                synthesized_output += sigma * noise
            
            intermediate_outputs.append(synthesized_output)
        return intermediate_outputs

    def forward(self, data, label):
        data = data.to(self.device).float()
        label = label.to(self.device).float()
        tp = torch.arange(self.latent_dim).unsqueeze(0).repeat(data.shape[0],1).to(self.device).float()
        side_info = self.get_side_info(tp)
        loss_func = self.calc_loss
        return loss_func(data, label, side_info)