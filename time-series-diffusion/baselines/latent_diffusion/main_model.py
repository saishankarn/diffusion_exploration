import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_models import diff_CSDI


class LDM_base(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config
 
        self.time_embedding_dim = self.config["model"]["time_embedding_dim"] # 128 
        self.latent_dim = self.config["model"]["latent_dim"] # 128
        # self.latent_feature_size = self.config["model"]["latent_feat_size"] # 1
        # self.latent_feature_dim = self.config["model"]["latent_feat_dim"] # 16

        self.total_embedding_dim = self.time_embedding_dim #+ self.latent_feature_dim # 144
        # self.latent_feature_embedding_layer = nn.Embedding(num_embeddings=self.latent_feature_size, embedding_dim=self.latent_feature_dim).to(self.device)

        self.config["model"]["side_dim"] = self.time_embedding_dim #+ self.latent_feature_dim
        self.config["model"]["input_dim"] = 1
        self.diffmodel = diff_CSDI(self.config)

        # parameters for diffusion models
        self.num_steps = self.config["diffusion"]["num_steps"]
        self.beta = np.linspace(self.config["diffusion"]["beta_start"], self.config["diffusion"]["beta_end"], self.config["diffusion"]["num_steps"])
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

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
        # contains the embedding for diffusion time step
        B = time_points.shape[0]
        time_embed = self.time_embedding(time_points, self.time_embedding_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2) # (B, L, 1, emb)
        # feature_embed = self.latent_feature_embedding_layer(torch.arange(self.latent_feature_size).to(self.device)) # (K,emb)
        # feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        # side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        # print(side_info.shape)
        # side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)
        # print(side_info.shape)
        side_info = time_embed.permute(0,3,2,1)

        return side_info.to(self.device)

    def calc_loss(self, batch, side_info):
        batch = batch.unsqueeze(1)
        B, K, L = batch.shape # B is the batch size, K is the number of features for each time step, L is the total no of time steps
        t = torch.randint(0, self.num_steps, [B]).to(self.device) # randomly sampling noise levels for each element in the batch 
        current_alpha = self.alpha_torch[t]  # (B,1,1)

        noise = torch.randn_like(batch).to(self.device).float()
        noisy_data = (current_alpha ** 0.5) * batch + (1.0 - current_alpha) ** 0.5 * noise
        total_input = noisy_data.unsqueeze(1) # process the input before forward pass to the neural network
        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)
        # print(predicted.shape, observed_data.shape)
        # print(predicted.shape, noise.shape)
        loss = F.mse_loss(predicted, noise)
        return loss

    def synthesize(self):
        B = self.config["train"]["num_plots"]
        L = self.latent_dim

        timepoints = torch.arange(self.latent_dim).unsqueeze(0).repeat(B, 1).to(self.device).float()
        side_info = self.get_side_info(timepoints)
        synthesized_output = torch.randn((B,1,L)).to(self.device) # step 1

        for tidx in reversed(range(self.num_steps)):            
            input = synthesized_output.unsqueeze(1)
            predicted_noise = self.diffmodel(input, side_info, torch.tensor([tidx]).to(self.device))
            coeff1 = 1 / self.alpha_hat[tidx] ** 0.5
            coeff2 = (1 - self.alpha_hat[tidx]) / (1 - self.alpha[tidx]) ** 0.5
            synthesized_output = coeff1 * (synthesized_output - coeff2 * predicted_noise)
            if tidx > 0:
                noise = torch.randn_like(synthesized_output).to(self.device)
                sigma = ((1.0 - self.alpha[tidx - 1]) / (1.0 - self.alpha[tidx]) * self.beta[tidx]) ** 0.5
                synthesized_output += sigma * noise

        return synthesized_output

    def forward(self, batch):
        batch = batch.to(self.device).float()
        tp = torch.arange(self.latent_dim).unsqueeze(0).repeat(batch.shape[0],1).to(self.device).float()
        side_info = self.get_side_info(tp)
        loss_func = self.calc_loss
        return loss_func(batch, side_info)

class LDM(LDM_base):
    def __init__(self, config, device):
        super(LDM, self).__init__(config, device)